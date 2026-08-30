from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from media_knowledge.config import AppConfig
from media_knowledge.documents import document_from_text
from media_knowledge.embedding import HashEmbeddingProvider
from media_knowledge.indexing import IndexingService
from media_knowledge.ingestion import IngestionService
from media_knowledge.ingestion.quality import QualityReport
from media_knowledge.ingestion.types import ExtractionResult
from media_knowledge.models import ContentSegment, KnowledgeDocument, SourceReference
from media_knowledge.product import DesktopSettings, ProductPaths
from media_knowledge.storage import KnowledgeDatabase, SQLiteVectorStore
from media_knowledge.transcripts import (
    TranscriptQuality,
    TranscriptRepository,
    TranscriptRun,
    TranscriptSegment,
    TranscriptSource,
    TranscriptV2,
)


class CountingEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dimensions=64, model="counting-hash")
        self.batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return super().embed(texts)


class FailingEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dimensions=64, model="failing-hash")

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated embedding outage")


class FailingSQLiteVectorStore(SQLiteVectorStore):
    """Write one vector, then fail so the surrounding transaction must undo it."""

    def upsert(
        self,
        chunk_id: str,
        vector: Sequence[float],
        *,
        provider: str,
        model: str,
        content_hash: str,
    ) -> None:
        super().upsert(
            chunk_id,
            vector,
            provider=provider,
            model=model,
            content_hash=content_hash,
        )
        raise RuntimeError("simulated vector-store failure")


class IndexingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.database = KnowledgeDatabase(Path(self.temp.name) / "knowledge.db")
        self.embedding = CountingEmbeddingProvider()
        self.service = IndexingService(self.database, self.embedding)

    def tearDown(self) -> None:
        self.database.close()
        self.temp.cleanup()

    def _index_snapshot(self) -> dict[str, list[tuple[object, ...]]]:
        queries = {
            "documents": "SELECT * FROM documents ORDER BY id",
            "knowledge_items": "SELECT * FROM knowledge_items ORDER BY id",
            "chunks": "SELECT * FROM chunks ORDER BY id",
            "source_references": "SELECT * FROM source_references ORDER BY chunk_id",
            "collections": "SELECT * FROM collections ORDER BY id",
            "document_collections": (
                "SELECT * FROM document_collections ORDER BY document_id, collection_id"
            ),
            "tags": "SELECT * FROM tags ORDER BY id",
            "document_tags": "SELECT * FROM document_tags ORDER BY document_id, tag_id",
            "embeddings": "SELECT * FROM embeddings ORDER BY chunk_id",
            "chunks_fts": (
                "SELECT chunk_id, document_id, title, content FROM chunks_fts ORDER BY chunk_id"
            ),
        }
        return {
            name: [tuple(row) for row in self.database.connection.execute(statement).fetchall()]
            for name, statement in queries.items()
        }

    def test_duplicate_import_does_not_add_document_or_chunks(self) -> None:
        first = document_from_text("same durable knowledge", title="A", source_id="source-a")
        second = document_from_text("same durable knowledge", title="B", source_id="source-b")
        created = self.service.index_document(first)
        duplicate = self.service.index_document(second)
        self.assertEqual(created.status, "created")
        self.assertEqual(duplicate.status, "duplicate")
        self.assertEqual(duplicate.duplicate_of, "source-a")
        self.assertEqual(self.database.status()["documents"], 1)
        self.assertEqual(self.database.status()["chunks"], 1)
        self.assertEqual(self.database.status()["source_references"], 1)

    def test_content_hash_deduplicates_reextraction_with_different_segment_ids(self) -> None:
        first_source = SourceReference(source_id="extract-a", media_type="pdf", title="A")
        second_source = SourceReference(source_id="extract-b", media_type="pdf", title="B")
        first = KnowledgeDocument(
            source_id="extract-a",
            title="A",
            media_type="pdf",
            source=first_source,
            segments=[ContentSegment("old-id", 1, "text", text="Stable page content", location={"page": 1})],
        )
        second = KnowledgeDocument(
            source_id="extract-b",
            title="B",
            media_type="pdf",
            source=second_source,
            segments=[ContentSegment("new-id", 1, "text", text="Stable page content", location={"page": 1})],
        )
        self.service.index_document(first)
        report = self.service.index_document(second)
        self.assertEqual(report.status, "duplicate")
        self.assertEqual(self.database.status()["documents"], 1)

    @staticmethod
    def _incremental_document(second_text: str) -> KnowledgeDocument:
        source = SourceReference(source_id="source-incremental", media_type="document", title="Incremental")
        return KnowledgeDocument(
            source_id=source.source_id,
            title=source.title,
            media_type="document",
            source=source,
            segments=[
                ContentSegment("stable-a", 1, "text", text="This segment stays unchanged."),
                ContentSegment("stable-b", 2, "text", text=second_text),
            ],
        )

    def test_incremental_update_embeds_only_changed_chunk(self) -> None:
        first = self.service.index_document(self._incremental_document("Old content."))
        self.embedding.batches.clear()
        second = self.service.index_document(self._incremental_document("New content."))
        self.assertEqual(first.embedded_chunks, 2)
        self.assertEqual(second.status, "updated")
        self.assertEqual(second.updated_chunks, 1)
        self.assertEqual(second.unchanged_chunks, 1)
        self.assertEqual(second.embedded_chunks, 1)
        self.assertEqual(self.embedding.batches, [["New content."]])

    def test_failed_embedding_leaves_new_document_completely_absent(self) -> None:
        document = document_from_text(
            "New content that must not become partially searchable.",
            title="Atomic create",
            source_id="atomic-create",
            collections=["New collection"],
            tags=["New tag"],
        )

        with self.assertRaisesRegex(RuntimeError, "embedding outage"):
            IndexingService(self.database, FailingEmbeddingProvider()).index_document(document)

        snapshot = self._index_snapshot()
        self.assertEqual(snapshot, {name: [] for name in snapshot})

    def test_failed_embedding_preserves_complete_previous_document_version(self) -> None:
        original = document_from_text(
            "The original durable evidence remains available.",
            title="Original title",
            source_id="atomic-update",
            collections=["Original collection"],
            tags=["Original tag"],
        )
        self.service.index_document(original)
        before = self._index_snapshot()
        replacement = document_from_text(
            "Replacement content must never appear after a failed embedding.",
            title="Replacement title",
            source_id="atomic-update",
            collections=["Replacement collection"],
            tags=["Replacement tag"],
        )

        with self.assertRaisesRegex(RuntimeError, "embedding outage"):
            IndexingService(self.database, FailingEmbeddingProvider()).index_document(replacement)

        self.assertEqual(self._index_snapshot(), before)

    def test_sqlite_write_failure_rolls_back_document_fts_facets_and_embedding(self) -> None:
        document = document_from_text(
            "A transaction failure must roll back every searchable representation.",
            title="Transactional create",
            source_id="transaction-create",
            collections=["Transactional collection"],
            tags=["Transactional tag"],
        )
        vector_store = FailingSQLiteVectorStore(
            self.database,
            provider=self.embedding.name,
            model=self.embedding.model,
        )

        with self.assertRaisesRegex(RuntimeError, "vector-store failure"):
            IndexingService(
                self.database,
                self.embedding,
                vector_store=vector_store,
            ).index_document(document)

        snapshot = self._index_snapshot()
        self.assertEqual(snapshot, {name: [] for name in snapshot})

    def test_sqlite_write_failure_restores_complete_previous_document_version(self) -> None:
        original = document_from_text(
            "The committed version must survive a later transaction failure.",
            title="Committed title",
            source_id="transaction-update",
            collections=["Committed collection"],
            tags=["Committed tag"],
        )
        self.service.index_document(original)
        before = self._index_snapshot()
        replacement = document_from_text(
            "This replacement must be rolled back with its FTS row and vector.",
            title="Uncommitted title",
            source_id="transaction-update",
            collections=["Uncommitted collection"],
            tags=["Uncommitted tag"],
        )
        vector_store = FailingSQLiteVectorStore(
            self.database,
            provider=self.embedding.name,
            model=self.embedding.model,
        )

        with self.assertRaisesRegex(RuntimeError, "vector-store failure"):
            IndexingService(
                self.database,
                self.embedding,
                vector_store=vector_store,
            ).index_document(replacement)

        self.assertEqual(self._index_snapshot(), before)

    def test_delete_cascades_chunks_embeddings_facets_and_fts(self) -> None:
        document = document_from_text(
            "cascade deletion evidence", title="Delete", source_id="delete-me", collections=["C"], tags=["T"]
        )
        report = self.service.index_document(document)
        self.assertTrue(self.service.delete_document(report.document_id))
        status = self.database.status()
        self.assertEqual(status["documents"], 0)
        self.assertEqual(status["chunks"], 0)
        self.assertEqual(status["source_references"], 0)
        self.assertEqual(status["embeddings"], 0)
        self.assertEqual(self.database.connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0], 0)
        self.assertEqual(
            self.database.connection.execute("SELECT COUNT(*) FROM document_collections").fetchone()[0], 0
        )
        self.assertEqual(self.database.connection.execute("SELECT COUNT(*) FROM document_tags").fetchone()[0], 0)

    def test_identical_source_is_unchanged(self) -> None:
        document = document_from_text("unchanged", title="Same", source_id="same-source")
        self.service.index_document(document)
        self.embedding.batches.clear()
        report = self.service.index_document(document)
        self.assertEqual(report.status, "unchanged")
        self.assertEqual(self.embedding.batches, [])

    def test_same_content_with_changed_locator_or_facets_updates_document_state(self) -> None:
        original = document_from_text(
            "stable content",
            title="Same source",
            source_id="relocated-source",
            collections=["Old collection"],
            tags=["Old tag"],
        )
        original.source.local_path = "/evidence/old.txt"
        self.service.index_document(original)
        self.embedding.batches.clear()

        relocated = document_from_text(
            "stable content",
            title="Same source",
            source_id="relocated-source",
            collections=["New collection"],
            tags=["New tag"],
        )
        relocated.source.local_path = "/evidence/repaired.txt"
        report = self.service.index_document(relocated)

        self.assertEqual(report.status, "updated")
        self.assertEqual(self.embedding.batches, [])
        stored = self.database.get_document_by_source_id("relocated-source")
        self.assertEqual(stored["local_path"], "/evidence/repaired.txt")
        self.assertEqual(
            self.database.document_facets(stored["id"]),
            {"collections": ["New collection"], "tags": ["New tag"]},
        )

    def test_real_ingestion_does_not_leave_database_pointing_to_rolled_back_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "atomic-ingestion.md"
            source.write_text(
                "# 原子入库\n\nEmbedding 服务失败时，数据库不能保留指向已回滚归档证据的记录。" * 4,
                encoding="utf-8",
            )
            paths = ProductPaths.resolve(root / "data")
            settings = DesktopSettings(
                archive_originals=True,
                create_source_notes=False,
                auto_synthesize_notes=False,
                enable_cloud_vision=False,
                embedding_provider="hash",
                embedding_model="hash-384-v1",
            )
            config = AppConfig(database_path=paths.database)

            with patch(
                "media_knowledge.ingestion.service.build_embedding_provider",
                return_value=FailingEmbeddingProvider(),
            ):
                summary = IngestionService(paths, config=config, settings=settings).ingest([source])

            self.assertEqual(summary.failed, 1)
            self.assertIn("embedding outage", summary.results[0].error or "")
            with KnowledgeDatabase(paths.database) as database:
                self.assertEqual(database.status()["documents"], 0)
                dangling = database.connection.execute(
                    "SELECT id, local_path FROM documents WHERE local_path IS NOT NULL"
                ).fetchall()
                self.assertEqual(dangling, [])
                fts_count = database.connection.execute(
                    "SELECT COUNT(*) FROM chunks_fts"
                ).fetchone()[0]
                self.assertEqual(fts_count, 0)
            evidence_files = [
                path
                for path in paths.archive.rglob("*")
                if path.is_file() and "source-packages" not in path.parts
            ]
            self.assertEqual(evidence_files, [])

    def test_review_transcript_persists_facts_without_fts_or_vectors(self) -> None:
        embedding = CountingEmbeddingProvider()

        class TranscriptExtractor:
            @staticmethod
            def extract(path: Path, _context) -> ExtractionResult:
                transcript = TranscriptV2(
                    source=TranscriptSource(path.name, "review-checksum", 3000),
                    run=TranscriptRun("review-run", "accuracy", "test", "tiny"),
                    speakers=[],
                    segments=[TranscriptSegment(
                        "review-segment", 0, 0, 3000, None,
                        "这是一段需要人工复核、但必须保留事实层的转写内容。",
                    )],
                    quality=TranscriptQuality("review", ("需要人工复核",)),
                )
                return ExtractionResult(
                    title="待复核录音",
                    media_type="audio",
                    segments=[ContentSegment(
                        "review-segment", 0, "speech",
                        text="这是一段需要人工复核、但必须保留事实层的转写内容。",
                        location={"timestamp_start": 0, "timestamp_end": 3},
                    )],
                    source_path=path,
                    checksum="review-checksum",
                    transcript_data=transcript.to_dict(),
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "review.audio"
            source.write_bytes(b"source evidence")
            paths = ProductPaths.resolve(root / "data")
            settings = DesktopSettings(
                archive_originals=False,
                create_source_notes=False,
                auto_synthesize_notes=False,
                enable_cloud_vision=False,
                transcript_quality_gate=True,
            )
            with patch(
                "media_knowledge.ingestion.service.extractor_for",
                return_value=TranscriptExtractor(),
            ), patch(
                "media_knowledge.ingestion.service.evaluate_extraction",
                return_value=QualityReport(True, 100, "A", []),
            ), patch(
                "media_knowledge.ingestion.service.build_embedding_provider",
                return_value=embedding,
            ):
                summary = IngestionService(paths, settings=settings).ingest([source])

            result = summary.results[0]
            self.assertEqual((result.status, result.chunks), ("created", 0))
            self.assertEqual(result.transcript_run_id, "review-run")
            self.assertEqual(embedding.batches, [])
            with KnowledgeDatabase(paths.database) as database:
                document = database.get_document(result.document_id or "")
                self.assertIsNotNone(document)
                self.assertFalse(bool(document["enabled"]))
                self.assertEqual(database.get_chunks(result.document_id or ""), {})
                self.assertEqual(
                    database.connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0],
                    0,
                )
                self.assertEqual(
                    database.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0],
                    0,
                )
                self.assertIsNotNone(TranscriptRepository(database).get_run("review-run"))


if __name__ == "__main__":
    unittest.main()
