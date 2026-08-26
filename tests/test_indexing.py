from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from media_knowledge.documents import document_from_text
from media_knowledge.embedding import HashEmbeddingProvider
from media_knowledge.indexing import IndexingService
from media_knowledge.models import ContentSegment, KnowledgeDocument, SourceReference
from media_knowledge.storage import KnowledgeDatabase


class CountingEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dimensions=64, model="counting-hash")
        self.batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return super().embed(texts)


class IndexingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.database = KnowledgeDatabase(Path(self.temp.name) / "knowledge.db")
        self.embedding = CountingEmbeddingProvider()
        self.service = IndexingService(self.database, self.embedding)

    def tearDown(self) -> None:
        self.database.close()
        self.temp.cleanup()

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


if __name__ == "__main__":
    unittest.main()
