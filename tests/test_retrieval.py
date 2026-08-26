from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from media_knowledge.documents import documents_from_ucb
from media_knowledge.embedding import HashEmbeddingProvider
from media_knowledge.indexing import IndexingService
from media_knowledge.rerank import LocalLexicalRerankProvider
from media_knowledge.retrieval import KnowledgeRetriever
from media_knowledge.storage import KnowledgeDatabase


def sample_bundle() -> dict:
    return {
        "schema_version": "1.0",
        "bundle_id": "bundle-test",
        "created_at": "2026-08-22T00:00:00Z",
        "sources": [
            {
                "id": "src-pdf",
                "kind": "pdf",
                "title": "FAST-LIVO2 Paper",
                "origin": {"uri": "https://example.test/paper.pdf", "local_path": "/archive/paper.pdf", "sha256": "pdf-checksum"},
                "metadata": {"collections": ["SLAM"], "tags": ["fusion"], "obsidian_path": "90_Sources/FAST-LIVO2.md"},
                "extraction": {"status": "complete", "methods": ["text", "render"], "limitations": []},
            },
            {
                "id": "src-video",
                "kind": "video",
                "title": "Cooking lesson",
                "origin": {"uri": None, "local_path": "/archive/cooking.mp4", "sha256": "video-checksum"},
                "metadata": {"collections": ["Cooking"], "tags": ["food"]},
                "extraction": {"status": "complete", "methods": ["asr", "frames"], "limitations": []},
            },
        ],
        "content": [
            {
                "id": "pdf-p6",
                "source_id": "src-pdf",
                "sequence": 6,
                "modality": "text",
                "text": "FAST-LIVO2 fuses LiDAR IMU and camera measurements for robust odometry.",
                "location": {"page": 6, "section": "System Architecture"},
            },
            {
                "id": "video-1",
                "source_id": "src-video",
                "sequence": 1,
                "modality": "speech",
                "text": "Add flour and water to prepare the dough.",
                "location": {"timestamp_start": 12.0, "timestamp_end": 24.0},
            },
        ],
    }


class HybridRetrievalIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.database = KnowledgeDatabase(Path(self.temp.name) / "knowledge.db")
        self.embedding = HashEmbeddingProvider(dimensions=128, model="test-hash")
        indexing = IndexingService(self.database, self.embedding)
        for document in documents_from_ucb(sample_bundle()):
            indexing.index_document(document)
        self.retriever = KnowledgeRetriever(
            self.database, self.embedding, rerank_provider=LocalLexicalRerankProvider()
        )

    def tearDown(self) -> None:
        self.database.close()
        self.temp.cleanup()

    def test_hybrid_search_returns_exact_pdf_provenance(self) -> None:
        results = self.retriever.search_knowledge("FAST-LIVO2 camera fusion", top_k=5)
        self.assertTrue(results)
        first = results[0]
        self.assertEqual(first.source.source_id, "src-pdf")
        self.assertEqual(first.page, 6)
        self.assertEqual(first.source.obsidian_path, "90_Sources/FAST-LIVO2.md")
        self.assertTrue(first.document_id)
        self.assertTrue(first.chunk_id)

    def test_collection_tag_and_media_filters_are_applied(self) -> None:
        self.assertEqual(
            self.retriever.search_knowledge("dough", collections=["SLAM"], top_k=5), []
        )
        results = self.retriever.search_knowledge(
            "dough flour", collections=["Cooking"], tags=["food"], media_types=["video"], top_k=5
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].timestamp_start, 12.0)
        self.assertEqual(results[0].timestamp_end, 24.0)

    def test_reindex_rebuilds_both_indexes(self) -> None:
        report = IndexingService(self.database, self.embedding).reindex()
        self.assertEqual(report, {"embedded_chunks": 2, "fts_rows": 2})


if __name__ == "__main__":
    unittest.main()
