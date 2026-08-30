from __future__ import annotations

import unittest

from media_knowledge.providers.web import WebSearchHit, WebSearchProvider
from media_knowledge.transcripts.evidence import collect_external_evidence, plan_external_queries
from media_knowledge.transcripts.schema import (
    TranscriptQuality,
    TranscriptRun,
    TranscriptSegment,
    TranscriptSource,
    TranscriptV2,
)


class _Search(WebSearchProvider):
    name = "fake"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []

    @property
    def available(self) -> bool:
        return True

    def search(self, query: str, top_k: int = 5) -> list[WebSearchHit]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("offline")
        return [WebSearchHit("Obsidian 官方帮助", "Obsidian 是一款知识管理工具。", "https://example.test/obsidian")]


def _transcript() -> TranscriptV2:
    return TranscriptV2(
        TranscriptSource("课程.wav", "a" * 64, 2000),
        TranscriptRun("run", "accuracy", "qwen3-mlx", "Qwen3-ASR", "zh"),
        [],
        [
            TranscriptSegment("s1", 0, 0, 1000, None, "这里使用 Obsidian 和 RAG 系统。"),
            TranscriptSegment("s2", 1, 1000, 2000, None, "检索召回率是 96.6%。"),
        ],
        TranscriptQuality("pass"),
    )


class ExternalEvidenceCollectionTests(unittest.TestCase):
    def test_queries_are_bounded_and_do_not_upload_whole_transcript(self) -> None:
        queries = plan_external_queries(_transcript(), max_queries=2)
        self.assertEqual(len(queries), 2)
        self.assertTrue(any("Obsidian" in item or "RAG" in item for item in queries))
        self.assertTrue(all(len(item) <= 160 for item in queries))
        self.assertTrue(all("这里使用" not in item for item in queries))

    def test_collects_deduplicated_traceable_hits(self) -> None:
        provider = _Search()
        result = collect_external_evidence(_transcript(), provider, max_queries=3)
        self.assertEqual(len(result.evidence), 1)
        item = result.evidence[0]
        self.assertTrue(item.id.startswith("web-"))
        self.assertEqual(item.url, "https://example.test/obsidian")
        self.assertIn(item.query, provider.queries)

    def test_network_failure_is_a_warning_not_fabricated_evidence(self) -> None:
        result = collect_external_evidence(_transcript(), _Search(fail=True), max_queries=1)
        self.assertEqual(result.evidence, ())
        self.assertTrue(result.warnings)


if __name__ == "__main__":
    unittest.main()
