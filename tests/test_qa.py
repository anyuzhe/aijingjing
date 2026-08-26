from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from media_knowledge.citations import CitationValidationError, CitationValidator
from media_knowledge.documents import documents_from_ucb
from media_knowledge.embedding import HashEmbeddingProvider
from media_knowledge.indexing import IndexingService
from media_knowledge.models import SourceReference
from media_knowledge.providers import AnswerProvider, CodexAnswerProvider, WebSearchHit, WebSearchProvider
from media_knowledge.qa.engine import KnowledgeQAEngine
from media_knowledge.qa.models import AnswerRequest, AnswerResponse, Evidence, TokenUsage
from media_knowledge.rerank import LocalLexicalRerankProvider
from media_knowledge.retrieval import KnowledgeRetriever
from media_knowledge.storage import ConversationRepository, KnowledgeDatabase


def qa_bundle() -> dict:
    return {
        "schema_version": "1.0",
        "bundle_id": "qa-bundle",
        "created_at": "2026-08-22T00:00:00Z",
        "sources": [
            {
                "id": "src-fast-livo2",
                "kind": "pdf",
                "title": "FAST-LIVO2 Synchronization Guide",
                "origin": {"uri": None, "local_path": "/archive/fast-livo2.pdf", "sha256": "fast-livo2-checksum"},
                "metadata": {"collections": ["SLAM"], "tags": ["synchronization"], "obsidian_path": "90_Sources/FAST-LIVO2.md"},
                "extraction": {"status": "complete", "methods": ["text", "render"], "limitations": []},
            }
        ],
        "content": [
            {
                "id": "fast-page-12",
                "source_id": "src-fast-livo2",
                "sequence": 12,
                "modality": "text",
                "text": "FAST-LIVO2 requires hardware time synchronization between the LiDAR and camera for reliable sensor fusion.",
                "location": {"page": 12, "section": "Time synchronization"},
            },
            {
                "id": "fast-page-16",
                "source_id": "src-fast-livo2",
                "sequence": 16,
                "modality": "image",
                "description": "The architecture routes synchronized LiDAR, IMU and camera measurements into the estimator.",
                "location": {"page": 16, "slide": 16, "section": "Architecture"},
                "asset": "slides/slide-016.png",
            },
        ],
    }


class EmptyRetriever:
    def search_knowledge(self, *args, **kwargs):
        return []


class InvalidCitationProvider(AnswerProvider):
    name = "invalid-test"
    model = "invalid-test"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: AnswerRequest) -> AnswerResponse:
        self.calls += 1
        return AnswerResponse("It requires synchronization. [S8]", self.model, self.name, TokenUsage())


class FakeWebProvider(WebSearchProvider):
    name = "fake-web"

    @property
    def available(self) -> bool:
        return True

    def search(self, query: str, top_k: int = 5) -> list[WebSearchHit]:
        return [
            WebSearchHit(
                title="External synchronization note",
                content="The external note says hardware synchronization is required.",
                url="https://example.test/sync",
                score=0.9,
            )
        ]


class KnowledgeQAIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.database = KnowledgeDatabase(Path(self.temp.name) / "qa.db")
        self.embedding = HashEmbeddingProvider(dimensions=128, model="qa-test-hash")
        indexing = IndexingService(self.database, self.embedding)
        for document in documents_from_ucb(qa_bundle()):
            indexing.index_document(document)
        self.retriever = KnowledgeRetriever(
            self.database, self.embedding, rerank_provider=LocalLexicalRerankProvider()
        )

    def tearDown(self) -> None:
        self.database.close()
        self.temp.cleanup()

    def test_answer_has_structured_traceable_citation(self) -> None:
        answer = KnowledgeQAEngine(self.database, self.retriever).ask(
            "FAST-LIVO2 是否需要硬件时间同步？", collections=["SLAM"]
        )
        self.assertIn("[S1]", answer.markdown)
        self.assertTrue(answer.citations)
        citation = answer.citations[0]
        self.assertEqual(citation.document_id, answer.evidence[0].document_id)
        self.assertEqual(citation.chunk_id, answer.evidence[0].chunk_id)
        self.assertEqual(citation.page_number, 12)
        self.assertEqual(citation.obsidian_path, "90_Sources/FAST-LIVO2.md")
        status = self.database.status()
        self.assertEqual(status["conversations"], 1)
        self.assertEqual(status["messages"], 2)
        self.assertEqual(status["answers"], 1)
        self.assertGreaterEqual(status["citations"], 1)

    def test_codex_provider_returns_chinese_for_english_evidence(self) -> None:
        root = Path(self.temp.name)
        fake_codex = root / "codex"
        fake_codex.write_text(
            """#!/usr/bin/env python3
import sys
from pathlib import Path
if "--ask-for-approval" in sys.argv:
    print("error: unsupported legacy approval flag", file=sys.stderr)
    print("For more information, try '--help'.", file=sys.stderr)
    raise SystemExit(2)
if 'model_reasoning_effort="low"' not in sys.argv:
    print("error: missing low reasoning override", file=sys.stderr)
    raise SystemExit(2)
required = ["--ignore-user-config", "--model", "gpt-5.6-luna", 'model_provider="openai-http"', "model_providers.openai-http.supports_websockets=false"]
if any(item not in sys.argv for item in required):
    print("error: missing direct HTTPS configuration", file=sys.stderr)
    raise SystemExit(2)
prompt = sys.stdin.read()
if "Simplified Chinese" not in prompt:
    raise SystemExit(2)
output = Path(sys.argv[sys.argv.index("--output-last-message") + 1])
output.write_text("FAST-LIVO2 需要激光雷达与相机进行硬件时间同步，以保证传感器融合可靠。[S1]", encoding="utf-8")
""",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)
        provider = CodexAnswerProvider(str(fake_codex), workspace_root=root)
        answer = KnowledgeQAEngine(
            self.database, self.retriever, answer_provider=provider
        ).ask("FAST-LIVO2 是否需要同步？", response_language="zh-CN")
        self.assertIn("需要", answer.markdown)
        self.assertNotIn("requires hardware", answer.markdown)
        self.assertEqual(answer.provider, "codex-local")
        self.assertEqual(answer.retrieval_info["response_language"], "zh-CN")

    def test_codex_provider_reports_the_actionable_cli_error(self) -> None:
        root = Path(self.temp.name)
        fake_codex = root / "failing-codex"
        fake_codex.write_text(
            """#!/usr/bin/env python3
import sys
print("error: unexpected argument '--legacy-option' found", file=sys.stderr)
print("For more information, try '--help'.", file=sys.stderr)
raise SystemExit(2)
""",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)
        provider = CodexAnswerProvider(str(fake_codex), workspace_root=root)
        request = AnswerRequest("测试", "请用中文回答", "测试", [])
        with self.assertRaises(RuntimeError) as raised:
            provider.generate(request)
        self.assertIn("unexpected argument", str(raised.exception))
        self.assertNotIn("For more information", str(raised.exception))

    def test_follow_up_query_is_rewritten_with_conversation_subject(self) -> None:
        engine = KnowledgeQAEngine(self.database, self.retriever)
        first = engine.ask("FAST-LIVO2 的传感器融合有什么要求？")
        second = engine.ask("它需要硬件同步吗？", conversation_id=first.conversation_id)
        self.assertIn("FAST-LIVO2", second.retrieval_info["rewritten_query"])
        self.assertNotEqual(
            second.retrieval_info["original_question"], second.retrieval_info["rewritten_query"]
        )
        record = ConversationRepository(self.database).conversation_record(first.conversation_id)
        self.assertEqual(len(record["messages"]), 4)

    def test_no_evidence_returns_explicit_insufficient_answer(self) -> None:
        answer = KnowledgeQAEngine(self.database, EmptyRetriever()).ask("不存在的量子香蕉协议是什么？")
        self.assertIn("没有足够资料", answer.markdown)
        self.assertEqual(answer.citations, [])
        self.assertEqual(answer.provider, "system")
        self.assertEqual(answer.confidence, 0.0)

    def test_invalid_citation_is_retried_then_rejected_without_answer_record(self) -> None:
        provider = InvalidCitationProvider()
        engine = KnowledgeQAEngine(self.database, self.retriever, answer_provider=provider)
        with self.assertRaises(CitationValidationError):
            engine.ask("FAST-LIVO2 是否需要同步？")
        self.assertEqual(provider.calls, 2)
        self.assertEqual(self.database.status()["answers"], 0)
        self.assertEqual(self.database.status()["citations"], 0)

    def test_validator_rejects_chunk_not_in_database(self) -> None:
        conversation_id = ConversationRepository(self.database).ensure_conversation()
        repository = ConversationRepository(self.database)
        validator = CitationValidator(repository)
        evidence = Evidence(
            evidence_id="S1",
            content="fabricated",
            title="Ghost",
            score=1.0,
            source=SourceReference(
                source_id="ghost", media_type="pdf", title="Ghost", document_id="doc-ghost", chunk_id="chunk-ghost"
            ),
        )
        result = validator.validate("Claim. [S1]", [evidence])
        self.assertFalse(result.valid)
        self.assertIn("current retrieved chunk", result.errors[0])
        self.assertTrue(conversation_id)

    def test_document_delete_preserves_historical_citation_snapshot(self) -> None:
        answer = KnowledgeQAEngine(self.database, self.retriever).ask(
            "FAST-LIVO2 是否需要硬件时间同步？"
        )
        document_id = answer.citations[0].document_id
        self.assertTrue(IndexingService(self.database, self.embedding).delete_document(document_id))
        row = self.database.connection.execute(
            "SELECT document_id, chunk_id, page_number, title FROM citations WHERE answer_id = ?",
            (answer.answer_id,),
        ).fetchone()
        self.assertIsNone(row["document_id"])
        self.assertIsNone(row["chunk_id"])
        self.assertEqual(row["page_number"], 12)
        self.assertEqual(row["title"], "FAST-LIVO2 Synchronization Guide")

    def test_knowledge_plus_web_falls_back_when_provider_is_disabled(self) -> None:
        answer = KnowledgeQAEngine(self.database, EmptyRetriever()).ask(
            "需要联网才能回答的问题", mode="knowledge+web"
        )
        self.assertFalse(answer.retrieval_info["web_available"])
        self.assertEqual(answer.retrieval_info["effective_mode"], "knowledge")

    def test_web_provider_evidence_can_be_cited_without_fake_chunk(self) -> None:
        answer = KnowledgeQAEngine(
            self.database, EmptyRetriever(), web_search_provider=FakeWebProvider()
        ).ask("外部资料怎么说？", mode="knowledge+web")
        self.assertEqual(answer.retrieval_info["effective_mode"], "knowledge+web")
        self.assertEqual(answer.citations[0].source_kind, "web")
        self.assertIsNone(answer.citations[0].chunk_id)
        self.assertEqual(answer.citations[0].original_uri, "https://example.test/sync")

    def test_long_conversation_uses_summary_plus_recent_context(self) -> None:
        repository = ConversationRepository(self.database)
        conversation_id = repository.ensure_conversation(title="Long conversation")
        for index in range(10):
            repository.add_message(conversation_id, "user" if index % 2 == 0 else "assistant", f"message {index}")
        summary = repository.refresh_summary(conversation_id, recent_limit=4)
        context = repository.context(conversation_id, recent_limit=4)
        self.assertIn("message 0", summary)
        self.assertEqual(len(context.recent_messages), 4)
        self.assertIn("Conversation summary", context.as_prompt())
        self.assertIn("Recent context", context.as_prompt())

    def test_answer_progress_reports_generation_after_retrieval(self) -> None:
        progress = []
        KnowledgeQAEngine(self.database, self.retriever).ask(
            "FAST-LIVO2 是否需要同步？",
            progress_callback=lambda stage, message: progress.append((stage, message)),
        )
        self.assertEqual(progress[0][0], "answering")
        self.assertIn("正在生成中文回答", progress[0][1])


if __name__ == "__main__":
    unittest.main()
