from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from media_knowledge.config import AppConfig
from media_knowledge.product import DesktopSettings, ProductPaths
from media_knowledge.providers.web import WebSearchHit, WebSearchProvider
from media_knowledge.storage import KnowledgeDatabase
from media_knowledge.transcripts import (
    DeepCorrectionRepository,
    TranscriptQuality,
    TranscriptRepository,
    TranscriptRun,
    TranscriptSegment,
    TranscriptSource,
    TranscriptSpeaker,
    TranscriptV2,
)
from media_knowledge.transcripts.deep_correction import LLMCorrectionRequest
from media_knowledge.transcripts.workflow import DeepCorrectionWorkflow


class _CorrectionLLM:
    def correct(self, request: LLMCorrectionRequest) -> str:
        core = list(request.chunk.core_segment_ids)
        by_id = {str(item["segment_id"]): item for item in request.segments}
        first = core[0]
        raw = str(by_id[first]["raw_text"])
        corrected = "这里使用 Obsidian 管理知识。" if "奥格森林" in raw else raw
        corrections = []
        if corrected != raw:
            corrections.append({
                "segment_id": first,
                "corrected_text": corrected,
                "reason": "结合相邻语境统一软件名称",
                "confidence": 0.98,
                "uncertain": False,
                "evidence": [{"kind": "source", "segment_id": first, "quote": raw}],
            })
        return json.dumps({
            "schema_version": request.schema_version,
            "chunk_id": request.chunk.id,
            "reviewed_segment_ids": core,
            "corrections": corrections,
            "chapters": [{
                "title": "知识库方法",
                "start_segment_id": core[0],
                "end_segment_id": core[-1],
                "summary": "介绍知识管理与检索。",
                "evidence_segment_ids": core,
            }],
            "knowledge_cards": [{
                "title": "Obsidian 知识管理",
                "content": "使用可链接笔记沉淀知识。",
                "evidence_segment_ids": core,
            }],
            "entities": [{
                "canonical": "Obsidian",
                "variants": ["奥格森林"],
                "segment_ids": [first],
            }],
        }, ensure_ascii=False)


class _WebEvidenceCorrectionLLM(_CorrectionLLM):
    def correct(self, request: LLMCorrectionRequest) -> str:
        payload = json.loads(super().correct(request))
        if payload["corrections"] and request.external_evidence:
            item = request.external_evidence[0]
            payload["corrections"][0]["evidence"] = [{
                "kind": "web",
                "evidence_id": item.id,
                "url": item.url,
                "quote": item.snippet,
            }]
        return json.dumps(payload, ensure_ascii=False)


class _FakeWebProvider(WebSearchProvider):
    name = "test"

    @property
    def available(self) -> bool:
        return True

    def search(self, query: str, top_k: int = 5) -> list[WebSearchHit]:
        return [WebSearchHit(
            "Obsidian 与 RAG 官方帮助",
            "Obsidian 是一款用于管理链接笔记并配合 RAG 检索的知识工具。",
            "https://example.com/obsidian",
        )]


class DeepCorrectionWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = ProductPaths.resolve(self.root / "data").ensure()
        self.media = self.root / "meeting.wav"
        self.media.write_bytes(b"test-audio")
        transcript = TranscriptV2(
            TranscriptSource("meeting.wav", "a" * 64, 4000, str(self.media)),
            TranscriptRun("asr-workflow", "accuracy", "qwen3-mlx", "Qwen3-ASR-1.7B", "zh"),
            [TranscriptSpeaker("spk_00", "S1", "automatic")],
            [
                TranscriptSegment(
                    "seg-a", 0, 0, 2000, "spk_00", "这里使用奥格森林管理知识。",
                    confidence=0.55,
                ),
                TranscriptSegment(
                    "seg-b", 1, 2000, 4000, "spk_00", "然后使用 RAG 检索。",
                    confidence=0.95,
                ),
            ],
            TranscriptQuality("review"),
        )
        with KnowledgeDatabase(self.paths.database) as database:
            TranscriptRepository(database).save_transcript(transcript)
        settings = DesktopSettings(
            deep_correction_model="compatible::deepseek::deepseek-v4-flash",
            deep_correction_retranscribe_anomalies=False,
            deep_correction_web_verification=False,
            deep_correction_confidence_threshold=0.92,
        )
        self.workflow = DeepCorrectionWorkflow(
            self.paths,
            AppConfig(self.paths.database),
            settings,
            llm_factory=lambda _config, _model: _CorrectionLLM(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_pipeline_persists_proposals_export_and_atomic_accept(self) -> None:
        events: list[tuple[str, int]] = []
        snapshot = self.workflow.run(
            "asr-workflow",
            progress=lambda stage, completed, _total, _message: events.append((stage, completed)),
        )
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(len(snapshot["changes"]), 1)
        change = snapshot["changes"][0]
        self.assertEqual(change["status"], "pending")
        self.assertEqual(change["corrected_text"], "这里使用 Obsidian 管理知识。")
        self.assertTrue(change["evidence"][0]["url"].startswith("file:"))
        output = Path(str(snapshot["output_path"]))
        self.assertTrue(output.is_file())
        markdown = output.read_text(encoding="utf-8")
        self.assertIn("完整精校正文", markdown)
        self.assertIn("原稿 / 精校稿差异审计", markdown)
        self.assertIn("待确认", markdown)
        self.assertIn("知识卡", markdown)
        self.assertIn("```mermaid", markdown)
        self.assertEqual(events[-1], ("quality_gate", 10))

        with KnowledgeDatabase(self.paths.database) as database:
            before = TranscriptRepository(database).get_segment("seg-a")
            assert before is not None
            self.assertEqual(before.raw_text, "这里使用奥格森林管理知识。")
            self.assertIsNone(before.corrected_text)

        reviewed = self.workflow.review_change(str(change["id"]), decision="accepted")
        self.assertEqual(reviewed["change"]["status"], "accepted")
        with KnowledgeDatabase(self.paths.database) as database:
            after = TranscriptRepository(database).get_segment("seg-a")
            assert after is not None
            self.assertEqual(after.raw_text, "这里使用奥格森林管理知识。")
            self.assertEqual(after.corrected_text, "这里使用 Obsidian 管理知识。")
            edits = TranscriptRepository(database).list_edits(run_id="asr-workflow")
            self.assertEqual(edits[-1].edit_type, "deep_correction_accept")
            correction_id = str(snapshot["correction_run_id"])
            changes = DeepCorrectionRepository(database).list_changes(correction_id)
            self.assertEqual(changes[0].status, "accepted")
        self.assertTrue((self.paths.transcripts / "asr-workflow.latest.v2.json").is_file())

    def test_model_selection_rejects_codex_before_any_run(self) -> None:
        settings = DesktopSettings(deep_correction_model="codex::gpt-5.6::low")
        workflow = DeepCorrectionWorkflow(
            self.paths, AppConfig(self.paths.database), settings,
            llm_factory=lambda _config, _model: _CorrectionLLM(),
        )
        with self.assertRaisesRegex(ValueError, "不调用 Codex CLI"):
            workflow.run("asr-workflow")
        with KnowledgeDatabase(self.paths.database) as database:
            self.assertEqual(DeepCorrectionRepository(database).list_runs(), [])

    def test_auto_accept_requires_verified_external_quote_and_is_audited(self) -> None:
        settings = DesktopSettings(
            deep_correction_model="compatible::deepseek::deepseek-v4-flash",
            deep_correction_retranscribe_anomalies=False,
            deep_correction_web_verification=True,
            deep_correction_auto_apply_high_confidence=True,
            deep_correction_confidence_threshold=0.92,
        )
        workflow = DeepCorrectionWorkflow(
            self.paths,
            AppConfig(self.paths.database),
            settings,
            llm_factory=lambda _config, _model: _WebEvidenceCorrectionLLM(),
            web_provider=_FakeWebProvider(),
        )

        snapshot = workflow.run("asr-workflow")

        self.assertEqual(len(snapshot["auto_accepted_change_ids"]), 1)
        self.assertEqual(snapshot["changes"][0]["status"], "accepted")
        with KnowledgeDatabase(self.paths.database) as database:
            segment = TranscriptRepository(database).get_segment("seg-a")
            assert segment is not None
            self.assertEqual(segment.raw_text, "这里使用奥格森林管理知识。")
            self.assertEqual(segment.corrected_text, "这里使用 Obsidian 管理知识。")
            change = DeepCorrectionRepository(database).list_changes(
                str(snapshot["correction_run_id"])
            )[0]
            events = DeepCorrectionRepository(database).list_change_events(change.id)
            self.assertEqual(events[-1].actor, "deep-correction-auto")
            self.assertTrue(events[-1].metadata["automatic"])

    def test_generation_switches_omit_cards_and_mermaid_from_export(self) -> None:
        settings = DesktopSettings(
            deep_correction_model="compatible::deepseek::deepseek-v4-flash",
            deep_correction_retranscribe_anomalies=False,
            deep_correction_web_verification=False,
            deep_correction_generate_knowledge_cards=False,
            deep_correction_generate_mermaid=False,
        )
        workflow = DeepCorrectionWorkflow(
            self.paths,
            AppConfig(self.paths.database),
            settings,
            llm_factory=lambda _config, _model: _CorrectionLLM(),
        )

        snapshot = workflow.run("asr-workflow")
        markdown = Path(str(snapshot["output_path"])).read_text(encoding="utf-8")
        self.assertIn("未生成知识卡。", markdown)
        self.assertNotIn("知识关系图", markdown)
        self.assertNotIn("```mermaid", markdown)


if __name__ == "__main__":
    unittest.main()
