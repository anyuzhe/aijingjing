from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from media_knowledge.config import AppConfig, CompatibleQAProviderConfig
from media_knowledge.ingestion.service import IngestionService
from media_knowledge.ingestion.types import ExtractionResult
from media_knowledge.models import ContentSegment, SourceReference
from media_knowledge.product import ProductPaths


LAYERED_SYNTHESIS = """## 已确认事实

- 文档确认了基准值。[P7]

## 推测与待验证

- 原始资料未提供。

## 争议与不同观点

- 原始资料未提供。

## 结论与决策

- 会议明确决定进入复核。[01:02.500–01:08.250]

## 行动项

- 复核基准值；责任人和时限未说明。[01:02.500–01:08.250]"""


class CapturingProvider:
    def __init__(self, markdown: str = LAYERED_SYNTHESIS) -> None:
        self.markdown = markdown
        self.request = None

    def generate(self, request):
        self.request = request
        return SimpleNamespace(markdown=self.markdown)


class IngestionSynthesisLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = ProductPaths.resolve(Path(self.temporary.name) / "data")
        self.config = AppConfig(
            database_path=self.paths.database,
            qa_compatible_providers=(
                CompatibleQAProviderConfig(
                    "deepseek",
                    "DeepSeek",
                    "https://api.deepseek.example",
                    "test-key",
                    ("deepseek-v4-flash",),
                ),
            ),
        )
        self.service = IngestionService(self.paths, config=self.config)

    @staticmethod
    def extraction() -> ExtractionResult:
        return ExtractionResult(
            title="评审记录",
            media_type="video",
            segments=[
                ContentSegment(
                    "page-7",
                    1,
                    "text",
                    text="文档确认基准值为 42。",
                    location={"page": 7},
                ),
                ContentSegment(
                    "speech-2",
                    2,
                    "speech",
                    text="会议决定进入复核。",
                    location={"timestamp_start": 62.5, "timestamp_end": 68.25},
                ),
            ],
        )

    def test_synthesis_contract_separates_claim_layers_and_preserves_locators(self) -> None:
        extracted = self.extraction()
        original = copy.deepcopy(extracted)
        provider = CapturingProvider()

        with patch(
            "media_knowledge.ingestion.service.build_answer_provider",
            return_value=provider,
        ) as factory:
            result = self.service._synthesize(extracted)

        self.assertEqual(result, LAYERED_SYNTHESIS)
        factory.assert_called_once_with(
            self.config,
            model_id="compatible::deepseek::deepseek-v4-flash",
        )
        self.assertIsNotNone(provider.request)
        assert provider.request is not None
        for heading in (
            "## 已确认事实",
            "## 推测与待验证",
            "## 争议与不同观点",
            "## 结论与决策",
            "## 行动项",
        ):
            self.assertIn(heading, provider.request.user_prompt)
        self.assertIn("[P7]\n文档确认基准值为 42", provider.request.user_prompt)
        self.assertIn(
            "[01:02.500–01:08.250]\n会议决定进入复核",
            provider.request.user_prompt,
        )
        self.assertIn("不得改写、纠正、覆盖或补齐原始识别文字", provider.request.system_prompt)
        self.assertEqual(extracted, original)

    def test_synthesis_rejects_provider_output_without_required_layers(self) -> None:
        provider = CapturingProvider("## 核心摘要\n\n- 缺少证据分层。")
        with patch(
            "media_knowledge.ingestion.service.build_answer_provider",
            return_value=provider,
        ):
            with self.assertRaisesRegex(RuntimeError, "缺少必需分层"):
                self.service._synthesize(self.extraction())

    def test_synthesis_rejects_missing_or_fabricated_evidence_locator(self) -> None:
        missing = LAYERED_SYNTHESIS.replace("。[P7]", "。")
        fabricated = LAYERED_SYNTHESIS.replace("[P7]", "[P99]")
        for markdown, message in (
            (missing, "缺少原始定位"),
            (fabricated, "不存在的定位"),
        ):
            with self.subTest(message=message), patch(
                "media_knowledge.ingestion.service.build_answer_provider",
                return_value=CapturingProvider(markdown),
            ):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.service._synthesize(self.extraction())

    def test_source_note_links_transcript_fact_quality_and_model_route(self) -> None:
        transcript_path = self.paths.transcripts / "评审记录-fact.v2.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text('{"format":"ai-jingjing-transcript-v2"}\n', encoding="utf-8")
        extracted = self.extraction()
        extracted.transcript_data = {
            "format": "ai-jingjing-transcript-v2",
            "run": {
                "id": "asr-run-test",
                "profile": "accurate",
                "provider": "qwen3-mlx",
                "model": "Qwen3-ASR-1.7B",
                "diarization_provider": "pyannote",
            },
            "quality": {"status": "review", "warnings": ["术语待确认"]},
            "segments": [{"raw_text": "原始识别文字", "corrected_text": "人工校订文字"}],
        }
        extracted.metadata["transcription"] = {
            "artifacts": {"v2": str(transcript_path)},
            "v2_run_id": "asr-run-test",
        }
        original_transcript = copy.deepcopy(extracted.transcript_data)
        source = SourceReference(
            "desktop-source-note-test",
            "video",
            extracted.title,
            local_path="/archive/source.mp4",
        )

        note = self.service._write_source_note(
            extracted,
            source,
            "document-test",
            None,
            LAYERED_SYNTHESIS,
        )
        markdown = note.read_text(encoding="utf-8")

        self.assertIn(f'transcript_v2_path: "{transcript_path}"', markdown)
        self.assertIn('transcript_run_id: "asr-run-test"', markdown)
        self.assertIn('transcript_quality: "review"', markdown)
        self.assertIn(
            'transcript_model_route: "accurate -> qwen3-mlx -> Qwen3-ASR-1.7B -> 说话人:pyannote"',
            markdown,
        )
        self.assertIn(f"[打开事实文件](<{transcript_path}>)", markdown)
        self.assertIn("## AI 知识提炼（派生层，不覆盖原始事实）", markdown)
        self.assertLess(markdown.index("## AI 知识提炼"), markdown.index("## 原始内容导航"))
        self.assertEqual(extracted.transcript_data, original_transcript)


if __name__ == "__main__":
    unittest.main()
