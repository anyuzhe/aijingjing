from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from media_knowledge.ingestion.extractors import (
    ExtractionContext,
    MissingExtractorDependency,
    PublicPlatformVideoExtractor,
    WebExtractor,
    url_extractor_for,
)
from media_knowledge.ingestion.ocr import (
    OCRLine,
    OCRResult,
    OCRUnavailable,
    extract_ocr,
    normalize_rapidocr_result,
)
from media_knowledge.ingestion.quality import (
    evaluate_extraction,
    evaluate_transcript_integrity,
)
from media_knowledge.ingestion.service import IngestionService
from media_knowledge.ingestion.transcription import (
    TranscriptSegment,
    TranscriptionPlan,
    TranscriptionResult,
    TranscriptionUnavailable,
    select_transcription_plan,
    transcribe_audio,
    write_transcript_artifacts,
)
from media_knowledge.ingestion.types import CancellationToken, ExtractionResult
from media_knowledge.ingestion.types import CancelledError
from media_knowledge.models import ContentSegment
from media_knowledge.product import DesktopSettings, ProductPaths


class OCRQualityTests(unittest.TestCase):
    def test_rapidocr_v3_lines_keep_coordinates_and_confidence(self) -> None:
        class Output:
            txts = ["第一行", "第二行"]
            scores = [0.91, 0.42]
            boxes = [
                [[1, 2], [30, 2], [30, 12], [1, 12]],
                [[1, 20], [40, 20], [40, 30], [1, 30]],
            ]

        lines = normalize_rapidocr_result(Output())
        report = OCRResult(
            "rapidocr", "auto", lines, low_confidence_threshold=0.65
        ).to_dict()
        self.assertEqual(lines[0].bbox, [[1.0, 2.0], [30.0, 2.0], [30.0, 12.0], [1.0, 12.0]])
        self.assertEqual(report["mean_confidence"], 0.665)
        self.assertEqual(report["min_confidence"], 0.42)
        self.assertEqual([line["text"] for line in report["low_confidence_lines"]], ["第二行"])

    def test_complex_layout_uses_paddle_and_retains_original_rapidocr(self) -> None:
        rapid = [OCRLine("原始 OCR", 0.72, [[0, 0], [10, 0], [10, 4], [0, 4]])]
        paddle = [OCRLine("结构化表格", 0.96, [[0, 0], [20, 0], [20, 6], [0, 6]])]
        with patch("media_knowledge.ingestion.ocr._run_rapidocr", return_value=rapid), patch(
            "media_knowledge.ingestion.ocr._run_paddle_structure", return_value=paddle
        ):
            result = extract_ocr("scan.png", complex_layout=True)
        metadata = result.to_dict()
        self.assertEqual(result.engine, "paddleocr_ppstructurev3")
        self.assertEqual(result.text, "结构化表格")
        self.assertEqual(metadata["original_rapidocr"]["lines"][0]["text"], "原始 OCR")

    def test_paddle_failure_keeps_rapid_and_records_reason(self) -> None:
        with patch(
            "media_knowledge.ingestion.ocr._run_rapidocr",
            return_value=[OCRLine("仍可检索的原始文字", 0.8)],
        ), patch(
            "media_knowledge.ingestion.ocr._run_paddle_structure",
            side_effect=OCRUnavailable("PaddleOCR PP-StructureV3 未安装"),
        ):
            result = extract_ocr("scan.png", complex_layout=True)
        self.assertEqual(result.engine, "rapidocr")
        self.assertEqual(result.text, "仍可检索的原始文字")
        self.assertIn("PaddleOCR PP-StructureV3 未安装", result.fallback_reasons)


class TranscriptionQualityTests(unittest.TestCase):
    def test_hardware_router_prefers_mlx_then_cuda_then_cpu(self) -> None:
        mlx = select_transcription_plan(
            "small",
            capabilities={
                "apple_silicon": True, "mlx_whisper": True,
                "faster_whisper": True, "cuda": False,
            },
        )
        self.assertEqual((mlx.engine, mlx.device), ("mlx-whisper", "metal"))
        cuda = select_transcription_plan(
            "small",
            capabilities={
                "apple_silicon": False, "mlx_whisper": False,
                "faster_whisper": True, "cuda": True,
            },
        )
        self.assertEqual((cuda.engine, cuda.device, cuda.compute_type), ("faster-whisper", "cuda", "float16"))
        cpu = select_transcription_plan(
            "small",
            capabilities={
                "apple_silicon": True, "mlx_whisper": False,
                "faster_whisper": True, "cuda": False,
            },
        )
        self.assertEqual((cpu.device, cpu.compute_type), ("cpu", "int8"))
        self.assertTrue(any("明显变慢" in reason for reason in cpu.fallback_reasons))

    def test_slow_fallback_can_be_forbidden(self) -> None:
        with self.assertRaisesRegex(TranscriptionUnavailable, "禁止 CPU"):
            select_transcription_plan(
                "small",
                allow_cpu_fallback=False,
                capabilities={
                    "apple_silicon": True, "mlx_whisper": False,
                    "faster_whisper": True, "cuda": False,
                },
            )

    def test_cpu_policy_distinguishes_explicit_choice_from_auto_fallback(self) -> None:
        explicit = select_transcription_plan(
            "small",
            preferred_engine="cpu",
            allow_cpu_fallback=False,
            capabilities={
                "apple_silicon": True,
                "mlx_whisper": False,
                "faster_whisper": True,
                "cuda": False,
            },
        )
        self.assertEqual((explicit.engine, explicit.device), ("faster-whisper", "cpu"))

        with self.assertRaisesRegex(TranscriptionUnavailable, "禁止 CPU"):
            select_transcription_plan(
                "small",
                preferred_engine="auto",
                allow_cpu_fallback=False,
                capabilities={
                    "apple_silicon": False,
                    "mlx_whisper": False,
                    "faster_whisper": True,
                    "cuda": False,
                },
            )

    def test_accelerator_runtime_failure_is_explicitly_recorded(self) -> None:
        mlx_plan = TranscriptionPlan("mlx-whisper", "metal", "float16", "mlx/model")
        with patch(
            "media_knowledge.ingestion.transcription.select_transcription_plan",
            return_value=mlx_plan,
        ), patch(
            "media_knowledge.ingestion.transcription.resolve_local_hf_model",
            return_value="/models/local-whisper-small",
        ), patch(
            "media_knowledge.ingestion.transcription._transcribe_mlx",
            side_effect=RuntimeError("metal error"),
        ), patch(
            "media_knowledge.ingestion.transcription._transcribe_faster_whisper",
            return_value=([TranscriptSegment(0, 2, "降级成功")], "zh"),
        ):
            result = transcribe_audio(
                "audio.wav",
                model="small",
                duration_seconds=2,
                capabilities={"faster_whisper": True},
            )
        self.assertEqual(result.plan.device, "cpu")
        self.assertTrue(any("明确切换到 CPU int8" in reason for reason in result.fallback_reasons))

    def test_cancellation_never_triggers_cpu_fallback(self) -> None:
        mlx_plan = TranscriptionPlan("mlx-whisper", "metal", "float16", "mlx/model")
        with patch(
            "media_knowledge.ingestion.transcription.select_transcription_plan",
            return_value=mlx_plan,
        ), patch(
            "media_knowledge.ingestion.transcription.resolve_local_hf_model",
            return_value="/models/local-whisper-small",
        ), patch(
            "media_knowledge.ingestion.transcription._transcribe_mlx",
            side_effect=CancelledError("任务已取消"),
        ), patch("media_knowledge.ingestion.transcription._transcribe_faster_whisper") as cpu:
            with self.assertRaises(CancelledError):
                transcribe_audio(
                    "audio.wav", model="small",
                    capabilities={"faster_whisper": True},
                )
        cpu.assert_not_called()

    def test_transcript_artifacts_are_structured_and_complete(self) -> None:
        result = TranscriptionResult(
            TranscriptionPlan("faster-whisper", "cpu", "int8", "small"),
            "zh",
            5.0,
            [TranscriptSegment(0, 2.5, "你好"), TranscriptSegment(2.5, 5, "世界")],
        )
        result.integrity = evaluate_transcript_integrity(
            [item.to_dict() for item in result.segments], duration_seconds=5
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_transcript_artifacts(result, temporary, "recording", source_name="录音.wav")
            self.assertEqual(set(paths), {"json", "md", "txt", "srt", "vtt"})
            self.assertTrue(all(path.is_file() for path in paths.values()))
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "ai-jingjing-transcript-v1")
            self.assertEqual(payload["engine"], "faster-whisper")
            self.assertEqual(payload["language"], "zh")
            self.assertEqual(payload["duration_seconds"], 5.0)
            self.assertEqual(payload["segments"][1]["end"], 5)
            self.assertIn("WEBVTT", paths["vtt"].read_text(encoding="utf-8"))

    def test_dotted_transcript_basenames_never_overwrite_each_other(self) -> None:
        first = TranscriptionResult(
            TranscriptionPlan("faster-whisper", "cpu", "int8", "small"),
            "zh", 1.0, [TranscriptSegment(0, 1, "版本一")],
        )
        second = TranscriptionResult(
            TranscriptionPlan("faster-whisper", "cpu", "int8", "small"),
            "zh", 1.0, [TranscriptSegment(0, 1, "版本二")],
        )
        with tempfile.TemporaryDirectory() as temporary:
            first_paths = write_transcript_artifacts(
                first, temporary, "meeting.v1-a1b2", source_name="meeting.v1.wav"
            )
            second_paths = write_transcript_artifacts(
                second, temporary, "meeting.v2-c3d4", source_name="meeting.v2.wav"
            )
            self.assertTrue(set(first_paths.values()).isdisjoint(second_paths.values()))
            self.assertIn("版本一", first_paths["txt"].read_text(encoding="utf-8"))
            self.assertIn("版本二", second_paths["txt"].read_text(encoding="utf-8"))

    def test_integrity_gate_detects_reversal_overlap_empty_and_edge_gaps(self) -> None:
        report = evaluate_transcript_integrity(
            [
                {"start": 5, "end": 10, "text": "later"},
                {"start": 1, "end": 9, "text": ""},
            ],
            duration_seconds=20,
        )
        self.assertFalse(report["accepted"])
        self.assertEqual(report["out_of_order_segments"], 1)
        self.assertEqual(report["empty_segments"], 1)
        self.assertEqual(report["abnormal_overlaps"], 1)
        self.assertEqual(report["status"], "fail")

    def test_extraction_quality_uses_transcript_integrity_as_a_gate(self) -> None:
        integrity = evaluate_transcript_integrity(
            [{"start": 3, "end": 2, "text": "非法"}], duration_seconds=4
        )
        extracted = ExtractionResult(
            title="bad timeline",
            media_type="audio",
            segments=[ContentSegment(
                "speech-1", 3, "speech", text="非法",
                location={"timestamp_start": 3, "timestamp_end": 2},
            )],
            checksum="abc",
            metadata={"transcription": {"duration_seconds": 4, "integrity": integrity}},
        )
        report = evaluate_extraction(extracted)
        self.assertFalse(report.accepted)
        self.assertTrue(any(check.name == "转写时间连续性" and check.status == "fail" for check in report.checks))


class PublicPlatformConnectorTests(unittest.TestCase):
    def _context(self, root: Path) -> ExtractionContext:
        return ExtractionContext(
            paths=ProductPaths.resolve(root).ensure(),
            settings=DesktopSettings(enable_cloud_vision=False),
            cancellation=CancellationToken(),
        )

    def test_router_has_a_strict_platform_allowlist(self) -> None:
        supported = (
            "https://youtu.be/abc",
            "https://www.bilibili.com/video/BV1xx",
            "https://v.douyin.com/abc/",
            "https://www.xiaohongshu.com/explore/abc",
            "https://x.com/user/status/123",
        )
        for url in supported:
            self.assertIsInstance(url_extractor_for(url), PublicPlatformVideoExtractor)
        self.assertFalse(PublicPlatformVideoExtractor.supports("https://youtube.com.evil.example/watch?v=abc"))
        self.assertFalse(PublicPlatformVideoExtractor.supports("https://user:pass@youtube.com/watch?v=abc"))
        self.assertIsInstance(url_extractor_for("https://example.com/video/abc"), WebExtractor)

    def test_missing_ytdlp_is_a_recoverable_dependency_error(self) -> None:
        extractor = PublicPlatformVideoExtractor()
        with patch(
            "media_knowledge.ingestion.extractors._youtube_dl_class",
            side_effect=MissingExtractorDependency("yt-dlp 未安装；请先安装或下载本地文件"),
        ):
            with self.assertRaisesRegex(MissingExtractorDependency, "下载本地文件"):
                extractor._run_ytdlp("https://youtu.be/abc", {}, download=False)

    def test_public_subtitle_is_preferred_preserved_and_never_uses_credentials(self) -> None:
        captured: list[dict[str, object]] = []

        def fake_run(_self, _url, options, *, download):
            captured.append(dict(options))
            info = {
                "id": "video123",
                "title": "公开课程",
                "uploader": "老师",
                "duration": 4,
                "subtitles": {"zh": [{"ext": "vtt"}]},
            }
            if download:
                directory = Path(str(options["outtmpl"])).parent
                (directory / "video123.zh.vtt").write_text(
                    "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n第一段\n\n"
                    "00:00:02.000 --> 00:00:04.000\n第二段\n",
                    encoding="utf-8",
                )
            return info

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            PublicPlatformVideoExtractor, "_run_ytdlp", autospec=True, side_effect=fake_run
        ):
            result = PublicPlatformVideoExtractor().extract(
                "https://www.youtube.com/watch?v=video123", self._context(Path(temporary))
            )
            self.assertEqual([item.text for item in result.segments], ["第一段", "第二段"])
            self.assertEqual(result.metadata["transcription"]["engine"], "platform-subtitle")
            self.assertTrue(Path(result.metadata["source_subtitle"]).is_file())
            self.assertTrue(Path(result.transcript_path).is_file())
            self.assertEqual(result.source_path, Path(result.metadata["source_subtitle"]))
            self.assertTrue(Path(result.source_path).is_file())
        self.assertGreaterEqual(len(captured), 2)
        for options in captured:
            self.assertIsNone(options["cookiefile"])
            self.assertIsNone(options["cookiesfrombrowser"])
            self.assertEqual(options["proxy"], "")
            self.assertFalse(options["usenetrc"])

    def test_cancelled_subtitle_download_does_not_fall_through_to_media(self) -> None:
        calls = 0

        def fake_run(_self, _url, _options, *, download):
            nonlocal calls
            calls += 1
            if download:
                raise CancelledError("任务已取消")
            return {"id": "video123", "subtitles": {"zh": [{"ext": "vtt"}]}}

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            PublicPlatformVideoExtractor, "_run_ytdlp", autospec=True, side_effect=fake_run
        ):
            with self.assertRaises(CancelledError):
                PublicPlatformVideoExtractor().extract(
                    "https://youtu.be/video123", self._context(Path(temporary))
                )
        self.assertEqual(calls, 2)

    def test_no_public_subtitle_falls_back_to_downloaded_media_pipeline(self) -> None:
        def fake_run(_self, _url, options, *, download):
            info = {"id": "video456", "title": "没有字幕的视频", "uploader": "作者", "duration": 3}
            if download:
                directory = Path(str(options["outtmpl"])).parent
                (directory / "video456.mp4").write_bytes(b"public-media")
            return info

        fake_extracted = ExtractionResult(
            title="video456",
            media_type="video",
            segments=[ContentSegment(
                "speech-1", 0, "speech", text="本地转写",
                location={"timestamp_start": 0, "timestamp_end": 3},
            )],
            source_path=Path("video456.mp4"),
            checksum="checksum",
            metadata={"transcription": {"engine": "faster-whisper"}},
        )
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            PublicPlatformVideoExtractor, "_run_ytdlp", autospec=True, side_effect=fake_run
        ), patch(
            "media_knowledge.ingestion.extractors.AudioVideoExtractor.extract",
            return_value=fake_extracted,
        ):
            result = PublicPlatformVideoExtractor().extract(
                "https://x.com/user/status/456", self._context(Path(temporary))
            )
        self.assertEqual(result.title, "没有字幕的视频")
        self.assertEqual(result.metadata["platform"], "x")
        self.assertEqual(result.metadata["content_scope"], "full_media")
        self.assertFalse(result.metadata["cookies_used"])

    def test_url_archive_fallback_is_always_an_existing_package_file(self) -> None:
        extracted = ExtractionResult(
            title="只有远程来源",
            media_type="web",
            segments=[ContentSegment("text-1", 0, "text", text="证据")],
            original_uri="https://example.com/source",
            checksum="checksum",
        )
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "package"
            package.mkdir()
            bundle = package / "bundle.json"
            bundle.write_text("{}", encoding="utf-8")
            owned = IngestionService._owned_source_path(extracted, package)
            self.assertEqual(owned, bundle)
            self.assertTrue(owned.is_file())


class MediaSettingsTests(unittest.TestCase):
    def test_new_settings_roundtrip_and_invalid_values_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            path.write_text(json.dumps({
                "ocr_engine": "invalid",
                "ocr_low_confidence_threshold": 3,
                "transcription_engine": "invalid",
                "transcription_allow_cpu_fallback": False,
            }), encoding="utf-8")
            settings = DesktopSettings.load(path)
            self.assertEqual(settings.ocr_engine, "auto")
            self.assertEqual(settings.ocr_low_confidence_threshold, 1.0)
            self.assertEqual(settings.transcription_engine, "auto")
            self.assertFalse(settings.transcription_allow_cpu_fallback)


if __name__ == "__main__":
    unittest.main()
