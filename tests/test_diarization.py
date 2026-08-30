from __future__ import annotations

import tempfile
import sys
import wave
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from media_knowledge.ingestion.diarization import (
    DIARIZATION_UNKNOWN_SPEAKER,
    DiarizationRequest,
    DiarizationResult,
    DiarizationRouter,
    DiarizationSegment,
    DiarizationUnavailable,
    PyannoteProvider,
    SherpaOnnxProvider,
    TimedWord,
    build_speaker_cues,
    fuse_words_with_speakers,
)
from media_knowledge.ingestion.audio import (
    AudioNormalizationResult,
    AudioPreparationResult,
    AudioProbeResult,
)
from media_knowledge.ingestion.extractors import _transcript_v2
from media_knowledge.ingestion.transcription import (
    TranscriptSegment,
    TranscriptionPlan,
    TranscriptionResult,
)
from media_knowledge.ingestion.types import CancelledError
from media_knowledge.product import DesktopSettings


class _Provider:
    def __init__(self, provider_id: str, *, available: bool = True, result=None, error=None):
        self.provider_id = provider_id
        self._available = available
        self._result = result
        self._error = error
        self.calls = 0

    def availability(self, _request):
        return self._available, None if self._available else "本地模型未安装"

    def diarize(self, request, progress=None, check_cancelled=None):
        self.calls += 1
        if check_cancelled:
            check_cancelled()
        if self._error:
            raise self._error
        return self._result


class DiarizationContractTests(unittest.TestCase):
    def test_segment_only_timing_never_invents_a_speaker_or_word_alignment(self) -> None:
        prepared = AudioPreparationResult(
            probe=AudioProbeResult(
                source="meeting.mp3",
                duration_seconds=4,
                sample_rate=44_100,
                channels=2,
                codec="mp3",
                bit_rate=None,
                format_name="mp3",
                decode_ok=True,
                loudness_dbfs=-20,
                silence_ratio=0.1,
                clipping_ratio=0,
            ),
            normalized=AudioNormalizationResult("normalized.wav", 4),
            vad_segments=(),
        )
        transcribed = TranscriptionResult(
            TranscriptionPlan("qwen3-mlx", "metal", "float16", "Qwen3-ASR"),
            "zh",
            4,
            [TranscriptSegment(0, 4, "甲说前半句，乙说后半句。")],
        )
        diarized = DiarizationResult(
            "test-diarization",
            "local",
            [
                DiarizationSegment(0, 2, "spk_00"),
                DiarizationSegment(2, 4, "spk_01"),
            ],
        )

        transcript = _transcript_v2(
            source_path=Path("meeting.mp3"),
            source_checksum="checksum",
            prepared=prepared,
            transcribed=transcribed,
            settings=DesktopSettings(diarization_enabled=True),
            diarization=diarized,
            pipeline_warnings=[],
        )

        segment = transcript.segments[0]
        self.assertEqual(segment.speaker_id, DIARIZATION_UNKNOWN_SPEAKER)
        self.assertIn("speaker_alignment_unavailable", segment.flags)
        self.assertEqual(segment.words, ())
        self.assertEqual(transcript.quality.status, "review")
        self.assertTrue(any("无法可靠对齐说话人" in item for item in transcript.quality.warnings))

    def test_diarized_transcript_keeps_finish_reason_and_truncation_quality(self) -> None:
        prepared = AudioPreparationResult(
            probe=AudioProbeResult(
                source="meeting.mp3",
                duration_seconds=2,
                sample_rate=44_100,
                channels=2,
                codec="mp3",
                bit_rate=None,
                format_name="mp3",
                decode_ok=True,
                loudness_dbfs=-20,
                silence_ratio=0.1,
                clipping_ratio=0,
            ),
            normalized=AudioNormalizationResult("normalized.wav", 2),
            vad_segments=(),
        )
        transcribed = TranscriptionResult(
            TranscriptionPlan("test", "cpu", "int8", "tiny"),
            "zh",
            2,
            [TranscriptSegment(0, 2, "未完整输出")],
            finish_reason="length",
            truncated=True,
        )
        diarized = DiarizationResult(
            "test-diarization",
            "local",
            [DiarizationSegment(0, 2, "spk_00")],
        )

        transcript = _transcript_v2(
            source_path=Path("meeting.mp3"),
            source_checksum="checksum",
            prepared=prepared,
            transcribed=transcribed,
            settings=DesktopSettings(diarization_enabled=True),
            diarization=diarized,
            pipeline_warnings=[],
        )

        self.assertIn("truncated", transcript.segments[-1].flags)
        self.assertEqual(transcript.segments[-1].metadata["finish_reason"], "length")
        self.assertEqual(transcript.quality.status, "review")
        self.assertTrue(transcript.quality.metrics["truncated"])

    def test_request_validates_speaker_constraints(self) -> None:
        request = DiarizationRequest(Path("meeting.wav"), min_speakers=2, max_speakers=4)
        self.assertEqual((request.min_speakers, request.max_speakers), (2, 4))
        with self.assertRaisesRegex(ValueError, "最少说话人数"):
            DiarizationRequest(Path("meeting.wav"), min_speakers=4, max_speakers=2)
        with self.assertRaisesRegex(ValueError, "预计说话人数"):
            DiarizationRequest(Path("meeting.wav"), expected_speakers=1, min_speakers=2)

    def test_result_anonymizes_provider_labels_by_first_appearance(self) -> None:
        result = DiarizationResult.normalized(
            provider_id="mock",
            model="local",
            segments=[
                DiarizationSegment(3.0, 5.0, "real-person-B"),
                DiarizationSegment(0.0, 2.0, "real-person-A"),
                DiarizationSegment(5.0, 7.0, "real-person-A"),
            ],
        )
        self.assertEqual(
            [(item.start, item.speaker_id) for item in result.segments],
            [(0.0, "spk_00"), (3.0, "spk_01"), (5.0, "spk_00")],
        )
        self.assertEqual(result.speaker_count, 2)
        self.assertNotIn("real-person", repr(result.segments))

    def test_router_does_not_fallback_after_user_cancellation(self) -> None:
        first = _Provider("first", error=CancelledError("任务已取消"))
        second = _Provider("second", result=DiarizationResult("second", "local", []))
        router = DiarizationRouter([first, second])
        with self.assertRaises(CancelledError):
            router.diarize(DiarizationRequest(Path("meeting.wav"), allow_fallback=True))
        self.assertEqual((first.calls, second.calls), (1, 0))

    def test_router_skips_missing_local_models_without_downloading(self) -> None:
        missing = _Provider("missing", available=False)
        fallback = _Provider(
            "fallback",
            result=DiarizationResult("fallback", "local", [DiarizationSegment(0, 1, "spk_00")]),
        )
        result = DiarizationRouter([missing, fallback]).diarize(
            DiarizationRequest(Path("meeting.wav"))
        )
        self.assertEqual(result.provider_id, "fallback")
        self.assertEqual(missing.calls, 0)
        self.assertTrue(any("本地模型未安装" in reason for reason in result.fallback_reasons))

    def test_explicit_missing_provider_reports_install_error(self) -> None:
        router = DiarizationRouter([_Provider("pyannote", available=False)])
        with self.assertRaisesRegex(DiarizationUnavailable, "本地模型未安装"):
            router.diarize(
                DiarizationRequest(Path("meeting.wav"), preferred_provider="pyannote")
            )


class DiarizationFusionTests(unittest.TestCase):
    def test_words_use_maximum_overlap_and_mark_overlap(self) -> None:
        speakers = [
            DiarizationSegment(0.0, 1.8, "spk_00"),
            DiarizationSegment(1.0, 3.0, "spk_01"),
        ]
        words = [
            TimedWord(0.2, 0.8, "第一句"),
            TimedWord(1.2, 2.5, "第二句"),
            TimedWord(4.0, 4.5, "无人区间"),
        ]
        fused = fuse_words_with_speakers(words, speakers)
        self.assertEqual([item.speaker_id for item in fused], ["spk_00", "spk_01", DIARIZATION_UNKNOWN_SPEAKER])
        self.assertFalse(fused[0].overlap)
        self.assertTrue(fused[1].overlap)
        self.assertFalse(fused[2].overlap)

    def test_cue_builder_splits_on_speaker_gap_punctuation_and_quality(self) -> None:
        words = [
            TimedWord(0.0, 0.3, "甲", speaker_id="spk_00"),
            TimedWord(0.3, 0.8, "说完。", speaker_id="spk_00"),
            TimedWord(0.9, 1.2, "乙", speaker_id="spk_01"),
            TimedWord(1.2, 1.8, "回答", speaker_id="spk_01"),
            TimedWord(3.0, 3.5, "间隔后", speaker_id="spk_01"),
            TimedWord(3.5, 4.0, "异常", speaker_id="spk_01", quality_status="review"),
        ]
        cues = build_speaker_cues(words, silence_gap_seconds=0.8)
        self.assertEqual([cue.speaker_id for cue in cues], ["spk_00", "spk_01", "spk_01", "spk_01"])
        self.assertEqual([cue.raw_text for cue in cues], ["甲说完。", "乙回答", "间隔后", "异常"])
        self.assertEqual(cues[-1].quality_status, "review")

    def test_fusion_honours_cancellation_during_long_lists(self) -> None:
        calls = 0

        def check() -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise CancelledError("任务已取消")

        with self.assertRaises(CancelledError):
            fuse_words_with_speakers(
                [TimedWord(index, index + 0.5, str(index)) for index in range(10)],
                [DiarizationSegment(0, 20, "spk_00")],
                check_cancelled=check,
            )


class LocalProviderTests(unittest.TestCase):
    def test_pyannote_requires_an_existing_local_model_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = DiarizationRequest(
                Path(temporary) / "meeting.wav",
                model_path=Path(temporary) / "missing-pyannote",
            )
            provider = PyannoteProvider()
            with patch("importlib.util.find_spec") as find_spec:
                available, reason = provider.availability(request)
            self.assertFalse(available)
            self.assertIn("本地模型", reason or "")
            find_spec.assert_not_called()

    def test_pyannote_adapter_passes_exact_speaker_count_and_anonymizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            audio = root / "meeting.wav"
            audio.touch()
            runner = Mock(return_value=[(0.0, 1.0, "Alice"), (1.0, 2.0, "Bob")])
            result = PyannoteProvider(runner=runner).diarize(
                DiarizationRequest(audio, model_path=model, expected_speakers=2)
            )
            runner.assert_called_once()
            call = runner.call_args
            self.assertEqual(call.kwargs["num_speakers"], 2)
            self.assertEqual([item.speaker_id for item in result.segments], ["spk_00", "spk_01"])

    def test_sherpa_adapter_never_accepts_a_remote_model_identifier(self) -> None:
        request = DiarizationRequest(
            Path("meeting.wav"),
            model_path=Path("hf://speaker-diarization"),
        )
        available, reason = SherpaOnnxProvider().availability(request)
        self.assertFalse(available)
        self.assertIn("本地模型", reason or "")

    def test_sherpa_1136_adapter_uses_nested_config_and_result_sorting(self) -> None:
        captured: dict[str, object] = {}

        class PyannoteConfig:
            def __init__(self, *, model: str) -> None:
                self.model = model

        class SegmentationConfig:
            def __init__(self, *, pyannote: PyannoteConfig) -> None:
                if not isinstance(pyannote, PyannoteConfig):
                    raise TypeError("pyannote must be a nested model config")
                self.pyannote = pyannote

        class EmbeddingConfig:
            def __init__(self, *, model: str) -> None:
                self.model = model

        class ClusteringConfig:
            def __init__(self, *, num_clusters: int, threshold: float) -> None:
                self.num_clusters = num_clusters
                self.threshold = threshold

        class DiarizationConfig:
            def __init__(
                self, *, segmentation, embedding, clustering,
                min_duration_on: float, min_duration_off: float,
            ) -> None:
                captured["segmentation"] = segmentation
                captured["embedding"] = embedding
                captured["clustering"] = clustering
                captured["durations"] = (min_duration_on, min_duration_off)

        class Result:
            def sort_by_start_time(self):
                return [
                    SimpleNamespace(start=0.0, end=0.5, speaker=1),
                    SimpleNamespace(start=0.5, end=1.0, speaker=0),
                ]

        class Diarizer:
            sample_rate = 16_000

            def __init__(self, _config) -> None:
                pass

            def process(self, samples, callback=None):
                captured["sample_count"] = len(samples)
                captured["callback"] = callback
                if callback:
                    self.assert_callback(callback)
                return Result()

            @staticmethod
            def assert_callback(callback) -> None:
                if callback(1, 2) != 0:
                    raise AssertionError("callback must return zero")

        fake_module = SimpleNamespace(
            OfflineSpeakerSegmentationPyannoteModelConfig=PyannoteConfig,
            OfflineSpeakerSegmentationModelConfig=SegmentationConfig,
            SpeakerEmbeddingExtractorConfig=EmbeddingConfig,
            FastClusteringConfig=ClusteringConfig,
            OfflineSpeakerDiarizationConfig=DiarizationConfig,
            OfflineSpeakerDiarization=Diarizer,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "pyannote-segmentation.onnx").write_bytes(b"seg")
            (model / "3dspeaker-embedding.onnx").write_bytes(b"emb")
            audio = root / "meeting.wav"
            with wave.open(str(audio), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(16_000)
                target.writeframes(b"\0\0" * 160)
            messages: list[str] = []
            checks = 0

            def check_cancelled() -> None:
                nonlocal checks
                checks += 1

            with patch.dict(sys.modules, {"sherpa_onnx": fake_module}):
                segments = SherpaOnnxProvider._run_local(
                    audio_path=audio,
                    model_path=model,
                    num_speakers=2,
                    min_speakers=None,
                    max_speakers=None,
                    progress=messages.append,
                    check_cancelled=check_cancelled,
                )

        segmentation = captured["segmentation"]
        self.assertIsInstance(segmentation.pyannote, PyannoteConfig)
        self.assertEqual(captured["clustering"].num_clusters, 2)
        self.assertEqual(captured["sample_count"], 160)
        self.assertGreaterEqual(checks, 2)
        self.assertIn("Sherpa-ONNX 说话人分段 50%", messages)
        self.assertEqual(
            segments,
            [(0.0, 0.5, "provider-speaker-1"), (0.5, 1.0, "provider-speaker-0")],
        )


if __name__ == "__main__":
    unittest.main()
