from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from media_knowledge.ingestion.asr import (
    AsrProviderError,
    AsrProviderRegistry,
    AsrResult,
    AsrRouter,
    AsrSegment,
    AsrWord,
    CueBuilder,
    FasterWhisperProvider,
    MlxWhisperProvider,
    Qwen3MlxProvider,
    TranscriptionRequest,
)
from media_knowledge.ingestion.asr.providers._shared import resolve_local_hf_model
from media_knowledge.ingestion.transcription import transcribe_audio
from media_knowledge.ingestion.types import CancelledError


class _Provider:
    def __init__(self, provider_id: str, outcome: object, *, available: bool = True) -> None:
        self.provider_id = provider_id
        self.outcome = outcome
        self._available = available
        self.calls: list[TranscriptionRequest] = []

    def available(self) -> bool:
        return self._available

    def transcribe(self, request, progress=None, check_cancelled=None):
        del progress
        self.calls.append(request)
        if check_cancelled:
            check_cancelled()
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _result(provider: str, text: str = "结果") -> AsrResult:
    return AsrResult(
        provider_id=provider,
        model="model",
        device="test",
        compute_type="test",
        language="Chinese",
        segments=[AsrSegment(0.0, 1.0, text)],
    )


class AsrRegistryAndRouterTests(unittest.TestCase):
    def test_registry_is_explicit_and_duplicate_registration_is_rejected(self) -> None:
        registry = AsrProviderRegistry()
        first = _Provider("first", _result("first"))
        registry.register(first)
        self.assertIs(registry.get("first"), first)
        self.assertEqual(registry.provider_ids(), ("first",))
        with self.assertRaisesRegex(ValueError, "已注册"):
            registry.register(_Provider("first", _result("first")))

    def test_profiles_resolve_to_expected_model_order(self) -> None:
        router = AsrRouter(AsrProviderRegistry())
        accurate = router.resolve_attempts(TranscriptionRequest(
            Path("audio.wav"), profile="chinese-accuracy", model_path=Path("/models/qwen-1.7b"),
            whisper_fallback_model_path=Path("/models/whisper-small"),
        ))
        self.assertEqual(
            [(item.provider, item.model) for item in accurate],
            [
                ("qwen3-mlx", "Qwen3-ASR-1.7B"),
                ("mlx-whisper", "small"),
                ("faster-whisper", "small"),
            ],
        )
        self.assertEqual(accurate[1].model_path, Path("/models/whisper-small"))
        quick = router.resolve_attempts(TranscriptionRequest(
            Path("audio.wav"), profile="fast-preview", model_path=Path("/models/qwen-0.6b")
        ))
        self.assertEqual(quick[0].model, "Qwen3-ASR-0.6B")
        compatibility = router.resolve_attempts(TranscriptionRequest(
            Path("audio.wav"), profile="compatibility", model="large-v3"
        ))
        self.assertEqual(
            [(item.provider, item.model) for item in compatibility],
            [("mlx-whisper", "large-v3"), ("faster-whisper", "large-v3")],
        )

    def test_technical_failure_records_explicit_fallback(self) -> None:
        registry = AsrProviderRegistry()
        qwen = _Provider(
            "qwen3-mlx",
            AsrProviderError("模型加载失败", reason_code="model_load_failed"),
        )
        whisper = _Provider("mlx-whisper", _result("mlx-whisper", "降级成功"))
        registry.register(qwen)
        registry.register(whisper)
        registry.register(_Provider("faster-whisper", _result("faster-whisper")))
        result = AsrRouter(registry).transcribe(TranscriptionRequest(
            Path("audio.wav"),
            profile="chinese-accuracy",
            model_path=Path("/models/qwen-1.7b"),
        ))
        self.assertEqual(result.provider_id, "mlx-whisper")
        self.assertEqual(len(result.fallback_history), 1)
        self.assertEqual(result.fallback_history[0].fallback_from, "qwen3-mlx")
        self.assertEqual(result.fallback_history[0].fallback_to, "mlx-whisper")
        self.assertEqual(result.fallback_history[0].reason_code, "model_load_failed")

    def test_cancelled_error_never_falls_back(self) -> None:
        registry = AsrProviderRegistry()
        qwen = _Provider("qwen3-mlx", CancelledError("用户取消"))
        whisper = _Provider("mlx-whisper", _result("mlx-whisper"))
        registry.register(qwen)
        registry.register(whisper)
        registry.register(_Provider("faster-whisper", _result("faster-whisper")))
        with self.assertRaises(CancelledError):
            AsrRouter(registry).transcribe(TranscriptionRequest(
                Path("audio.wav"),
                profile="chinese-accuracy",
                model_path=Path("/models/qwen-1.7b"),
            ))
        self.assertEqual(len(whisper.calls), 0)

    def test_fallback_can_be_disabled(self) -> None:
        registry = AsrProviderRegistry()
        registry.register(_Provider(
            "qwen3-mlx",
            AsrProviderError("模型损坏", reason_code="model_corrupt"),
        ))
        registry.register(_Provider("mlx-whisper", _result("mlx-whisper")))
        with self.assertRaisesRegex(AsrProviderError, "模型损坏"):
            AsrRouter(registry).transcribe(TranscriptionRequest(
                Path("audio.wav"),
                profile="chinese-accuracy",
                model_path=Path("/models/qwen-1.7b"),
                allow_fallback=False,
            ))

    def test_public_transcription_api_forwards_new_model_selection_fields(self) -> None:
        captured: list[TranscriptionRequest] = []

        def fake_transcribe(_self, request, progress=None, check_cancelled=None):
            del progress, check_cancelled
            captured.append(request)
            return AsrResult(
                provider_id="qwen3-mlx",
                model=request.model,
                device="metal",
                compute_type="8bit",
                language="Chinese",
                segments=[AsrSegment(0, 2, "结构面")],
                finish_reason="length",
                truncated=True,
            )

        with patch.object(Qwen3MlxProvider, "transcribe", autospec=True, side_effect=fake_transcribe):
            result = transcribe_audio(
                "audio.wav",
                profile="chinese-accuracy",
                provider="qwen3-mlx",
                model="Qwen3-ASR-1.7B",
                model_path="/models/qwen3-asr-1.7b",
                whisper_fallback_model_path="/models/whisper-small",
                language="Chinese",
                context_terms=("FLAC3D", "结构面"),
                word_timestamps=True,
                duration_seconds=2,
                capabilities={
                    "apple_silicon": True,
                    "mlx_audio": True,
                    "mlx_whisper": True,
                    "faster_whisper": True,
                },
            )
        self.assertEqual(captured[0].model_path, Path("/models/qwen3-asr-1.7b"))
        self.assertEqual(
            captured[0].whisper_fallback_model_path,
            Path("/models/whisper-small"),
        )
        self.assertEqual(captured[0].context_terms, ("FLAC3D", "结构面"))
        self.assertTrue(captured[0].word_timestamps)
        self.assertEqual(result.provider, "qwen3-mlx")
        self.assertEqual(result.finish_reason, "length")
        self.assertTrue(result.metadata()["truncated"])


class Qwen3MlxProviderTests(unittest.TestCase):
    def test_remote_identifier_is_rejected_before_import_or_download(self) -> None:
        provider = Qwen3MlxProvider(available_override=True)
        with patch("media_knowledge.ingestion.asr.providers.qwen3_mlx.import_module") as importer:
            with self.assertRaisesRegex(AsrProviderError, "本地模型目录") as caught:
                provider.transcribe(TranscriptionRequest(
                    Path("audio.wav"),
                    provider="qwen3-mlx",
                    model="mlx-community/Qwen3-ASR-1.7B-8bit",
                ))
        self.assertEqual(caught.exception.reason_code, "model_not_local")
        importer.assert_not_called()

    def test_local_path_is_passed_as_path_and_context_is_forwarded(self) -> None:
        class Output:
            text = "FLAC3D 结构面"
            language = ["Chinese"]
            generation_tokens = 8192
            segments = [{
                "start": 0.0,
                "end": 2.0,
                "text": "FLAC3D 结构面",
                "words": [
                    {"start": 0.0, "end": 0.8, "text": "FLAC3D"},
                    {"start": 0.8, "end": 2.0, "text": "结构面"},
                ],
            }]

        class Model:
            def __init__(self) -> None:
                self.kwargs = {}

            def generate(
                self,
                audio,
                *,
                max_tokens,
                temperature,
                language=None,
                hotwords=None,
                stream=False,
                chunk_duration=None,
            ):
                self.audio = audio
                self.kwargs = {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "language": language,
                    "hotwords": hotwords,
                    "stream": stream,
                    "chunk_duration": chunk_duration,
                }
                return Output()

        model = Model()
        loaded: list[object] = []

        class SttModule:
            @staticmethod
            def load(path):
                loaded.append(path)
                return model

        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / "Qwen3-ASR-1.7B-8bit"
            local.mkdir()
            (local / "config.json").write_text("{}", encoding="utf-8")
            with patch(
                "media_knowledge.ingestion.asr.providers.qwen3_mlx.import_module",
                return_value=SttModule,
            ):
                result = Qwen3MlxProvider(available_override=True).transcribe(
                    TranscriptionRequest(
                        Path("audio.wav"),
                        provider="qwen3-mlx",
                        model="Qwen3-ASR-1.7B",
                        model_path=local,
                        language="zh",
                        context_terms=("FLAC3D", "结构面", "FLAC3D"),
                        word_timestamps=True,
                    )
                )
        self.assertIsInstance(loaded[0], Path)
        self.assertEqual(model.kwargs["language"], "Chinese")
        self.assertEqual(model.kwargs["hotwords"], ["FLAC3D", "结构面"])
        self.assertIs(model.kwargs["stream"], True)
        self.assertEqual(model.kwargs["chunk_duration"], 60.0)
        self.assertEqual(result.finish_reason, "length")
        self.assertTrue(result.truncated)
        self.assertEqual(result.segments[0].words[1].text, "结构面")

    def test_mapping_output_from_older_runtime_is_preserved(self) -> None:
        class Model:
            @staticmethod
            def generate(_audio, **_kwargs):
                return {
                    "text": "旧版兼容结果",
                    "language": "Chinese",
                    "generation_tokens": 4,
                    "segments": [
                        {"start": 0.0, "end": 1.0, "text": "旧版兼容结果"}
                    ],
                }

        class SttModule:
            @staticmethod
            def load(_path):
                return Model()

        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / "Qwen3-ASR-0.6B-8bit"
            local.mkdir()
            (local / "config.json").write_text("{}", encoding="utf-8")
            with patch(
                "media_knowledge.ingestion.asr.providers.qwen3_mlx.import_module",
                return_value=SttModule,
            ):
                result = Qwen3MlxProvider(available_override=True).transcribe(
                    TranscriptionRequest(
                        Path("audio.wav"),
                        provider="qwen3-mlx",
                        model="Qwen3-ASR-0.6B",
                        model_path=local,
                    )
                )
        self.assertEqual(result.segments[0].text, "旧版兼容结果")
        self.assertEqual(result.language, "Chinese")
        self.assertEqual(result.finish_reason, "stop")

    def test_progress_and_cancellation_are_preserved_at_runtime_boundaries(self) -> None:
        class Output:
            text = "结构面"
            language = ["Chinese"]
            generation_tokens = 1
            segments = [{"start": 0.0, "end": 1.0, "text": "结构面"}]

        class Model:
            called = False

            def generate(self, audio, **kwargs):
                del audio, kwargs
                self.called = True
                return Output()

        model = Model()

        class SttModule:
            @staticmethod
            def load(path):
                del path
                return model

        messages: list[str] = []
        checks = 0

        def cancel_after_load() -> None:
            nonlocal checks
            checks += 1
            if checks == 3:
                raise CancelledError("用户取消")

        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / "Qwen3-ASR-1.7B-8bit"
            local.mkdir()
            (local / "config.json").write_text("{}", encoding="utf-8")
            with patch(
                "media_knowledge.ingestion.asr.providers.qwen3_mlx.import_module",
                return_value=SttModule,
            ):
                with self.assertRaises(CancelledError):
                    Qwen3MlxProvider(available_override=True).transcribe(
                        TranscriptionRequest(
                            Path("audio.wav"),
                            provider="qwen3-mlx",
                            model="Qwen3-ASR-1.7B",
                            model_path=local,
                        ),
                        messages.append,
                        cancel_after_load,
                    )

        self.assertFalse(model.called)
        self.assertEqual(len(messages), 1)
        self.assertIn("正在加载本地 Qwen3-ASR", messages[0])

    def test_streaming_generation_can_be_cancelled_between_decoder_tokens(self) -> None:
        class Model:
            closed = False

            def generate(self, _audio, **_kwargs):
                try:
                    yield types.SimpleNamespace(
                        text="结", is_final=False, start_time=0.0, end_time=0.1,
                        language="Chinese", generation_tokens=0,
                    )
                    yield types.SimpleNamespace(
                        text="构", is_final=False, start_time=0.1, end_time=0.2,
                        language="Chinese", generation_tokens=0,
                    )
                    yield types.SimpleNamespace(
                        text="", is_final=True, start_time=0.0, end_time=1.0,
                        language="Chinese", generation_tokens=2,
                    )
                finally:
                    self.closed = True

        model = Model()

        class SttModule:
            @staticmethod
            def load(_path):
                return model

        checks = 0

        def cancel_during_generation() -> None:
            nonlocal checks
            checks += 1
            if checks == 6:
                raise CancelledError("用户在推理中取消")

        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / "Qwen3-ASR-1.7B-8bit"
            local.mkdir()
            (local / "config.json").write_text("{}", encoding="utf-8")
            with patch(
                "media_knowledge.ingestion.asr.providers.qwen3_mlx.import_module",
                return_value=SttModule,
            ):
                with self.assertRaises(CancelledError):
                    Qwen3MlxProvider(available_override=True).transcribe(
                        TranscriptionRequest(
                            Path("audio.wav"),
                            provider="qwen3-mlx",
                            model="Qwen3-ASR-1.7B",
                            model_path=local,
                        ),
                        check_cancelled=cancel_during_generation,
                    )

        self.assertTrue(model.closed)

    def test_missing_word_alignment_is_reported_without_fabrication(self) -> None:
        class Output:
            text = "结构面"
            language = ["Chinese"]
            generation_tokens = 1
            segments = [{"start": 0.0, "end": 1.0, "text": "结构面"}]

        class Model:
            @staticmethod
            def generate(audio, **kwargs):
                del audio, kwargs
                return Output()

        class SttModule:
            @staticmethod
            def load(path):
                del path
                return Model()

        messages: list[str] = []
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / "Qwen3-ASR-1.7B-8bit"
            local.mkdir()
            (local / "config.json").write_text("{}", encoding="utf-8")
            with patch(
                "media_knowledge.ingestion.asr.providers.qwen3_mlx.import_module",
                return_value=SttModule,
            ):
                result = Qwen3MlxProvider(available_override=True).transcribe(
                    TranscriptionRequest(
                        Path("audio.wav"),
                        provider="qwen3-mlx",
                        model="Qwen3-ASR-1.7B",
                        model_path=local,
                        word_timestamps=True,
                    ),
                    messages.append,
                )

        self.assertEqual(result.segments[0].words, ())
        self.assertTrue(any(
            "word_timestamps_unavailable" in item for item in result.warnings
        ))
        self.assertIn("本地 Qwen3-ASR 模型已加载", messages[1])
        self.assertIn("本地转写完成", messages[2])


class OfflineInferenceTests(unittest.TestCase):
    def test_public_mlx_route_resolves_local_cache_before_legacy_worker(self) -> None:
        with patch(
            "media_knowledge.ingestion.transcription.resolve_local_hf_model",
            return_value="/models/local-whisper-small",
        ) as resolver, patch(
            "media_knowledge.ingestion.transcription._transcribe_mlx",
            return_value=([], "zh"),
        ) as transcribe:
            transcribe_audio(
                "audio.wav",
                profile="compatibility",
                provider="mlx-whisper",
                model="small",
                capabilities={
                    "apple_silicon": True,
                    "mlx_audio": False,
                    "mlx_whisper": True,
                    "faster_whisper": False,
                },
            )

        resolver.assert_called_once_with(
            "mlx-community/whisper-small-mlx",
            provider_label="mlx-whisper",
        )
        self.assertEqual(transcribe.call_args.args[1].model, "/models/local-whisper-small")

    def test_huggingface_resolution_is_cache_only(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        with tempfile.TemporaryDirectory() as temporary:
            cached = Path(temporary) / "snapshot"
            cached.mkdir()

            class Hub:
                @staticmethod
                def snapshot_download(repo_id, **kwargs):
                    calls.append((repo_id, dict(kwargs)))
                    return str(cached)

            with patch(
                "media_knowledge.ingestion.asr.providers._shared.import_module",
                return_value=Hub,
            ):
                resolved = resolve_local_hf_model(
                    "mlx-community/whisper-small-mlx",
                    provider_label="mlx-whisper",
                )
        self.assertEqual(resolved, str(cached.resolve()))
        self.assertEqual(calls[0][0], "mlx-community/whisper-small-mlx")
        self.assertIs(calls[0][1]["local_files_only"], True)

    def test_mlx_whisper_receives_resolved_local_snapshot_not_repo_id(self) -> None:
        captured: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as temporary:
            cached = Path(temporary) / "snapshot"
            cached.mkdir()

            class Hub:
                @staticmethod
                def snapshot_download(repo_id, **kwargs):
                    self.assertEqual(repo_id, "mlx-community/whisper-small-mlx")
                    self.assertIs(kwargs["local_files_only"], True)
                    return str(cached)

            fake_mlx = types.SimpleNamespace(
                transcribe=lambda audio, **kwargs: (
                    captured.append({"audio": audio, **kwargs})
                    or {"language": "zh", "segments": []}
                )
            )
            with patch(
                "media_knowledge.ingestion.asr.providers._shared.import_module",
                return_value=Hub,
            ), patch.dict(sys.modules, {"mlx_whisper": fake_mlx}):
                MlxWhisperProvider(available_override=True).transcribe(
                    TranscriptionRequest(
                        Path("audio.wav"),
                        provider="mlx-whisper",
                        model="small",
                    )
                )
        self.assertEqual(captured[0]["path_or_hf_repo"], str(cached.resolve()))
        self.assertIs(captured[0]["condition_on_previous_text"], False)

    def test_faster_whisper_constructor_forbids_model_download(self) -> None:
        captured: list[dict[str, object]] = []

        class Model:
            def __init__(
                self,
                model,
                *,
                device,
                compute_type,
                local_files_only=False,
            ) -> None:
                captured.append({
                    "model": model,
                    "device": device,
                    "compute_type": compute_type,
                    "local_files_only": local_files_only,
                })

            def transcribe(self, audio, **kwargs):
                del audio
                captured.append({"transcribe_kwargs": kwargs})
                return [], types.SimpleNamespace(language="zh")

        fake_module = types.SimpleNamespace(WhisperModel=Model)
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            result = FasterWhisperProvider(available_override=True).transcribe(
                TranscriptionRequest(
                    Path("audio.wav"),
                    provider="faster-whisper",
                    model="small",
                    device="cpu",
                    compute_type="int8",
                )
            )
        self.assertEqual(result.provider_id, "faster-whisper")
        self.assertIs(captured[0]["local_files_only"], True)
        self.assertIs(captured[1]["transcribe_kwargs"]["condition_on_previous_text"], False)


class CueBuilderTests(unittest.TestCase):
    def test_words_are_merged_by_punctuation_pause_and_duration(self) -> None:
        words = [
            AsrWord(0.0, 0.4, "结构面"),
            AsrWord(0.4, 0.8, "参数。"),
            AsrWord(2.0, 2.4, "需要"),
            AsrWord(2.4, 2.9, "复核"),
        ]
        cues = CueBuilder(max_duration_seconds=15, max_characters=60, pause_seconds=0.8).build(words)
        self.assertEqual([cue.text for cue in cues], ["结构面参数。", "需要复核"])
        self.assertEqual((cues[1].start, cues[1].end), (2.0, 2.9))

    def test_speaker_change_forces_a_new_cue(self) -> None:
        cues = CueBuilder().build([
            AsrWord(0, 0.5, "甲", speaker_id="spk_00"),
            AsrWord(0.5, 1, "乙", speaker_id="spk_01"),
        ])
        self.assertEqual(len(cues), 2)
        self.assertEqual([item.speaker_id for item in cues], ["spk_00", "spk_01"])


if __name__ == "__main__":
    unittest.main()
