from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from media_knowledge.ingestion.asr import AsrResult, AsrSegment
from media_knowledge.ingestion.audio.normalize import AudioNormalizationResult
from media_knowledge.ingestion.audio.pipeline import prepare_audio
from media_knowledge.ingestion.audio.probe import AudioProbeResult
from media_knowledge.ingestion.audio.vad import VadSegment
from media_knowledge.ingestion.checkpoints import (
    CheckpointIdentity,
    MediaCheckpointStore,
    configuration_hash,
    glossary_version,
)
from media_knowledge.ingestion.extractors import AudioVideoExtractor, ExtractionContext
from media_knowledge.ingestion.transcription import transcribe_audio
from media_knowledge.ingestion.types import CancellationToken
from media_knowledge.product import DesktopSettings, ProductPaths


def _write_wav(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = array("h", [800] * int(16_000 * seconds))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())


def _probe(path: Path) -> AudioProbeResult:
    return AudioProbeResult(
        source=str(path.resolve()),
        duration_seconds=1.0,
        sample_rate=16_000,
        channels=1,
        codec="pcm_s16le",
        bit_rate=256_000,
        format_name="wav",
        decode_ok=True,
    )


def _identity(source: Path, config: str = "default") -> CheckpointIdentity:
    return CheckpointIdentity(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        config_hash=configuration_hash({"config": config}),
        asr_provider="mock-asr",
        asr_model="mock-model",
        speaker_provider="none",
        glossary_version=glossary_version(["结构面"]),
        asr_model_sha256="a" * 64,
        speaker_model_sha256="b" * 64,
    )


class MediaCheckpointTests(unittest.TestCase):
    def _normalizer(self, _source, destination, **_kwargs):
        target = Path(destination)
        _write_wav(target)
        return AudioNormalizationResult(str(target), 1.0)

    def test_audio_stages_hit_persistent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            _write_wav(source)
            store = MediaCheckpointStore(root / "cache", _identity(source))
            with patch(
                "media_knowledge.ingestion.audio.pipeline.probe_audio",
                return_value=_probe(source),
            ) as probe, patch(
                "media_knowledge.ingestion.audio.pipeline.normalize_audio",
                side_effect=self._normalizer,
            ) as normalize, patch(
                "media_knowledge.ingestion.audio.pipeline.detect_voice_activity",
                return_value=[VadSegment(0.0, 1.0, 800.0)],
            ) as vad:
                first = prepare_audio(source, root / "work", ffmpeg="ffmpeg", checkpoint_store=store)
                second = prepare_audio(source, root / "work", ffmpeg="ffmpeg", checkpoint_store=store)

            self.assertEqual(first.vad_segments, second.vad_segments)
            self.assertEqual(probe.call_count, 1)
            self.assertEqual(normalize.call_count, 1)
            self.assertEqual(vad.call_count, 1)
            self.assertTrue(store.path("audio_probe.json").is_file())
            self.assertTrue(store.path("normalized.wav").is_file())
            self.assertTrue(store.path("vad_segments.json").is_file())
            manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_sha256"], store.identity.source_sha256)
            self.assertEqual(manifest["config_hash"], store.identity.config_hash)
            self.assertEqual(manifest["artifacts"]["normalized.wav"]["asr_model"], "mock-model")
            self.assertEqual(
                manifest["artifacts"]["normalized.wav"]["asr_model_sha256"],
                "a" * 64,
            )

    def test_source_and_configuration_changes_do_not_hit_old_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            _write_wav(source)
            with patch(
                "media_knowledge.ingestion.audio.pipeline.probe_audio",
                side_effect=lambda value, **_kwargs: _probe(Path(value)),
            ) as probe, patch(
                "media_knowledge.ingestion.audio.pipeline.normalize_audio",
                side_effect=self._normalizer,
            ) as normalize, patch(
                "media_knowledge.ingestion.audio.pipeline.detect_voice_activity",
                return_value=[],
            ):
                prepare_audio(
                    source,
                    root / "work",
                    ffmpeg="ffmpeg",
                    checkpoint_store=MediaCheckpointStore(root / "cache", _identity(source, "a")),
                )
                prepare_audio(
                    source,
                    root / "work",
                    ffmpeg="ffmpeg",
                    checkpoint_store=MediaCheckpointStore(root / "cache", _identity(source, "b")),
                )
                _write_wav(source, seconds=1.1)
                prepare_audio(
                    source,
                    root / "work",
                    ffmpeg="ffmpeg",
                    checkpoint_store=MediaCheckpointStore(root / "cache", _identity(source, "b")),
                )
            self.assertEqual(probe.call_count, 3)
            self.assertEqual(normalize.call_count, 3)

    def test_crash_after_normalization_resumes_without_redecoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            _write_wav(source)
            store = MediaCheckpointStore(root / "cache", _identity(source))
            with patch(
                "media_knowledge.ingestion.audio.pipeline.probe_audio",
                return_value=_probe(source),
            ), patch(
                "media_knowledge.ingestion.audio.pipeline.normalize_audio",
                side_effect=self._normalizer,
            ) as normalize, patch(
                "media_knowledge.ingestion.audio.pipeline.detect_voice_activity",
                side_effect=[RuntimeError("simulated crash"), [VadSegment(0.0, 1.0, 800.0)]],
            ) as vad:
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    prepare_audio(source, root / "work", ffmpeg="ffmpeg", checkpoint_store=store)
                recovered = prepare_audio(
                    source, root / "work", ffmpeg="ffmpeg", checkpoint_store=store
                )
            self.assertEqual(normalize.call_count, 1)
            self.assertEqual(vad.call_count, 2)
            self.assertEqual(len(recovered.vad_segments), 1)

    def test_asr_raw_checkpoint_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            _write_wav(source)
            store = MediaCheckpointStore(root / "cache", _identity(source))
            raw = AsrResult(
                provider_id="mock-asr",
                model="mock-model",
                device="test",
                compute_type="test",
                language="zh",
                segments=[AsrSegment(0.0, 1.0, "可恢复结果")],
            )
            with patch(
                "media_knowledge.ingestion.transcription.AsrRouter.transcribe",
                return_value=raw,
            ) as routed:
                first = transcribe_audio(
                    source,
                    profile="compatibility",
                    provider="mlx-whisper",
                    duration_seconds=1,
                    capabilities={"apple_silicon": True, "mlx_whisper": True},
                    checkpoint_store=store,
                )
                second = transcribe_audio(
                    source,
                    profile="compatibility",
                    provider="mlx-whisper",
                    duration_seconds=1,
                    capabilities={},
                    checkpoint_store=store,
                )
            self.assertEqual(routed.call_count, 1)
            self.assertEqual(first.segments[0].text, second.segments[0].text)
            self.assertTrue(store.path("asr_raw.json").is_file())

    def test_extractor_retains_all_named_checkpoint_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "meeting.wav"
            _write_wav(source)
            paths = ProductPaths.resolve(root / "product").ensure()
            context = ExtractionContext(
                paths=paths,
                settings=DesktopSettings(
                    enable_cloud_vision=False,
                    diarization_enabled=False,
                    asr_provider="mlx-whisper",
                    transcription_profile="compatibility",
                ),
                cancellation=CancellationToken(),
            )
            raw = AsrResult(
                provider_id="mlx-whisper",
                model="small",
                device="test",
                compute_type="test",
                language="zh",
                segments=[AsrSegment(0.0, 1.0, "会议内容")],
            )
            with patch(
                "media_knowledge.ingestion.extractors._ffmpeg_executable",
                return_value="ffmpeg",
            ), patch(
                "media_knowledge.ingestion.audio.pipeline.probe_audio",
                return_value=_probe(source),
            ), patch(
                "media_knowledge.ingestion.audio.pipeline.normalize_audio",
                side_effect=self._normalizer,
            ), patch(
                "media_knowledge.ingestion.audio.pipeline.detect_voice_activity",
                return_value=[VadSegment(0.0, 1.0, 800.0)],
            ), patch(
                "media_knowledge.ingestion.transcription.AsrRouter.transcribe",
                return_value=raw,
            ):
                AudioVideoExtractor().extract(source, context)

            manifests = list((paths.cache / "media-checkpoints").rglob("manifest.json"))
            self.assertEqual(len(manifests), 1)
            directory = manifests[0].parent
            expected = {
                "audio_probe.json",
                "normalized.wav",
                "vad_segments.json",
                "asr_raw.json",
                "diarization.json",
                "transcript-v2.json",
                "quality.json",
            }
            self.assertTrue(all((directory / name).is_file() for name in expected))
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["artifacts"]), expected)


if __name__ == "__main__":
    unittest.main()
