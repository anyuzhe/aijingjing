from __future__ import annotations

import math
import io
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import Mock, patch

from media_knowledge.ingestion.audio.normalize import normalize_audio
from media_knowledge.ingestion.audio.pipeline import prepare_audio
from media_knowledge.ingestion.audio.normalize import AudioNormalizationResult
from media_knowledge.ingestion.audio.probe import AudioProbeResult, probe_audio
from media_knowledge.ingestion.audio.vad import detect_voice_activity, write_vad_checkpoint
from media_knowledge.ingestion.types import CancelledError


def _write_pcm(path: Path, *, seconds: float = 1.0, speech_start: float = 0.2) -> None:
    rate = 16_000
    samples = array("h")
    for index in range(int(rate * seconds)):
        time = index / rate
        value = 0 if time < speech_start else int(7_000 * math.sin(2 * math.pi * 220 * time))
        samples.append(value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


class AudioPipelineTests(unittest.TestCase):
    def test_prepare_audio_stops_before_normalization_on_decode_failure(self) -> None:
        failed = AudioProbeResult(
            source="broken.mp3",
            duration_seconds=0,
            sample_rate=None,
            channels=None,
            codec=None,
            bit_rate=None,
            format_name=None,
            decode_ok=False,
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "media_knowledge.ingestion.audio.pipeline.probe_audio",
            return_value=failed,
        ), patch("media_knowledge.ingestion.audio.pipeline.normalize_audio") as normalize:
            source = Path(temporary) / "broken.mp3"
            source.write_bytes(b"broken")
            with self.assertRaisesRegex(RuntimeError, "解码预检失败"):
                prepare_audio(source, Path(temporary) / "work", ffmpeg="ffmpeg")
        normalize.assert_not_called()

    def test_non_wav_quality_metrics_are_backfilled_from_normalized_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "recording.mp3"
            source.write_bytes(b"container-bytes")
            normalized = root / "normalized-source.wav"
            _write_pcm(normalized, speech_start=0.5)
            original_probe = AudioProbeResult(
                source=str(source),
                duration_seconds=1.0,
                sample_rate=44_100,
                channels=2,
                codec="mp3",
                bit_rate=128_000,
                format_name="mp3",
                decode_ok=True,
            )
            with patch(
                "media_knowledge.ingestion.audio.pipeline.probe_audio",
                return_value=original_probe,
            ), patch(
                "media_knowledge.ingestion.audio.pipeline.normalize_audio",
                return_value=AudioNormalizationResult(str(normalized), 1.0),
            ), patch(
                "media_knowledge.ingestion.audio.pipeline.detect_voice_activity",
                return_value=[],
            ):
                prepared = prepare_audio(source, root / "work", ffmpeg="ffmpeg")

            self.assertIsNotNone(prepared.probe.loudness_dbfs)
            self.assertAlmostEqual(prepared.probe.silence_ratio or 0.0, 0.5, delta=0.02)
            self.assertIsNotNone(prepared.probe.clipping_ratio)
            self.assertEqual(prepared.probe.source, str(source))

    @patch("media_knowledge.ingestion.audio.probe.subprocess.run")
    def test_probe_and_vad_keep_reproducible_audio_metadata(self, run: Mock) -> None:
        run.return_value = Mock(
            returncode=0,
            stdout='{"streams": [{"sample_rate": "16000", "channels": 1, "codec_name": "pcm_s16le"}], "format": {"duration": "1.0", "format_name": "wav"}}',
            stderr=b"",
        )
        with tempfile.TemporaryDirectory() as temporary:
            wav = Path(temporary) / "recording.wav"
            _write_pcm(wav)
            result = probe_audio(wav, ffmpeg="ffmpeg", ffprobe="ffprobe")
            self.assertEqual(result.sample_rate, 16_000)
            self.assertEqual(result.channels, 1)
            self.assertAlmostEqual(result.duration_seconds, 1.0, places=2)

            segments = detect_voice_activity(wav)
            self.assertEqual(len(segments), 1)
            self.assertLessEqual(segments[0].start, 0.21)
            checkpoint = write_vad_checkpoint(segments, Path(temporary) / "vad.json")
            self.assertIn("ai-jingjing-vad-v1", checkpoint.read_text(encoding="utf-8"))

    @patch("media_knowledge.ingestion.audio.probe.subprocess.run")
    def test_probe_decode_failure_is_explicit(self, run: Mock) -> None:
        run.return_value = Mock(returncode=1, stdout="", stderr=b"bad codec")
        with tempfile.TemporaryDirectory() as temporary:
            wav = Path(temporary) / "broken.wav"
            wav.write_bytes(b"not-a-wav")
            result = probe_audio(wav, ffmpeg="ffmpeg", ffprobe="ffprobe")
        self.assertFalse(result.decode_ok)
        self.assertTrue(any("解码预检失败" in item for item in result.warnings))

    @patch("media_knowledge.ingestion.audio.normalize.subprocess.Popen")
    def test_normalization_cancellation_never_leaves_partial_file(self, popen: Mock) -> None:
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        process.stderr = Mock()
        popen.return_value = process
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.mp3"
            target = Path(temporary) / "normalized.wav"
            source.write_bytes(b"media")
            cancellation = Mock(side_effect=[None, CancelledError("任务已取消")])
            with self.assertRaises(CancelledError):
                normalize_audio(
                    source,
                    target,
                    ffmpeg="ffmpeg",
                    check_cancelled=cancellation,
                )
            self.assertFalse(target.exists())
            process.terminate.assert_called_once()

    def test_normalization_rejects_invalid_ffmpeg_output(self) -> None:
        def fake_popen(command, **_kwargs):
            Path(command[-1]).write_bytes(b"not-a-valid-wav")
            process = Mock()
            process.poll.return_value = 0
            process.returncode = 0
            process.stderr = io.BytesIO()
            return process

        with tempfile.TemporaryDirectory() as temporary, patch(
            "media_knowledge.ingestion.audio.normalize.subprocess.Popen",
            side_effect=fake_popen,
        ):
            source = Path(temporary) / "source.mp3"
            target = Path(temporary) / "normalized.wav"
            source.write_bytes(b"media")
            with self.assertRaisesRegex(RuntimeError, "未生成有效"):
                normalize_audio(source, target, ffmpeg="ffmpeg")
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
