from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from media_knowledge.ingestion.retranscription import LocalASRReRecognizer
from media_knowledge.ingestion.transcription import (
    TranscriptSegment,
    TranscriptionPlan,
    TranscriptionResult,
)
from media_knowledge.product import DesktopSettings
from media_knowledge.transcripts.deep_correction import ReRecognitionRequest


class LocalASRReRecognizerTests(unittest.TestCase):
    def test_extracts_only_interval_and_uses_alternate_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "source.mp4"
            media.write_bytes(b"media")
            qwen = root / "qwen"
            qwen.mkdir()
            calls: dict[str, object] = {}

            def command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
                calls["command"] = command
                Path(command[-1]).write_bytes(b"wav")
                return subprocess.CompletedProcess(command, 0, "", "")

            def transcriber(path: Path, **kwargs: object) -> TranscriptionResult:
                calls["path"] = path
                calls["kwargs"] = kwargs
                return TranscriptionResult(
                    TranscriptionPlan("qwen3-mlx", "metal", "mlx", "Qwen3-ASR-1.7B"),
                    "zh",
                    4.0,
                    [TranscriptSegment(0, 4, "Obsidian 与 RAG", confidence=0.91)],
                    provider="qwen3-mlx",
                )

            recognizer = LocalASRReRecognizer(
                media,
                DesktopSettings(asr_model_path=str(qwen)),
                original_provider="faster-whisper",
                ffmpeg="/fake/ffmpeg",
                command_runner=command_runner,
                transcriber=transcriber,
            )
            result = recognizer.rerecognize(ReRecognitionRequest(
                str(media), 1_000, 5_000, ("seg-1",), "zh", ("Obsidian",), ("low_confidence",),
            ))
            command = calls["command"]
            assert isinstance(command, list)
            self.assertIn("1.000", command)
            self.assertIn("4.000", command)
            kwargs = calls["kwargs"]
            assert isinstance(kwargs, dict)
            self.assertEqual(kwargs["provider"], "qwen3-mlx")
            self.assertFalse(kwargs["allow_fallback"])
            self.assertEqual(result.text, "Obsidian 与 RAG")
            self.assertEqual(result.confidence, 0.91)
            self.assertIn("qwen3-mlx", result.model)

    def test_rejects_unbounded_interval_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "source.wav"
            media.write_bytes(b"audio")
            recognizer = LocalASRReRecognizer(
                media,
                DesktopSettings(),
                ffmpeg="/fake/ffmpeg",
                command_runner=lambda command: subprocess.CompletedProcess(command, 0, "", ""),
                max_interval_ms=2_000,
            )
            with self.assertRaisesRegex(ValueError, "安全时长"):
                recognizer.rerecognize(ReRecognitionRequest(
                    str(media), 0, 5_000, ("seg-1",), "zh", (), ("truncated",),
                ))

    def test_preserves_faster_whisper_model_identity_for_local_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "source.wav"
            media.write_bytes(b"audio")
            model = root / "large-v3"
            model.mkdir()
            recognizer = LocalASRReRecognizer(
                media,
                DesktopSettings(
                    asr_provider="faster-whisper",
                    asr_model="large-v3",
                    asr_model_path=str(model),
                    transcription_engine="cuda",
                ),
                original_provider="faster-whisper",
                ffmpeg="/fake/ffmpeg",
            )

            route = recognizer._route()

            self.assertEqual(route["provider"], "faster-whisper")
            self.assertEqual(route["model"], "large-v3")
            self.assertEqual(route["preferred_engine"], "cuda")


if __name__ == "__main__":
    unittest.main()
