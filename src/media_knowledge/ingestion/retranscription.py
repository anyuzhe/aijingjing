from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from ..product import DesktopSettings
from ..transcripts.deep_correction import ReRecognitionRequest, ReRecognitionResult
from .transcription import TranscriptionResult, transcribe_audio


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
Transcriber = Callable[..., TranscriptionResult]


def _ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


class LocalASRReRecognizer:
    """Re-recognize a bounded interval with an alternate, local ASR route.

    The source media path is supplied by the controller after it has resolved the
    database document.  ``source_uri`` from model-facing data is never trusted as
    a filesystem target.  No model is downloaded by this adapter.
    """

    def __init__(
        self,
        media_path: str | Path,
        settings: DesktopSettings,
        *,
        original_provider: str = "",
        ffmpeg: str | None = None,
        command_runner: CommandRunner | None = None,
        transcriber: Transcriber | None = None,
        check_cancelled: Callable[[], None] | None = None,
        max_interval_ms: int = 10 * 60 * 1000,
    ) -> None:
        try:
            source = Path(media_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("局部重识别的源音视频不存在或不可访问") from exc
        if not source.is_file():
            raise ValueError("局部重识别的源位置不是文件")
        executable = ffmpeg or _ffmpeg_executable()
        if not executable:
            raise RuntimeError("局部重识别需要内置或系统 FFmpeg")
        self.media_path = source
        self.settings = settings
        self.original_provider = str(original_provider or "").strip().casefold()
        self.ffmpeg = executable
        self.command_runner = command_runner or _run_command
        self.transcriber = transcriber or transcribe_audio
        self.check_cancelled = check_cancelled
        self.max_interval_ms = max(1_000, min(30 * 60 * 1000, int(max_interval_ms)))

    def _route(self) -> dict[str, object]:
        configured_provider = str(self.settings.asr_provider or "").strip().casefold()
        primary_path = str(self.settings.asr_model_path or "").strip()
        qwen_path = primary_path if configured_provider.startswith("qwen") else ""
        whisper_path = str(self.settings.asr_whisper_fallback_model_path or "").strip()
        if not whisper_path and configured_provider in {"mlx-whisper", "faster-whisper"}:
            whisper_path = primary_path
        original_is_qwen = self.original_provider.startswith("qwen")
        if not original_is_qwen and qwen_path:
            return {
                "profile": "custom",
                "provider": "qwen3-mlx",
                "model": self.settings.asr_model or "Qwen3-ASR-1.7B",
                "model_path": qwen_path,
                "preferred_engine": "mlx",
            }
        if whisper_path:
            primary_is_whisper = (
                whisper_path == primary_path
                and configured_provider in {"mlx-whisper", "faster-whisper"}
            )
            whisper_provider = (
                configured_provider if primary_is_whisper else "mlx-whisper"
            )
            whisper_model = (
                self.settings.asr_model or self.settings.whisper_model or "large-v3"
                if primary_is_whisper
                else "small"
            )
            return {
                "profile": "custom",
                "provider": whisper_provider,
                "model": whisper_model,
                "model_path": whisper_path,
                "preferred_engine": (
                    "mlx"
                    if whisper_provider == "mlx-whisper"
                    else self.settings.transcription_engine
                ),
            }
        # The normal ASR router may use an already cached large-v3 model.  It is
        # intentionally not allowed to fall back to a small model here.
        return {
            "profile": "custom",
            "provider": "mlx-whisper" if self.settings.transcription_engine in {"auto", "mlx"} else "faster-whisper",
            "model": "large-v3",
            "model_path": None,
            "preferred_engine": self.settings.transcription_engine,
        }

    def rerecognize(self, request: ReRecognitionRequest) -> ReRecognitionResult:
        start_ms = max(0, int(request.start_ms))
        end_ms = max(start_ms, int(request.end_ms))
        if end_ms <= start_ms:
            raise ValueError("局部重识别区间无效")
        if end_ms - start_ms > self.max_interval_ms:
            raise ValueError("单个局部重识别区间超过安全时长限制")
        if self.check_cancelled:
            self.check_cancelled()
        with tempfile.TemporaryDirectory(prefix="ai-jingjing-rerecognize-") as temporary:
            clip = Path(temporary) / "interval.wav"
            duration = (end_ms - start_ms) / 1000.0
            command = [
                self.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_ms / 1000.0:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(self.media_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(clip),
            ]
            completed = self.command_runner(command)
            if completed.returncode != 0 or not clip.is_file():
                detail = (completed.stderr or "FFmpeg 未生成音频区间").strip().splitlines()
                raise RuntimeError(f"局部音频截取失败：{(detail[-1] if detail else '未知错误')[:240]}")
            if self.check_cancelled:
                self.check_cancelled()
            route = self._route()
            result = self.transcriber(
                clip,
                model=str(route["model"]),
                profile=str(route["profile"]),
                provider=str(route["provider"]),
                model_path=route["model_path"],
                whisper_fallback_model_path=None,
                language=request.language,
                context_terms=request.context_terms,
                word_timestamps=True,
                preferred_engine=str(route["preferred_engine"]),
                allow_cpu_fallback=False,
                allow_fallback=False,
                duration_seconds=duration,
                check_cancelled=self.check_cancelled,
            )
        text = " ".join(segment.text.strip() for segment in result.segments if segment.text.strip()).strip()
        if not text:
            raise RuntimeError("备用 ASR 没有返回可用文字")
        confidences = [
            float(segment.confidence)
            for segment in result.segments
            if segment.confidence is not None
        ]
        confidence = sum(confidences) / len(confidences) if confidences else None
        provider = result.provider or result.plan.engine
        evidence = (
            f"local_interval_ms={start_ms}-{end_ms}",
            f"provider={provider}",
            f"model={result.plan.model}",
        )
        return ReRecognitionResult(
            text=text,
            confidence=confidence,
            model=f"{provider}/{result.plan.model}",
            evidence=evidence,
        )


__all__ = ["LocalASRReRecognizer"]
