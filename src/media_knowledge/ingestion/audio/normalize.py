from __future__ import annotations

import os
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .probe import _ffmpeg_executable


@dataclass(frozen=True, slots=True)
class AudioNormalizationResult:
    path: str
    duration_seconds: float
    sample_rate: int = 16_000
    channels: int = 1
    sample_format: str = "pcm_s16le"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else 0.0
    except (OSError, EOFError, wave.Error):
        return 0.0


def normalize_audio(
    source: str | Path,
    destination: str | Path,
    *,
    ffmpeg: str | None = None,
    check_cancelled: Callable[[], None] | None = None,
    progress: Callable[[str], None] | None = None,
    timeout_seconds: int = 30 * 60,
) -> AudioNormalizationResult:
    """Decode media to deterministic 16 kHz mono PCM WAV atomically."""

    source_path = Path(source).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    executable = ffmpeg or _ffmpeg_executable()
    if not executable:
        raise RuntimeError("音视频组件 FFmpeg 未安装或未随应用打包")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.part.wav")
    temporary.unlink(missing_ok=True)
    if check_cancelled:
        check_cancelled()
    if progress:
        progress("正在标准化音轨（16 kHz · 单声道 · PCM16）")
    process = subprocess.Popen(
        [
            executable, "-nostdin", "-y", "-v", "error", "-i", str(source_path),
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(temporary),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    started = time.monotonic()
    try:
        while process.poll() is None:
            if check_cancelled:
                try:
                    check_cancelled()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                    raise
            if time.monotonic() - started > timeout_seconds:
                process.kill()
                process.wait(timeout=3)
                raise TimeoutError("音轨标准化超时")
            time.sleep(0.1)
        stderr = (process.stderr.read() if process.stderr else b"").decode("utf-8", errors="replace")
        if process.returncode != 0 or not temporary.is_file():
            detail = stderr.strip().splitlines()[-1] if stderr.strip() else "未知 FFmpeg 错误"
            raise RuntimeError(f"FFmpeg 无法提取音轨：{detail}")
        duration = _duration(temporary)
        if duration <= 0:
            raise RuntimeError("音轨标准化失败：FFmpeg 未生成有效的 PCM WAV 音频")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
        if process.stderr:
            process.stderr.close()
    return AudioNormalizationResult(path=str(target), duration_seconds=duration)
