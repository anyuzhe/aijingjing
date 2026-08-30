from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import wave
from array import array
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable


@dataclass(frozen=True, slots=True)
class AudioProbeResult:
    source: str
    duration_seconds: float
    sample_rate: int | None
    channels: int | None
    codec: str | None
    bit_rate: int | None
    format_name: str | None
    decode_ok: bool
    loudness_dbfs: float | None = None
    silence_ratio: float | None = None
    clipping_ratio: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def _number(value: object, cast: Callable[[object], object], default: object) -> object:
    try:
        return cast(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _wav_statistics(path: Path, *, max_frames: int = 16_000 * 60 * 10) -> tuple[float | None, float | None, float | None]:
    """Return dBFS, silence and clipping ratios for PCM WAV without NumPy."""

    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getsampwidth() != 2:
                return None, None, None
            frame_count = min(handle.getnframes(), max_frames)
            raw = handle.readframes(frame_count)
    except (OSError, EOFError, wave.Error):
        return None, None, None
    samples = array("h")
    samples.frombytes(raw)
    if not samples:
        return None, None, None
    square_sum = sum(int(sample) * int(sample) for sample in samples)
    rms = math.sqrt(square_sum / len(samples))
    dbfs = 20.0 * math.log10(max(rms, 1.0) / 32768.0)
    silence = sum(1 for sample in samples if abs(sample) < 328) / len(samples)
    clipping = sum(1 for sample in samples if abs(sample) >= 32700) / len(samples)
    return round(dbfs, 2), round(silence, 5), round(clipping, 5)


def with_normalized_wav_statistics(
    probe: AudioProbeResult,
    normalized_wav: str | Path,
) -> AudioProbeResult:
    """Backfill comparable quality metrics from the normalized PCM WAV.

    Container formats such as MP3/MP4 cannot be inspected by ``wave`` directly.
    Measuring the deterministic PCM file also makes every supported input use the
    exact samples subsequently consumed by VAD and ASR.
    """

    loudness, silence_ratio, clipping_ratio = _wav_statistics(Path(normalized_wav))
    if loudness is None or silence_ratio is None or clipping_ratio is None:
        raise RuntimeError("标准化音轨质量检测失败：normalized.wav 不是有效的 PCM16 音频")
    warnings = list(probe.warnings)
    if silence_ratio >= 0.98:
        warnings.append("音轨几乎全是静音")
    if clipping_ratio >= 0.01:
        warnings.append("音轨存在明显削波，转写准确率可能下降")
    if loudness < -45:
        warnings.append("音量很低，建议先检查录音质量")
    return replace(
        probe,
        loudness_dbfs=loudness,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def probe_audio(
    source: str | Path,
    *,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    timeout_seconds: int = 60,
    check_cancelled: Callable[[], None] | None = None,
) -> AudioProbeResult:
    """Inspect an existing media file and perform a short decoder preflight."""

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if check_cancelled:
        check_cancelled()
    ffmpeg = ffmpeg or _ffmpeg_executable()
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffprobe and ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe")
        if sibling.is_file():
            ffprobe = str(sibling)

    duration = 0.0
    sample_rate: int | None = None
    channels: int | None = None
    codec: str | None = None
    bit_rate: int | None = None
    format_name: str | None = None
    warnings: list[str] = []
    if ffprobe:
        process = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_streams", "-show_format",
                "-select_streams", "a:0", "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if process.returncode == 0:
            try:
                payload = json.loads(process.stdout or "{}")
                streams = payload.get("streams") or []
                stream = streams[0] if streams else {}
                format_data = payload.get("format") or {}
                duration = float(_number(stream.get("duration") or format_data.get("duration"), float, 0.0))
                sample_rate = int(_number(stream.get("sample_rate"), int, 0)) or None
                channels = int(_number(stream.get("channels"), int, 0)) or None
                codec = str(stream.get("codec_name") or "") or None
                bit_rate = int(_number(stream.get("bit_rate") or format_data.get("bit_rate"), int, 0)) or None
                format_name = str(format_data.get("format_name") or "") or None
            except (json.JSONDecodeError, AttributeError, IndexError):
                warnings.append("媒体探测结果无法解析，已使用兼容检测")
        else:
            warnings.append("ffprobe 无法读取媒体元数据，已使用兼容检测")

    if path.suffix.casefold() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                duration = duration or (handle.getnframes() / rate if rate else 0.0)
                sample_rate = sample_rate or rate
                channels = channels or handle.getnchannels()
                codec = codec or f"pcm_s{handle.getsampwidth() * 8}le"
                format_name = format_name or "wav"
        except (OSError, EOFError, wave.Error):
            pass

    decode_ok = False
    if ffmpeg:
        if check_cancelled:
            check_cancelled()
        preflight = subprocess.run(
            [ffmpeg, "-v", "error", "-t", "5", "-i", str(path), "-vn", "-f", "null", "-"],
            capture_output=True,
            timeout=timeout_seconds,
        )
        decode_ok = preflight.returncode == 0
        if not decode_ok:
            warnings.append("音轨解码预检失败；文件可能损坏或编码不受支持")
    elif path.suffix.casefold() == ".wav":
        decode_ok = duration > 0
    else:
        warnings.append("未找到 FFmpeg，无法验证音轨能否解码")

    loudness, silence_ratio, clipping_ratio = _wav_statistics(path)
    if silence_ratio is not None and silence_ratio >= 0.98:
        warnings.append("音轨几乎全是静音")
    if clipping_ratio is not None and clipping_ratio >= 0.01:
        warnings.append("音轨存在明显削波，转写准确率可能下降")
    if loudness is not None and loudness < -45:
        warnings.append("音量很低，建议先检查录音质量")
    if duration <= 0:
        warnings.append("无法确定音视频时长")

    return AudioProbeResult(
        source=str(path),
        duration_seconds=max(0.0, duration),
        sample_rate=sample_rate,
        channels=channels,
        codec=codec,
        bit_rate=bit_rate,
        format_name=format_name,
        decode_ok=decode_ok,
        loudness_dbfs=loudness,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        warnings=tuple(dict.fromkeys(warnings)),
    )
