from __future__ import annotations

import json
import math
import os
import tempfile
import wave
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Callable


@dataclass(frozen=True, slots=True)
class VadSegment:
    start: float
    end: float
    mean_rms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _rms(raw: bytes) -> float:
    samples = array("h")
    samples.frombytes(raw)
    if not samples:
        return 0.0
    return math.sqrt(sum(int(value) * int(value) for value in samples) / len(samples))


def detect_voice_activity(
    wav_path: str | Path,
    *,
    frame_ms: int = 30,
    min_speech_ms: int = 240,
    min_silence_ms: int = 420,
    padding_ms: int = 180,
    check_cancelled: Callable[[], None] | None = None,
) -> list[VadSegment]:
    """Small dependency-free energy VAD for chunk planning and diagnostics."""

    path = Path(wav_path)
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("VAD 需要 PCM16 单声道 WAV")
        rate = handle.getframerate()
        frame_samples = max(1, int(rate * frame_ms / 1000))
        frames: list[tuple[float, float, float]] = []
        index = 0
        while True:
            if check_cancelled and index % 100 == 0:
                check_cancelled()
            raw = handle.readframes(frame_samples)
            if not raw:
                break
            start = index * frame_samples / rate
            end = start + len(raw) / (2 * rate)
            frames.append((start, end, _rms(raw)))
            index += 1
    if not frames:
        return []
    energies = [item[2] for item in frames]
    quiet = sorted(energies)[: max(1, len(energies) // 5)]
    threshold = max(220.0, median(quiet) * 3.0)
    speech = [energy >= threshold for _, _, energy in frames]
    bridge = max(1, min_silence_ms // frame_ms)
    cursor = 0
    while cursor < len(speech):
        if speech[cursor]:
            cursor += 1
            continue
        end = cursor
        while end < len(speech) and not speech[end]:
            end += 1
        if cursor > 0 and end < len(speech) and end - cursor <= bridge:
            for index in range(cursor, end):
                speech[index] = True
        cursor = end

    minimum = max(1, min_speech_ms // frame_ms)
    padding = padding_ms / 1000.0
    result: list[VadSegment] = []
    cursor = 0
    while cursor < len(speech):
        if not speech[cursor]:
            cursor += 1
            continue
        end = cursor
        while end < len(speech) and speech[end]:
            end += 1
        if end - cursor >= minimum:
            start_seconds = max(0.0, frames[cursor][0] - padding)
            end_seconds = min(frames[-1][1], frames[end - 1][1] + padding)
            mean_rms = sum(energies[cursor:end]) / (end - cursor)
            if result and start_seconds <= result[-1].end:
                previous = result.pop()
                result.append(VadSegment(previous.start, end_seconds, max(previous.mean_rms, mean_rms)))
            else:
                result.append(VadSegment(start_seconds, end_seconds, round(mean_rms, 2)))
        cursor = end
    return result


def write_vad_checkpoint(segments: list[VadSegment], destination: str | Path) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {"format": "ai-jingjing-vad-v1", "segments": [item.to_dict() for item in segments]},
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target
