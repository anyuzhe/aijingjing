from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .normalize import AudioNormalizationResult, normalize_audio
from .probe import AudioProbeResult, probe_audio, with_normalized_wav_statistics
from .vad import VadSegment, detect_voice_activity, write_vad_checkpoint

if TYPE_CHECKING:
    from ..checkpoints import MediaCheckpointStore


@dataclass(frozen=True, slots=True)
class AudioPreparationResult:
    probe: AudioProbeResult
    normalized: AudioNormalizationResult
    vad_segments: tuple[VadSegment, ...]
    vad_checkpoint: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "probe": self.probe.to_dict(),
            "normalized": self.normalized.to_dict(),
            "vad_segments": [item.to_dict() for item in self.vad_segments],
            "vad_checkpoint": self.vad_checkpoint,
        }


def prepare_audio(
    source: str | Path,
    work_directory: str | Path,
    *,
    ffmpeg: str | None = None,
    check_cancelled: Callable[[], None] | None = None,
    progress: Callable[[str], None] | None = None,
    checkpoint_store: MediaCheckpointStore | None = None,
) -> AudioPreparationResult:
    if check_cancelled:
        check_cancelled()
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    probe: AudioProbeResult | None = None
    probe_payload = (
        checkpoint_store.read_json("audio_probe.json", "audio_probe")
        if checkpoint_store else None
    )
    if isinstance(probe_payload, dict):
        try:
            probe = AudioProbeResult(
                **{**probe_payload, "warnings": tuple(probe_payload.get("warnings") or ())}
            )
            if progress:
                progress("已复用音视频探测检查点")
        except (TypeError, ValueError):
            probe = None
    if probe is None:
        if progress:
            progress("正在检查音视频编码与音轨质量")
        probe = probe_audio(source, ffmpeg=ffmpeg, check_cancelled=check_cancelled)
        if checkpoint_store:
            checkpoint_store.write_json("audio_probe.json", "audio_probe", probe.to_dict())
    if not probe.decode_ok:
        raise RuntimeError("音轨解码预检失败，无法继续转写")
    if check_cancelled:
        check_cancelled()
    normalized_path = (
        checkpoint_store.path("normalized.wav")
        if checkpoint_store else work / "normalized.wav"
    )
    normalized: AudioNormalizationResult | None = None
    if checkpoint_store and checkpoint_store.is_file_valid("normalized.wav"):
        from .normalize import _duration

        duration = _duration(normalized_path)
        if duration > 0:
            normalized = AudioNormalizationResult(str(normalized_path), duration)
            if progress:
                progress("已复用标准化音轨检查点")
    if normalized is None:
        normalized = normalize_audio(
            source,
            normalized_path,
            ffmpeg=ffmpeg,
            check_cancelled=check_cancelled,
            progress=progress,
        )
        if checkpoint_store:
            checkpoint_store.record_file("normalized.wav", "normalized_audio")
    if check_cancelled:
        check_cancelled()

    # WAV-derived levels are more useful than compressed-source statistics. A
    # complete cached probe already contains them, so a hit does not rescan a
    # potentially multi-hour WAV.
    if not isinstance(probe_payload, dict):
        probe = with_normalized_wav_statistics(probe, normalized.path)
        if checkpoint_store:
            checkpoint_store.write_json("audio_probe.json", "audio_probe", probe.to_dict())

    segments: list[VadSegment] | None = None
    vad_payload = (
        checkpoint_store.read_json("vad_segments.json", "vad")
        if checkpoint_store else None
    )
    if isinstance(vad_payload, dict) and isinstance(vad_payload.get("segments"), list):
        try:
            segments = [
                VadSegment(**item)
                for item in vad_payload["segments"]
                if isinstance(item, dict)
            ]
            if progress:
                progress("已复用语音区间检查点")
        except (TypeError, ValueError):
            segments = None
    if segments is None:
        if progress:
            progress("正在检测语音区间")
        segments = detect_voice_activity(normalized.path, check_cancelled=check_cancelled)
        if checkpoint_store:
            checkpoint = checkpoint_store.write_json(
                "vad_segments.json",
                "vad",
                {
                    "format": "ai-jingjing-vad-v1",
                    "segments": [item.to_dict() for item in segments],
                },
            )
        else:
            checkpoint = write_vad_checkpoint(segments, work / "vad_segments.json")
    else:
        checkpoint = checkpoint_store.path("vad_segments.json") if checkpoint_store else None
    return AudioPreparationResult(
        probe=probe,
        normalized=normalized,
        vad_segments=tuple(segments),
        vad_checkpoint=str(checkpoint),
    )
