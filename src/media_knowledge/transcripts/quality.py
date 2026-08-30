from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from .schema import TranscriptQuality, TranscriptSegment, TranscriptV2


@dataclass(frozen=True, slots=True)
class TranscriptQualityIssue:
    code: str
    severity: str
    message: str
    segment_ids: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranscriptQualityReport:
    status: str
    issues: tuple[TranscriptQualityIssue, ...]
    metrics: dict[str, object] = field(default_factory=dict)

    @property
    def accepted_for_indexing(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "accepted_for_indexing": self.accepted_for_indexing,
            "warnings": [issue.message for issue in self.issues],
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
        }

    def to_quality(self) -> TranscriptQuality:
        return TranscriptQuality(
            status=self.status,
            warnings=tuple(issue.message for issue in self.issues),
            metrics=dict(self.metrics),
        )


def _overlap_ms(start: int, end: int, intervals: Sequence[tuple[int, int]]) -> int:
    return sum(max(0, min(end, interval_end) - max(start, interval_start)) for interval_start, interval_end in intervals)


def _repeated_phrase(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    # Detect a phrase of at least two characters repeated three or more times.
    return bool(re.search(r"(.{2,16}?)\1{2,}", compact))


def evaluate_transcript_quality(
    transcript: TranscriptV2,
    *,
    expected_speakers: int | None = None,
    expected_language: str | None = None,
    silence_intervals_ms: Sequence[tuple[int, int]] = (),
    audio_metrics: Mapping[str, object] | None = None,
) -> TranscriptQualityReport:
    """Run deterministic, offline quality checks without rewriting any text."""

    segments = sorted(transcript.segments, key=lambda item: (item.ordinal, item.start_ms, item.id))
    duration_ms = max(0, int(transcript.source.duration_ms))
    issues: list[TranscriptQualityIssue] = []
    hard_timeline_ids: list[str] = []
    out_of_bounds: list[str] = []
    out_of_order: list[str] = []
    empty_ids: list[str] = []
    truncated_ids: list[str] = []
    repetition_ids: list[str] = []
    silence_ids: list[str] = []
    abnormal_rate_ids: list[str] = []
    previous_start = -1
    previous_normalized = ""
    same_text_streak = 0
    short_speaker_segments = 0
    unknown_duration = 0
    overlap_duration = 0
    speech_duration = 0
    all_text: list[str] = []
    flagged: dict[str, list[str]] = {
        "generated_loop": [],
        "language_mismatch": [],
        "professional_term": [],
        "number_unit": [],
        "speaker_alignment_unavailable": [],
    }

    for segment in segments:
        text = segment.effective_text.strip()
        all_text.append(text)
        segment_duration = max(0, segment.end_ms - segment.start_ms)
        speech_duration += segment_duration
        if not text:
            empty_ids.append(segment.id)
        if segment.start_ms < 0 or segment.end_ms <= segment.start_ms:
            hard_timeline_ids.append(segment.id)
        if duration_ms and segment.end_ms > duration_ms + max(1000, int(duration_ms * 0.01)):
            out_of_bounds.append(segment.id)
        if segment.start_ms < previous_start:
            out_of_order.append(segment.id)
        previous_start = segment.start_ms
        normalized = re.sub(r"\W+", "", text, flags=re.UNICODE).casefold()
        if normalized and normalized == previous_normalized:
            same_text_streak += 1
        else:
            same_text_streak = 1
        previous_normalized = normalized
        if same_text_streak >= 3 or _repeated_phrase(text):
            repetition_ids.append(segment.id)
        flags = {str(item).strip().casefold() for item in segment.flags}
        if "truncated" in flags or "finish_reason_truncated" in flags:
            truncated_ids.append(segment.id)
        for code in flagged:
            if code in flags:
                flagged[code].append(segment.id)
        if "overlap" in flags:
            overlap_duration += segment_duration
        if not segment.speaker_id or segment.speaker_id == "speaker_unknown":
            unknown_duration += segment_duration
        if segment_duration < 1000 and segment.speaker_id:
            short_speaker_segments += 1
        seconds = segment_duration / 1000.0
        if seconds > 0 and len(re.sub(r"\s+", "", text)) / seconds > 30:
            abnormal_rate_ids.append(segment.id)
        if segment_duration and silence_intervals_ms:
            silent = _overlap_ms(segment.start_ms, segment.end_ms, silence_intervals_ms)
            if silent / segment_duration >= 0.8 and len(text) >= 8:
                silence_ids.append(segment.id)

    if not segments or not any(all_text):
        issues.append(TranscriptQualityIssue("empty_transcript", "error", "转写结果为空"))
    if hard_timeline_ids:
        issues.append(TranscriptQualityIssue(
            "invalid_timestamp", "error", "存在起止时间无效的片段", tuple(hard_timeline_ids)
        ))
    if out_of_bounds:
        issues.append(TranscriptQualityIssue(
            "timestamp_out_of_bounds", "error", "片段时间超出原始媒体时长", tuple(out_of_bounds)
        ))
    if out_of_order:
        issues.append(TranscriptQualityIssue(
            "timestamp_reversal", "error", "片段时间顺序发生倒退", tuple(out_of_order)
        ))
    if empty_ids and len(empty_ids) >= max(1, len(segments) // 3):
        issues.append(TranscriptQualityIssue(
            "empty_segments", "warning", "较多片段没有可用文字", tuple(empty_ids)
        ))
    if repetition_ids:
        issues.append(TranscriptQualityIssue(
            "repetition", "warning", "检测到连续重复短语或疑似生成循环", tuple(dict.fromkeys(repetition_ids))
        ))
    if truncated_ids:
        issues.append(TranscriptQualityIssue(
            "truncated", "warning", "模型报告结果可能被截断，需要人工核验", tuple(truncated_ids)
        ))
    if silence_ids:
        issues.append(TranscriptQualityIssue(
            "silence_hallucination", "warning", "静音区间出现了较多转写文字", tuple(silence_ids)
        ))
    if abnormal_rate_ids:
        issues.append(TranscriptQualityIssue(
            "abnormal_character_rate", "warning", "部分片段的字符速度异常", tuple(abnormal_rate_ids)
        ))
    for code, message in (
        ("generated_loop", "模型标记了疑似生成循环"),
        ("language_mismatch", "模型标记了语言与指定语言不符"),
        ("professional_term", "存在需要人工确认的专业术语"),
        ("number_unit", "存在需要人工确认的数字或单位"),
        ("speaker_alignment_unavailable", "缺少词级时间戳，无法可靠对齐说话人"),
    ):
        if flagged[code]:
            issues.append(TranscriptQualityIssue(
                code, "warning", message, tuple(flagged[code])
            ))

    audio = dict(audio_metrics or {})
    if audio.get("decode_ok") is False:
        issues.append(TranscriptQualityIssue(
            "audio_decode_failed", "error", "原始音频无法完整解码"
        ))
    if "duration_ms" in audio:
        try:
            probed_duration = int(audio["duration_ms"])  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            probed_duration = -1
        if probed_duration == 0:
            issues.append(TranscriptQualityIssue(
                "audio_zero_duration", "error", "音频时长为 0"
            ))
    for key, threshold, code, message in (
        ("silence_ratio", 0.85, "audio_long_silence", "音频中长时间静音比例过高"),
        ("clipping_ratio", 0.10, "audio_clipping", "音频存在严重削波"),
    ):
        try:
            value = float(audio.get(key, 0.0))
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if value >= threshold:
            issues.append(TranscriptQualityIssue(code, "warning", message))
    loudness_key = "loudness_lufs" if "loudness_lufs" in audio else "loudness_dbfs"
    try:
        loudness = float(audio.get(loudness_key, 0.0))
    except (TypeError, ValueError, OverflowError):
        loudness = 0.0
    if loudness_key in audio and audio.get(loudness_key) is not None and loudness < -45:
        issues.append(TranscriptQualityIssue(
            "audio_too_quiet", "warning", "音频音量过低，识别结果需要核验"
        ))
    if audio.get("corrupt") is True:
        issues.append(TranscriptQualityIssue(
            "audio_corrupt", "error", "音频文件可能中途损坏"
        ))

    speaker_ids = {
        segment.speaker_id
        for segment in segments
        if segment.speaker_id and segment.speaker_id != "speaker_unknown"
    }
    diarization_expected = bool(
        transcript.run.diarization_provider
        or expected_speakers is not None
        or transcript.speakers
    )
    if expected_speakers is not None and expected_speakers > 0 and len(speaker_ids) != expected_speakers:
        issues.append(TranscriptQualityIssue(
            "speaker_count_mismatch",
            "warning",
            f"预期 {expected_speakers} 位说话人，当前识别到 {len(speaker_ids)} 位",
        ))
    fragmentation_ratio = short_speaker_segments / len(segments) if segments else 0.0
    if diarization_expected and len(segments) >= 4 and fragmentation_ratio > 0.5:
        issues.append(TranscriptQualityIssue(
            "speaker_fragmentation", "warning", "说话人片段过度碎片化"
        ))
    unknown_ratio = unknown_duration / speech_duration if speech_duration else 0.0
    if diarization_expected and unknown_ratio > 0.2:
        issues.append(TranscriptQualityIssue(
            "speaker_unknown", "warning", "无法确定说话人的语音比例过高"
        ))
    overlap_ratio = overlap_duration / speech_duration if speech_duration else 0.0
    if diarization_expected and overlap_ratio > 0.15:
        issues.append(TranscriptQualityIssue(
            "speaker_overlap", "warning", "重叠讲话比例较高，需要人工核验"
        ))

    merged_text = "".join(all_text)
    language = str(expected_language or transcript.run.language or "").casefold()
    if len(merged_text) >= 20 and language in {"zh", "chinese", "中文", "zh-cn"}:
        meaningful = re.findall(r"[A-Za-z\u3400-\u9fff]", merged_text)
        chinese = re.findall(r"[\u3400-\u9fff]", merged_text)
        ratio = len(chinese) / len(meaningful) if meaningful else 0.0
        if ratio < 0.15:
            issues.append(TranscriptQualityIssue(
                "language_mismatch", "warning", "转写文本与指定中文语言明显不符"
            ))

    if any(issue.severity == "error" for issue in issues):
        status = "fail"
    elif issues:
        status = "review"
    else:
        status = "pass"
    metrics: dict[str, object] = {
        "segment_count": len(segments),
        "duration_ms": duration_ms,
        "speaker_count": len(speaker_ids),
        "speaker_unknown_ratio": round(unknown_ratio, 4),
        "speaker_overlap_ratio": round(overlap_ratio, 4),
        "speaker_fragmentation_ratio": round(fragmentation_ratio, 4),
    }
    return TranscriptQualityReport(status, tuple(issues), metrics)


__all__ = [
    "TranscriptQualityIssue",
    "TranscriptQualityReport",
    "evaluate_transcript_quality",
]
