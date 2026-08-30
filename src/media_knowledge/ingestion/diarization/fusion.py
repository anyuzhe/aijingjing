from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence

from .base import DIARIZATION_UNKNOWN_SPEAKER, DiarizationSegment


@dataclass(frozen=True, slots=True)
class TimedWord:
    start: float
    end: float
    text: str
    confidence: float | None = None
    speaker_id: str | None = None
    overlap: bool = False
    quality_status: str = "pass"
    flags: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        start, end = float(self.start), float(self.end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("词时间戳无效")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "text", str(self.text or ""))
        object.__setattr__(self, "quality_status", str(self.quality_status or "pass").casefold())
        object.__setattr__(self, "flags", tuple(dict.fromkeys(str(item) for item in self.flags if str(item))))


@dataclass(frozen=True, slots=True)
class SpeakerCue:
    start: float
    end: float
    speaker_id: str
    raw_text: str
    confidence: float | None = None
    overlap: bool = False
    quality_status: str = "pass"
    flags: tuple[str, ...] = ()
    words: tuple[TimedWord, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "start_ms": round(self.start * 1000),
            "end_ms": round(self.end * 1000),
            "speaker_id": self.speaker_id,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "overlap": self.overlap,
            "quality_status": self.quality_status,
            "flags": list(self.flags),
            "words": [
                {
                    "start_ms": round(item.start * 1000),
                    "end_ms": round(item.end * 1000),
                    "text": item.text,
                    "confidence": item.confidence,
                    "speaker_id": item.speaker_id,
                    "overlap": item.overlap,
                }
                for item in self.words
            ],
        }


def _overlap(start: float, end: float, segment: DiarizationSegment) -> float:
    if end == start:
        return 1e-9 if segment.start <= start < segment.end else 0.0
    return max(0.0, min(end, segment.end) - max(start, segment.start))


def _has_simultaneous_speakers(
    start: float,
    end: float,
    candidates: Sequence[DiarizationSegment],
) -> bool:
    if any(item.overlap for item in candidates):
        return True
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            if first.speaker_id == second.speaker_id:
                continue
            shared_start = max(start, first.start, second.start)
            shared_end = min(end, first.end, second.end)
            if shared_end > shared_start:
                return True
    return False


def fuse_words_with_speakers(
    words: Iterable[TimedWord],
    diarization: Iterable[DiarizationSegment],
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> list[TimedWord]:
    """Assign every word to the speaker with the greatest temporal overlap."""

    speaker_segments = sorted(diarization, key=lambda item: (item.start, item.end, item.speaker_id))
    fused: list[TimedWord] = []
    for word in words:
        if check_cancelled:
            check_cancelled()
        if "speaker_alignment_unavailable" in word.flags:
            # A whole ASR segment is not a word-alignment fact.  Assigning it by
            # maximum overlap would invent speaker certainty whenever the segment
            # spans turns, so preserve only its real segment timing and text.
            fused.append(replace(
                word,
                speaker_id=DIARIZATION_UNKNOWN_SPEAKER,
                overlap=False,
            ))
            continue
        candidates = [
            item for item in speaker_segments if _overlap(word.start, word.end, item) > 0
        ]
        totals: dict[str, float] = {}
        first_start: dict[str, float] = {}
        for item in candidates:
            totals[item.speaker_id] = totals.get(item.speaker_id, 0.0) + _overlap(word.start, word.end, item)
            first_start[item.speaker_id] = min(first_start.get(item.speaker_id, item.start), item.start)
        speaker = DIARIZATION_UNKNOWN_SPEAKER
        if totals:
            speaker = min(totals, key=lambda item: (-totals[item], first_start[item], item))
        fused.append(
            replace(
                word,
                speaker_id=speaker,
                overlap=_has_simultaneous_speakers(word.start, word.end, candidates),
            )
        )
    return fused


def _join_word(previous: str, current: str) -> str:
    if not previous:
        return current
    if not current:
        return previous
    separator = " " if re.search(r"[A-Za-z0-9]$", previous) and re.match(r"[A-Za-z0-9]", current) else ""
    return previous + separator + current


def _cue(words: Sequence[TimedWord]) -> SpeakerCue:
    confidences = [item.confidence for item in words if item.confidence is not None]
    text = ""
    for item in words:
        text = _join_word(text, item.text.strip())
    return SpeakerCue(
        start=words[0].start,
        end=words[-1].end,
        speaker_id=words[0].speaker_id or DIARIZATION_UNKNOWN_SPEAKER,
        raw_text=text,
        confidence=sum(confidences) / len(confidences) if confidences else None,
        overlap=any(item.overlap for item in words),
        quality_status=words[0].quality_status,
        flags=tuple(dict.fromkeys(flag for item in words for flag in item.flags)),
        words=tuple(words),
    )


def build_speaker_cues(
    words: Iterable[TimedWord],
    *,
    silence_gap_seconds: float = 0.8,
    max_duration_seconds: float = 15.0,
    max_characters: int = 60,
    check_cancelled: Callable[[], None] | None = None,
) -> list[SpeakerCue]:
    """Merge word timestamps into readable, speaker-preserving transcript cues."""

    if silence_gap_seconds < 0 or max_duration_seconds <= 0 or max_characters < 1:
        raise ValueError("CueBuilder 参数无效")
    ordered = sorted(words, key=lambda item: (item.start, item.end))
    cues: list[SpeakerCue] = []
    current: list[TimedWord] = []
    for word in ordered:
        if check_cancelled:
            check_cancelled()
        if not word.text.strip():
            continue
        if current:
            previous = current[-1]
            candidate_text = _join_word(_cue(current).raw_text, word.text.strip())
            must_split = (
                word.speaker_id != previous.speaker_id
                or word.quality_status != previous.quality_status
                or word.overlap != previous.overlap
                or word.start - previous.end > silence_gap_seconds
                or word.end - current[0].start > max_duration_seconds
                or len(candidate_text) > max_characters
            )
            if must_split:
                cues.append(_cue(current))
                current = []
        current.append(word)
        if re.search(r"[。！？!?]\s*$", word.text):
            cues.append(_cue(current))
            current = []
    if current:
        cues.append(_cue(current))
    return cues
