from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from .types import AsrSegment, AsrWord


_ENDING_PUNCTUATION = re.compile(r"[。！？!?；;]$", re.UNICODE)
_NO_SPACE_BEFORE = re.compile(r"^[，。！？、；：,.!?;:)）】》]", re.UNICODE)


def _join_token(current: str, token: str) -> str:
    token = str(token or "").strip()
    if not current or not token:
        return current + token
    if _NO_SPACE_BEFORE.search(token):
        return current + token
    if current[-1].isascii() and current[-1].isalnum() and token[0].isascii() and token[0].isalnum():
        return current + " " + token
    return current + token


@dataclass(frozen=True, slots=True)
class CueBuilder:
    pause_seconds: float = 0.8
    max_duration_seconds: float = 15.0
    max_characters: int = 60

    def build(self, values: Iterable[AsrWord]) -> list[AsrSegment]:
        words = [item for item in values if item.text.strip()]
        if not words:
            return []
        cues: list[AsrSegment] = []
        group: list[AsrWord] = []
        text = ""

        def flush() -> None:
            nonlocal group, text
            if not group:
                return
            confidence_values = [
                float(item.confidence)
                for item in group
                if item.confidence is not None and math.isfinite(float(item.confidence))
            ]
            cues.append(AsrSegment(
                start=max(0.0, float(group[0].start)),
                end=max(float(group[0].start), float(group[-1].end)),
                text=text.strip(),
                confidence=(sum(confidence_values) / len(confidence_values)) if confidence_values else None,
                words=tuple(group),
                speaker_id=group[0].speaker_id,
            ))
            group = []
            text = ""

        for word in words:
            normalized = AsrWord(
                max(0.0, float(word.start)),
                max(float(word.start), float(word.end)),
                str(word.text or "").strip(),
                word.confidence,
                word.speaker_id,
            )
            candidate = _join_token(text, normalized.text)
            if group:
                pause = normalized.start - group[-1].end
                duration = normalized.end - group[0].start
                speaker_changed = normalized.speaker_id != group[0].speaker_id
                if (
                    speaker_changed
                    or pause > max(0.0, self.pause_seconds)
                    or duration > max(0.1, self.max_duration_seconds)
                    or len(candidate) > max(1, self.max_characters)
                ):
                    flush()
                    candidate = normalized.text
            group.append(normalized)
            text = candidate
            if _ENDING_PUNCTUATION.search(normalized.text):
                flush()
        flush()
        return cues
