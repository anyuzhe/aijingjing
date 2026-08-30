from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class CorrectionSuggestion:
    """A reviewable correction proposal, never an instruction to replace globally."""

    before: str
    after: str
    reason: str
    confidence: float | None = None
    confirmed: bool = False
    start_char: int | None = None
    end_char: int | None = None
    segment_id: str | None = None

    def confirm(self) -> "CorrectionSuggestion":
        return replace(self, confirmed=True)

    def reject(self) -> "CorrectionSuggestion":
        return replace(self, confirmed=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "before": self.before,
            "after": self.after,
            "reason": self.reason,
            "confidence": self.confidence,
            "confirmed": self.confirmed,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "segment_id": self.segment_id,
        }


class _GlossaryTerm(Protocol):
    canonical_term: str
    variants: Sequence[str]


def suggest_glossary_corrections(
    text: str,
    terms: Iterable[_GlossaryTerm | Mapping[str, object]],
    *,
    segment_id: str | None = None,
    confidence: float = 0.75,
) -> tuple[CorrectionSuggestion, ...]:
    """Locate variant occurrences and return unconfirmed, span-specific proposals.

    This deliberately does not mutate text. A glossary hit is contextual evidence,
    not proof that the recognized phrase is wrong.
    """

    suggestions: list[CorrectionSuggestion] = []
    seen: set[tuple[int, int, str]] = set()
    for term in terms:
        if isinstance(term, Mapping):
            canonical = str(term.get("canonical_term") or term.get("term") or "").strip()
            raw_variants = term.get("variants") or term.get("aliases") or ()
        else:
            canonical = str(term.canonical_term or "").strip()
            raw_variants = term.variants
        if not canonical or not isinstance(raw_variants, Sequence) or isinstance(raw_variants, (str, bytes)):
            continue
        for raw_variant in raw_variants:
            variant = str(raw_variant or "").strip()
            if not variant or variant.casefold() == canonical.casefold():
                continue
            for match in re.finditer(re.escape(variant), text, flags=re.IGNORECASE):
                key = (match.start(), match.end(), canonical)
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append(
                    CorrectionSuggestion(
                        before=match.group(0),
                        after=canonical,
                        reason=f"命中术语词库候选：{canonical}",
                        confidence=max(0.0, min(1.0, float(confidence))),
                        confirmed=False,
                        start_char=match.start(),
                        end_char=match.end(),
                        segment_id=segment_id,
                    )
                )
    return tuple(sorted(suggestions, key=lambda item: (item.start_char or 0, item.end_char or 0)))


def apply_confirmed_corrections(
    text: str,
    suggestions: Iterable[CorrectionSuggestion],
) -> str:
    """Apply only confirmed, non-overlapping suggestions at validated spans.

    Suggestions without a span are rejected instead of falling back to global
    ``str.replace``. This protects legitimate occurrences elsewhere in a segment.
    """

    confirmed = [item for item in suggestions if item.confirmed]
    if not confirmed:
        return text
    ranges: list[tuple[int, int, CorrectionSuggestion]] = []
    for item in confirmed:
        if item.start_char is None or item.end_char is None:
            raise ValueError("已确认的术语校订必须包含精确字符范围")
        start, end = int(item.start_char), int(item.end_char)
        if start < 0 or end <= start or end > len(text):
            raise ValueError("术语校订字符范围无效")
        if text[start:end] != item.before:
            raise ValueError("术语校订原文与指定字符范围不一致")
        ranges.append((start, end, item))
    ranges.sort(key=lambda value: (value[0], value[1]))
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError("术语校订范围互相重叠")
    result = text
    for start, end, item in reversed(ranges):
        result = result[:start] + item.after + result[end:]
    return result


__all__ = [
    "CorrectionSuggestion",
    "apply_confirmed_corrections",
    "suggest_glossary_corrections",
]
