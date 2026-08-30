from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Protocol, Sequence, runtime_checkable


DIARIZATION_UNKNOWN_SPEAKER = "speaker_unknown"

ProgressCallback = Callable[[str], None]
CancellationCallback = Callable[[], None]


class DiarizationUnavailable(RuntimeError):
    """Raised when no installed, local diarization provider can run."""


def _finite_time(value: float, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} 必须是有效时间") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} 必须是非负有限数")
    return number


def is_existing_local_model(path: Path | None) -> bool:
    """Accept model assets only from an explicit existing filesystem path.

    Provider execution deliberately does not accept repository identifiers.  Model
    installation is a separate, user-triggered operation; an ingestion task must
    never turn a missing model into an implicit network download.
    """

    if path is None:
        return False
    raw = str(path).strip()
    if not raw or "://" in raw or raw.casefold().startswith(("hf:", "http:", "https:")):
        return False
    try:
        return path.expanduser().is_absolute() and path.expanduser().exists()
    except OSError:
        return False


@dataclass(frozen=True, slots=True)
class DiarizationRequest:
    audio_path: Path
    model_path: Path | None = None
    preferred_provider: str = "auto"
    expected_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    allow_fallback: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio_path", Path(self.audio_path).expanduser())
        if self.model_path is not None:
            object.__setattr__(self, "model_path", Path(self.model_path).expanduser())
        provider = str(self.preferred_provider or "auto").strip().casefold().replace("_", "-")
        object.__setattr__(self, "preferred_provider", provider or "auto")

        for name, value in (
            ("预计说话人数", self.expected_speakers),
            ("最少说话人数", self.min_speakers),
            ("最多说话人数", self.max_speakers),
        ):
            if value is not None and (isinstance(value, bool) or int(value) != value or value < 1):
                raise ValueError(f"{name}必须是正整数")
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("最少说话人数不能大于最多说话人数")
        if self.expected_speakers is not None:
            if self.min_speakers is not None and self.expected_speakers < self.min_speakers:
                raise ValueError("预计说话人数不能小于最少说话人数")
            if self.max_speakers is not None and self.expected_speakers > self.max_speakers:
                raise ValueError("预计说话人数不能大于最多说话人数")

    @property
    def exact_speakers(self) -> int | None:
        if self.expected_speakers is not None:
            return self.expected_speakers
        if self.min_speakers is not None and self.min_speakers == self.max_speakers:
            return self.min_speakers
        return None


@dataclass(frozen=True, slots=True)
class DiarizationSegment:
    start: float
    end: float
    speaker_id: str
    confidence: float | None = None
    overlap: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        start = _finite_time(self.start, field_name="说话人片段开始时间")
        end = _finite_time(self.end, field_name="说话人片段结束时间")
        if end <= start:
            raise ValueError("说话人片段结束时间必须晚于开始时间")
        speaker = str(self.speaker_id or "").strip()
        if not speaker:
            speaker = DIARIZATION_UNKNOWN_SPEAKER
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "speaker_id", speaker)
        if self.confidence is not None:
            confidence = float(self.confidence)
            object.__setattr__(self, "confidence", max(0.0, min(1.0, confidence)))


@dataclass(slots=True)
class DiarizationResult:
    provider_id: str
    model: str
    segments: list[DiarizationSegment]
    fallback_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def speaker_count(self) -> int:
        return len(
            {
                item.speaker_id
                for item in self.segments
                if item.speaker_id != DIARIZATION_UNKNOWN_SPEAKER
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider_id,
            "model": self.model,
            "speaker_count": self.speaker_count,
            "segments": [
                {
                    "start": item.start,
                    "end": item.end,
                    "speaker_id": item.speaker_id,
                    "confidence": item.confidence,
                    "overlap": item.overlap,
                    "metadata": dict(item.metadata),
                }
                for item in self.segments
            ],
            "fallback_reasons": list(self.fallback_reasons),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def normalized(
        cls,
        *,
        provider_id: str,
        model: str,
        segments: Sequence[DiarizationSegment],
        fallback_reasons: Sequence[str] = (),
        warnings: Sequence[str] = (),
        metadata: dict[str, object] | None = None,
    ) -> "DiarizationResult":
        ordered = sorted(segments, key=lambda item: (item.start, item.end, item.speaker_id))
        aliases: dict[str, str] = {}
        normalized: list[DiarizationSegment] = []
        for item in ordered:
            raw = item.speaker_id
            if raw == DIARIZATION_UNKNOWN_SPEAKER:
                anonymous = raw
            else:
                anonymous = aliases.setdefault(raw, f"spk_{len(aliases):02d}")
            # Never retain provider labels or inferred identities in persisted
            # metadata.  Display names are a separate, manual user decision.
            cleaned_metadata = {
                key: value
                for key, value in item.metadata.items()
                if key.casefold() not in {"speaker", "speaker_id", "label", "name", "display_name"}
            }
            normalized.append(replace(item, speaker_id=anonymous, metadata=cleaned_metadata))
        return cls(
            provider_id=str(provider_id),
            model=str(model),
            segments=normalized,
            fallback_reasons=list(dict.fromkeys(str(item) for item in fallback_reasons if str(item))),
            warnings=list(dict.fromkeys(str(item) for item in warnings if str(item))),
            metadata=dict(metadata or {}),
        )


@runtime_checkable
class DiarizationProvider(Protocol):
    provider_id: str

    def availability(self, request: DiarizationRequest) -> tuple[bool, str | None]:
        ...

    def diarize(
        self,
        request: DiarizationRequest,
        progress: ProgressCallback | None = None,
        check_cancelled: CancellationCallback | None = None,
    ) -> DiarizationResult:
        ...
