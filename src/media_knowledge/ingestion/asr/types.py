from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable


ASR_PROFILES = ("chinese-accuracy", "fast-preview", "compatibility", "custom")


def normalize_profile(value: object) -> str:
    normalized = str(value or "custom").strip().casefold().replace("_", "-")
    return normalized if normalized in ASR_PROFILES else "custom"


def normalize_provider_id(value: object) -> str | None:
    normalized = str(value or "").strip().casefold().replace("_", "-")
    aliases = {
        "": None,
        "auto": None,
        "qwen": "qwen3-mlx",
        "qwen3": "qwen3-mlx",
        "qwen3-asr": "qwen3-mlx",
        "mlx": "mlx-whisper",
        "whisper-mlx": "mlx-whisper",
        "faster": "faster-whisper",
        "cpu": "faster-whisper",
        "cuda": "faster-whisper",
    }
    return aliases.get(normalized, normalized or None)


def normalize_context_terms(values: Iterable[object] | None) -> tuple[str, ...]:
    """Return a small, deterministic hotword list safe to pass to a model prompt."""

    unique: list[str] = []
    seen: set[str] = set()
    for raw in values or ():
        value = " ".join(str(raw or "").replace("\x00", " ").split()).strip()[:128]
        folded = value.casefold()
        if not value or folded in seen:
            continue
        seen.add(folded)
        unique.append(value)
        if len(unique) >= 200:
            break
    return tuple(unique)


@dataclass(frozen=True, slots=True)
class AsrWord:
    start: float
    end: float
    text: str
    confidence: float | None = None
    speaker_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AsrSegment:
    start: float
    end: float
    text: str
    confidence: float | None = None
    avg_logprob: float | None = None
    words: tuple[AsrWord, ...] = ()
    speaker_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AsrFallback:
    fallback_from: str
    fallback_to: str
    reason_code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    audio_path: Path
    profile: str = "custom"
    provider: str | None = None
    model: str = "small"
    model_path: Path | None = None
    whisper_fallback_model_path: Path | None = None
    language: str | None = None
    context_terms: tuple[str, ...] = ()
    word_timestamps: bool = False
    allow_fallback: bool = True
    duration_seconds: float = 0.0
    device: str = "auto"
    compute_type: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio_path", Path(self.audio_path).expanduser())
        object.__setattr__(self, "profile", normalize_profile(self.profile))
        object.__setattr__(self, "provider", normalize_provider_id(self.provider))
        object.__setattr__(self, "model", str(self.model or "small").strip() or "small")
        if self.model_path is not None:
            object.__setattr__(self, "model_path", Path(self.model_path).expanduser())
        if self.whisper_fallback_model_path is not None:
            object.__setattr__(
                self,
                "whisper_fallback_model_path",
                Path(self.whisper_fallback_model_path).expanduser(),
            )
        language = str(self.language or "").strip()
        object.__setattr__(self, "language", language or None)
        object.__setattr__(self, "context_terms", normalize_context_terms(self.context_terms))
        object.__setattr__(self, "duration_seconds", max(0.0, float(self.duration_seconds or 0.0)))
        object.__setattr__(self, "device", str(self.device or "auto").strip().casefold())
        object.__setattr__(self, "compute_type", str(self.compute_type or "auto").strip().casefold())

    def with_attempt(
        self,
        *,
        provider: str,
        model: str,
        model_path: Path | None,
        device: str,
        compute_type: str,
        language: str | None = None,
        word_timestamps: bool | None = None,
    ) -> "TranscriptionRequest":
        return replace(
            self,
            provider=provider,
            model=model,
            model_path=model_path,
            device=device,
            compute_type=compute_type,
            language=self.language if language is None else language,
            word_timestamps=self.word_timestamps if word_timestamps is None else word_timestamps,
        )


@dataclass(frozen=True, slots=True)
class AsrAttempt:
    provider: str
    model: str
    model_path: Path | None = None
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None
    word_timestamps: bool | None = None


@dataclass(slots=True)
class AsrResult:
    provider_id: str
    model: str
    device: str
    compute_type: str
    language: str | None
    segments: list[AsrSegment]
    finish_reason: str = "stop"
    truncated: bool = False
    fallback_history: list[AsrFallback] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, object]:
        return {
            "provider": self.provider_id,
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "finish_reason": self.finish_reason,
            "truncated": bool(self.truncated),
            "fallback_history": [item.to_dict() for item in self.fallback_history],
            "warnings": list(dict.fromkeys(self.warnings)),
        }
