from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


TRANSCRIPT_V1_FORMAT = "ai-jingjing-transcript-v1"
TRANSCRIPT_V2_FORMAT = "ai-jingjing-transcript-v2"
QUALITY_STATUSES = frozenset({"pass", "review", "fail"})
SPEAKER_NAME_SOURCES = frozenset({"automatic", "manual"})


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_float(value: object, default: float | None = None) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _milliseconds(value: object, *, seconds: bool = False) -> int:
    number = _finite_float(value, 0.0) or 0.0
    if seconds:
        number *= 1000.0
    return max(0, int(round(number)))


def _text(value: object) -> str:
    return str(value or "").strip()


def _json_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class TranscriptWord:
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None
    speaker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TranscriptWord":
        seconds = "start_ms" not in value and "start" in value
        confidence = _finite_float(value.get("confidence"))
        return cls(
            start_ms=_milliseconds(value.get("start_ms", value.get("start")), seconds=seconds),
            end_ms=_milliseconds(value.get("end_ms", value.get("end")), seconds=seconds),
            text=_text(value.get("text")),
            confidence=confidence,
            speaker_id=_text(value.get("speaker_id")) or None,
            metadata=_mapping(value.get("metadata")),
        )


@dataclass(slots=True)
class TranscriptSource:
    name: str
    sha256: str
    duration_ms: int
    original_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TranscriptSource":
        duration_ms = value.get("duration_ms")
        if duration_ms is None:
            duration_ms = _milliseconds(value.get("duration_seconds"), seconds=True)
        return cls(
            name=_text(value.get("name") or value.get("source")) or "未命名音视频",
            sha256=_text(value.get("sha256") or value.get("checksum")),
            duration_ms=_milliseconds(duration_ms),
            original_uri=_text(value.get("original_uri")) or None,
            metadata=_mapping(value.get("metadata")),
        )


@dataclass(slots=True)
class TranscriptRun:
    id: str
    profile: str
    provider: str
    model: str
    language: str | None = None
    word_timestamps: bool = False
    diarization_provider: str | None = None
    context_profile: str | None = None
    fallback: dict[str, Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TranscriptRun":
        fallback = value.get("fallback")
        return cls(
            id=_text(value.get("id")),
            profile=_text(value.get("profile")) or "compatibility",
            provider=_text(value.get("provider") or value.get("engine")) or "unknown",
            model=_text(value.get("model")) or "unknown",
            language=_text(value.get("language")) or None,
            word_timestamps=bool(value.get("word_timestamps", False)),
            diarization_provider=_text(value.get("diarization_provider")) or None,
            context_profile=_text(value.get("context_profile")) or None,
            fallback=_mapping(fallback) if isinstance(fallback, Mapping) else None,
            config=_mapping(value.get("config")),
        )


@dataclass(slots=True)
class TranscriptSpeaker:
    id: str
    display_name: str | None = None
    name_source: str = "automatic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TranscriptSpeaker":
        source = _text(value.get("name_source")) or "automatic"
        if source not in SPEAKER_NAME_SOURCES:
            source = "automatic"
        return cls(
            id=_text(value.get("id") or value.get("speaker_id")),
            display_name=_text(value.get("display_name")) or None,
            name_source=source,
            metadata=_mapping(value.get("metadata")),
        )


@dataclass(slots=True)
class TranscriptSegment:
    id: str
    ordinal: int
    start_ms: int
    end_ms: int
    speaker_id: str | None
    raw_text: str
    corrected_text: str | None = None
    confidence: float | None = None
    flags: tuple[str, ...] = ()
    words: tuple[TranscriptWord, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_text(self) -> str:
        return self.corrected_text if self.corrected_text is not None else self.raw_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ordinal": self.ordinal,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "speaker_id": self.speaker_id,
            "raw_text": self.raw_text,
            "corrected_text": self.corrected_text,
            "confidence": self.confidence,
            "flags": list(self.flags),
            "words": [word.to_dict() for word in self.words],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        fallback_id: str = "",
        fallback_ordinal: int = 0,
    ) -> "TranscriptSegment":
        seconds = "start_ms" not in value and "start" in value
        flags = value.get("flags")
        words = value.get("words")
        raw_text = _text(value.get("raw_text"))
        if not raw_text:
            raw_text = _text(value.get("text"))
        corrected = value.get("corrected_text")
        return cls(
            id=_text(value.get("id")) or fallback_id,
            ordinal=int(value.get("ordinal", fallback_ordinal)),
            start_ms=_milliseconds(value.get("start_ms", value.get("start")), seconds=seconds),
            end_ms=_milliseconds(value.get("end_ms", value.get("end")), seconds=seconds),
            speaker_id=_text(value.get("speaker_id")) or None,
            raw_text=raw_text,
            corrected_text=None if corrected is None else str(corrected),
            confidence=_finite_float(value.get("confidence")),
            flags=tuple(_text(item) for item in flags if _text(item))
            if isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)) else (),
            words=tuple(
                TranscriptWord.from_dict(item)
                for item in words
                if isinstance(item, Mapping)
            ) if isinstance(words, Sequence) and not isinstance(words, (str, bytes)) else (),
            metadata=_mapping(value.get("metadata")),
        )


@dataclass(slots=True)
class TranscriptQuality:
    status: str = "review"
    warnings: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status if self.status in QUALITY_STATUSES else "review"
        return {"status": status, "warnings": list(self.warnings), "metrics": dict(self.metrics)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "TranscriptQuality":
        data = _mapping(value)
        raw_status = _text(data.get("status")).casefold()
        status = {
            "pass": "pass",
            "passed": "pass",
            "ok": "pass",
            "warn": "review",
            "warning": "review",
            "review": "review",
            "fail": "fail",
            "failed": "fail",
        }.get(raw_status, "review")
        warnings = data.get("warnings")
        if not isinstance(warnings, Sequence) or isinstance(warnings, (str, bytes)):
            warnings = []
        return cls(
            status=status,
            warnings=tuple(_text(item) for item in warnings if _text(item)),
            metrics=_mapping(data.get("metrics")),
        )


@dataclass(slots=True)
class TranscriptV2:
    source: TranscriptSource
    run: TranscriptRun
    speakers: list[TranscriptSpeaker]
    segments: list[TranscriptSegment]
    quality: TranscriptQuality = field(default_factory=TranscriptQuality)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def format(self) -> str:
        return TRANSCRIPT_V2_FORMAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": TRANSCRIPT_V2_FORMAT,
            "source": self.source.to_dict(),
            "run": self.run.to_dict(),
            "speakers": [speaker.to_dict() for speaker in self.speakers],
            "segments": [segment.to_dict() for segment in self.segments],
            "quality": self.quality.to_dict(),
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TranscriptV2":
        return transcript_from_dict(value)

    @classmethod
    def from_json(cls, value: str | bytes) -> "TranscriptV2":
        parsed = json.loads(value)
        if not isinstance(parsed, Mapping):
            raise ValueError("转写 JSON 顶层必须是对象")
        return transcript_from_dict(parsed)


def _v1_to_v2(value: Mapping[str, Any]) -> TranscriptV2:
    digest = _json_digest(value)
    source_value = value.get("source")
    if isinstance(source_value, Mapping):
        source_data = dict(source_value)
    else:
        source_data = {"name": _text(source_value)}
    source_data.setdefault("duration_seconds", value.get("duration_seconds"))
    source_data.setdefault("sha256", _text(value.get("source_checksum")) or digest)
    source = TranscriptSource.from_dict(source_data)
    fallback_reasons = value.get("fallback_reasons")
    fallback: dict[str, Any] | None = None
    if isinstance(fallback_reasons, Sequence) and not isinstance(fallback_reasons, (str, bytes)):
        reasons = [_text(item) for item in fallback_reasons if _text(item)]
        if reasons:
            fallback = {"reasons": reasons}
    run = TranscriptRun(
        id=f"asr-run-v1-{digest[:16]}",
        profile="compatibility",
        provider=_text(value.get("engine")) or "legacy",
        model=_text(value.get("model")) or "unknown",
        language=_text(value.get("language")) or None,
        word_timestamps=False,
        fallback=fallback,
        config={
            "legacy_format": TRANSCRIPT_V1_FORMAT,
            "device": value.get("device"),
            "compute_type": value.get("compute_type"),
        },
    )
    raw_segments = value.get("segments")
    segments = [
        TranscriptSegment.from_dict(item, fallback_id=f"seg_{index:04d}", fallback_ordinal=index - 1)
        for index, item in enumerate(
            raw_segments if isinstance(raw_segments, Sequence) and not isinstance(raw_segments, (str, bytes)) else (),
            1,
        )
        if isinstance(item, Mapping)
    ]
    integrity = value.get("integrity")
    quality = TranscriptQuality.from_dict(integrity if isinstance(integrity, Mapping) else None)
    return TranscriptV2(
        source=source,
        run=run,
        speakers=[],
        segments=segments,
        quality=quality,
        metadata={"converted_from": TRANSCRIPT_V1_FORMAT},
    )


def transcript_from_dict(value: Mapping[str, Any]) -> TranscriptV2:
    format_name = _text(value.get("format"))
    if format_name in {"", TRANSCRIPT_V1_FORMAT} and "run" not in value:
        return _v1_to_v2(value)
    if format_name not in {"", TRANSCRIPT_V2_FORMAT}:
        raise ValueError(f"不支持的转写格式：{format_name}")
    source_raw = value.get("source")
    run_raw = value.get("run")
    if not isinstance(source_raw, Mapping) or not isinstance(run_raw, Mapping):
        raise ValueError("Transcript V2 必须包含 source 和 run")
    source = TranscriptSource.from_dict(source_raw)
    run = TranscriptRun.from_dict(run_raw)
    if not run.id:
        run.id = f"asr-run-{_json_digest(value)[:16]}"
    if not source.sha256:
        source.sha256 = _json_digest(source_raw)
    speakers_raw = value.get("speakers")
    segments_raw = value.get("segments")
    speakers = [
        TranscriptSpeaker.from_dict(item)
        for item in speakers_raw
        if isinstance(item, Mapping)
    ] if isinstance(speakers_raw, Sequence) and not isinstance(speakers_raw, (str, bytes)) else []
    segments = [
        TranscriptSegment.from_dict(item, fallback_id=f"seg_{index:04d}", fallback_ordinal=index - 1)
        for index, item in enumerate(
            segments_raw if isinstance(segments_raw, Sequence) and not isinstance(segments_raw, (str, bytes)) else (),
            1,
        )
        if isinstance(item, Mapping)
    ]
    quality_raw = value.get("quality")
    return TranscriptV2(
        source=source,
        run=run,
        speakers=speakers,
        segments=segments,
        quality=TranscriptQuality.from_dict(quality_raw if isinstance(quality_raw, Mapping) else None),
        metadata=_mapping(value.get("metadata")),
    )


def read_transcript(path: str | Path) -> TranscriptV2:
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError("转写 JSON 顶层必须是对象")
    return transcript_from_dict(parsed)


def write_transcript(transcript: TranscriptV2, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(transcript.to_json())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.close(handle)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "QUALITY_STATUSES",
    "SPEAKER_NAME_SOURCES",
    "TRANSCRIPT_V1_FORMAT",
    "TRANSCRIPT_V2_FORMAT",
    "TranscriptQuality",
    "TranscriptRun",
    "TranscriptSegment",
    "TranscriptSource",
    "TranscriptSpeaker",
    "TranscriptV2",
    "TranscriptWord",
    "read_transcript",
    "transcript_from_dict",
    "write_transcript",
]
