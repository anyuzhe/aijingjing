from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from importlib import import_module
from pathlib import Path

from ..base import AsrProviderError
from ..types import AsrSegment, AsrWord


def safe_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def value_of(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def parse_words(raw: object) -> tuple[AsrWord, ...]:
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes, Mapping)):
        return ()
    words: list[AsrWord] = []
    for value in raw:
        start = safe_float(value_of(value, "start"))
        end = safe_float(value_of(value, "end"))
        text = str(value_of(value, "word", value_of(value, "text", "")) or "").strip()
        if start is None or end is None or not text:
            continue
        probability = safe_float(value_of(value, "probability", value_of(value, "confidence")))
        speaker = str(value_of(value, "speaker_id", "") or "").strip() or None
        words.append(AsrWord(start, max(start, end), text, probability, speaker))
    return tuple(words)


def parse_segments(raw: object) -> list[AsrSegment]:
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes, Mapping)):
        return []
    segments: list[AsrSegment] = []
    for value in raw:
        start = safe_float(value_of(value, "start"))
        end = safe_float(value_of(value, "end"))
        text = str(value_of(value, "text", "") or "").strip()
        if start is None or end is None:
            continue
        confidence = safe_float(value_of(value, "confidence"))
        avg_logprob = safe_float(value_of(value, "avg_logprob"))
        speaker = str(value_of(value, "speaker_id", "") or "").strip() or None
        segments.append(AsrSegment(
            start=start,
            end=max(start, end),
            text=text,
            confidence=confidence,
            avg_logprob=avg_logprob,
            words=parse_words(value_of(value, "words", ())),
            speaker_id=speaker,
        ))
    return segments


def normalize_language(value: object) -> str | None:
    if isinstance(value, (list, tuple)):
        unique: list[str] = []
        for raw in value:
            language = str(raw or "").strip()
            if language and language not in unique:
                unique.append(language)
        return ",".join(unique) or None
    return str(value or "").strip() or None


def whisper_language(value: str | None) -> str | None:
    normalized = str(value or "").strip().casefold()
    if not normalized or normalized in {"auto", "automatic", "自动"}:
        return None
    aliases = {
        "chinese": "zh",
        "中文": "zh",
        "mandarin": "zh",
        "cantonese": "yue",
        "english": "en",
        "英语": "en",
        "japanese": "ja",
        "日语": "ja",
        "korean": "ko",
        "韩语": "ko",
    }
    return aliases.get(normalized, normalized)


def resolve_local_hf_model(reference: str | Path, *, provider_label: str) -> str:
    """Resolve a local directory/cache snapshot without permitting a download."""

    raw = str(reference or "").strip()
    candidate = Path(raw).expanduser()
    if raw and candidate.is_dir():
        try:
            return str(candidate.resolve(strict=True))
        except (OSError, RuntimeError):
            pass
    if not raw:
        raise AsrProviderError(
            f"{provider_label} 尚未选择本地模型",
            reason_code="model_not_local",
        )
    try:
        hub = import_module("huggingface_hub")
        cached = hub.snapshot_download(raw, local_files_only=True)
        resolved = Path(cached).expanduser().resolve(strict=True)
    except Exception as exc:
        raise AsrProviderError(
            f"{provider_label} 模型不在本地；请先在模型管理中安装",
            reason_code="model_not_local",
        ) from exc
    if not resolved.is_dir():
        raise AsrProviderError(
            f"{provider_label} 本地模型目录无效",
            reason_code="model_invalid",
        )
    return str(resolved)
