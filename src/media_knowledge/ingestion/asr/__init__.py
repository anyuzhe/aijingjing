from __future__ import annotations

from .base import AsrProvider, AsrProviderError, AsrRoutingError
from .cue_builder import CueBuilder
from .providers import FasterWhisperProvider, MlxWhisperProvider, Qwen3MlxProvider
from .router import AsrProviderRegistry, AsrRouter
from .types import (
    ASR_PROFILES,
    AsrAttempt,
    AsrFallback,
    AsrResult,
    AsrSegment,
    AsrWord,
    TranscriptionRequest,
    normalize_context_terms,
    normalize_profile,
    normalize_provider_id,
)


def create_default_registry() -> AsrProviderRegistry:
    registry = AsrProviderRegistry()
    registry.register(Qwen3MlxProvider())
    registry.register(MlxWhisperProvider())
    registry.register(FasterWhisperProvider())
    return registry


__all__ = [
    "ASR_PROFILES",
    "AsrAttempt",
    "AsrFallback",
    "AsrProvider",
    "AsrProviderError",
    "AsrProviderRegistry",
    "AsrResult",
    "AsrRouter",
    "AsrRoutingError",
    "AsrSegment",
    "AsrWord",
    "CueBuilder",
    "FasterWhisperProvider",
    "MlxWhisperProvider",
    "Qwen3MlxProvider",
    "TranscriptionRequest",
    "create_default_registry",
    "normalize_context_terms",
    "normalize_profile",
    "normalize_provider_id",
]
