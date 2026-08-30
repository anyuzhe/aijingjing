from __future__ import annotations

from collections.abc import Iterable

from ..types import CancelledError
from ...resource_scheduler import LOCAL_HEAVY_TASKS
from .base import (
    CancellationCallback,
    DiarizationProvider,
    DiarizationRequest,
    DiarizationResult,
    DiarizationUnavailable,
    ProgressCallback,
)


class DiarizationRouter:
    """Select an already-installed local provider with explicit fallback history."""

    def __init__(self, providers: Iterable[DiarizationProvider] | None = None) -> None:
        if providers is None:
            from .pyannote_provider import PyannoteProvider
            from .sherpa_provider import SherpaOnnxProvider

            providers = (PyannoteProvider(), SherpaOnnxProvider())
        self.providers = list(providers)

    def _ordered(self, preferred: str) -> list[DiarizationProvider]:
        preferred = {
            "sherpa": "sherpa-onnx",
            "pyannote-community-1": "pyannote",
        }.get(preferred, preferred)
        if preferred == "auto":
            return list(self.providers)
        selected = [item for item in self.providers if item.provider_id.casefold() == preferred]
        if not selected:
            raise DiarizationUnavailable(f"未知说话人分段 Provider：{preferred}")
        return [*selected, *(item for item in self.providers if item not in selected)]

    def diarize(
        self,
        request: DiarizationRequest,
        progress: ProgressCallback | None = None,
        check_cancelled: CancellationCallback | None = None,
    ) -> DiarizationResult:
        attempts: list[str] = []
        providers = self._ordered(request.preferred_provider)
        for index, provider in enumerate(providers):
            if check_cancelled:
                check_cancelled()
            available, reason = provider.availability(request)
            if not available:
                detail = reason or "不可用"
                attempts.append(f"{provider.provider_id}：{detail}")
                if not request.allow_fallback or (
                    request.preferred_provider != "auto" and index == 0 and len(providers) == 1
                ):
                    break
                continue
            if progress:
                progress(f"正在使用 {provider.provider_id} 进行本地说话人分段")
            try:
                with LOCAL_HEAVY_TASKS.reserve(
                    f"diarization:{provider.provider_id}",
                    progress=progress,
                    check_cancelled=check_cancelled,
                ):
                    result = provider.diarize(
                        request,
                        progress=progress,
                        check_cancelled=check_cancelled,
                    )
            except CancelledError:
                # User cancellation is a terminal control signal, never a
                # technical failure and therefore never eligible for fallback.
                raise
            except (DiarizationUnavailable, RuntimeError, OSError, ValueError) as error:
                attempts.append(f"{provider.provider_id}：{error}")
                if not request.allow_fallback:
                    break
                continue
            if check_cancelled:
                check_cancelled()
            result.fallback_reasons = list(
                dict.fromkeys([*attempts, *result.fallback_reasons])
            )
            return result
        detail = "；".join(attempts) or "没有配置可用的本地说话人分段 Provider"
        raise DiarizationUnavailable(detail)
