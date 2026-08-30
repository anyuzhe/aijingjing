from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from .types import AsrResult, TranscriptionRequest


ProgressCallback = Callable[[str], None]
CancellationCallback = Callable[[], None]


class AsrProviderError(RuntimeError):
    """A provider failure with a stable, persistable reason code."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "provider_failed",
        technical: bool = True,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "provider_failed")
        self.technical = bool(technical)


class AsrRoutingError(AsrProviderError):
    pass


@runtime_checkable
class AsrProvider(Protocol):
    provider_id: str

    def available(self) -> bool:
        ...

    def transcribe(
        self,
        request: TranscriptionRequest,
        progress: ProgressCallback | None = None,
        check_cancelled: CancellationCallback | None = None,
    ) -> AsrResult:
        ...
