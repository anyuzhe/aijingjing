from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import ContentSegment


class CancelledError(RuntimeError):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._pause = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def check(self) -> None:
        if self.cancelled:
            raise CancelledError("任务已取消")
        while self.paused and not self.cancelled:
            self._event.wait(0.1)
        if self.cancelled:
            raise CancelledError("任务已取消")


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    item: str
    stage: str
    percent: int
    message: str


@dataclass(slots=True)
class ExtractionResult:
    title: str
    media_type: str
    segments: list[ContentSegment]
    source_path: Path | None = None
    original_uri: str | None = None
    checksum: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    retained_assets: list[Path] = field(default_factory=list)
    transcript_path: Path | None = None

    @property
    def extracted_characters(self) -> int:
        return sum(len(segment.retrieval_text) for segment in self.segments)
