from __future__ import annotations

import gc
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator


ProgressCallback = Callable[[str], None]
CancellationCallback = Callable[[], None]


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    active_task: str | None
    waiting_tasks: int
    concurrency_limit: int = 1


class LocalHeavyTaskScheduler:
    """One memory-heavy local inference task at a time.

    M1-class machines with unified memory can become unstable when ASR,
    diarization and a large embedding model load concurrently.  All three use
    this cooperative process-wide slot.  Waiting remains cancellable.
    """

    def __init__(self) -> None:
        self._slot = threading.Semaphore(1)
        self._state_lock = threading.Lock()
        self._active_task: str | None = None
        self._waiting_tasks = 0

    def snapshot(self) -> ResourceSnapshot:
        with self._state_lock:
            return ResourceSnapshot(self._active_task, self._waiting_tasks)

    @contextmanager
    def reserve(
        self,
        task: str,
        *,
        progress: ProgressCallback | None = None,
        check_cancelled: CancellationCallback | None = None,
    ) -> Iterator[None]:
        acquired = False
        announced = False
        with self._state_lock:
            self._waiting_tasks += 1
        try:
            while not acquired:
                if check_cancelled:
                    check_cancelled()
                acquired = self._slot.acquire(timeout=0.1)
                if not acquired and progress and not announced:
                    progress("正在等待本地推理资源；AI静静一次只运行一个高内存任务")
                    announced = True
            with self._state_lock:
                self._waiting_tasks = max(0, self._waiting_tasks - 1)
                self._active_task = str(task)
            if check_cancelled:
                check_cancelled()
            yield
        finally:
            if not acquired:
                with self._state_lock:
                    self._waiting_tasks = max(0, self._waiting_tasks - 1)
            else:
                with self._state_lock:
                    self._active_task = None
                self._slot.release()
                self.release_runtime_caches()

    @staticmethod
    def release_runtime_caches() -> None:
        # Do not import MLX merely to clear it: that would make diagnostics or
        # hash embeddings load a platform-specific runtime unnecessarily.
        mlx = sys.modules.get("mlx.core")
        clear = getattr(mlx, "clear_cache", None) if mlx is not None else None
        if callable(clear):
            try:
                clear()
            except (RuntimeError, OSError):
                pass
        gc.collect()


LOCAL_HEAVY_TASKS = LocalHeavyTaskScheduler()


__all__ = ["LOCAL_HEAVY_TASKS", "LocalHeavyTaskScheduler", "ResourceSnapshot"]
