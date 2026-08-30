from __future__ import annotations

import threading
import time
import unittest

from media_knowledge.ingestion.types import CancelledError
from media_knowledge.resource_scheduler import LocalHeavyTaskScheduler


class LocalHeavyTaskSchedulerTests(unittest.TestCase):
    def test_memory_heavy_tasks_never_overlap(self) -> None:
        scheduler = LocalHeavyTaskScheduler()
        first_entered = threading.Event()
        release_first = threading.Event()
        order: list[str] = []

        def first() -> None:
            with scheduler.reserve("asr"):
                order.append("first-enter")
                first_entered.set()
                release_first.wait(2)
                order.append("first-exit")

        def second() -> None:
            first_entered.wait(2)
            with scheduler.reserve("embedding"):
                order.append("second-enter")

        one = threading.Thread(target=first)
        two = threading.Thread(target=second)
        one.start()
        two.start()
        self.assertTrue(first_entered.wait(2))
        time.sleep(0.05)
        self.assertEqual(scheduler.snapshot().active_task, "asr")
        self.assertEqual(scheduler.snapshot().waiting_tasks, 1)
        release_first.set()
        one.join(2)
        two.join(2)
        self.assertEqual(order, ["first-enter", "first-exit", "second-enter"])

    def test_waiting_for_resource_is_cancellable(self) -> None:
        scheduler = LocalHeavyTaskScheduler()
        cancelled = threading.Event()
        result: list[str] = []

        def check_cancelled() -> None:
            if cancelled.is_set():
                raise CancelledError("用户取消")

        def waiter() -> None:
            try:
                with scheduler.reserve("diarization", check_cancelled=check_cancelled):
                    result.append("entered")
            except CancelledError:
                result.append("cancelled")

        with scheduler.reserve("asr"):
            thread = threading.Thread(target=waiter)
            thread.start()
            time.sleep(0.15)
            cancelled.set()
            thread.join(2)

        self.assertEqual(result, ["cancelled"])
        self.assertEqual(scheduler.snapshot().waiting_tasks, 0)


if __name__ == "__main__":
    unittest.main()
