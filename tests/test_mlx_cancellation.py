from __future__ import annotations

import queue
import unittest
from pathlib import Path
from unittest.mock import patch

from media_knowledge.ingestion.transcription import TranscriptionPlan, _transcribe_mlx
from media_knowledge.ingestion.types import CancelledError


class _FakeQueue:
    def __init__(self, messages: list[object]) -> None:
        self.messages = list(messages)
        self.closed = False
        self.cancelled_join = False

    def get(self, *, timeout: float) -> object:
        del timeout
        if not self.messages:
            raise queue.Empty
        value = self.messages.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def get_nowait(self) -> object:
        return self.get(timeout=0.0)

    def close(self) -> None:
        self.closed = True

    def cancel_join_thread(self) -> None:
        self.cancelled_join = True


class _FakeProcess:
    def __init__(self, *, alive: bool = True) -> None:
        self.alive = alive
        self.started = False
        self.terminated = False
        self.killed = False
        self.join_calls = 0

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def join(self, *, timeout: float) -> None:
        del timeout
        self.join_calls += 1
        if self.started and not self.terminated and not self.killed:
            self.alive = False


class _FakeContext:
    def __init__(self, result_queue: _FakeQueue, process: _FakeProcess) -> None:
        self.result_queue = result_queue
        self.process = process
        self.process_kwargs: dict[str, object] = {}

    def Queue(self, *, maxsize: int) -> _FakeQueue:  # noqa: N802 - multiprocessing API
        self.maxsize = maxsize
        return self.result_queue

    def Process(self, **kwargs: object) -> _FakeProcess:  # noqa: N802 - multiprocessing API
        self.process_kwargs = kwargs
        return self.process


class MlxCancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = TranscriptionPlan("mlx-whisper", "metal", "float16", "mlx/model")

    def test_cancellation_terminates_and_reaps_worker(self) -> None:
        result_queue = _FakeQueue([])
        process = _FakeProcess()
        context = _FakeContext(result_queue, process)

        def cancel() -> None:
            raise CancelledError("用户已取消")

        with patch("media_knowledge.ingestion.transcription.mp.get_context", return_value=context):
            with self.assertRaises(CancelledError):
                _transcribe_mlx(Path("private-recording.wav"), self.plan, cancel)

        self.assertTrue(process.started)
        self.assertTrue(process.terminated)
        self.assertGreaterEqual(process.join_calls, 1)
        self.assertFalse(process.killed)
        self.assertTrue(result_queue.closed)
        self.assertTrue(result_queue.cancelled_join)
        self.assertEqual(context.maxsize, 1)
        self.assertEqual(context.process_kwargs["daemon"], True)

    def test_worker_result_is_parsed_into_transcript_segments(self) -> None:
        payload = {
            "language": "zh",
            "segments": [
                {
                    "start": 0.25,
                    "end": 1.75,
                    "text": " 冬日知识 ",
                    "confidence": 0.92,
                    "avg_logprob": -0.08,
                }
            ],
        }
        result_queue = _FakeQueue([("ok", payload)])
        process = _FakeProcess()
        context = _FakeContext(result_queue, process)
        checks: list[bool] = []

        with patch("media_knowledge.ingestion.transcription.mp.get_context", return_value=context):
            segments, language = _transcribe_mlx(
                Path("recording.wav"),
                self.plan,
                lambda: checks.append(True),
            )

        self.assertEqual(language, "zh")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "冬日知识")
        self.assertEqual(segments[0].start, 0.25)
        self.assertEqual(segments[0].end, 1.75)
        self.assertEqual(segments[0].confidence, 0.92)
        self.assertEqual(segments[0].avg_logprob, -0.08)
        self.assertEqual(checks, [True])
        self.assertFalse(process.terminated)
        self.assertGreaterEqual(process.join_calls, 1)


if __name__ == "__main__":
    unittest.main()
