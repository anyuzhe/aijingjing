from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from media_knowledge.ingestion import ocr
from media_knowledge.ingestion.quality import evaluate_extraction
from media_knowledge.ingestion.types import ExtractionResult
from media_knowledge.models import ContentSegment


class PaddleStructureHardeningTests(unittest.TestCase):
    def test_structured_blocks_keep_order_markdown_formula_and_bbox(self) -> None:
        payload = {
            "parsing_res_list": [
                {
                    "block_order": 2,
                    "block_content": "$$E = mc^2$$",
                    "block_bbox": [10, 20, 80, 40],
                    "block_score": 0.91,
                },
                {
                    "block_order": "1",
                    "block_content": "| 参数 | 值 |\n| --- | --- |\n| A | 1 |",
                    "block_bbox": [0, 0, 100, 18],
                    "block_score": 0.96,
                },
            ],
            "overall_ocr_res": {
                "rec_texts": ["扁平污染 A", "扁平污染 B"],
                "rec_scores": [0.99, 0.99],
            },
        }

        lines = ocr._lines_from_paddle_payload(payload)

        self.assertEqual(
            [line.text for line in lines],
            ["| 参数 | 值 |\n| --- | --- |\n| A | 1 |", "$$E = mc^2$$"],
        )
        self.assertEqual(lines[0].bbox, [[0.0, 0.0], [100.0, 0.0], [100.0, 18.0], [0.0, 18.0]])
        self.assertEqual(lines[0].confidence, 0.96)
        self.assertNotIn("扁平污染", "\n".join(line.text for line in lines))

    def test_pipeline_is_created_once_per_process(self) -> None:
        constructions: list[int] = []

        class FakePipeline:
            def __init__(self) -> None:
                constructions.append(1)

        fake_module = types.SimpleNamespace(PPStructureV3=FakePipeline)
        with patch.dict(sys.modules, {"paddleocr": fake_module}), patch.object(
            ocr, "_PADDLE_PIPELINE", None
        ):
            first = ocr._get_paddle_pipeline()
            second = ocr._get_paddle_pipeline()

        self.assertIs(first, second)
        self.assertEqual(len(constructions), 1)

    def test_shared_pipeline_prediction_is_serialized(self) -> None:
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        class FakePipeline:
            def predict(self, input: str) -> list[dict[str, str]]:
                nonlocal active, max_active
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.03)
                with state_lock:
                    active -= 1
                return [{"text": input}]

        errors: list[Exception] = []

        def run() -> None:
            try:
                ocr._run_paddle_structure(Path("scan.png"))
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        with patch.object(ocr, "_get_paddle_pipeline", return_value=FakePipeline()):
            workers = [threading.Thread(target=run) for _ in range(3)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

        self.assertEqual(errors, [])
        self.assertEqual(max_active, 1)


class OCRQualityGateHardeningTests(unittest.TestCase):
    @staticmethod
    def _result(*, fallback: bool, confidence: float = 0.01) -> ExtractionResult:
        return ExtractionResult(
            title="低置信度扫描件",
            media_type="image",
            segments=[ContentSegment(
                "image-1",
                1,
                "image",
                text="可能完全识别错误的文字",
                description="视觉模型看到了原图并给出描述" if fallback else "",
            )],
            checksum="abc",
            metadata={
                "ocr": {
                    "line_count": 1,
                    "mean_confidence": confidence,
                    "min_confidence": confidence,
                    "low_confidence_threshold": 0.65,
                    "low_confidence_lines": [{"text": "错误", "confidence": confidence}],
                    "vision_fallback_used": fallback,
                }
            },
        )

    def test_extremely_low_confidence_without_vision_is_rejected(self) -> None:
        report = evaluate_extraction(self._result(fallback=False))

        self.assertFalse(report.accepted)
        self.assertLess(report.score, 60)
        self.assertTrue(any(check.name == "OCR 置信度" and check.status == "fail" for check in report.checks))
        self.assertEqual(report.metrics["ocr"]["min_confidence"], 0.01)

    def test_vision_fallback_still_warns_and_reduces_score(self) -> None:
        report = evaluate_extraction(self._result(fallback=True))

        self.assertTrue(report.accepted)
        self.assertLess(report.score, 100)
        self.assertTrue(any(check.name == "OCR 置信度" and check.status == "warn" for check in report.checks))


if __name__ == "__main__":
    unittest.main()
