from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


_PADDLE_PIPELINE: object | None = None
_PADDLE_PIPELINE_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class OCRLine:
    """One OCR line in a JSON-safe, engine-neutral representation."""

    text: str
    confidence: float | None = None
    bbox: list[list[float]] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class OCRResult:
    engine: str
    requested_engine: str
    lines: list[OCRLine] = field(default_factory=list)
    low_confidence_threshold: float = 0.65
    fallback_reasons: list[str] = field(default_factory=list)
    original_rapidocr: dict[str, object] | None = None
    complex_layout: bool = False
    vision_fallback_used: bool = False

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text.strip())

    @property
    def confidences(self) -> list[float]:
        return [line.confidence for line in self.lines if line.confidence is not None]

    def to_dict(self, *, include_original: bool = True) -> dict[str, object]:
        scores = self.confidences
        low_lines = [
            line.to_dict()
            for line in self.lines
            if line.confidence is not None and line.confidence < self.low_confidence_threshold
        ]
        payload: dict[str, object] = {
            "engine": self.engine,
            "requested_engine": self.requested_engine,
            "lines": [line.to_dict() for line in self.lines],
            "line_count": len(self.lines),
            "mean_confidence": round(sum(scores) / len(scores), 6) if scores else None,
            "min_confidence": round(min(scores), 6) if scores else None,
            "low_confidence_threshold": self.low_confidence_threshold,
            "low_confidence_lines": low_lines,
            "fallback_reasons": list(dict.fromkeys(self.fallback_reasons)),
            "complex_layout": self.complex_layout,
            "vision_fallback_used": self.vision_fallback_used,
        }
        if include_original and self.original_rapidocr is not None:
            payload["original_rapidocr"] = self.original_rapidocr
        return payload


class OCRUnavailable(RuntimeError):
    pass


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    # OCR libraries normally return 0..1. Clamp malformed plugin values so
    # downstream quality metrics remain meaningful and JSON-safe.
    return round(min(1.0, max(0.0, number)), 6)


def _bbox(value: object) -> list[list[float]] | None:
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except (TypeError, ValueError):
            return None
    if not isinstance(value, (list, tuple)):
        return None
    points: list[list[float]] = []
    # Some engines return [x1, y1, x2, y2] instead of four points.
    if len(value) == 4 and all(not isinstance(item, (list, tuple)) for item in value):
        numbers = [_finite_coordinate(item) for item in value]
        if all(number is not None for number in numbers):
            left, top, right, bottom = numbers
            return [[left, top], [right, top], [right, bottom], [left, bottom]]  # type: ignore[list-item]
        return None
    for point in value:
        if hasattr(point, "tolist"):
            try:
                point = point.tolist()
            except (TypeError, ValueError):
                return None
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        x = _finite_coordinate(point[0])
        y = _finite_coordinate(point[1])
        if x is None or y is None:
            return None
        points.append([x, y])
    return points or None


def _finite_coordinate(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return round(number, 3) if math.isfinite(number) else None


def _line(text: object, confidence: object = None, bbox: object = None) -> OCRLine | None:
    value = str(text or "").strip()
    if not value:
        return None
    return OCRLine(value, _finite_float(confidence), _bbox(bbox))


def _sequence(value: object) -> list[object]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except (TypeError, ValueError):
            return []
    return list(value) if isinstance(value, (list, tuple)) else []


def _first_present(payload: dict[object, object], *keys: str) -> object:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def normalize_rapidocr_result(raw: object) -> list[OCRLine]:
    """Normalize RapidOCR 2.x tuples and 3.x output objects."""

    texts = getattr(raw, "txts", None)
    if texts is not None:
        boxes = _sequence(getattr(raw, "boxes", None))
        scores = _sequence(getattr(raw, "scores", None))
        lines = []
        for index, text in enumerate(_sequence(texts)):
            item = _line(
                text,
                scores[index] if index < len(scores) else None,
                boxes[index] if index < len(boxes) else None,
            )
            if item:
                lines.append(item)
        return lines

    payload = raw
    if isinstance(raw, tuple) and raw:
        payload = raw[0]
    if not isinstance(payload, (list, tuple)):
        return []
    lines: list[OCRLine] = []
    for item in payload:
        if not isinstance(item, (list, tuple)):
            continue
        # Legacy RapidOCR: [box, text, confidence]. Some wrappers omit box.
        if len(item) >= 3:
            value = _line(item[1], item[2], item[0])
        elif len(item) >= 2:
            value = _line(item[1], None, item[0])
        else:
            value = None
        if value:
            lines.append(value)
    return lines


def _run_rapidocr(path: Path) -> list[OCRLine]:
    try:
        try:
            from rapidocr import RapidOCR  # type: ignore
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except ImportError as exc:
        raise OCRUnavailable("RapidOCR 未安装") from exc
    try:
        return normalize_rapidocr_result(RapidOCR()(str(path)))
    except Exception as exc:
        raise OCRUnavailable(f"RapidOCR 运行失败：{type(exc).__name__}") from exc


def _payload_dict(value: object) -> object:
    candidate = getattr(value, "json", value)
    if callable(candidate):
        try:
            candidate = candidate()
        except Exception:
            return value
    return candidate


def _block_text(value: object) -> str:
    """Preserve PP-Structure Markdown/formula content without flattening it."""

    value = _payload_dict(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(filter(None, (_block_text(item) for item in value))).strip()
    if isinstance(value, dict):
        preferred = (
            "markdown",
            "block_content",
            "content",
            "latex",
            "formula",
            "text",
            "rec_text",
        )
        parts = [_block_text(value[key]) for key in preferred if key in value]
        if any(parts):
            return "\n".join(part for part in parts if part).strip()
        return "\n".join(filter(None, (_block_text(item) for item in value.values()))).strip()
    return str(value).strip() if value is not None else ""


def _deduplicate_lines(lines: list[OCRLine]) -> list[OCRLine]:
    unique: list[OCRLine] = []
    seen: set[tuple[str, str]] = set()
    for item in lines:
        key = (item.text, repr(item.bbox))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _structured_paddle_lines(payload: object) -> list[OCRLine]:
    """Collect ordered PP-Structure blocks before considering flat OCR data."""

    payload = _payload_dict(payload)
    lines: list[OCRLine] = []
    if isinstance(payload, dict):
        blocks = _sequence(payload.get("parsing_res_list"))
        if blocks:
            ordered: list[tuple[float, int, dict[object, object]]] = []
            for index, raw_block in enumerate(blocks):
                block = _payload_dict(raw_block)
                if not isinstance(block, dict):
                    continue
                try:
                    order = float(block.get("block_order"))
                except (TypeError, ValueError, OverflowError):
                    order = math.inf
                if not math.isfinite(order):
                    order = math.inf
                ordered.append((order, index, block))
            for _, _, block in sorted(ordered, key=lambda item: (item[0], item[1])):
                item = _line(
                    _block_text(block.get("block_content")),
                    _first_present(
                        block,
                        "block_confidence",
                        "block_score",
                        "confidence",
                        "score",
                        "rec_score",
                    ),
                    _first_present(block, "block_bbox", "bbox", "box", "poly"),
                )
                if item:
                    lines.append(item)
        for key, value in payload.items():
            if key == "parsing_res_list":
                continue
            if isinstance(value, (dict, list, tuple)):
                lines.extend(_structured_paddle_lines(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            lines.extend(_structured_paddle_lines(value))
    return _deduplicate_lines(lines)


def _lines_from_paddle_payload(payload: object) -> list[OCRLine]:
    """Best-effort normalizer for PP-StructureV3 across PaddleOCR releases."""

    payload = _payload_dict(payload)
    structured = _structured_paddle_lines(payload)
    if structured:
        # A structured parse is higher-fidelity than ``overall_ocr_res``: it
        # retains reading order, Markdown tables/formulas, and block geometry.
        # Do not append the flattened OCR copy, which would duplicate and
        # scramble the page content.
        return structured
    lines: list[OCRLine] = []
    if isinstance(payload, dict):
        texts = _first_present(payload, "rec_texts", "texts")
        scores = _first_present(payload, "rec_scores", "scores")
        boxes = _first_present(payload, "rec_boxes", "dt_polys", "boxes")
        text_values = _sequence(texts)
        if text_values:
            score_values = _sequence(scores)
            box_values = _sequence(boxes)
            for index, text in enumerate(text_values):
                item = _line(
                    text,
                    score_values[index] if index < len(score_values) else None,
                    box_values[index] if index < len(box_values) else None,
                )
                if item:
                    lines.append(item)
        direct_text = _first_present(payload, "text", "rec_text")
        if direct_text and not isinstance(direct_text, (list, tuple, dict)):
            item = _line(
                direct_text,
                payload.get("confidence", payload.get("score", payload.get("rec_score"))),
                payload.get("bbox", payload.get("box", payload.get("poly"))),
            )
            if item:
                lines.append(item)
        for key, value in payload.items():
            if key in {"rec_texts", "texts", "rec_scores", "scores", "rec_boxes", "dt_polys", "boxes"}:
                continue
            if isinstance(value, (dict, list, tuple)):
                lines.extend(_lines_from_paddle_payload(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            lines.extend(_lines_from_paddle_payload(value))
    # Nested layout results can repeat the same OCR block. Preserve reading
    # order but remove exact duplicate records.
    return _deduplicate_lines(lines)


def _get_paddle_pipeline() -> object:
    """Create PP-StructureV3 once per process; model loading is expensive."""

    global _PADDLE_PIPELINE
    with _PADDLE_PIPELINE_LOCK:
        if _PADDLE_PIPELINE is None:
            try:
                from paddleocr import PPStructureV3  # type: ignore
            except ImportError as exc:
                raise OCRUnavailable("PaddleOCR PP-StructureV3 未安装") from exc
            _PADDLE_PIPELINE = PPStructureV3()
        return _PADDLE_PIPELINE


def _run_paddle_structure(path: Path) -> list[OCRLine]:
    try:
        pipeline = _get_paddle_pipeline()
        # Paddle pipelines are not documented as thread-safe. Keep prediction
        # and lazy-result iteration under the same lock so parallel PDF pages
        # cannot race the shared model instance.
        with _PADDLE_PIPELINE_LOCK:
            try:
                output: Iterable[object] = pipeline.predict(input=str(path))  # type: ignore[attr-defined]
            except TypeError:
                output = pipeline.predict(str(path))  # type: ignore[attr-defined]
            materialized = list(output)
        lines: list[OCRLine] = []
        for value in materialized:
            lines.extend(_lines_from_paddle_payload(value))
        return lines
    except OCRUnavailable:
        raise
    except Exception as exc:
        raise OCRUnavailable(f"PaddleOCR PP-StructureV3 运行失败：{type(exc).__name__}") from exc


def looks_like_complex_layout(lines: list[OCRLine]) -> bool:
    """Conservative signal for tables, formulas, or multi-column documents."""

    if any(any(marker in line.text for marker in ("∑", "√", "≈", "≠", "≤", "≥", "|", "＝")) for line in lines):
        return True
    positioned = [line for line in lines if line.bbox]
    if len(positioned) < 8:
        return False
    extents = [
        (
            min(point[0] for point in line.bbox or []),
            max(point[0] for point in line.bbox or []),
            min(point[1] for point in line.bbox or []),
            max(point[1] for point in line.bbox or []),
        )
        for line in positioned
    ]
    page_width = max((right for _, right, _, _ in extents), default=0.0)
    if page_width <= 0:
        return False
    left_column = [item for item in extents if item[1] <= page_width * 0.58]
    right_column = [item for item in extents if item[0] >= page_width * 0.42]
    if len(left_column) < 3 or len(right_column) < 3:
        return False
    return any(
        max(left_top, right_top) <= min(left_bottom, right_bottom)
        for _, _, left_top, left_bottom in left_column
        for _, _, right_top, right_bottom in right_column
    )


def extract_ocr(
    path: str | Path,
    *,
    requested_engine: str = "auto",
    complex_layout: bool = False,
    allow_paddle: bool = True,
    low_confidence_threshold: float = 0.65,
) -> OCRResult:
    """Run local OCR while retaining the baseline RapidOCR evidence.

    RapidOCR is always the baseline for ordinary images. PP-StructureV3 is only
    attempted for an explicit Paddle request or a detected/declared complex
    layout, and its absence is reported rather than hidden.
    """

    requested = str(requested_engine or "auto").strip().casefold()
    if requested not in {"auto", "rapidocr", "paddleocr"}:
        requested = "auto"
    threshold = min(1.0, max(0.0, float(low_confidence_threshold)))
    target = Path(path)
    reasons: list[str] = []
    rapid_lines: list[OCRLine] = []
    try:
        rapid_lines = _run_rapidocr(target)
        if not rapid_lines:
            reasons.append("RapidOCR 未识别到文字")
    except OCRUnavailable as exc:
        reasons.append(str(exc))

    detected_complex = complex_layout or looks_like_complex_layout(rapid_lines)
    rapid = OCRResult(
        engine="rapidocr" if rapid_lines else "none",
        requested_engine=requested,
        lines=rapid_lines,
        low_confidence_threshold=threshold,
        fallback_reasons=list(reasons),
        complex_layout=detected_complex,
    )
    should_use_paddle = allow_paddle and (requested == "paddleocr" or (requested == "auto" and detected_complex))
    if should_use_paddle:
        try:
            paddle_lines = _run_paddle_structure(target)
            if paddle_lines:
                return OCRResult(
                    engine="paddleocr_ppstructurev3",
                    requested_engine=requested,
                    lines=paddle_lines,
                    low_confidence_threshold=threshold,
                    fallback_reasons=reasons,
                    original_rapidocr=rapid.to_dict(include_original=False),
                    complex_layout=True,
                )
            reasons.append("PaddleOCR PP-StructureV3 未识别到文字")
        except OCRUnavailable as exc:
            reasons.append(str(exc))
    elif requested == "paddleocr" and not allow_paddle:
        reasons.append("PaddleOCR PP-StructureV3 已在设置中禁用")

    rapid.fallback_reasons = reasons
    return rapid
