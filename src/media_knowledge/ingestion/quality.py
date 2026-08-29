from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable

from .types import ExtractionResult


@dataclass(slots=True)
class QualityCheck:
    name: str
    status: str
    detail: str


@dataclass(slots=True)
class QualityReport:
    accepted: bool
    score: int
    grade: str
    checks: list[QualityCheck] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class QualityGateError(RuntimeError):
    def __init__(self, message: str, report: QualityReport) -> None:
        super().__init__(message)
        self.report = report


def _timeline_number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def evaluate_transcript_integrity(
    segments: Iterable[dict[str, object]],
    *,
    duration_seconds: float = 0.0,
) -> dict[str, object]:
    """Validate transcript timing without inventing or rewriting source text.

    Long silence is a warning rather than an automatic rejection. Invalid time
    ranges, time reversal, or pervasive overlaps are hard failures because they
    make timestamp citations unsafe.
    """

    raw = [dict(item) for item in segments]
    duration = max(0.0, _timeline_number(duration_seconds) or 0.0)
    empty_segments = 0
    invalid_ranges = 0
    beyond_duration = 0
    out_of_order = 0
    valid: list[tuple[float, float, str]] = []
    previous_start: float | None = None
    for item in raw:
        text = str(item.get("text") or "").strip()
        if not text:
            empty_segments += 1
        start = _timeline_number(item.get("start"))
        end = _timeline_number(item.get("end"))
        if start is None or end is None or start < 0 or end <= start:
            invalid_ranges += 1
            continue
        if duration and end > duration + max(1.0, duration * 0.01):
            beyond_duration += 1
        if previous_start is not None and start + 0.05 < previous_start:
            out_of_order += 1
        previous_start = start
        valid.append((start, end, text))

    abnormal_overlaps = 0
    overlap_seconds = 0.0
    discontinuity_gaps = 0
    max_gap = 0.0
    merged: list[list[float]] = []
    previous_end: float | None = None
    gap_threshold = max(30.0, min(120.0, duration * 0.05)) if duration else 30.0
    for start, end, _ in valid:
        if previous_end is not None:
            overlap = previous_end - start
            if overlap > 1.0:
                abnormal_overlaps += 1
                overlap_seconds += overlap
            gap = start - previous_end
            if gap > gap_threshold:
                discontinuity_gaps += 1
                max_gap = max(max_gap, gap)
        previous_end = max(previous_end or end, end)
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    covered_seconds = sum(end - start for start, end in merged)
    first_start = valid[0][0] if valid else None
    last_end = max((item[1] for item in valid), default=None)
    head_gap = first_start if first_start is not None else duration
    tail_gap = max(0.0, duration - last_end) if duration and last_end is not None else 0.0
    edge_tolerance = max(5.0, min(30.0, duration * 0.05)) if duration else 5.0
    head_covered = first_start is not None and first_start <= edge_tolerance
    tail_covered = not duration or (last_end is not None and tail_gap <= edge_tolerance)
    coverage_ratio = min(1.0, covered_seconds / duration) if duration else None
    pervasive_overlaps = abnormal_overlaps > max(2, round(len(valid) * 0.2))
    nonempty_valid = sum(bool(text) for _, _, text in valid)
    hard_failure = (
        not valid or not nonempty_valid or invalid_ranges > 0
        or beyond_duration > 0 or out_of_order > 0 or pervasive_overlaps
    )

    checks: list[dict[str, str]] = []
    if not valid:
        checks.append({"name": "时间连续性", "status": "fail", "detail": "没有有效的带时间轴转写片段"})
    elif invalid_ranges or beyond_duration or out_of_order:
        checks.append({
            "name": "时间连续性",
            "status": "fail",
            "detail": (
                f"发现 {invalid_ranges} 个非法时间段、{beyond_duration} 个超出媒体时长的时间段、"
                f"{out_of_order} 次时间倒序"
            ),
        })
    elif discontinuity_gaps:
        checks.append({
            "name": "时间连续性",
            "status": "warn",
            "detail": f"发现 {discontinuity_gaps} 个超过阈值的长间隙，最大 {max_gap:.2f} 秒",
        })
    else:
        checks.append({"name": "时间连续性", "status": "pass", "detail": "时间轴顺序有效，未发现异常长间隙"})

    if head_covered and tail_covered:
        checks.append({"name": "首尾覆盖", "status": "pass", "detail": "转写时间轴覆盖媒体首尾允许范围"})
    else:
        missing = []
        if not head_covered:
            missing.append(f"开头空缺 {head_gap:.2f} 秒")
        if not tail_covered:
            missing.append(f"结尾空缺 {tail_gap:.2f} 秒")
        checks.append({"name": "首尾覆盖", "status": "warn", "detail": "、".join(missing)})

    if pervasive_overlaps:
        overlap_status = "fail"
    elif abnormal_overlaps or empty_segments:
        overlap_status = "warn"
    else:
        overlap_status = "pass"
    checks.append({
        "name": "重叠与空段",
        "status": overlap_status,
        "detail": f"异常重叠 {abnormal_overlaps} 个、空文字片段 {empty_segments} 个",
    })

    status = "fail" if hard_failure else "warn" if any(item["status"] == "warn" for item in checks) else "pass"
    return {
        "accepted": not hard_failure,
        "status": status,
        "checks": checks,
        "segment_count": len(raw),
        "valid_segment_count": len(valid),
        "empty_segments": empty_segments,
        "invalid_time_ranges": invalid_ranges,
        "beyond_duration_segments": beyond_duration,
        "out_of_order_segments": out_of_order,
        "abnormal_overlaps": abnormal_overlaps,
        "overlap_seconds": round(overlap_seconds, 3),
        "discontinuity_gaps": discontinuity_gaps,
        "max_gap_seconds": round(max_gap, 3),
        "head_gap_seconds": round(head_gap, 3),
        "tail_gap_seconds": round(tail_gap, 3),
        "duration_seconds": round(duration, 3),
        "covered_seconds": round(covered_seconds, 3),
        "coverage_ratio": round(coverage_ratio, 6) if coverage_ratio is not None else None,
    }


def evaluate_extraction(extracted: ExtractionResult) -> QualityReport:
    checks: list[QualityCheck] = []
    hard_failure = False
    score = 100
    characters = extracted.extracted_characters
    segment_count = len(extracted.segments)

    if segment_count and characters:
        checks.append(QualityCheck("内容提取", "pass", f"提取 {segment_count} 个片段、{characters} 个可检索字符"))
    else:
        checks.append(QualityCheck("内容提取", "fail", "没有提取到可检索正文"))
        hard_failure = True
        score = 0

    scope = str(extracted.metadata.get("content_scope") or "full_source")
    if scope in {"platform_description_only", "metadata_only", "cover_only"}:
        checks.append(QualityCheck("原始内容真实性", "fail", "只取得简介、元数据或封面，未取得原始正文"))
        hard_failure = True
        score = min(score, 20)
    else:
        checks.append(QualityCheck("原始内容真实性", "pass", "内容来自原文件或真实媒体流"))

    ocr = extracted.metadata.get("ocr")
    if isinstance(ocr, dict):
        line_count = max(0, int(_timeline_number(ocr.get("line_count")) or 0))
        mean_confidence = _timeline_number(ocr.get("mean_confidence"))
        min_confidence = _timeline_number(ocr.get("min_confidence"))
        threshold = _timeline_number(ocr.get("low_confidence_threshold"))
        pages = ocr.get("pages")
        if threshold is None and isinstance(pages, list):
            threshold = next(
                (
                    value
                    for page in pages
                    if isinstance(page, dict)
                    for value in [_timeline_number(page.get("low_confidence_threshold"))]
                    if value is not None
                ),
                None,
            )
        threshold = min(1.0, max(0.0, threshold if threshold is not None else 0.65))
        raw_low_lines = ocr.get("low_confidence_lines")
        low_confidence_count = len(raw_low_lines) if isinstance(raw_low_lines, (list, tuple)) else 0
        low_confidence_ratio = min(1.0, low_confidence_count / line_count) if line_count else 0.0
        declared_vision_fallback = bool(ocr.get("vision_fallback_used"))
        if isinstance(pages, list):
            declared_vision_fallback = declared_vision_fallback or any(
                bool(page.get("vision_fallback_used")) for page in pages if isinstance(page, dict)
            )
        has_vision_evidence = any(segment.description.strip() for segment in extracted.segments)
        vision_fallback = declared_vision_fallback and has_vision_evidence

        if line_count:
            if mean_confidence is None or min_confidence is None:
                checks.append(QualityCheck("OCR 置信度", "warn", "OCR 未提供完整置信度，建议核对原图"))
                score -= 5
            else:
                severe = mean_confidence < 0.35 or low_confidence_ratio >= 0.8
                uncertain = (
                    mean_confidence < threshold
                    or min_confidence < threshold
                    or low_confidence_ratio >= 0.35
                )
                detail = (
                    f"平均 {mean_confidence:.1%}、最低 {min_confidence:.1%}，"
                    f"低置信度 {low_confidence_count}/{line_count} 行"
                )
                if severe and not vision_fallback:
                    checks.append(QualityCheck("OCR 置信度", "fail", detail + "；无有效视觉模型兜底"))
                    hard_failure = True
                    score = min(score, 45)
                elif severe:
                    checks.append(QualityCheck("OCR 置信度", "warn", detail + "；已用视觉模型补充，仍需核对"))
                    score -= 20
                elif uncertain:
                    checks.append(QualityCheck("OCR 置信度", "warn", detail + "；部分文字可能识别不准"))
                    score -= 12
                else:
                    checks.append(QualityCheck("OCR 置信度", "pass", detail))

    speech_segments = sum(segment.modality == "speech" for segment in extracted.segments)
    visual_segments = sum(segment.modality in {"image", "slide", "page"} for segment in extracted.segments)
    if extracted.media_type in {"audio", "video"}:
        if speech_segments:
            checks.append(QualityCheck("音视频转写", "pass", f"生成 {speech_segments} 个带时间轴的语音片段"))
            transcription = extracted.metadata.get("transcription")
            integrity = transcription.get("integrity") if isinstance(transcription, dict) else None
            if not isinstance(integrity, dict):
                duration = transcription.get("duration_seconds", 0.0) if isinstance(transcription, dict) else 0.0
                integrity = evaluate_transcript_integrity(
                    [
                        {
                            "start": segment.location.get("timestamp_start"),
                            "end": segment.location.get("timestamp_end"),
                            "text": segment.text,
                        }
                        for segment in extracted.segments
                        if segment.modality == "speech"
                    ],
                    duration_seconds=_timeline_number(duration) or 0.0,
                )
            for item in integrity.get("checks", []):
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status") or "warn")
                checks.append(QualityCheck(
                    f"转写{str(item.get('name') or '完整性')}",
                    status,
                    str(item.get("detail") or "未提供检查说明"),
                ))
                if status == "fail":
                    hard_failure = True
                    score = min(score, 45)
                elif status == "warn":
                    score -= 5
        elif extracted.media_type == "video" and visual_segments:
            checks.append(QualityCheck("音视频转写", "warn", "没有识别到语音，仅保留真实视频关键帧理解"))
            score -= 20
        else:
            checks.append(QualityCheck("音视频转写", "fail", "没有得到语音转写或真实视频画面理解"))
            hard_failure = True
            score = min(score, 30)

    expected = 0
    located = 0
    if extracted.media_type == "pdf":
        expected = int(extracted.metadata.get("page_count") or 0)
        located = len({segment.location.get("page") for segment in extracted.segments if segment.location.get("page")})
    elif extracted.media_type == "presentation":
        expected = int(extracted.metadata.get("slide_count") or 0)
        located = len({segment.location.get("slide") for segment in extracted.segments if segment.location.get("slide")})
    coverage = located / expected if expected else 1.0
    if expected:
        if coverage >= 0.85:
            checks.append(QualityCheck("页面覆盖", "pass", f"覆盖 {located}/{expected} 页"))
        elif coverage >= 0.5:
            checks.append(QualityCheck("页面覆盖", "warn", f"只覆盖 {located}/{expected} 页，请检查扫描页或空白页"))
            score -= 15
        else:
            checks.append(QualityCheck("页面覆盖", "fail", f"仅覆盖 {located}/{expected} 页，解析结果不完整"))
            hard_failure = True
            score = min(score, 45)

    if extracted.checksum:
        checks.append(QualityCheck("来源校验", "pass", "已生成原始内容校验值，可用于去重和变更检测"))
    else:
        checks.append(QualityCheck("来源校验", "warn", "来源缺少内容校验值"))
        score -= 10

    if characters < 80 and extracted.media_type not in {"image"}:
        checks.append(QualityCheck("内容规模", "warn", "正文很短，问答覆盖范围可能有限"))
        score -= 10

    warning_count = len(extracted.warnings)
    if warning_count:
        checks.append(QualityCheck("解析提醒", "warn", f"解析器报告 {warning_count} 条提醒"))
        score -= min(15, warning_count * 3)

    score = max(0, min(100, score))
    accepted = not hard_failure and score >= 60
    grade = "优秀" if score >= 90 else "良好" if score >= 75 else "可用" if accepted else "不合格"
    metrics = {
        "segments": segment_count,
        "characters": characters,
        "speech_segments": speech_segments,
        "visual_segments": visual_segments,
        "expected_pages": expected,
        "covered_pages": located,
        "coverage": round(coverage, 4),
        "warnings": warning_count,
        "content_scope": scope,
    }
    if isinstance(ocr, dict):
        metrics["ocr"] = {
            "line_count": line_count,
            "mean_confidence": mean_confidence,
            "min_confidence": min_confidence,
            "low_confidence_lines": low_confidence_count,
            "low_confidence_ratio": round(low_confidence_ratio, 6),
            "threshold": threshold,
            "vision_fallback_used": vision_fallback,
        }
    transcription = extracted.metadata.get("transcription")
    if isinstance(transcription, dict) and isinstance(transcription.get("integrity"), dict):
        metrics["transcript_integrity"] = transcription["integrity"]
    return QualityReport(accepted, score, grade, checks, metrics)
