from __future__ import annotations

from dataclasses import asdict, dataclass, field

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

    speech_segments = sum(segment.modality == "speech" for segment in extracted.segments)
    visual_segments = sum(segment.modality in {"image", "slide", "page"} for segment in extracted.segments)
    if extracted.media_type in {"audio", "video"}:
        if speech_segments:
            checks.append(QualityCheck("音视频转写", "pass", f"生成 {speech_segments} 个带时间轴的语音片段"))
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
    return QualityReport(accepted, score, grade, checks, metrics)
