from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .models import Citation, Evidence


_CITATION = re.compile(r"\[S\d+\]")
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?])\s+|\n+")
_INSUFFICIENT = (
    "没有足够资料",
    "资料不足",
    "无法从知识库确认",
    "insufficient evidence",
    "does not contain enough information",
)


@dataclass(frozen=True, slots=True)
class EvidenceQuality:
    """Deterministic support diagnostics, never a probability of truth."""

    level: str
    citation_coverage: float
    evidence_utilization: float
    source_diversity: float
    evidence_count: int
    cited_evidence_count: int
    citation_count: int
    source_count: int
    claim_count: int
    cited_claim_count: int
    unsupported_claim_count: int
    instruction_risk_count: int
    retrieval_strategy: str
    explanation: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def _claim_units(markdown: str) -> list[str]:
    """Return deterministic answer units that ought to carry a citation.

    This is deliberately a formatting/coverage diagnostic. It is not a semantic
    fact checker and must not be presented as a calibrated confidence score.
    """

    plain = _CODE_BLOCK.sub("", markdown)
    units: list[str] = []
    for raw in _SENTENCE_BOUNDARY.split(plain):
        value = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|>\s*)", "", raw).strip()
        value = re.sub(r"^#{1,6}\s+", "", value).strip()
        without_markers = _CITATION.sub("", value).strip(" \t:：;；")
        if len(without_markers) < 5:
            continue
        if value.rstrip().endswith((":", "：")):
            continue
        if any(phrase.casefold() in value.casefold() for phrase in _INSUFFICIENT):
            continue
        units.append(value)
    return units


def evaluate_evidence_quality(
    markdown: str,
    evidence: list[Evidence],
    citations: list[Citation],
    *,
    retrieval_strategy: str,
    image_count: int = 0,
) -> EvidenceQuality:
    evidence_ids = {item.evidence_id for item in evidence}
    cited_ids = {item.evidence_id for item in citations if item.evidence_id in evidence_ids}
    source_ids = {
        item.source.document_id
        or item.source.original_uri
        or item.source.local_path
        or item.source.source_id
        for item in evidence
    }
    claims = _claim_units(markdown)
    cited_claim_count = sum(bool(_CITATION.search(claim)) for claim in claims)
    claim_count = len(claims)
    coverage = cited_claim_count / claim_count if claim_count else 0.0
    utilization = len(cited_ids) / len(evidence_ids) if evidence_ids else 0.0
    source_diversity = len(source_ids) / len(evidence) if evidence else 0.0

    if not evidence:
        level = "image_only" if image_count else "insufficient"
        explanation = (
            "回答仅基于用户图片观察，未使用知识库证据。"
            if image_count
            else "未检索到可用于回答的知识库证据。"
        )
    elif not citations:
        level = "insufficient"
        explanation = "检索到了候选证据，但回答没有采用可验证引用。"
    elif coverage >= 0.8:
        level = "well_supported"
        explanation = "回答中的可识别知识陈述大多带有当前证据集中的有效引用。"
    elif coverage >= 0.4:
        level = "partially_supported"
        explanation = "回答只对部分可识别知识陈述提供了有效引用。"
    else:
        level = "limited"
        explanation = "回答采用了引用，但多数可识别知识陈述仍缺少逐句证据标记。"

    return EvidenceQuality(
        level=level,
        citation_coverage=round(coverage, 3),
        evidence_utilization=round(utilization, 3),
        source_diversity=round(source_diversity, 3),
        evidence_count=len(evidence),
        cited_evidence_count=len(cited_ids),
        citation_count=len(citations),
        source_count=len(source_ids),
        claim_count=claim_count,
        cited_claim_count=cited_claim_count,
        unsupported_claim_count=max(0, claim_count - cited_claim_count),
        instruction_risk_count=sum(item.instruction_risk for item in evidence),
        retrieval_strategy=retrieval_strategy,
        explanation=explanation,
        reasons=(
            explanation,
            f"{cited_claim_count}/{claim_count} 个可识别知识陈述带引用。"
            if claim_count
            else "未识别出需要引用的知识陈述。",
            f"引用覆盖 {len(cited_ids)}/{len(evidence_ids)} 个候选证据。"
            if evidence_ids
            else "没有候选证据可供引用。",
        ),
    )


__all__ = ["EvidenceQuality", "evaluate_evidence_quality"]
