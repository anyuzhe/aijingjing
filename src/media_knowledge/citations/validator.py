from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..qa.models import Citation, Evidence
from ..storage.conversations import ConversationRepository


@dataclass(slots=True)
class CitationValidationResult:
    valid: bool
    citation_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class CitationValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("citation validation failed: " + "; ".join(errors))


class CitationValidator:
    MARKER = re.compile(r"\[S\d+\]")
    INSUFFICIENT_PHRASES = (
        "没有足够资料",
        "资料不足",
        "insufficient evidence",
        "does not contain enough information",
    )

    def __init__(self, repository: ConversationRepository):
        self.repository = repository

    def validate(self, markdown: str, evidence: list[Evidence]) -> CitationValidationResult:
        evidence_map = {item.evidence_id: item for item in evidence}
        marker_ids = [marker[1:-1] for marker in self.MARKER.findall(markdown)]
        citation_ids = list(dict.fromkeys(marker_ids))
        errors: list[str] = []
        for citation_id in citation_ids:
            item = evidence_map.get(citation_id)
            if item is None:
                errors.append(f"citation [{citation_id}] is not in the current evidence set")
                continue
            if item.source_kind == "knowledge":
                if not item.document_id or not item.chunk_id:
                    errors.append(f"citation [{citation_id}] has no document/chunk identity")
                elif not self.repository.chunk_belongs_to_document(item.chunk_id, item.document_id):
                    errors.append(f"citation [{citation_id}] does not point to a current retrieved chunk")
            elif item.source_kind == "web" and not item.source.original_uri:
                errors.append(f"citation [{citation_id}] has no web URI")
        is_insufficient = any(phrase.casefold() in markdown.casefold() for phrase in self.INSUFFICIENT_PHRASES)
        if evidence and not citation_ids and not is_insufficient:
            errors.append("answer contains no citation markers")
        return CitationValidationResult(not errors, citation_ids, errors)

    @staticmethod
    def citations(result: CitationValidationResult, evidence: list[Evidence]) -> list[Citation]:
        evidence_map = {item.evidence_id: item for item in evidence}
        return [Citation.from_evidence(evidence_map[item]) for item in result.citation_ids]
