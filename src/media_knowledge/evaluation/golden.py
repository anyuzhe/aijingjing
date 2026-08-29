from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..qa import KnowledgeQAEngine
from ..retrieval import KnowledgeRetriever


@dataclass(frozen=True, slots=True)
class CitationTarget:
    target_id: str
    chunk_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any], index: int) -> "CitationTarget":
        return cls(
            target_id=str(value.get("target_id") or value.get("claim_id") or f"target-{index}"),
            chunk_ids=tuple(str(item) for item in value.get("chunk_ids", []) if item),
            document_ids=tuple(str(item) for item in value.get("document_ids", []) if item),
        )


@dataclass(frozen=True, slots=True)
class GoldenCase:
    case_id: str
    query: str
    relevant_chunk_ids: tuple[str, ...] = ()
    relevant_document_ids: tuple[str, ...] = ()
    citation_targets: tuple[CitationTarget, ...] = ()
    filters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], index: int) -> "GoldenCase":
        case_id = str(value.get("id") or value.get("case_id") or f"case-{index}")
        query = str(value.get("query") or "").strip()
        if not query:
            raise ValueError(f"golden case {case_id!r} has no query")
        chunk_ids = tuple(str(item) for item in value.get("relevant_chunk_ids", []) if item)
        document_ids = tuple(str(item) for item in value.get("relevant_document_ids", []) if item)
        targets = tuple(
            CitationTarget.from_dict(item, target_index)
            for target_index, item in enumerate(value.get("citation_targets", []), 1)
            if isinstance(item, dict)
        )
        if targets:
            chunk_ids = tuple(
                dict.fromkeys([*chunk_ids, *(item for target in targets for item in target.chunk_ids)])
            )
            document_ids = tuple(
                dict.fromkeys(
                    [*document_ids, *(item for target in targets for item in target.document_ids)]
                )
            )
        if not chunk_ids and not document_ids and not targets:
            raise ValueError(f"golden case {case_id!r} has no relevance targets")
        return cls(
            case_id=case_id,
            query=query,
            relevant_chunk_ids=chunk_ids,
            relevant_document_ids=document_ids,
            citation_targets=targets,
            filters=dict(value.get("filters") or {}),
        )


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    cases: tuple[GoldenCase, ...]
    schema_version: str = "1.0"
    name: str = "golden-evaluation"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationDataset":
        raw_cases = value.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("golden dataset must contain a non-empty cases list")
        cases = tuple(
            GoldenCase.from_dict(item, index)
            for index, item in enumerate(raw_cases, 1)
            if isinstance(item, dict)
        )
        if not cases:
            raise ValueError("golden dataset contains no valid case objects")
        return cls(
            cases=cases,
            schema_version=str(value.get("schema_version") or "1.0"),
            name=str(value.get("name") or "golden-evaluation"),
        )


def load_golden_dataset(path: str | Path) -> EvaluationDataset:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid golden dataset JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("golden dataset root must be a JSON object")
    return EvaluationDataset.from_dict(payload)


class GoldenEvaluator:
    """Run deterministic retrieval and citation checks over a local golden set."""

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        *,
        qa_engine: KnowledgeQAEngine | None = None,
    ) -> None:
        self.retriever = retriever
        self.qa_engine = qa_engine

    def evaluate(
        self,
        dataset: EvaluationDataset,
        *,
        top_k: int = 10,
        evaluate_citations: bool = True,
    ) -> dict[str, Any]:
        if top_k < 1 or top_k > 12:
            raise ValueError("evaluation top_k must be between 1 and 12")
        if evaluate_citations and self.qa_engine is None:
            raise ValueError("qa_engine is required when citation evaluation is enabled")

        case_reports: list[dict[str, Any]] = []
        reciprocal_ranks: list[float] = []
        hits: list[float] = []
        correct_citations = 0
        citation_total = 0
        covered_targets = 0
        target_total = 0

        for case in dataset.cases:
            kwargs = self._search_kwargs(case.filters)
            results = self.retriever.search_knowledge(case.query, top_k=top_k, **kwargs)
            first_relevant_rank = next(
                (
                    rank
                    for rank, result in enumerate(results, 1)
                    if self._matches(
                        result.chunk_id,
                        result.document_id,
                        case.relevant_chunk_ids,
                        case.relevant_document_ids,
                    )
                ),
                None,
            )
            reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            reciprocal_ranks.append(reciprocal_rank)
            hits.append(float(first_relevant_rank is not None))

            cited: list[dict[str, str | None]] = []
            case_correct = 0
            targets = self._targets(case)
            case_covered = 0
            if evaluate_citations and self.qa_engine is not None:
                answer = self.qa_engine.ask(case.query, top_k=top_k, **kwargs)
                cited = [
                    {
                        "citation_id": citation.citation_id,
                        "chunk_id": citation.chunk_id,
                        "document_id": citation.document_id,
                    }
                    for citation in answer.citations
                ]
                for citation in answer.citations:
                    if self._matches(
                        citation.chunk_id,
                        citation.document_id,
                        case.relevant_chunk_ids,
                        case.relevant_document_ids,
                    ):
                        case_correct += 1
                for target in targets:
                    if any(
                        self._matches(
                            citation.chunk_id,
                            citation.document_id,
                            target.chunk_ids,
                            target.document_ids,
                        )
                        for citation in answer.citations
                    ):
                        case_covered += 1
                correct_citations += case_correct
                citation_total += len(answer.citations)
                covered_targets += case_covered
                target_total += len(targets)

            case_reports.append(
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "retrieved_chunk_ids": [item.chunk_id for item in results],
                    "first_relevant_rank": first_relevant_rank,
                    "hit": first_relevant_rank is not None,
                    "reciprocal_rank": round(reciprocal_rank, 6),
                    "citations": cited,
                    "citation_precision": round(case_correct / len(cited), 6) if cited else 0.0,
                    "citation_coverage": (
                        round(case_covered / len(targets), 6) if evaluate_citations and targets else 0.0
                    ),
                }
            )

        count = len(case_reports)
        metrics = {
            f"hit_rate@{top_k}": round(sum(hits) / count, 6) if count else 0.0,
            "mrr": round(sum(reciprocal_ranks) / count, 6) if count else 0.0,
            "citation_precision": (
                round(correct_citations / citation_total, 6) if citation_total else 0.0
            ),
            "citation_coverage": (
                round(covered_targets / target_total, 6) if target_total else 0.0
            ),
        }
        return {
            "schema_version": "1.0",
            "dataset": dataset.name,
            "case_count": count,
            "top_k": top_k,
            "metrics": metrics,
            "counts": {
                "retrieval_hits": int(sum(hits)),
                "correct_citations": correct_citations,
                "total_citations": citation_total,
                "covered_citation_targets": covered_targets,
                "total_citation_targets": target_total,
            },
            "metric_definitions": {
                f"hit_rate@{top_k}": "fraction of cases with a relevant chunk or document in top-k",
                "mrr": "mean reciprocal rank of the first relevant retrieval result",
                "citation_precision": "fraction of emitted citations matching a golden relevance target",
                "citation_coverage": "fraction of golden citation target groups covered by at least one citation",
            },
            "cases": case_reports,
        }

    @staticmethod
    def _search_kwargs(filters: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "collections",
            "tags",
            "media_types",
            "folders",
            "document_ids",
            "date_range",
        }
        return {key: value for key, value in filters.items() if key in allowed}

    @staticmethod
    def _matches(
        chunk_id: str | None,
        document_id: str | None,
        relevant_chunk_ids: tuple[str, ...],
        relevant_document_ids: tuple[str, ...],
    ) -> bool:
        return bool(
            (chunk_id and chunk_id in relevant_chunk_ids)
            or (document_id and document_id in relevant_document_ids)
        )

    @staticmethod
    def _targets(case: GoldenCase) -> tuple[CitationTarget, ...]:
        if case.citation_targets:
            return case.citation_targets
        if case.relevant_chunk_ids:
            return tuple(
                CitationTarget(f"chunk:{chunk_id}", chunk_ids=(chunk_id,))
                for chunk_id in case.relevant_chunk_ids
            )
        return tuple(
            CitationTarget(f"document:{document_id}", document_ids=(document_id,))
            for document_id in case.relevant_document_ids
        )


__all__ = [
    "CitationTarget",
    "EvaluationDataset",
    "GoldenCase",
    "GoldenEvaluator",
    "load_golden_dataset",
]
