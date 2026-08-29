from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..models import SearchResult, SourceReference
from ..storage import KnowledgeDatabase
from .models import QuestionAnalysis


@dataclass(slots=True)
class RetrievalSelection:
    results: list[SearchResult]
    strategy: str
    details: dict[str, Any] = field(default_factory=dict)


class AdaptiveRetrievalPlanner:
    """Select focused, full-document, or hierarchical context deterministically."""

    SUMMARY_TASKS = {"synthesis"}

    def __init__(
        self,
        database: KnowledgeDatabase,
        *,
        full_context_token_budget: int = 12_000,
        hierarchical_chunk_limit: int = 12,
    ) -> None:
        self.database = database
        self.full_context_token_budget = max(1_000, full_context_token_budget)
        self.hierarchical_chunk_limit = max(4, hierarchical_chunk_limit)

    def select(
        self,
        analysis: QuestionAnalysis,
        focused_results: list[SearchResult],
        *,
        document_ids: list[str] | None,
        top_k: int,
        collections: list[str] | None = None,
        tags: list[str] | None = None,
        media_types: list[str] | None = None,
        folders: list[str] | None = None,
        date_range: tuple[str | None, str | None] | None = None,
    ) -> RetrievalSelection:
        candidate_document_id = self._candidate_document_id(
            analysis, focused_results, document_ids or []
        )
        if candidate_document_id is None:
            return RetrievalSelection(
                focused_results,
                "focused",
                {"reason": "question_requires_relevance-focused_retrieval"},
            )

        document = self.database.get_document(candidate_document_id)
        if document is None or not bool(document["enabled"]):
            return RetrievalSelection(
                focused_results,
                "focused",
                {"reason": "selected_document_is_missing_or_disabled"},
            )
        if not self._matches_filters(
            candidate_document_id,
            document,
            collections=collections or [],
            tags=tags or [],
            media_types=media_types or [],
            folders=folders or [],
            date_range=date_range,
        ):
            return RetrievalSelection(
                focused_results,
                "focused",
                {"reason": "selected_document_does_not_match_active_filters"},
            )
        rows = self.database.list_chunks(candidate_document_id)
        total_tokens = sum(int(row.get("token_count") or 0) for row in rows)
        common = {
            "document_id": candidate_document_id,
            "total_chunk_count": len(rows),
            "estimated_context_tokens": total_tokens,
            "full_context_token_budget": self.full_context_token_budget,
        }
        if rows and total_tokens <= self.full_context_token_budget:
            results = self._rows_to_results(rows, strategy="full_context")
            return RetrievalSelection(
                results,
                "full_context",
                {**common, "selected_chunk_count": len(results)},
            )

        if analysis.task_type in self.SUMMARY_TASKS and rows:
            limit = min(len(rows), max(top_k, self.hierarchical_chunk_limit))
            selected_rows = self._hierarchical_rows(rows, focused_results, limit)
            results = self._rows_to_results(selected_rows, strategy="hierarchical")
            return RetrievalSelection(
                results,
                "hierarchical",
                {
                    **common,
                    "selected_chunk_count": len(results),
                    "reason": "document_exceeds_full_context_budget",
                },
            )

        return RetrievalSelection(
            focused_results,
            "focused",
            {**common, "reason": "document_exceeds_full_context_budget"},
        )

    def _matches_filters(
        self,
        document_id: str,
        document: Any,
        *,
        collections: list[str],
        tags: list[str],
        media_types: list[str],
        folders: list[str],
        date_range: tuple[str | None, str | None] | None,
    ) -> bool:
        if media_types and str(document["media_type"]) not in media_types:
            return False
        if collections or tags:
            facets = self.database.document_facets(document_id)
            if collections and not set(collections).intersection(facets["collections"]):
                return False
            if tags and not set(tags).intersection(facets["tags"]):
                return False
        if folders:
            paths = [str(document[key] or "") for key in ("local_path", "obsidian_path")]
            normalized_folders = [value.rstrip("/\\") + "/" for value in folders]
            if not any(
                path == folder[:-1] or path.startswith(folder)
                for path in paths
                for folder in normalized_folders
            ):
                return False
        date_from, date_to = date_range or (None, None)
        updated_at = str(document["updated_at"])
        if date_from and updated_at < date_from:
            return False
        if date_to and updated_at > date_to:
            return False
        return True

    @classmethod
    def _candidate_document_id(
        cls,
        analysis: QuestionAnalysis,
        focused_results: list[SearchResult],
        document_ids: list[str],
    ) -> str | None:
        explicit = list(dict.fromkeys(value for value in document_ids if value))
        if len(explicit) == 1:
            return explicit[0]
        if len(explicit) > 1 or analysis.task_type not in cls.SUMMARY_TASKS:
            return None
        retrieved = list(dict.fromkeys(result.document_id for result in focused_results))
        return retrieved[0] if len(retrieved) == 1 else None

    @staticmethod
    def _hierarchical_rows(
        rows: list[dict[str, Any]],
        focused_results: list[SearchResult],
        limit: int,
    ) -> list[dict[str, Any]]:
        if len(rows) <= limit:
            return rows
        focused_ids = {item.chunk_id for item in focused_results}
        selected_indexes = {
            index for index, row in enumerate(rows) if str(row["id"]) in focused_ids
        }
        if limit == 1:
            selected_indexes.add(0)
        else:
            selected_indexes.update(
                round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)
            )
        if len(selected_indexes) > limit:
            focused_indexes = sorted(
                (index for index in selected_indexes if str(rows[index]["id"]) in focused_ids)
            )[: max(1, limit // 2)]
            evenly_spaced = sorted(selected_indexes - set(focused_indexes))
            remaining = limit - len(focused_indexes)
            selected_indexes = set(focused_indexes + evenly_spaced[:remaining])
        return [rows[index] for index in sorted(selected_indexes)[:limit]]

    def _rows_to_results(
        self, rows: list[dict[str, Any]], *, strategy: str
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        total = max(1, len(rows))
        for index, row in enumerate(rows):
            raw_source = row.get("source_reference_json") or "{}"
            source = SourceReference.from_dict(json.loads(str(raw_source)))
            chunk_id = str(row["id"])
            document_id = str(row["document_id"])
            source = source.with_chunk(document_id, chunk_id)
            results.append(
                SearchResult(
                    score=round(1.0 - (index / (total * 1000)), 6),
                    content=str(row["content"]),
                    title=source.title,
                    source=source,
                    page=source.page_number,
                    slide=source.slide_number,
                    timestamp_start=source.timestamp_start,
                    timestamp_end=source.timestamp_end,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    debug={"retrieval_strategy": strategy, "ordinal": int(row["ordinal"])},
                )
            )
        return results


__all__ = ["AdaptiveRetrievalPlanner", "RetrievalSelection"]
