from __future__ import annotations

import re
from collections import defaultdict

from ..embedding import EmbeddingProvider
from ..models import SearchCandidate, SearchFilters, SearchResult
from ..rerank import DisabledRerankProvider, RerankProvider
from ..storage import KnowledgeDatabase, SQLiteVectorStore, VectorStore


class KnowledgeRetriever:
    def __init__(
        self,
        database: KnowledgeDatabase,
        embedding_provider: EmbeddingProvider,
        *,
        vector_store: VectorStore | None = None,
        rerank_provider: RerankProvider | None = None,
        rrf_k: int = 60,
    ):
        self.database = database
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store or SQLiteVectorStore(
            database,
            provider=embedding_provider.name,
            model=embedding_provider.model,
        )
        self.rerank_provider = rerank_provider or DisabledRerankProvider()
        self.rrf_k = rrf_k

    @staticmethod
    def normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query).strip()

    @staticmethod
    def _score(candidate: SearchCandidate) -> float:
        return candidate.rerank_score if candidate.rerank_score is not None else candidate.fused_score

    def _rank_relevant(self, candidates: list[SearchCandidate], top_k: int) -> list[SearchCandidate]:
        ranked = sorted(
            candidates,
            key=lambda item: (-self._score(item), -item.fused_score, item.chunk_id),
        )
        if not ranked:
            return []

        if self.rerank_provider.name == "local-lexical":
            strongest_overlap = max((item.lexical_overlap or 0.0) for item in ranked)
            if strongest_overlap > 0:
                overlap_floor = max(0.03, strongest_overlap * 0.25)
                score_floor = self._score(ranked[0]) * 0.25
                ranked = [
                    item for item in ranked
                    if (item.lexical_overlap or 0.0) >= overlap_floor
                    and self._score(item) >= score_floor
                ]
            elif self.embedding_provider.name == "local-hash":
                # Feature hashing is not semantic enough to justify unrelated zero-overlap hits.
                ranked = []
            else:
                score_floor = self._score(ranked[0]) * 0.55
                ranked = [item for item in ranked if self._score(item) >= score_floor]
        elif self.rerank_provider.name == "api":
            score_floor = max(0.15, self._score(ranked[0]) * 0.35)
            ranked = [item for item in ranked if self._score(item) >= score_floor]

        return ranked[:top_k]

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_+-]+|[\u3400-\u9fff]+", query)
        safe = [token.replace('"', '""') for token in tokens if token.strip()]
        return " OR ".join(f'"{token}"' for token in safe)

    def search_knowledge(
        self,
        query: str,
        *,
        collections: list[str] | None = None,
        tags: list[str] | None = None,
        media_types: list[str] | None = None,
        folders: list[str] | None = None,
        document_ids: list[str] | None = None,
        date_range: tuple[str | None, str | None] | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        normalized = self.normalize_query(query)
        if not normalized:
            raise ValueError("query must not be empty")
        if top_k < 1 or top_k > 50:
            raise ValueError("top_k must be between 1 and 50")
        date_from, date_to = date_range or (None, None)
        filters = SearchFilters(
            collections=collections or [],
            tags=tags or [],
            media_types=media_types or [],
            folders=folders or [],
            document_ids=document_ids or [],
            date_from=date_from,
            date_to=date_to,
        )
        query_vectors = self.embedding_provider.embed([normalized])
        if len(query_vectors) != 1:
            raise RuntimeError("embedding provider did not return one query vector")
        vector_hits = self.vector_store.search(query_vectors[0], 40, filters)
        fts_query = self._fts_query(normalized)
        keyword_rows = self.database.keyword_search(fts_query, 40, filters) if fts_query else []

        scores: dict[str, float] = defaultdict(float)
        vector_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}
        for rank, hit in enumerate(vector_hits, 1):
            scores[hit.chunk_id] += 1.0 / (self.rrf_k + rank)
            vector_scores[hit.chunk_id] = hit.score
        for rank, row in enumerate(keyword_rows, 1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            keyword_scores[chunk_id] = -float(row["rank"])

        fused_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:30]
        rows = self.database.fetch_candidates(fused_ids)
        candidates = []
        for chunk_id in fused_ids:
            row = rows.get(chunk_id)
            if row is None:
                continue
            candidates.append(
                SearchCandidate(
                    chunk_id=chunk_id,
                    content=row["content"],
                    title=row["title"],
                    source_reference=row["source_reference"],
                    fused_score=scores[chunk_id],
                    vector_score=vector_scores.get(chunk_id),
                    keyword_score=keyword_scores.get(chunk_id),
                )
            )
        ranked = self._rank_relevant(
            self.rerank_provider.rerank(normalized, candidates),
            top_k,
        )
        results = []
        for candidate in ranked:
            source = candidate.source_reference
            score = candidate.rerank_score if candidate.rerank_score is not None else candidate.fused_score
            results.append(
                SearchResult(
                    score=score,
                    content=candidate.content,
                    title=candidate.title,
                    source=source,
                    page=source.page_number,
                    slide=source.slide_number,
                    timestamp_start=source.timestamp_start,
                    timestamp_end=source.timestamp_end,
                    document_id=source.document_id or "",
                    chunk_id=candidate.chunk_id,
                    debug={
                        "fused_score": candidate.fused_score,
                        "vector_score": candidate.vector_score,
                        "keyword_score": candidate.keyword_score,
                        "rerank_score": candidate.rerank_score,
                        "lexical_overlap": candidate.lexical_overlap,
                    },
                )
            )
        return results

    search = search_knowledge
