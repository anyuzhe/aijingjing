from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from ..models import SearchCandidate
from .base import RerankProvider


def _terms(value: str) -> set[str]:
    lowered = value.casefold()
    words = set(re.findall(r"[a-z0-9_]+", lowered))
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    cjk = {char for run in cjk_runs for char in run}
    cjk.update(run[index : index + 2] for run in cjk_runs for index in range(len(run) - 1))
    return words | cjk


class DisabledRerankProvider(RerankProvider):
    name = "disabled"

    def rerank(self, query: str, candidates: list[SearchCandidate]) -> list[SearchCandidate]:
        for candidate in candidates:
            candidate.rerank_score = candidate.fused_score
        return candidates


class LocalLexicalRerankProvider(RerankProvider):
    name = "local-lexical"

    def rerank(self, query: str, candidates: list[SearchCandidate]) -> list[SearchCandidate]:
        query_terms = _terms(query)
        max_fused = max((candidate.fused_score for candidate in candidates), default=1.0) or 1.0
        max_keyword = max((candidate.keyword_score or 0.0 for candidate in candidates), default=1.0) or 1.0
        max_vector = max((candidate.vector_score or 0.0 for candidate in candidates), default=1.0) or 1.0
        for candidate in candidates:
            title_overlap = len(query_terms & _terms(candidate.title)) / max(1, len(query_terms))
            content_overlap = len(query_terms & _terms(candidate.content)) / max(1, len(query_terms))
            candidate.lexical_overlap = max(content_overlap, title_overlap * 0.65)
            candidate.rerank_score = (
                0.45 * content_overlap
                + 0.10 * title_overlap
                + 0.20 * (candidate.fused_score / max_fused)
                + 0.15 * ((candidate.keyword_score or 0.0) / max_keyword)
                + 0.10 * ((candidate.vector_score or 0.0) / max_vector)
            )
        return sorted(
            candidates,
            key=lambda item: (-(item.rerank_score or 0.0), -item.fused_score, item.chunk_id),
        )


class HTTPRerankProvider(RerankProvider):
    name = "api"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        if not base_url or not api_key or not model:
            raise ValueError("base_url, api_key, and model are required")
        self.endpoint = base_url.rstrip("/") if base_url.rstrip("/").endswith("/rerank") else base_url.rstrip("/") + "/rerank"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def rerank(self, query: str, candidates: list[SearchCandidate]) -> list[SearchCandidate]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                {"model": self.model, "query": query, "documents": [candidate.content for candidate in candidates]}
            ).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"rerank request failed: {exc}") from exc
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise RuntimeError("rerank response does not contain a results array")
        ranked: list[SearchCandidate] = []
        seen: set[int] = set()
        for result in results:
            index = result.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(candidates) or index in seen:
                continue
            seen.add(index)
            candidate = candidates[index]
            candidate.rerank_score = float(result.get("relevance_score", result.get("score", 0.0)))
            ranked.append(candidate)
        if not ranked:
            raise RuntimeError("rerank response contained no usable results")
        return ranked
