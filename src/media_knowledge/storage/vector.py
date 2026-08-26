from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from ..models import SearchFilters
from .database import KnowledgeDatabase


@dataclass(slots=True)
class VectorHit:
    chunk_id: str
    score: float


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        chunk_id: str,
        vector: Sequence[float],
        *,
        provider: str,
        model: str,
        content_hash: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: Sequence[float], top_k: int, filters: SearchFilters) -> list[VectorHit]:
        raise NotImplementedError


class SQLiteVectorStore(VectorStore):
    def __init__(
        self,
        database: KnowledgeDatabase,
        *,
        provider: str | None = None,
        model: str | None = None,
    ):
        self.database = database
        self.provider = provider
        self.model = model

    def upsert(
        self,
        chunk_id: str,
        vector: Sequence[float],
        *,
        provider: str,
        model: str,
        content_hash: str,
    ) -> None:
        self.database.upsert_embedding(chunk_id, provider, model, vector, content_hash)

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def search(self, query_vector: Sequence[float], top_k: int, filters: SearchFilters) -> list[VectorHit]:
        hits = []
        for row in self.database.iter_embeddings(
            filters,
            provider=self.provider,
            model=self.model,
        ):
            score = self._cosine(query_vector, json.loads(row["vector_json"]))
            if score > 0.0:
                hits.append(VectorHit(row["chunk_id"], score))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return hits[:top_k]
