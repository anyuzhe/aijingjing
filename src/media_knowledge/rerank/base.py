from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import SearchCandidate


class RerankProvider(ABC):
    name: str

    @abstractmethod
    def rerank(self, query: str, candidates: list[SearchCandidate]) -> list[SearchCandidate]:
        raise NotImplementedError
