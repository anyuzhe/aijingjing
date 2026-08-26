from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
