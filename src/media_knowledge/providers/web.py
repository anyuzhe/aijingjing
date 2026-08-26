from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class WebSearchHit:
    title: str
    content: str
    url: str
    score: float = 0.0
    published_at: str | None = None


class WebSearchProvider(ABC):
    name: str

    @property
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[WebSearchHit]:
        raise NotImplementedError


class DisabledWebSearchProvider(WebSearchProvider):
    name = "disabled"

    @property
    def available(self) -> bool:
        return False

    def search(self, query: str, top_k: int = 5) -> list[WebSearchHit]:
        return []
