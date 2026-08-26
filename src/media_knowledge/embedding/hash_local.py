from __future__ import annotations

import hashlib
import math
import re

from .base import EmbeddingProvider


class HashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free local feature hashing; deterministic and private by default."""

    name = "local-hash"

    def __init__(self, dimensions: int = 384, model: str = "hash-384-v1"):
        if dimensions < 32:
            raise ValueError("embedding dimensions must be at least 32")
        self.dimensions = dimensions
        self.model = model

    @staticmethod
    def _features(text: str) -> list[str]:
        lowered = text.casefold()
        words = re.findall(r"[a-z0-9_]+", lowered)
        cjk_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
        cjk = [char for run in cjk_runs for char in run]
        cjk_bigrams = [run[index : index + 2] for run in cjk_runs for index in range(len(run) - 1)]
        word_bigrams = [f"{words[index]}::{words[index + 1]}" for index in range(len(words) - 1)]
        return words + word_bigrams + cjk + cjk_bigrams

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                index = value % self.dimensions
                sign = 1.0 if value & (1 << 63) else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors
