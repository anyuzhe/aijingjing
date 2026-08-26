from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import EmbeddingProvider


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        if not base_url or not api_key or not model:
            raise ValueError("base_url, api_key, and model are required")
        self.endpoint = base_url.rstrip("/") if base_url.rstrip("/").endswith("/embeddings") else base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"embedding request failed: {exc}") from exc
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("embedding response does not contain a data array")
        ordered = sorted(rows, key=lambda row: row.get("index", 0))
        vectors = [row.get("embedding") for row in ordered]
        if len(vectors) != len(texts) or not all(isinstance(vector, list) for vector in vectors):
            raise RuntimeError("embedding response count or shape is invalid")
        return vectors
