from __future__ import annotations

from pathlib import Path
import warnings

from .base import EmbeddingProvider
from ..resource_scheduler import LOCAL_HEAVY_TASKS


class FastEmbedProvider(EmbeddingProvider):
    """Lazy, CPU-only semantic embeddings backed by an installed ONNX model.

    Knowledge ingestion and retrieval are execution paths, not model installers.
    They therefore always ask FastEmbed for local files only.  This prevents a
    first document import from unexpectedly opening a network connection.
    """

    name = "fastembed"

    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        *,
        cache_dir: str | Path | None = None,
        threads: int | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.threads = threads
        self.local_files_only = bool(local_files_only)
        self._backend = None

    def _load(self):
        if self._backend is not None:
            return self._backend
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ImportError as exc:
            raise RuntimeError("本地语义检索组件 FastEmbed 未安装") from exc
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*now uses mean pooling instead of CLS embedding.*",
                    category=UserWarning,
                )
                self._backend = TextEmbedding(
                    model_name=self.model,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    threads=self.threads,
                    local_files_only=self.local_files_only,
                )
        except Exception as exc:
            raise RuntimeError(
                "本地语义模型尚未安装或校验失败。正式导入不会联网下载模型；"
                "请先显式准备 FastEmbed 模型，或在设置中切换为本地哈希检索。"
            ) from exc
        return self._backend

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        backend = self._load()
        try:
            with LOCAL_HEAVY_TASKS.reserve("embedding:fastembed"):
                return [
                    vector.astype("float32").tolist()
                    for vector in backend.embed(texts, batch_size=16)
                ]
        except Exception as exc:
            raise RuntimeError("本地语义模型生成向量失败") from exc
