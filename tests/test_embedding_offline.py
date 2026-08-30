from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from media_knowledge.embedding.fastembed_local import FastEmbedProvider
from media_knowledge.product import DesktopSettings


class _FakeVector:
    def astype(self, _kind: str):
        return self

    def tolist(self) -> list[float]:
        return [0.0, 1.0]


class FastEmbedOfflineTests(unittest.TestCase):
    def test_runtime_load_is_local_only(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeTextEmbedding:
            def __init__(self, **kwargs: object) -> None:
                calls.append(dict(kwargs))

            def embed(self, texts: list[str], batch_size: int = 16):
                self.texts = texts
                self.batch_size = batch_size
                return [_FakeVector() for _ in texts]

        fake_module = types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            sys.modules, {"fastembed": fake_module}
        ):
            provider = FastEmbedProvider(cache_dir=Path(temporary))
            self.assertEqual(provider.embed(["离线知识"]), [[0.0, 1.0]])

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["local_files_only"], True)

    def test_missing_model_error_never_suggests_implicit_download(self) -> None:
        class MissingTextEmbedding:
            def __init__(self, **_kwargs: object) -> None:
                raise FileNotFoundError("not cached")

        fake_module = types.SimpleNamespace(TextEmbedding=MissingTextEmbedding)
        with patch.dict(sys.modules, {"fastembed": fake_module}):
            with self.assertRaisesRegex(RuntimeError, "不会联网下载"):
                FastEmbedProvider().embed(["证据"])

    def test_fresh_desktop_install_defaults_to_hash_embeddings(self) -> None:
        settings = DesktopSettings()
        self.assertEqual(settings.embedding_provider, "hash")
        self.assertEqual(settings.embedding_model, "hash-384-v1")


if __name__ == "__main__":
    unittest.main()
