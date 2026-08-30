from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from media_knowledge.desktop.model_manager import LocalModelManager
from media_knowledge.desktop.controller import DesktopController
from media_knowledge.product import ProductPaths


def _fake_model(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def _fake_sherpa_model(path: Path, *, include_embedding: bool = True) -> Path:
    segmentation = path / "segmentation"
    segmentation.mkdir(parents=True)
    (segmentation / "pyannote-segmentation.onnx").write_bytes(b"segmentation")
    if include_embedding:
        embedding = path / "embedding"
        embedding.mkdir(parents=True)
        (embedding / "3dspeaker-embedding.onnx").write_bytes(b"embedding")
    return path


class LocalModelManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.paths = ProductPaths.resolve(self.temporary.name).ensure()
        self.manager = LocalModelManager(self.paths)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_status_is_local_only_and_never_downloads(self) -> None:
        with patch("media_knowledge.desktop.model_manager.LocalModelManager._cached_huggingface_path", return_value=None) as local:
            status = self.manager.status("qwen3-asr-1.7b-mlx")
        self.assertFalse(status.installed)
        local.assert_called_once()

    def test_external_registration_is_preserved_when_removed(self) -> None:
        external = _fake_model(Path(self.temporary.name) / "external-qwen")
        status = self.manager.register_path("qwen3-asr-1.7b-mlx", external)
        self.assertTrue(status.installed)
        self.assertEqual(status.source, "external")
        self.assertTrue(status.content_verified)
        self.assertEqual(len(status.content_sha256 or ""), 64)
        self.assertFalse(self.manager.remove("qwen3-asr-1.7b-mlx"))
        self.assertTrue(external.is_dir())

    def test_registering_default_directory_is_managed_and_removable(self) -> None:
        model_id = "whisper-base-mlx"
        managed = _fake_model(self.paths.models / model_id)
        status = self.manager.register_path(model_id, managed)
        self.assertEqual(status.source, "managed")
        self.assertTrue(self.manager.remove(model_id))
        self.assertFalse(managed.exists())

    def test_sherpa_registration_requires_both_onnx_model_roles(self) -> None:
        model_id = "sherpa-speaker-diarization-zh"
        spec = self.manager.spec(model_id)
        self.assertEqual((spec.provider, spec.kind, spec.repo_id), (
            "sherpa-onnx", "diarization", None,
        ))
        incomplete = _fake_sherpa_model(
            Path(self.temporary.name) / "sherpa-incomplete",
            include_embedding=False,
        )
        with self.assertRaisesRegex(ValueError, "segmentation.*embedding ONNX"):
            self.manager.register_path(model_id, incomplete)

        complete = _fake_sherpa_model(
            Path(self.temporary.name) / "sherpa-complete"
        )
        status = self.manager.register_path(model_id, complete)
        self.assertTrue(status.verified)
        self.assertTrue(status.content_verified)
        self.assertEqual(status.path, str(complete.resolve()))

    def test_verified_identity_lookup_never_rehashes_or_downloads(self) -> None:
        external = _fake_model(Path(self.temporary.name) / "identity-model")
        status = self.manager.register_path("whisper-large-v3-mlx", external)
        with (
            patch("media_knowledge.desktop.model_manager._content_sha256") as content_hash,
            patch.object(self.manager, "_cached_huggingface_path") as cached_lookup,
        ):
            actual = self.manager.verified_content_sha256_for_path(external)
        self.assertEqual(actual, status.content_sha256)
        content_hash.assert_not_called()
        cached_lookup.assert_not_called()

    def test_controller_backfills_registered_model_identity_for_old_settings(self) -> None:
        root = Path(self.temporary.name) / "controller-data"
        controller = DesktopController(root, migrate_legacy=False)
        external = _fake_model(Path(self.temporary.name) / "controller-model")
        status = controller.local_models.register_path("whisper-large-v3-mlx", external)
        controller.save_settings(replace(
            controller.settings,
            asr_model_path=str(external),
            asr_model_sha256=None,
        ))

        reopened = DesktopController(root, migrate_legacy=False)
        self.assertEqual(reopened.settings.asr_model_sha256, status.content_sha256)

    def test_import_creates_app_managed_model_and_remove_is_scoped(self) -> None:
        source = _fake_model(Path(self.temporary.name) / "source-model")
        status = self.manager.import_model("whisper-small-mlx", source)
        self.assertTrue(status.verified)
        self.assertTrue(Path(status.path or "").is_relative_to(self.paths.models))
        self.assertTrue(self.manager.remove("whisper-small-mlx"))
        self.assertFalse(Path(status.path or "").exists())
        self.assertTrue(source.exists())

    def test_download_requires_explicit_call_and_uses_staging(self) -> None:
        def fake_snapshot(_repo_id: str, *, local_dir: str, token: str | None = None) -> str:
            _fake_model(Path(local_dir))
            return local_dir

        with patch.dict("sys.modules", {}):
            with patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot):
                status = self.manager.download("qwen3-asr-0.6b-mlx")
        self.assertTrue(status.installed)
        self.assertEqual(status.source, "downloaded")
        self.assertTrue(status.content_verified)

    def test_verification_detects_changed_model_bytes(self) -> None:
        source = _fake_model(Path(self.temporary.name) / "verified-model")
        registered = self.manager.register_path("qwen3-asr-0.6b-mlx", source)
        self.assertTrue(registered.content_verified)
        (source / "model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "SHA-256 不匹配"):
            self.manager.verify("qwen3-asr-0.6b-mlx")
        status = self.manager.status("qwen3-asr-0.6b-mlx")
        self.assertFalse(status.verified)
        self.assertIn("SHA-256 不匹配", status.error or "")

    def test_fingerprint_hashes_content_not_only_file_sizes(self) -> None:
        source = _fake_model(Path(self.temporary.name) / "fingerprint-model")
        self.manager.register_path("whisper-base-mlx", source)
        before = self.manager.fingerprint("whisper-base-mlx")
        (source / "model.safetensors").write_bytes(b"WEIGHTS")
        after = self.manager.fingerprint("whisper-base-mlx")
        self.assertNotEqual(before, after)

    def test_register_reports_hash_progress_and_can_be_cancelled(self) -> None:
        source = _fake_model(Path(self.temporary.name) / "cancel-model")
        messages: list[str] = []
        checks = 0

        def cancel() -> None:
            nonlocal checks
            checks += 1
            if checks > 1:
                raise RuntimeError("模型安装已取消")

        with self.assertRaisesRegex(RuntimeError, "已取消"):
            self.manager.register_path(
                "whisper-tiny-mlx",
                source,
                progress=messages.append,
                check_cancelled=cancel,
            )
        self.assertTrue(messages)
        self.assertFalse(self.manager.status("whisper-tiny-mlx").installed)


if __name__ == "__main__":
    unittest.main()
