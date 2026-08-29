from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import urllib.error
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import media_knowledge.desktop.backup as backup_module
from media_knowledge.desktop import DesktopController
from media_knowledge.desktop.update import (
    _HTTPSOnlyRedirectHandler,
    _version,
    check_for_update,
    download_verified_update,
    verify_download_sha256,
)
from media_knowledge.product import DesktopSettings, ProductPaths
from media_knowledge.storage import KnowledgeDatabase


class BackupSecurityTests(unittest.TestCase):
    def _controller(self, root: Path) -> DesktopController:
        controller = DesktopController(root, migrate_legacy=False)
        controller.save_settings(
            DesktopSettings(
                default_model="local-extractive",
                embedding_provider="hash",
                embedding_model="hash-384-v1",
                answer_language="zh-CN",
            )
        )
        with KnowledgeDatabase(controller.paths.database):
            pass
        return controller

    def test_v2_backup_contains_all_knowledge_roots_and_per_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(Path(temporary) / "data")
            expected = {
                "notes/note.md": b"note",
                "archive/source.bin": b"archive",
                "assets/page.png": b"asset",
                "transcripts/audio.txt": b"transcript",
            }
            for name, content in expected.items():
                path = controller.paths.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            controller.paths.providers.write_text(
                '{"providers":[{"api_key":"must-not-leak"}]}', encoding="utf-8"
            )
            raw_settings = json.loads(controller.paths.settings.read_text(encoding="utf-8"))
            raw_settings["api_key"] = "must-not-leak-from-settings"
            controller.paths.settings.write_text(json.dumps(raw_settings), encoding="utf-8")

            target = controller.create_backup()
            with zipfile.ZipFile(target) as archive:
                names = set(archive.namelist())
                self.assertNotIn("providers.json", names)
                self.assertNotIn(b"must-not-leak", archive.read("settings.json"))
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["format"], "ai-jingjing-backup-v2")
                self.assertFalse(manifest["includes_credentials"])
                self.assertEqual(
                    manifest["content_roots"],
                    ["notes", "archive", "assets", "transcripts"],
                )
                records = {item["path"]: item for item in manifest["files"]}
                self.assertTrue({"knowledge.db", "settings.json", *expected}.issubset(records))
                for name, record in records.items():
                    payload = archive.read(name)
                    self.assertEqual(record["size"], len(payload))
                    self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())

    def test_v2_restore_replaces_all_roots_after_complete_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(Path(temporary) / "data")
            for root_name in backup_module.BACKUP_CONTENT_ROOTS:
                (getattr(controller.paths, root_name) / "kept.txt").write_text(
                    f"original-{root_name}", encoding="utf-8"
                )
            backup = controller.create_backup()
            for root_name in backup_module.BACKUP_CONTENT_ROOTS:
                base = getattr(controller.paths, root_name)
                (base / "kept.txt").write_text("changed", encoding="utf-8")
                (base / "stale.txt").write_text("stale", encoding="utf-8")
            controller.save_settings(DesktopSettings(answer_language="en-US"))

            report = controller.restore_backup(backup)
            self.assertEqual(report["format"], "ai-jingjing-backup-v2")
            self.assertTrue(Path(str(report["safety_backup"])).is_file())
            for root_name in backup_module.BACKUP_CONTENT_ROOTS:
                base = getattr(controller.paths, root_name)
                self.assertEqual(
                    (base / "kept.txt").read_text(encoding="utf-8"), f"original-{root_name}"
                )
                self.assertFalse((base / "stale.txt").exists())
            self.assertEqual(controller.settings.answer_language, "zh-CN")

    def test_v1_restore_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self._controller(root / "data")
            legacy = root / "legacy.aijjbackup"
            with zipfile.ZipFile(legacy, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(controller.paths.database, "knowledge.db")
                archive.writestr(
                    "manifest.json",
                    json.dumps(
                        {
                            "format": "ai-jingjing-backup-v1",
                            "includes_credentials": False,
                        }
                    ),
                )
                archive.writestr("notes/legacy.md", "legacy note")
                archive.writestr("archive/legacy.txt", "legacy source")
            (controller.paths.assets / "v2-only.txt").write_text("keep", encoding="utf-8")

            report = controller.restore_backup(legacy)
            self.assertEqual(report["format"], "ai-jingjing-backup-v1")
            self.assertEqual(
                (controller.paths.notes / "legacy.md").read_text(encoding="utf-8"),
                "legacy note",
            )
            self.assertTrue((controller.paths.assets / "v2-only.txt").is_file())

    def test_hash_mismatch_is_rejected_before_live_data_or_safety_backup_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self._controller(root / "data")
            live = controller.paths.notes / "evidence.md"
            live.write_text("trusted", encoding="utf-8")
            valid = controller.create_backup()
            tampered = root / "tampered.aijjbackup"
            with zipfile.ZipFile(valid) as source, zipfile.ZipFile(
                tampered, "w", zipfile.ZIP_DEFLATED
            ) as destination:
                for info in source.infolist():
                    payload = source.read(info.filename)
                    if info.filename == "notes/evidence.md":
                        payload = b"corrupt"
                    destination.writestr(info.filename, payload)
            before_backups = set(controller.paths.backups.iterdir())

            with self.assertRaisesRegex(ValueError, "哈希校验失败"):
                controller.restore_backup(tampered)
            self.assertEqual(live.read_text(encoding="utf-8"), "trusted")
            self.assertEqual(before_backups, set(controller.paths.backups.iterdir()))

    def test_zip_slip_and_oversized_entries_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self._controller(root / "data")
            malicious = root / "slip.aijjbackup"
            with zipfile.ZipFile(malicious, "w") as archive:
                archive.write(controller.paths.database, "knowledge.db")
                archive.writestr("manifest.json", '{"format":"ai-jingjing-backup-v1"}')
                archive.writestr("../escaped.txt", "escape")
            with self.assertRaisesRegex(ValueError, "不安全路径"):
                controller.restore_backup(malicious)
            self.assertFalse((root / "escaped.txt").exists())

            oversized = root / "oversized.aijjbackup"
            with zipfile.ZipFile(oversized, "w") as archive:
                archive.write(controller.paths.database, "knowledge.db")
                archive.writestr("manifest.json", '{"format":"ai-jingjing-backup-v1"}')
            with patch.object(backup_module, "MAX_ENTRY_UNCOMPRESSED_BYTES", 4):
                with self.assertRaisesRegex(ValueError, "超大条目"):
                    controller.restore_backup(oversized)

            compression_bomb = root / "compression-bomb.aijjbackup"
            with zipfile.ZipFile(compression_bomb, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(controller.paths.database, "knowledge.db")
                archive.writestr("manifest.json", '{"format":"ai-jingjing-backup-v1"}')
                archive.writestr("notes/repeated.txt", b"0" * 4096)
            with patch.object(backup_module, "MAX_COMPRESSION_RATIO", 2):
                with self.assertRaisesRegex(ValueError, "压缩比异常"):
                    controller.restore_backup(compression_bomb)


class UpdateSecurityTests(unittest.TestCase):
    class FakeResponse(BytesIO):
        status = 200

        def __init__(self, payload: bytes, *, final_url: str = "https://updates.example/update.json"):
            super().__init__(payload)
            self.headers = {"Content-Length": str(len(payload))}
            self._final_url = final_url

        def geturl(self) -> str:
            return self._final_url

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def test_update_report_exposes_required_sha256(self) -> None:
        checksum = "a" * 64
        payload = json.dumps(
            {
                "version": "2.1.0",
                "download_url": "https://updates.example/AI-Jingjing.dmg",
                "sha256": checksum,
                "notes": "security update",
            }
        ).encode()
        with patch(
            "media_knowledge.desktop.update._secure_urlopen",
            return_value=self.FakeResponse(payload),
        ):
            report = check_for_update("2.0.5", "https://updates.example/update.json")
        self.assertEqual(report.status, "available")
        self.assertEqual(report.sha256, checksum)
        self.assertEqual(report.to_dict()["sha256"], checksum)

    def test_each_update_redirect_hop_must_remain_https(self) -> None:
        handler = _HTTPSOnlyRedirectHandler()
        with self.assertRaisesRegex(urllib.error.HTTPError, "HTTPS"):
            handler.redirect_request(
                None, None, 302, "Found", {}, "http://downgrade.example/update"
            )

    def test_update_manifest_rejects_invalid_versions_missing_hash_and_http_redirect(self) -> None:
        base = {
            "version": "2.1",
            "download_url": "https://updates.example/app.dmg",
            "sha256": "a" * 64,
        }
        with patch(
            "media_knowledge.desktop.update._secure_urlopen",
            return_value=self.FakeResponse(json.dumps(base).encode()),
        ):
            with self.assertRaisesRegex(RuntimeError, "版本号"):
                check_for_update("2.0.5", "https://updates.example/update.json")

        base["version"] = "2.1.0"
        base.pop("sha256")
        with patch(
            "media_knowledge.desktop.update._secure_urlopen",
            return_value=self.FakeResponse(json.dumps(base).encode()),
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                check_for_update("2.0.5", "https://updates.example/update.json")

        base["sha256"] = "b" * 64
        with patch(
            "media_knowledge.desktop.update._secure_urlopen",
            return_value=self.FakeResponse(
                json.dumps(base).encode(), final_url="http://updates.example/update.json"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTPS"):
                check_for_update("2.0.5", "https://updates.example/update.json")

    def test_semver_prerelease_order_and_download_hash_verification(self) -> None:
        self.assertTrue(_version("2.1.0").newer_than(_version("2.1.0-rc.2")))
        self.assertTrue(_version("2.1.0-rc.10").newer_than(_version("2.1.0-rc.2")))
        with self.assertRaises(ValueError):
            _version("2.01.0")
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "AI-Jingjing.dmg"
            package.write_bytes(b"signed release payload")
            checksum = hashlib.sha256(package.read_bytes()).hexdigest()
            self.assertEqual(verify_download_sha256(package, checksum), package.resolve())
            with self.assertRaisesRegex(ValueError, "不匹配"):
                verify_download_sha256(package, "0" * 64)

    def test_update_download_is_streamed_verified_and_atomically_published(self) -> None:
        package = b"verified installer payload" * 100
        checksum = hashlib.sha256(package).hexdigest()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "media_knowledge.desktop.update._secure_urlopen",
            return_value=self.FakeResponse(
                package, final_url="https://cdn.example/releases/AI-Jingjing-2.1.0.dmg"
            ),
        ):
            target = download_verified_update(
                "https://updates.example/download", checksum, temporary
            )
            self.assertEqual(target.name, "AI-Jingjing-2.1.0.dmg")
            self.assertEqual(target.read_bytes(), package)
            self.assertEqual(
                [path.name for path in Path(temporary).iterdir()],
                ["AI-Jingjing-2.1.0.dmg"],
            )

    def test_update_download_rejects_bad_hash_size_and_insecure_redirect_without_residue(self) -> None:
        package = b"installer"
        checksum = hashlib.sha256(package).hexdigest()
        cases = (
            ("0" * 64, len(package) + 1, "https://cdn.example/app.dmg", "不匹配"),
            (checksum, len(package) - 1, "https://cdn.example/app.dmg", "大小上限"),
            (checksum, len(package) + 1, "http://cdn.example/app.dmg", "HTTPS"),
        )
        for expected, size_limit, final_url, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary, patch(
                "media_knowledge.desktop.update._secure_urlopen",
                return_value=self.FakeResponse(package, final_url=final_url),
            ):
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    download_verified_update(
                        "https://updates.example/download",
                        expected,
                        temporary,
                        max_bytes=size_limit,
                    )
                self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
