from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from email.message import Message
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from media_knowledge.config import AppConfig, CompatibleQAProviderConfig
from media_knowledge.desktop import DesktopController, ProviderConfigStore
from media_knowledge.ingestion import CancellationToken, IngestionService
from media_knowledge.ingestion.quality import evaluate_extraction
from media_knowledge.ingestion.types import ExtractionResult
from media_knowledge.ingestion.extractors import (
    DirectMediaURLExtractor,
    ExtractionContext,
    WeixinChannelsExtractor,
    WebExtractor,
    url_extractor_for,
)
from media_knowledge.product import DesktopSettings, ProductPaths, PRODUCT_NAME, PRODUCT_SLUG
from media_knowledge.models import ContentSegment
from media_knowledge.sync import scan_folder


class DesktopProductTests(unittest.TestCase):
    def test_video_link_router_recognizes_weixin_and_direct_media(self) -> None:
        self.assertIsInstance(
            url_extractor_for("https://weixin.qq.com/sph/AciGNsUoaW"),
            WeixinChannelsExtractor,
        )
        self.assertIsInstance(
            url_extractor_for("https://cdn.example.test/course.mp4?token=temporary"),
            DirectMediaURLExtractor,
        )
        self.assertIsInstance(url_extractor_for("https://example.test/article"), WebExtractor)

    def test_weixin_public_share_metadata_is_rejected_without_media_stream(self) -> None:
        payload = {
            "data": {
                "authorInfo": {"nickname": "数分魔"},
                "feedInfo": {
                    "description": "Level2逐笔数据到底该怎么存。视频比较多种时序存储方案。",
                    "coverUrl": "https://finder.video.qq.com/cover.jpg",
                    "createtime": 1787600000,
                },
                "errMsg": {"type": 0},
            },
            "errCode": 0,
            "errMsg": "",
        }

        class FakeResponse(BytesIO):
            def __init__(self, body: bytes) -> None:
                super().__init__(body)
                self.headers = Message()
                self.headers["Content-Type"] = "application/json; charset=utf-8"

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            context = ExtractionContext(
                paths=paths,
                settings=DesktopSettings(auto_synthesize_notes=False, enable_cloud_vision=False),
                cancellation=CancellationToken(),
            )
            with patch(
                "media_knowledge.ingestion.extractors.urllib.request.urlopen",
                return_value=FakeResponse(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
            ):
                with self.assertRaisesRegex(RuntimeError, "本次不入库"):
                    WeixinChannelsExtractor().extract(
                        "https://weixin.qq.com/sph/AciGNsUoaW",
                        context,
                    )

    def test_brand_and_renamed_product_state_migration(self) -> None:
        self.assertEqual(PRODUCT_NAME, "AI知识库-AI静静")
        self.assertEqual(PRODUCT_SLUG, "AI-Jingjing")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old-product"
            new = ProductPaths.resolve(root / "new-product")
            (old / "notes" / "Sources").mkdir(parents=True)
            (old / "notes" / "Sources" / "existing.md").write_text("existing", encoding="utf-8")
            (old / "settings.json").write_text('{"whisper_model":"tiny"}', encoding="utf-8")
            (old / "providers.json").write_text('{"providers":[]}', encoding="utf-8")
            self.assertEqual(new.migrate_renamed_product(old), old.resolve())
            self.assertEqual((new.notes / "Sources" / "existing.md").read_text(encoding="utf-8"), "existing")
            self.assertEqual(DesktopSettings.load(new.settings).whisper_model, "tiny")
            self.assertEqual(new.providers.stat().st_mode & 0o777, 0o600)

    def test_settings_and_provider_status_never_expose_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            settings = DesktopSettings(auto_synthesize_notes=False, obsidian_vault="/tmp/vault")
            settings.save(paths.settings)
            self.assertEqual(DesktopSettings.load(paths.settings).obsidian_vault, "/tmp/vault")
            providers = ProviderConfigStore(paths.providers)
            with patch.object(ProviderConfigStore, "_set_secret", return_value=False):
                providers.update("deepseek", api_key="secret-value")
            status = providers.status()
            self.assertTrue(status[0]["configured"])
            self.assertNotIn("secret-value", json.dumps(status))
            self.assertEqual(paths.providers.stat().st_mode & 0o777, 0o600)

    def test_quality_gate_rejects_metadata_or_cover_only_sources(self) -> None:
        extracted = ExtractionResult(
            title="Only cover",
            media_type="video",
            segments=[ContentSegment("cover", 1, "image", description="封面说明")],
            metadata={"content_scope": "cover_only"},
        )
        report = evaluate_extraction(extracted)
        self.assertFalse(report.accepted)
        self.assertTrue(any(check.name == "原始内容真实性" and check.status == "fail" for check in report.checks))

    def test_watched_folder_scan_is_incremental_and_reports_removed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "notes.md"
            source.write_text("first", encoding="utf-8")
            first = scan_folder(root)
            self.assertEqual(first.changed, [str(source.resolve())])
            second = scan_folder(root, first.current)
            self.assertEqual(second.changed, [])
            self.assertEqual(second.unchanged, 1)
            source.unlink()
            third = scan_folder(root, first.current)
            self.assertEqual(third.removed, ["notes.md"])

    def test_controller_management_backup_and_restore_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = DesktopController(root / "data", migrate_legacy=False)
            controller.save_settings(DesktopSettings(
                auto_synthesize_notes=False, enable_cloud_vision=False,
                embedding_provider="hash", embedding_model="hash-384-v1",
            ))
            source = root / "managed.md"
            source.write_text("# 管理测试\n\n这是一段可以检索和恢复的正文。", encoding="utf-8")
            result = controller.ingest([source]).results[0]
            controller.rename_document(str(result.document_id), "新标题")
            controller.update_document_facets(
                str(result.document_id), collections=["研发"], tags=["重要"]
            )
            document = controller.documents()[0]
            self.assertEqual(document["title"], "新标题")
            self.assertEqual(document["collections"], ["研发"])
            controller.set_document_enabled(str(result.document_id), False)
            self.assertEqual(controller.search("可以检索"), [])
            controller.set_document_enabled(str(result.document_id), True)
            backup = controller.create_backup()
            self.assertTrue(backup.is_file())
            controller.delete_document(str(result.document_id))
            self.assertEqual(controller.status()["documents"], 0)
            controller.restore_backup(backup)
            self.assertEqual(controller.status()["documents"], 1)
            self.assertTrue(controller.database_health()["ok"])

    def test_batch_ingestion_archives_indexes_notes_and_groups_related_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            slides = root / "课程.md"
            recording = root / "课程录音.txt"
            slides.write_text("# 课程\n\n有效应力是学习要点。", encoding="utf-8")
            recording.write_text("讲师说明了孔隙水压力与有效应力的关系。", encoding="utf-8")
            controller = DesktopController(data, migrate_legacy=False)
            controller.save_settings(
                DesktopSettings(
                    auto_synthesize_notes=False,
                    enable_cloud_vision=False,
                    embedding_provider="hash",
                    embedding_model="hash-384-v1",
                )
            )
            summary = controller.ingest([slides, recording])
            self.assertEqual(summary.succeeded, 2)
            self.assertEqual(summary.results[0].package_id, summary.results[1].package_id)
            self.assertEqual(controller.status()["documents"], 2)
            self.assertTrue(Path(summary.results[0].archive_path).is_dir())
            self.assertTrue(Path(summary.results[0].note_path).is_file())
            manifest = data / "archive" / "source-packages" / summary.results[0].package_id / "manifest.json"
            self.assertTrue(manifest.is_file())
            self.assertTrue(json.loads(manifest.read_text(encoding="utf-8"))["multimodal"])
            self.assertEqual(controller.search("有效应力")[0].title, "课程")

    def test_cancelled_token_stops_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.md"
            source.write_text("content", encoding="utf-8")
            paths = ProductPaths.resolve(root / "data")
            settings = DesktopSettings(auto_synthesize_notes=False, enable_cloud_vision=False)
            token = CancellationToken()
            token.cancel()
            summary = IngestionService(paths, settings=settings).ingest([source], cancellation=token)
            self.assertEqual(summary.cancelled, 1)
            self.assertEqual(summary.results[0].status, "cancelled")

    def test_ingestion_synthesis_is_pinned_to_deepseek_without_codex_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(Path(temporary) / "data")
            config = AppConfig(
                database_path=paths.database,
                qa_compatible_providers=(
                    CompatibleQAProviderConfig(
                        "deepseek",
                        "DeepSeek",
                        "https://api.deepseek.com",
                        "test-key",
                        ("deepseek-v4-pro", "deepseek-v4-flash"),
                    ),
                ),
            )
            service = IngestionService(paths, config=config)
            self.assertEqual(
                service._deepseek_synthesis_model_id(),
                "compatible::deepseek::deepseek-v4-flash",
            )

            unavailable = IngestionService(paths, config=AppConfig(database_path=paths.database))
            with self.assertRaisesRegex(ValueError, "DeepSeek"):
                unavailable._deepseek_synthesis_model_id()

    def test_legacy_database_migration_uses_online_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy.db"
            connection = sqlite3.connect(legacy)
            connection.execute("CREATE TABLE documents(id TEXT)")
            connection.execute("INSERT INTO documents VALUES ('doc-1')")
            connection.commit()
            connection.close()
            paths = ProductPaths.resolve(root / "product")
            selected = paths.migrate_legacy_database([legacy])
            self.assertEqual(selected, legacy.resolve())
            migrated = sqlite3.connect(paths.database)
            try:
                self.assertEqual(migrated.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 1)
            finally:
                migrated.close()


if __name__ == "__main__":
    unittest.main()
