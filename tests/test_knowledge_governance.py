from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from media_knowledge.documents import document_from_text
from media_knowledge.embedding import HashEmbeddingProvider
from media_knowledge.indexing import IndexingService
from media_knowledge.storage import (
    KNOWLEDGE_ITEM_TYPES,
    KNOWLEDGE_MATURITIES,
    KNOWLEDGE_RELATION_TYPES,
    KNOWLEDGE_STATUSES,
    KnowledgeDatabase,
    KnowledgeGovernanceRepository,
)


class KnowledgeGovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "knowledge.db"
        self.database = KnowledgeDatabase(self.path)
        self.repository = KnowledgeGovernanceRepository(self.database)

    def tearDown(self) -> None:
        self.database.close()
        self.temporary.cleanup()

    def test_schema_migration_and_status_expose_governance_tables(self) -> None:
        versions = {
            int(row["version"])
            for row in self.database.connection.execute(
                "SELECT version FROM schema_migrations"
            ).fetchall()
        }
        self.assertIn(11, versions)
        status = self.database.status()
        self.assertEqual(status["schema_version"], 14)
        for table in (
            "knowledge_items",
            "knowledge_aliases",
            "knowledge_item_tags",
            "knowledge_relations",
            "knowledge_proposals",
            "source_assessments",
            "workflow_templates",
            "knowledge_events",
        ):
            self.assertEqual(status[table], 0)
        self.assertEqual(status["knowledge_space_policies"], 1)

    def test_item_crud_accepts_formal_taxonomy_and_validates_links(self) -> None:
        created = []
        for item_type in sorted(KNOWLEDGE_ITEM_TYPES):
            created.append(
                self.repository.create_item(
                    item_type=item_type,
                    title=f"{item_type} 条目",
                    metadata={"owner": "test"},
                )
            )
        self.assertEqual(
            {item.item_type for item in self.repository.list_items(limit=100)},
            KNOWLEDGE_ITEM_TYPES,
        )
        item = created[0]
        for status in sorted(KNOWLEDGE_STATUSES):
            item = self.repository.update_item(item.id, status=status)
            self.assertEqual(item.status, status)
        for maturity in sorted(KNOWLEDGE_MATURITIES):
            item = self.repository.update_item(item.id, maturity=maturity)
            self.assertEqual(item.maturity, maturity)
        item = self.repository.update_item(
            item.id,
            title="更新后的标题",
            summary="摘要",
            body="正文",
            aliases=["First Alias", "first   alias", "第二别名"],
            tags=["governance", "治理"],
            high_value=True,
            metadata={"owner": "jingjing"},
        )
        self.assertEqual(item.title, "更新后的标题")
        self.assertEqual(item.aliases, ("First Alias", "第二别名"))
        self.assertTrue(item.high_value)
        self.assertEqual(item.to_dict()["id"], item.id)
        self.assertTrue(self.repository.delete_item(item.id))
        self.assertFalse(self.repository.delete_item(item.id))
        with self.assertRaises(ValueError):
            self.repository.create_item(item_type="note", title="非法")
        with self.assertRaises(ValueError):
            self.repository.create_item(item_type="topic", title=" ")
        with self.assertRaises(ValueError):
            self.repository.create_item(
                item_type="topic", title="非法关联", document_id="missing"
            )
        with self.assertRaises(ValueError):
            self.repository.update_item(created[1].id, item_type="invalid")

    def test_alias_tag_search_filters_and_unicode_ranking(self) -> None:
        target = self.repository.create_item(
            item_type="entity",
            title="FDE Industry Notes",
            body="飞控数据引擎的产业资料",
            aliases=["飞控数据引擎", "FDE"],
            tags=["fde", "行业研究"],
            status="current",
            maturity="summarized",
        )
        other = self.repository.create_item(
            item_type="topic",
            title="FDE 扩展阅读",
            body="只是扩展内容",
            tags=["fde"],
        )
        self.assertEqual(self.repository.search("飞控数据")[0].id, target.id)
        self.assertEqual(self.repository.search("fde")[0].id, target.id)
        self.assertEqual(
            {item.id for item in self.repository.list_items(tags=["FDE"])},
            {target.id, other.id},
        )
        self.assertEqual(
            [item.id for item in self.repository.search("产业", item_types=["entity"])],
            [target.id],
        )
        self.assertTrue(self.repository.add_alias(target.id, "Flight Data Engine"))
        self.assertFalse(self.repository.add_alias(target.id, "flight   data engine"))
        self.assertTrue(self.repository.remove_alias(target.id, "FLIGHT DATA ENGINE"))
        self.assertTrue(self.repository.add_tag(target.id, "knowledge-graph"))
        self.assertFalse(self.repository.add_tag(target.id, "KNOWLEDGE-GRAPH"))
        self.assertTrue(self.repository.remove_tag(target.id, "Knowledge-Graph"))

    def test_relation_crud_direction_queries_and_cascade(self) -> None:
        source = self.repository.create_item(item_type="source", title="原始报告")
        targets = [
            self.repository.create_item(item_type="analysis", title=f"分析 {index}")
            for index in range(len(KNOWLEDGE_RELATION_TYPES))
        ]
        relations = []
        for relation_type, target in zip(sorted(KNOWLEDGE_RELATION_TYPES), targets):
            relations.append(
                self.repository.create_relation(
                    source.id,
                    target.id,
                    relation_type,
                    summary=f"{relation_type} 关系",
                )
            )
        outgoing = self.repository.list_relations(source.id, direction="outgoing")
        self.assertEqual({relation.relation_type for relation in outgoing}, KNOWLEDGE_RELATION_TYPES)
        incoming = self.repository.related_items(targets[0].id, direction="incoming")
        self.assertEqual(incoming[0].item.id, source.id)
        self.assertEqual(incoming[0].direction, "incoming")
        updated = self.repository.update_relation(
            relations[0].id, summary="新摘要", metadata={"reviewed": True}
        )
        self.assertEqual(updated.summary, "新摘要")
        self.assertTrue(updated.metadata["reviewed"])
        with self.assertRaises(ValueError):
            self.repository.create_relation(source.id, source.id, "supports")
        with self.assertRaises(ValueError):
            self.repository.create_relation(
                source.id, targets[0].id, relations[0].relation_type
            )
        with self.assertRaises(ValueError):
            self.repository.list_relations(direction="sideways")
        self.assertTrue(self.repository.delete_relation(relations[0].id))
        self.assertFalse(self.repository.delete_relation(relations[0].id))
        self.assertTrue(self.repository.delete_item(source.id))
        self.assertEqual(self.repository.list_relations(), [])
        self.assertIsNotNone(self.repository.get_item(targets[0].id))

    def test_item_snapshot_restores_identity_facets_and_live_relations(self) -> None:
        source = self.repository.create_item(
            item_id="kg-restore-source",
            item_type="analysis",
            title="可恢复分析",
            status="current",
            maturity="compiled",
            summary="完整快照",
            body="应恢复原正文",
            aliases=["Restore Alias", "恢复别名"],
            tags=["restore", "知识治理"],
            high_value=True,
            metadata={"note_relative_path": "正式知识/analysis/restore.md"},
        )
        target = self.repository.create_item(item_type="topic", title="仍存在的端点")
        relation = self.repository.create_relation(
            source.id,
            target.id,
            "supports",
            summary="恢复关系",
            metadata={"origin": "test"},
        )
        snapshot = self.repository.snapshot_item(source.id)
        self.assertEqual(snapshot["item"]["id"], source.id)
        self.assertEqual(
            {row["alias"] for row in snapshot["aliases"]},
            {"Restore Alias", "恢复别名"},
        )
        self.assertEqual(snapshot["relations"][0]["id"], relation.id)

        self.assertTrue(self.repository.delete_item(source.id))
        restored = self.repository.restore_item_snapshot(snapshot)

        self.assertEqual(restored.item.id, source.id)
        self.assertEqual(restored.item.created_at, source.created_at)
        self.assertEqual(restored.item.aliases, source.aliases)
        self.assertEqual(restored.item.tags, source.tags)
        self.assertTrue(restored.item.high_value)
        self.assertEqual([value.id for value in restored.restored_relations], [relation.id])
        self.assertEqual(restored.skipped_relation_ids, ())

    def test_item_snapshot_skips_relation_when_other_endpoint_was_deleted(self) -> None:
        source = self.repository.create_item(item_type="analysis", title="待恢复")
        target = self.repository.create_item(item_type="topic", title="随后删除")
        relation = self.repository.create_relation(source.id, target.id, "extends")
        snapshot = self.repository.snapshot_item(source.id)
        self.assertTrue(self.repository.delete_item(source.id))
        self.assertTrue(self.repository.delete_item(target.id))

        restored = self.repository.restore_item_snapshot(snapshot)

        self.assertEqual(restored.item.id, source.id)
        self.assertEqual(restored.restored_relations, ())
        self.assertEqual(restored.skipped_relation_ids, (relation.id,))
        self.assertEqual(self.repository.list_relations(source.id), [])

    def test_documents_and_artifacts_are_automatically_governed(self) -> None:
        indexing = IndexingService(
            self.database, HashEmbeddingProvider(dimensions=32, model="governance-test")
        )
        report = indexing.index_document(
            document_from_text(
                "可追溯来源正文", title="原始资料", source_id="governance-source"
            )
        )
        source = self.repository.get_item_for_document(report.document_id)
        self.assertIsNotNone(source)
        assert source is not None
        self.assertEqual((source.item_type, source.status, source.maturity), ("source", "current", "indexed"))
        self.assertTrue(self.database.rename_document(report.document_id, "重命名资料"))
        self.assertEqual(self.repository.get_item(source.id).title, "重命名资料")  # type: ignore[union-attr]

        self.database.save_artifact(
            "artifact-1",
            "source-note",
            "知识产物",
            "第一版正文",
            [report.document_id],
        )
        output = self.repository.get_item_for_artifact("artifact-1")
        self.assertIsNotNone(output)
        assert output is not None
        self.assertEqual((output.item_type, output.maturity), ("output", "compiled"))
        self.assertEqual(output.body, "第一版正文")
        relation = self.repository.list_relations(output.id, direction="incoming")
        self.assertEqual([(item.source_item_id, item.relation_type) for item in relation], [(source.id, "supports")])

        self.database.save_artifact(
            "artifact-1", "source-note", "知识产物二版", "第二版正文", []
        )
        output = self.repository.get_item_for_artifact("artifact-1")
        assert output is not None
        self.assertEqual((output.title, output.body), ("知识产物二版", "第二版正文"))
        self.assertEqual(self.repository.list_relations(output.id), [])

        self.assertTrue(self.repository.delete_item(source.id))
        self.assertIsNotNone(self.database.get_document(report.document_id))

    def test_version_11_backfills_existing_documents_artifacts_and_support_edges(self) -> None:
        indexing = IndexingService(
            self.database, HashEmbeddingProvider(dimensions=32, model="migration-test")
        )
        report = indexing.index_document(
            document_from_text("迁移正文", title="迁移资料", source_id="legacy-source")
        )
        self.database.save_artifact(
            "legacy-artifact", "note", "旧知识产物", "产物正文", [report.document_id]
        )
        self.database.close()
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys=OFF")
        for table in (
            "knowledge_relations",
            "knowledge_item_tags",
            "knowledge_aliases",
            "knowledge_items",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.execute("DELETE FROM schema_migrations WHERE version=11")
        connection.commit()
        connection.close()

        self.database = KnowledgeDatabase(self.path)
        self.repository = KnowledgeGovernanceRepository(self.database)
        source = self.repository.get_item_for_document(report.document_id)
        output = self.repository.get_item_for_artifact("legacy-artifact")
        self.assertIsNotNone(source)
        self.assertIsNotNone(output)
        assert source is not None and output is not None
        relations = self.repository.list_relations(output.id, direction="incoming")
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].source_item_id, source.id)
        self.assertEqual(relations[0].relation_type, "supports")

    def test_health_report_detects_actionable_governance_problems(self) -> None:
        source = self.repository.create_item(
            item_type="source",
            title="高价值旧来源",
            status="current",
            maturity="indexed",
            high_value=True,
            metadata={"url": "https://example.test/source"},
        )
        self.repository.create_item(item_type="source", title="无证据来源")
        compiled = self.repository.create_item(
            item_type="analysis",
            title="无来源编译结果",
            status="current",
            maturity="compiled",
            tags=["Bad Tag"],
        )
        stale = self.repository.create_item(
            item_type="decision", title="明确过期决策", status="stale", body="旧决策"
        )
        first = self.repository.create_item(
            item_type="entity", title="实体 A", aliases=["公共别名"], body="A"
        )
        second = self.repository.create_item(
            item_type="entity", title="实体 B", aliases=["公共别名"], body="B"
        )
        self.database.connection.execute(
            "UPDATE knowledge_items SET updated_at='2020-01-01T00:00:00Z' WHERE id=?",
            (source.id,),
        )
        self.database.connection.commit()
        report = self.repository.health_report(
            stale_after_days=30,
            as_of=datetime(2026, 8, 29, tzinfo=UTC),
        )
        codes = {issue.code for issue in report.issues}
        self.assertTrue(
            {
                "stale_current",
                "high_value_uncompiled",
                "source_without_evidence",
                "orphan_item",
                "missing_summary",
                "missing_body",
                "compiled_without_source",
                "marked_stale",
                "noncanonical_tag",
                "ambiguous_alias",
            }.issubset(codes)
        )
        self.assertEqual(report.total_items, 6)
        self.assertEqual(report.counts["by_type"]["entity"], 2)
        self.assertEqual(report.to_dict()["issues"][0]["code"], report.issues[0].code)
        self.assertFalse(report.healthy)
        self.assertEqual(compiled.tags, ("Bad Tag",))
        self.assertEqual(stale.status, "stale")
        self.assertNotEqual(first.id, second.id)
        with self.assertRaises(ValueError):
            self.repository.health_report(stale_after_days=0)

    def test_health_report_accepts_transitive_source_provenance(self) -> None:
        source = self.repository.create_item(
            item_type="source",
            title="原始资料",
            maturity="compiled",
            metadata={"url": "https://example.test/evidence"},
        )
        topic = self.repository.create_item(
            item_type="topic",
            title="来源主题",
            summary="主题摘要",
            body="主题正文",
            metadata={"owner": "test"},
        )
        analysis = self.repository.create_item(
            item_type="analysis",
            title="二级分析",
            maturity="compiled",
            summary="分析摘要",
            body="分析正文",
            metadata={"owner": "test"},
        )
        self.repository.create_relation(source.id, topic.id, "supports")
        self.repository.create_relation(topic.id, analysis.id, "extends")
        issues = self.repository.health_report().issues
        self.assertFalse(
            any(
                issue.code == "compiled_without_source" and issue.item_id == analysis.id
                for issue in issues
            )
        )


if __name__ == "__main__":
    unittest.main()
