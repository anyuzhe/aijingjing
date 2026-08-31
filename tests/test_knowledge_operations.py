from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from media_knowledge.documents import document_from_text
from media_knowledge.embedding import HashEmbeddingProvider
from media_knowledge.indexing import IndexingService
from media_knowledge.storage import (
    KnowledgeDatabase,
    KnowledgeGovernanceRepository,
    KnowledgeOperationsRepository,
)
from media_knowledge.wiki import PortableWikiCompiler


class KnowledgeOperationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = KnowledgeDatabase(self.root / "knowledge.db")
        self.governance = KnowledgeGovernanceRepository(self.database)
        self.operations = KnowledgeOperationsRepository(self.database)
        document = document_from_text(
            "FDE 使用可追溯的知识卡片管理行业资料。",
            title="FDE Industry Notes",
            source_id="source-fde",
            media_type="markdown",
            local_path=str(self.root / "fde.md"),
            collections=["FDE"],
            tags=["fde"],
        )
        IndexingService(
            self.database, HashEmbeddingProvider(dimensions=384)
        ).index_document(document)
        self.document_id = document.document_id

    def tearDown(self) -> None:
        self.database.close()
        self.temporary.cleanup()

    def test_policy_is_allowlisted_and_source_assessment_is_auditable(self) -> None:
        policy = self.operations.upsert_policy(
            "FDE", "FDE 研究",
            {
                "auto_propose": True,
                "external_verification": True,
                "conflict_policy": "require-review",
                "default_source_reliability": "medium",
                "require_review": True,
                "arbitrary_shell": "rm -rf /",
            },
        )
        self.assertNotIn("arbitrary_shell", policy.policy)
        self.assertEqual(policy.policy["conflict_policy"], "require-review")
        assessment = self.operations.upsert_source_assessment(
            self.document_id,
            source_class="industry",
            reliability="medium",
            extraction_completeness=0.94,
            valid_until="2099-12-31",
            notes="行业白皮书，核心结论需与官方数据交叉核验",
            checked=True,
        )
        self.assertEqual(assessment.reliability, "medium")
        self.assertNotIn(
            self.document_id,
            {item.get("document_id") for item in self.operations.source_quality_issues()},
        )

    def test_candidate_is_deduplicated_reviewed_and_linked_to_source(self) -> None:
        first = self.operations.create_proposal(
            title="FDE 行业知识",
            body="FDE 资料需要保留来源并经人工复核。",
            source_document_id=self.document_id,
            correction_run_id=None,
            source_segment_ids=["seg-1"],
            tags=["fde", "知识候选"],
        )
        same = self.operations.create_proposal(
            title="FDE 行业知识",
            body="FDE 资料需要保留来源并经人工复核。",
            source_document_id=self.document_id,
            source_segment_ids=["seg-1"],
            tags=["fde"],
        )
        self.assertEqual(first.id, same.id)
        item = self.operations.accept_proposal(first.id)
        self.assertEqual((item.status, item.maturity), ("needs-review", "summarized"))
        source = self.governance.get_item_for_document(self.document_id)
        assert source is not None
        relations = self.governance.list_relations(source.item_id, direction="outgoing")
        self.assertTrue(any(relation.target_item_id == item.item_id for relation in relations))
        self.assertEqual(self.operations.get_proposal(first.id).status, "accepted")
        self.assertTrue(self.operations.list_events())

    def test_duplicate_candidate_requires_explicit_merge_choice(self) -> None:
        existing = self.governance.create_item(
            item_type="topic", title="FDE", aliases=["飞控数据引擎"], body="旧内容"
        )
        proposal = self.operations.create_proposal(
            title="飞控数据引擎", body="新证据内容", source_document_id=self.document_id
        )
        self.assertEqual(proposal.duplicate_item_id, existing.item_id)
        merged = self.operations.accept_proposal(proposal.id, merge_duplicate=True)
        self.assertEqual(merged.item_id, existing.item_id)
        self.assertIn("新证据内容", merged.body)
        self.assertEqual(self.operations.get_proposal(proposal.id).status, "merged")

    def test_sop_and_portable_wiki_are_structured_and_linked(self) -> None:
        workflow = self.operations.upsert_workflow(
            name="课程录音深度精校",
            description="从原始音频到可审核知识",
            trigger={"media_type": "audio"},
            steps=["本地 ASR", "说话人分离", "候选知识人工复核"],
            model_policy={"asr": "large-v3"},
            privacy={"boundary": "原始音频不离开本机"},
        )
        self.assertEqual(len(workflow.steps), 3)
        topic = self.governance.create_item(
            item_type="topic",
            title="FDE 知识治理",
            status="current",
            maturity="compiled",
            body="知识必须保留证据关系。",
            tags=["fde", "knowledge-governance"],
        )
        source = self.governance.get_item_for_document(self.document_id)
        assert source is not None
        self.governance.create_relation(source.item_id, topic.item_id, "supports")
        result = PortableWikiCompiler(
            self.database, self.root / "notes" / "LLM-Wiki"
        ).compile()
        self.assertEqual(result.item_count, 2)
        self.assertTrue((result.root / "wiki" / "index.md").is_file())
        self.assertTrue((result.root / "wiki" / "indexes" / "tag-index.md").is_file())
        item_files = list((result.root / "wiki" / "topics").glob("*.md"))
        self.assertEqual(len(item_files), 1)
        self.assertIn("supports", item_files[0].read_text(encoding="utf-8"))
        self.assertFalse(any("断链" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
