from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from media_knowledge.config import AppConfig
from media_knowledge.documents import document_from_text
from media_knowledge.embedding import HashEmbeddingProvider
from media_knowledge.indexing import IndexingService
from media_knowledge.storage import KnowledgeDatabase
from media_knowledge.ui.knowledge_workbench.server import WorkbenchHTTPServer
from media_knowledge.ui.knowledge_workbench.skills import SkillRunResult


class WorkbenchIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source_path = root / "sources" / "FAST-LIVO2.txt"
        self.source_path.parent.mkdir()
        self.source_path.write_text(
            "FAST-LIVO2 requires hardware synchronization between LiDAR and camera.", encoding="utf-8"
        )
        self.vault = root / "Obsidian Vault"
        self.vault.mkdir()
        self.database_path = root / "workbench.db"
        with KnowledgeDatabase(self.database_path) as database:
            IndexingService(database, HashEmbeddingProvider(dimensions=128, model="ui-test")).index_document(
                document_from_text(
                    self.source_path.read_text(encoding="utf-8"),
                    title="FAST-LIVO2 Synchronization",
                    source_id="ui-fast-livo2",
                    media_type="pdf",
                    local_path=str(self.source_path),
                    collections=["SLAM"],
                    tags=["hardware-sync"],
                )
            )
        config = AppConfig(
            database_path=self.database_path,
            embedding_dimensions=128,
            embedding_model="ui-test",
            obsidian_vault_root=self.vault,
        )
        self.server = WorkbenchHTTPServer(("127.0.0.1", 0), config)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def get(self, path: str, headers: dict[str, str] | None = None):
        return urllib.request.urlopen(urllib.request.Request(self.base + path, headers=headers or {}))

    def post(self, path: str, payload: dict):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(request)

    def test_static_workbench_and_bootstrap_use_real_database(self) -> None:
        html = self.get("/").read().decode("utf-8")
        self.assertIn("知识工作台", html)
        self.assertIn("搜索", html)
        self.assertIn("问 AI", html)
        bootstrap = json.load(self.get("/api/bootstrap"))
        self.assertEqual(bootstrap["stats"]["documents"], 1)
        self.assertEqual(bootstrap["collections"], [{"name": "SLAM", "count": 1}])
        self.assertEqual(bootstrap["tags"], [{"name": "hardware-sync", "count": 1}])
        self.assertTrue(bootstrap["capabilities"]["obsidian"])
        self.assertEqual(bootstrap["capabilities"]["answer_language"], "zh-CN")
        self.assertIn("local-extractive", [model["id"] for model in bootstrap["capabilities"]["models"]])
        self.assertEqual(bootstrap["capabilities"]["default_model"], "local-extractive")

    def test_search_respects_collection_folder_and_document_scope(self) -> None:
        payload = {
            "query": "FAST-LIVO2 synchronization",
            "filters": {
                "collections": ["SLAM"],
                "folders": [str(self.source_path.parent)],
            },
        }
        result = json.load(self.post("/api/search", payload))
        self.assertEqual(result["count"], 1)
        document_id = result["results"][0]["document_id"]
        payload["filters"]["document_ids"] = [document_id]
        self.assertEqual(json.load(self.post("/api/search", payload))["count"], 1)
        payload["filters"]["collections"] = ["Different"]
        self.assertEqual(json.load(self.post("/api/search", payload))["count"], 0)

    def test_search_does_not_require_a_qa_provider(self) -> None:
        self.server.config.qa_provider = "openai-compatible"
        result = json.load(
            self.post("/api/search", {"query": "FAST-LIVO2 synchronization", "filters": {}})
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["title"], "FAST-LIVO2 Synchronization")

    def test_streamed_answer_conversation_and_obsidian_save(self) -> None:
        response = self.post(
            "/api/ask/stream",
            {"question": "Does FAST-LIVO2 require synchronization?", "filters": {"collections": ["SLAM"]}},
        )
        events = [json.loads(line) for line in response.read().decode("utf-8").splitlines()]
        self.assertIn("delta", {event["type"] for event in events})
        statuses = [event for event in events if event["type"] == "status"]
        self.assertEqual([event["stage"] for event in statuses], ["retrieving", "answering", "finalizing"])
        self.assertIn("正在生成中文回答", statuses[1]["message"])
        answer = next(event["answer"] for event in events if event["type"] == "final")
        self.assertTrue(answer["citations"])
        self.assertEqual(answer["retrieval_info"]["response_language"], "zh-CN")
        self.assertEqual(answer["citations"][0]["title"], "FAST-LIVO2 Synchronization")
        conversation = json.load(self.get(f"/api/conversations/{answer['conversation_id']}"))
        self.assertEqual(len(conversation["messages"]), 2)
        self.assertEqual(conversation["answers"][0]["answer_id"], answer["answer_id"])

        saved = json.load(self.post("/api/obsidian/save", {"answer_id": answer["answer_id"], "tags": ["SLAM"]}))
        note = self.vault / saved["path"]
        self.assertTrue(note.is_file())
        content = note.read_text(encoding="utf-8")
        self.assertIn("## 问题", content)
        self.assertIn("## 回答", content)
        self.assertIn("## 引用", content)
        self.assertIn("FAST-LIVO2 Synchronization", content)
        self.assertTrue(saved["obsidian_uri"].startswith("obsidian://open?"))

    def test_source_endpoint_supports_byte_ranges(self) -> None:
        result = json.load(
            self.post("/api/search", {"query": "FAST-LIVO2 synchronization", "filters": {}})
        )["results"][0]
        response = self.get(
            f"/api/source/content?chunk_id={result['chunk_id']}", headers={"Range": "bytes=0-9"}
        )
        self.assertEqual(response.status, 206)
        self.assertEqual(response.headers["Accept-Ranges"], "bytes")
        self.assertEqual(response.read(), b"FAST-LIVO2")

    def test_manual_obsidian_sync_indexes_markdown_and_exposes_open_path(self) -> None:
        note = self.vault / "10_Knowledge" / "AI" / "FDE.md"
        note.parent.mkdir(parents=True)
        note.write_text("# FDE 实践\n\nSkill 是把一线经验沉淀成可复用工作流。", encoding="utf-8")

        result = json.load(self.post("/api/obsidian/sync", {}))
        self.assertEqual(result["created"], 1)
        bootstrap = json.load(self.get("/api/bootstrap"))
        self.assertEqual(bootstrap["stats"]["documents"], 2)
        self.assertEqual(bootstrap["capabilities"]["obsidian_sync"]["last_result"]["created"], 1)
        search = json.load(self.post("/api/search", {"query": "FDE Skill 一线经验", "filters": {}}))
        synced = next(item for item in search["results"] if item["title"] == "FDE 实践")
        self.assertEqual(synced["source"]["obsidian_path"], "10_Knowledge/AI/FDE.md")

    @mock.patch("media_knowledge.ui.knowledge_workbench.server.KnowledgeIngestorBridge.run")
    def test_skill_result_streams_and_is_saved_in_the_conversation(self, run_skill) -> None:
        def finish_skill(*_args, **_kwargs):
            note = self.vault / "90_Sources" / "Document" / "Skill Output.md"
            note.parent.mkdir(parents=True)
            note.write_text("# Skill 输出\n\n已沉淀到 Obsidian。", encoding="utf-8")
            return SkillRunResult(
                "knowledge-ingestor",
                "## 入库完成\n\n已创建来源笔记。",
                [str(self.source_path)],
            )

        run_skill.side_effect = finish_skill
        response = self.post(
            "/api/skills/invoke/stream",
            {
                "skill": "knowledge-ingestor",
                "instruction": "整理入库这份资料",
                "sources": [str(self.source_path)],
            },
        )
        events = [json.loads(line) for line in response.read().decode("utf-8").splitlines()]
        result = next(event["result"] for event in events if event["type"] == "final")
        self.assertEqual(result["skill"], "knowledge-ingestor")
        self.assertEqual(result["sync"]["created"], 1)
        self.assertIn("syncing", [event.get("stage") for event in events if event["type"] == "status"])
        conversation = json.load(self.get(f"/api/conversations/{result['conversation_id']}"))
        self.assertEqual(len(conversation["messages"]), 2)
        self.assertEqual(conversation["messages"][0]["metadata"]["skill"], "knowledge-ingestor")
        self.assertIn("入库完成", conversation["messages"][1]["content"])
        self.assertEqual(conversation["messages"][1]["metadata"]["sync"]["created"], 1)


if __name__ == "__main__":
    unittest.main()
