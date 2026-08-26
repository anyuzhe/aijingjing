from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from media_knowledge.embedding import HashEmbeddingProvider
from media_knowledge.indexing import IndexingService
from media_knowledge.storage import KnowledgeDatabase
from media_knowledge.sync import ObsidianMarkdownSync


class CountingEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dimensions=64, model="obsidian-sync-test")
        self.batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return super().embed(texts)


class ObsidianMarkdownSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.vault = self.root / "Vault"
        self.vault.mkdir()
        self.database = KnowledgeDatabase(self.root / "knowledge.db")
        self.embedding = CountingEmbeddingProvider()
        self.indexing = IndexingService(self.database, self.embedding)

    def tearDown(self) -> None:
        self.database.close()
        self.temp.cleanup()

    def sync(self):
        return ObsidianMarkdownSync(self.database, self.indexing, self.vault).sync()

    def test_incremental_sync_preserves_obsidian_path_tags_and_deletions(self) -> None:
        note = self.vault / "10_Knowledge" / "AI" / "FDE Skills.md"
        note.parent.mkdir(parents=True)
        note.write_text(
            '---\ntype: "knowledge"\ntags: ["ai-agent/fde", "skills"]\n---\n\n'
            "# FDE 的 Skill 资产\n\n高质量 Skill 来源于领域知识、失败经验和可验证工作流。",
            encoding="utf-8",
        )

        created = self.sync()
        self.assertEqual(created.created, 1)
        self.assertGreater(created.embedded_chunks, 0)
        row = self.database.connection.execute(
            "SELECT * FROM documents WHERE title = ?", ("FDE 的 Skill 资产",)
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["obsidian_path"], "10_Knowledge/AI/FDE Skills.md")
        tags = {
            item["name"]
            for item in self.database.connection.execute(
                """SELECT t.name FROM tags t
                   JOIN document_tags dt ON dt.tag_id=t.id WHERE dt.document_id=?""",
                (row["id"],),
            ).fetchall()
        }
        self.assertEqual(tags, {"ai-agent/fde", "skills"})

        self.embedding.batches.clear()
        unchanged = self.sync()
        self.assertEqual(unchanged.unchanged, 1)
        self.assertEqual(self.embedding.batches, [])

        note.write_text(note.read_text(encoding="utf-8") + "\n\n新增一条可复用知识。", encoding="utf-8")
        updated = self.sync()
        self.assertEqual(updated.updated, 1)
        self.assertGreater(updated.embedded_chunks, 0)

        note.unlink()
        deleted = self.sync()
        self.assertEqual(deleted.deleted, 1)
        self.assertEqual(self.database.status()["documents"], 0)

    def test_hidden_assets_and_saved_ai_answers_are_not_indexed(self) -> None:
        ignored = [
            self.vault / ".obsidian" / "internal.md",
            self.vault / "_assets" / "caption.md",
            self.vault / "10_Knowledge" / "AI Answers" / "generated.md",
        ]
        for path in ignored:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# 不应索引", encoding="utf-8")
        source = self.vault / "90_Sources" / "Document" / "Source.md"
        source.parent.mkdir(parents=True)
        source.write_text("# 应索引的来源\n\n真实资料。", encoding="utf-8")

        report = self.sync()
        self.assertEqual(report.scanned, 1)
        self.assertEqual(report.created, 1)
        titles = [row["title"] for row in self.database.connection.execute("SELECT title FROM documents")]
        self.assertEqual(titles, ["应索引的来源"])


if __name__ == "__main__":
    unittest.main()
