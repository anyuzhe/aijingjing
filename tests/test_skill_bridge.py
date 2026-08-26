from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from media_knowledge.ui.knowledge_workbench.skills import KnowledgeIngestorBridge


class KnowledgeIngestorBridgeTests(unittest.TestCase):
    def test_bridge_invokes_only_the_installed_skill_with_validated_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skill_root = root / "knowledge-ingestor"
            skill_root.mkdir()
            (skill_root / "SKILL.md").write_text("# test skill", encoding="utf-8")
            archive = root / "archive"
            vault = root / "vault"
            archive.mkdir()
            vault.mkdir()
            (skill_root / "config.yaml").write_text(
                f'archive:\n  root: "{archive}"\nobsidian:\n  vault_root: "{vault}"\n',
                encoding="utf-8",
            )
            source = root / "课程资料.txt"
            source.write_text("知识内容", encoding="utf-8")
            fake_codex = root / "codex"
            fake_codex.write_text(
                """#!/usr/bin/env python3
import sys
from pathlib import Path
if "--ask-for-approval" in sys.argv or "--approve-for-me" not in sys.argv or "model_providers.openai-http.supports_websockets=false" not in sys.argv:
    print("error: invalid approval option", file=sys.stderr)
    print("For more information, try '--help'.", file=sys.stderr)
    raise SystemExit(2)
prompt = sys.stdin.read()
if "$knowledge-ingestor" not in prompt:
    raise SystemExit(2)
output = Path(sys.argv[sys.argv.index("--output-last-message") + 1])
output.write_text("## 已完成\\n\\nSkill 成功", encoding="utf-8")
""",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)

            bridge = KnowledgeIngestorBridge(
                workspace_root=root,
                skill_root=skill_root,
                codex_executable=str(fake_codex),
            )
            result = bridge.run("整理入库这些课程资料", [str(source)])

            self.assertTrue(bridge.available)
            self.assertEqual(result.skill, "knowledge-ingestor")
            self.assertEqual(result.sources, [str(source.resolve())])
            self.assertIn("Skill 成功", result.markdown)

    def test_bridge_rejects_missing_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "所选文件不存在"):
            KnowledgeIngestorBridge.validate_sources(["/definitely/missing/source.pdf"])


if __name__ == "__main__":
    unittest.main()
