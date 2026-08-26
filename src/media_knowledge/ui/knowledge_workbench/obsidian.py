from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode


class ObsidianAnswerWriter:
    def __init__(self, vault_root: str | Path | None, directory: str = "10_Knowledge/AI Answers"):
        self.vault_root = Path(vault_root).expanduser().resolve() if vault_root else None
        self.directory = PurePosixPath(directory)

    @property
    def available(self) -> bool:
        return bool(self.vault_root and self.vault_root.is_dir())

    def uri(self, note_path: str, section: str | None = None) -> str | None:
        if not self.vault_root:
            return None
        file_value = note_path.replace("\\", "/")
        if file_value.endswith(".md"):
            file_value = file_value[:-3]
        if section:
            file_value += "#" + section
        return "obsidian://open?" + urlencode({"vault": self.vault_root.name, "file": file_value})

    @staticmethod
    def _safe_title(value: str) -> str:
        compact = re.sub(r"\s+", " ", value).strip()
        compact = re.sub(r"[/:*?\"<>|\\]", "-", compact).strip(" .-")
        return compact[:72] or "知识问答"

    @staticmethod
    def _locator(citation: dict) -> str:
        parts = []
        if citation.get("page_number") is not None:
            parts.append(f"第 {citation['page_number']} 页")
        if citation.get("slide_number") is not None:
            parts.append(f"第 {citation['slide_number']} 张幻灯片")
        if citation.get("timestamp_start") is not None:
            start = citation["timestamp_start"]
            end = citation.get("timestamp_end")
            parts.append(f"{start:g}s" + (f"–{end:g}s" if end is not None else ""))
        if citation.get("section"):
            parts.append(str(citation["section"]))
        return ", ".join(parts)

    def save(self, answer: dict, *, title: str | None = None, tags: list[str] | None = None) -> dict:
        if not self.available or self.vault_root is None:
            raise ValueError("尚未配置 Obsidian 仓库，或仓库路径不存在")
        note_title = self._safe_title(title or answer["question"])
        note_tags = [self._safe_title(tag).replace(" ", "-") for tag in (tags or ["ai-answer"]) if tag.strip()]
        relative_directory = Path(*self.directory.parts)
        destination_directory = (self.vault_root / relative_directory).resolve()
        if self.vault_root not in destination_directory.parents and destination_directory != self.vault_root:
            raise ValueError("Obsidian 保存位置超出了已配置的仓库")
        destination_directory.mkdir(parents=True, exist_ok=True)
        destination = destination_directory / f"{note_title}--{answer['answer_id'][-8:]}.md"
        related = []
        citation_lines = []
        for citation in answer.get("citations", []):
            locator = self._locator(citation)
            source_path = citation.get("obsidian_path")
            source_link = f"[[{source_path[:-3] if source_path and source_path.endswith('.md') else source_path}]]" if source_path else citation["title"]
            citation_lines.append(
                f"- [{citation['citation_id']}] {source_link}"
                + (f" — {locator}" if locator else "")
            )
            related.append(citation["title"])
        related = list(dict.fromkeys(related))
        frontmatter = [
            "---",
            f"title: {json.dumps(note_title, ensure_ascii=False)}",
            f"created: {answer['created_at']}",
            f"source_answer_id: {answer['answer_id']}",
            f"conversation_id: {answer['conversation_id']}",
            "tags: " + json.dumps(note_tags, ensure_ascii=False),
            "---",
        ]
        body = "\n".join(
            [
                *frontmatter,
                "",
                f"# {note_title}",
                "",
                "## 问题",
                "",
                answer["question"],
                "",
                "## 回答",
                "",
                answer["markdown"],
                "",
                "## 引用",
                "",
                *(citation_lines or ["- 无引用"]),
                "",
                "## 相关文档",
                "",
                *([f"- {item}" for item in related] or ["- 无"]),
                "",
            ]
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination_directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        relative = destination.relative_to(self.vault_root).as_posix()
        return {
            "title": note_title,
            "path": relative,
            "absolute_path": str(destination),
            "obsidian_uri": self.uri(relative),
        }
