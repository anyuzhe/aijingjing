from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..models import utcnow_iso
from ..storage import KnowledgeDatabase, KnowledgeGovernanceRepository, KnowledgeOperationsRepository
from ..storage.governance import KnowledgeItem, KnowledgeRelation


_TYPE_FOLDERS = {
    "source": "sources",
    "topic": "topics",
    "entity": "entities",
    "analysis": "analyses",
    "decision": "decisions",
    "output": "outputs",
}
_SAFE_RE = re.compile(r"[^\w\-\u3400-\u9fff]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class WikiCompileResult:
    root: Path
    written_files: tuple[Path, ...]
    removed_files: tuple[Path, ...]
    item_count: int
    relation_count: int
    warnings: tuple[str, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "written_files": [str(path) for path in self.written_files],
            "removed_files": [str(path) for path in self.removed_files],
            "item_count": self.item_count,
            "relation_count": self.relation_count,
            "warnings": list(self.warnings),
            "manifest_sha256": self.manifest_sha256,
        }


def _slug(value: object, fallback: str = "knowledge") -> str:
    clean = _SAFE_RE.sub("-", str(value or "").strip()).strip("-._")[:72]
    return clean or fallback


def _yaml(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class PortableWikiCompiler:
    """Compile governed SQLite facts into a portable, linked Markdown mirror."""

    MANIFEST = ".ai-jingjing-wiki-manifest.json"

    def __init__(self, database: KnowledgeDatabase, root: str | Path):
        self.database = database
        self.root = Path(root).expanduser().resolve(strict=False)

    def compile(self) -> WikiCompileResult:
        repository = KnowledgeGovernanceRepository(self.database)
        operations = KnowledgeOperationsRepository(self.database)
        items = repository.list_items(limit=1000)
        relations = repository.list_relations(limit=1000)
        events = operations.list_events(limit=200)
        generated_at = utcnow_iso()
        item_paths = {item.item_id: self._item_path(item) for item in items}
        payloads: dict[Path, str] = {}
        for item in items:
            payloads[item_paths[item.item_id]] = self._render_item(
                item,
                item_paths,
                [
                    relation
                    for relation in relations
                    if item.item_id in {relation.source_item_id, relation.target_item_id}
                ],
                generated_at,
            )
        payloads[self.root / "wiki" / "index.md"] = self._render_index(items, item_paths, generated_at)
        payloads[self.root / "wiki" / "overview.md"] = self._render_overview(items, relations, generated_at)
        payloads[self.root / "wiki" / "log.md"] = self._render_log(events, generated_at)
        payloads[self.root / "index" / "home.md"] = self._render_home(items, generated_at)
        payloads[self.root / "wiki" / "indexes" / "wiki-catalog.md"] = self._render_catalog(items, item_paths, generated_at)
        payloads[self.root / "wiki" / "indexes" / "tag-index.md"] = self._render_tag_index(items, item_paths, generated_at)
        payloads[self.root / "wiki" / "indexes" / "raw-status.md"] = self._render_status(
            [item for item in items if item.item_type == "source"], item_paths, "原始资料状态", generated_at
        )
        payloads[self.root / "wiki" / "indexes" / "output-status.md"] = self._render_status(
            [item for item in items if item.item_type == "output"], item_paths, "成果状态", generated_at
        )
        manifest_path = self.root / self.MANIFEST
        previous = self._previous_files(manifest_path)
        current = {path.relative_to(self.root).as_posix() for path in payloads}
        removed: list[Path] = []
        for relative in sorted(previous - current):
            candidate = (self.root / relative).resolve(strict=False)
            try:
                candidate.relative_to(self.root)
            except ValueError:
                continue
            if candidate.is_file() and candidate.suffix.casefold() == ".md":
                candidate.unlink()
                removed.append(candidate)
        for path, text in sorted(payloads.items(), key=lambda pair: pair[0].as_posix()):
            _atomic_text(path, text)
        manifest = {
            "format": "ai-jingjing-portable-wiki-v1",
            "generated_at": generated_at,
            "files": sorted(current),
            "item_count": len(items),
            "relation_count": len(relations),
        }
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_text(manifest_path, manifest_text)
        warnings = tuple(self._lint(payloads, item_paths))
        return WikiCompileResult(
            root=self.root,
            written_files=tuple(sorted(payloads, key=lambda path: path.as_posix())),
            removed_files=tuple(removed),
            item_count=len(items),
            relation_count=len(relations),
            warnings=warnings,
            manifest_sha256=hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        )

    def _item_path(self, item: KnowledgeItem) -> Path:
        folder = _TYPE_FOLDERS.get(item.item_type, "items")
        suffix = hashlib.sha256(item.item_id.encode("utf-8")).hexdigest()[:8]
        return self.root / "wiki" / folder / f"{_slug(item.title)}--{suffix}.md"

    def _wikilink(self, path: Path, title: str) -> str:
        relative = path.relative_to(self.root).with_suffix("").as_posix()
        return f"[[{relative}|{title}]]"

    def _frontmatter(self, item: KnowledgeItem, generated_at: str) -> list[str]:
        return [
            "---",
            f"id: {_yaml(item.item_id)}",
            f"title: {_yaml(item.title)}",
            f"type: {_yaml(item.item_type)}",
            f"status: {_yaml(item.status)}",
            f"maturity: {_yaml(item.maturity)}",
            f"high_value: {'true' if item.high_value else 'false'}",
            f"aliases: {_yaml(list(item.aliases))}",
            f"tags: {_yaml(list(item.tags))}",
            f"updated: {_yaml(item.updated_at)}",
            f"compiled_at: {_yaml(generated_at)}",
            "generated_by: AI静静",
            "---",
            "",
        ]

    def _render_item(
        self,
        item: KnowledgeItem,
        paths: dict[str, Path],
        relations: Iterable[KnowledgeRelation],
        generated_at: str,
    ) -> str:
        lines = [*self._frontmatter(item, generated_at), f"# {item.title}", ""]
        if item.summary:
            lines.extend(["> " + item.summary.replace("\n", "\n> "), ""])
        if item.body:
            lines.extend([item.body, ""])
        if item.document_id:
            row = self.database.get_document(item.document_id)
            if row is not None:
                source = row["original_uri"] or row["local_path"] or "本地资料"
                lines.extend(["## 来源", "", f"- 资料 ID：`{item.document_id}`", f"- 原始位置：{source}", ""])
        linked: list[str] = []
        for relation in relations:
            outgoing = relation.source_item_id == item.item_id
            other_id = relation.target_item_id if outgoing else relation.source_item_id
            if other_id not in paths:
                continue
            other = KnowledgeGovernanceRepository(self.database).get_item(other_id)
            if other is None:
                continue
            arrow = "→" if outgoing else "←"
            detail = f" — {relation.summary}" if relation.summary else ""
            linked.append(
                f"- `{arrow} {relation.relation_type}` {self._wikilink(paths[other_id], other.title)}{detail}"
            )
        lines.extend(["## 关系", "", *(linked or ["- 暂无关系"]), ""])
        lines.extend(["## 治理", "", f"- 状态：`{item.status}`", f"- 成熟度：`{item.maturity}`", f"- 最后更新：`{item.updated_at}`", ""])
        return "\n".join(lines)

    def _render_index(self, items: list[KnowledgeItem], paths: dict[str, Path], generated_at: str) -> str:
        lines = ["---", 'title: "LLM Wiki"', 'type: "index"', f"updated: {_yaml(generated_at)}", "generated_by: AI静静", "---", "", "# LLM Wiki", "", "这是 AI静静从 SQLite 治理事实自动编译的便携 Markdown 镜像。请在应用中修改知识，随后重新编译；不要直接把本目录当唯一事实源。", ""]
        for item_type, folder in _TYPE_FOLDERS.items():
            selected = [item for item in items if item.item_type == item_type]
            lines.extend([f"## {folder}", ""])
            lines.extend(
                f"- {self._wikilink(paths[item.item_id], item.title)} · `{item.status}` / `{item.maturity}`"
                for item in selected
            )
            if not selected:
                lines.append("- 暂无")
            lines.append("")
        lines.extend(["## 导航", "", "- [[wiki/overview|知识概览]]", "- [[wiki/log|变更日志]]", "- [[wiki/indexes/wiki-catalog|完整目录]]", "- [[wiki/indexes/tag-index|标签索引]]", "- [[index/home|入口首页]]", ""])
        return "\n".join(lines)

    @staticmethod
    def _render_overview(items: list[KnowledgeItem], relations: list[KnowledgeRelation], generated_at: str) -> str:
        counts: dict[str, int] = {}
        statuses: dict[str, int] = {}
        for item in items:
            counts[item.item_type] = counts.get(item.item_type, 0) + 1
            statuses[item.status] = statuses.get(item.status, 0) + 1
        lines = ["---", 'title: "知识概览"', 'type: "index"', f"updated: {_yaml(generated_at)}", "generated_by: AI静静", "---", "", "# 知识概览", "", f"- 正式知识：{len(items)}", f"- 知识关系：{len(relations)}", f"- 待复核：{statuses.get('needs-review', 0)}", "", "## 类型", ""]
        lines.extend(f"- `{key}`：{value}" for key, value in sorted(counts.items()))
        lines.extend(["", "## 状态", ""])
        lines.extend(f"- `{key}`：{value}" for key, value in sorted(statuses.items()))
        return "\n".join(lines)

    @staticmethod
    def _render_log(events: list[dict[str, object]], generated_at: str) -> str:
        lines = ["---", 'title: "知识变更日志"', 'type: "index"', f"updated: {_yaml(generated_at)}", "generated_by: AI静静", "---", "", "# 知识变更日志", ""]
        if not events:
            lines.append("尚无结构化知识操作记录。")
        else:
            lines.extend(
                f"- {event.get('created_at', '')} · `{event.get('event_type', '')}` · {event.get('summary', '')}"
                for event in events
            )
        return "\n".join(lines)

    @staticmethod
    def _render_home(items: list[KnowledgeItem], generated_at: str) -> str:
        return "\n".join([
            "---", 'title: "AI静静知识入口"', 'type: "index"', f"updated: {_yaml(generated_at)}", "generated_by: AI静静", "---", "", "# AI静静知识入口", "", f"当前编译 {len(items)} 条正式知识。", "", "- [[wiki/index|进入 LLM Wiki]]", "- [[wiki/overview|查看知识概览]]", "- [[wiki/log|查看变更日志]]", "",
        ])

    def _render_catalog(self, items: list[KnowledgeItem], paths: dict[str, Path], generated_at: str) -> str:
        lines = ["---", 'title: "Wiki 完整目录"', 'type: "index"', f"updated: {_yaml(generated_at)}", "generated_by: AI静静", "---", "", "# Wiki 完整目录", "", "| 类型 | 标题 | 状态 | 成熟度 |", "|---|---|---|---|"]
        lines.extend(
            f"| {item.item_type} | {self._wikilink(paths[item.item_id], item.title)} | {item.status} | {item.maturity} |"
            for item in sorted(items, key=lambda value: (value.item_type, value.title.casefold()))
        )
        return "\n".join(lines)

    def _render_tag_index(self, items: list[KnowledgeItem], paths: dict[str, Path], generated_at: str) -> str:
        tagged: dict[str, list[KnowledgeItem]] = {}
        for item in items:
            for tag in item.tags:
                tagged.setdefault(tag, []).append(item)
        lines = ["---", 'title: "标签索引"', 'type: "index"', f"updated: {_yaml(generated_at)}", "generated_by: AI静静", "---", "", "# 标签索引", ""]
        for tag, values in sorted(tagged.items(), key=lambda pair: pair[0].casefold()):
            lines.extend([f"## #{tag}", ""])
            lines.extend(f"- {self._wikilink(paths[item.item_id], item.title)}" for item in values)
            lines.append("")
        if not tagged:
            lines.append("暂无标签。")
        return "\n".join(lines)

    def _render_status(self, items: list[KnowledgeItem], paths: dict[str, Path], title: str, generated_at: str) -> str:
        lines = ["---", f"title: {_yaml(title)}", 'type: "index"', f"updated: {_yaml(generated_at)}", "generated_by: AI静静", "---", "", f"# {title}", "", "| 标题 | 状态 | 成熟度 |", "|---|---|---|"]
        lines.extend(
            f"| {self._wikilink(paths[item.item_id], item.title)} | {item.status} | {item.maturity} |"
            for item in items
        )
        if not items:
            lines.append("| 暂无 | — | — |")
        return "\n".join(lines)

    def _previous_files(self, manifest_path: Path) -> set[str]:
        if not manifest_path.is_file():
            return set()
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return set()
        raw = payload.get("files", []) if isinstance(payload, dict) else []
        return {str(value) for value in raw if isinstance(value, str)}

    def _lint(self, payloads: dict[Path, str], paths: dict[str, Path]) -> list[str]:
        warnings: list[str] = []
        known = {path.relative_to(self.root).with_suffix("").as_posix() for path in payloads}
        link_re = re.compile(r"\[\[([^\]|#]+)")
        for path, content in payloads.items():
            if not content.startswith("---\n"):
                warnings.append(f"缺少 frontmatter：{path.relative_to(self.root)}")
            for link in link_re.findall(content):
                if link not in known:
                    warnings.append(f"断链：{path.relative_to(self.root)} → {link}")
        linked_ids = {
            relation[0]
            for relation in self.database.connection.execute(
                "SELECT source_item_id FROM knowledge_relations UNION SELECT target_item_id FROM knowledge_relations"
            ).fetchall()
        }
        for item_id in paths:
            if item_id not in linked_ids:
                warnings.append(f"孤立知识：{item_id}")
        return warnings


__all__ = ["PortableWikiCompiler", "WikiCompileResult"]
