from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

from ..documents import document_from_text
from ..indexing import IndexingService
from ..models import sha256_text, utcnow_iso
from ..storage import KnowledgeDatabase


_EXCLUDED_DIRECTORIES = {".git", ".obsidian", ".trash", "_assets"}
_EXCLUDED_PREFIXES = ("10_Knowledge/AI Answers/",)
_MARKDOWN_SUFFIXES = {".md", ".markdown"}


@dataclass(slots=True)
class ObsidianSyncReport:
    scanned: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    duplicate: int = 0
    deleted: int = 0
    skipped: int = 0
    failed: int = 0
    embedded_chunks: int = 0
    errors: list[str] = field(default_factory=list)
    completed_at: str = field(default_factory=utcnow_iso)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = "partial" if self.failed else "complete"
        return payload


class ObsidianMarkdownSync:
    """Incrementally mirror Obsidian Markdown notes into the local retrieval index."""

    def __init__(
        self,
        database: KnowledgeDatabase,
        indexing: IndexingService,
        vault_root: str | Path,
        *,
        max_file_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.database = database
        self.indexing = indexing
        self.vault_root = Path(vault_root).expanduser().resolve()
        self.max_file_bytes = max_file_bytes
        if not self.vault_root.is_dir():
            raise ValueError("尚未配置可用的 Obsidian 仓库")

    def _paths(self) -> tuple[list[tuple[str, Path]], list[str]]:
        values: list[tuple[str, Path]] = []
        scan_errors: list[str] = []

        def onerror(error: OSError) -> None:
            scan_errors.append(f"扫描失败：{Path(error.filename or '').name or type(error).__name__}")

        for directory, child_directories, filenames in os.walk(
            self.vault_root, topdown=True, followlinks=False, onerror=onerror
        ):
            directory_path = Path(directory)
            relative_directory = directory_path.relative_to(self.vault_root)
            child_directories[:] = sorted(
                name
                for name in child_directories
                if name not in _EXCLUDED_DIRECTORIES and not name.startswith(".")
            )
            if relative_directory.as_posix() == "10_Knowledge":
                child_directories[:] = [name for name in child_directories if name != "AI Answers"]
            for filename in sorted(filenames):
                path = directory_path / filename
                if path.suffix.casefold() not in _MARKDOWN_SUFFIXES or filename.startswith("."):
                    continue
                relative = path.relative_to(self.vault_root).as_posix()
                if any(relative.startswith(prefix) for prefix in _EXCLUDED_PREFIXES):
                    continue
                resolved = path.resolve()
                if self.vault_root not in resolved.parents:
                    scan_errors.append(f"已跳过仓库外链接：{relative}")
                    continue
                values.append((relative, resolved))
        return values, scan_errors

    @staticmethod
    def _frontmatter(text: str) -> dict[str, object]:
        if not text.startswith("---"):
            return {}
        match = re.match(r"\A---\s*\n(.*?)\n---(?:\s*\n|\Z)", text, re.DOTALL)
        if not match:
            return {}
        values: dict[str, object] = {}
        for line in match.group(1).splitlines():
            item = re.match(r"^([A-Za-z_][\w-]*):\s*(.*?)\s*$", line)
            if not item:
                continue
            key, raw = item.groups()
            if not raw:
                continue
            try:
                values[key] = json.loads(raw)
            except json.JSONDecodeError:
                values[key] = raw.strip("\"'")
        return values

    @staticmethod
    def _title(text: str, fallback: str, frontmatter: dict[str, object]) -> str:
        configured = frontmatter.get("title")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        heading = re.search(r"(?m)^#\s+(.+?)\s*$", text)
        return heading.group(1).strip() if heading else fallback

    @staticmethod
    def _tags(frontmatter: dict[str, object]) -> list[str]:
        raw = frontmatter.get("tags")
        if isinstance(raw, list):
            return list(dict.fromkeys(str(tag).strip() for tag in raw if str(tag).strip()))
        if isinstance(raw, str):
            return list(
                dict.fromkeys(
                    tag.strip().strip("\"'")
                    for tag in raw.strip("[]").split(",")
                    if tag.strip().strip("\"'")
                )
            )
        return []

    def _managed_documents(self) -> dict[str, str]:
        managed: dict[str, str] = {}
        rows = self.database.connection.execute(
            "SELECT id, metadata_json FROM documents ORDER BY id"
        ).fetchall()
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            if metadata.get("sync_origin") != "obsidian":
                continue
            if metadata.get("sync_root") != str(self.vault_root):
                continue
            relative = metadata.get("obsidian_path")
            if isinstance(relative, str) and relative:
                managed[relative] = row["id"]
        return managed

    def sync(self) -> ObsidianSyncReport:
        started = perf_counter()
        report = ObsidianSyncReport()
        paths, scan_errors = self._paths()
        report.errors.extend(scan_errors[:20])
        if scan_errors:
            report.failed += len(scan_errors)
        report.scanned = len(paths)
        present = {relative for relative, _ in paths}

        # Only delete stale notes after a complete directory walk. A partial scan must never
        # turn a transient filesystem error into data loss from the retrieval index.
        if not scan_errors:
            for relative, document_id in self._managed_documents().items():
                if relative not in present and self.indexing.delete_document(document_id):
                    report.deleted += 1

        for relative, path in paths:
            try:
                if path.stat().st_size > self.max_file_bytes:
                    report.skipped += 1
                    report.errors.append(f"文件过大，已跳过：{relative}")
                    continue
                text = path.read_text(encoding="utf-8")
                if not text.strip():
                    report.skipped += 1
                    continue
                frontmatter = self._frontmatter(text)
                source_id = "obsidian-" + sha256_text(f"{self.vault_root}\0{relative}")[:20]
                collection = relative.split("/", 1)[0] if "/" in relative else "Obsidian"
                document = document_from_text(
                    text,
                    title=self._title(text, path.stem, frontmatter),
                    source_id=source_id,
                    media_type="markdown",
                    local_path=str(path),
                    obsidian_path=relative,
                    collections=[collection],
                    tags=self._tags(frontmatter),
                )
                document.metadata.update(
                    {
                        "sync_origin": "obsidian",
                        "sync_root": str(self.vault_root),
                        "obsidian_path": relative,
                    }
                )
                indexed = self.indexing.index_document(document)
                if indexed.status == "created":
                    report.created += 1
                elif indexed.status == "updated":
                    report.updated += 1
                elif indexed.status == "unchanged":
                    report.unchanged += 1
                elif indexed.status == "duplicate":
                    report.duplicate += 1
                report.embedded_chunks += indexed.embedded_chunks
            except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
                report.failed += 1
                if len(report.errors) < 20:
                    report.errors.append(f"{relative}：{type(exc).__name__}")

        report.duration_ms = round((perf_counter() - started) * 1000, 3)
        report.completed_at = utcnow_iso()
        return report
