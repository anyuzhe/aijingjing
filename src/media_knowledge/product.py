from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


PRODUCT_NAME = "AI知识库-AI静静"
PRODUCT_SLUG = "AI-Jingjing"
LEGACY_PRODUCT_SLUG = "AI-Xiaopang"
DEFAULT_ANSWER_MODEL = "compatible::deepseek::deepseek-v4-flash-vision-exp"
LEGACY_DEFAULT_ANSWER_MODELS = {"compatible::deepseek::deepseek-v4-flash"}


def _platform_product_root(slug: str) -> Path:
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / slug).resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return (base / slug).resolve()
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return (base / slug).resolve()


def default_product_root() -> Path:
    override = os.environ.get("AI_JINGJING_DATA_DIR") or os.environ.get("AI_XIAOPANG_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _platform_product_root(PRODUCT_SLUG)


def legacy_product_root() -> Path:
    return _platform_product_root(LEGACY_PRODUCT_SLUG)


@dataclass(frozen=True, slots=True)
class ProductPaths:
    root: Path

    @classmethod
    def resolve(cls, root: str | Path | None = None) -> "ProductPaths":
        return cls(Path(root).expanduser().resolve() if root else default_product_root())

    @property
    def archive(self) -> Path:
        return self.root / "archive"

    @property
    def notes(self) -> Path:
        return self.root / "notes"

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def transcripts(self) -> Path:
        return self.root / "transcripts"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def backups(self) -> Path:
        return self.root / "backups"

    @property
    def trash(self) -> Path:
        return self.root / "trash"

    @property
    def database(self) -> Path:
        return self.root / "knowledge.db"

    @property
    def providers(self) -> Path:
        return self.root / "providers.json"

    @property
    def settings(self) -> Path:
        return self.root / "settings.json"

    def ensure(self) -> "ProductPaths":
        for path in (
            self.root, self.archive, self.notes, self.assets, self.transcripts,
            self.cache, self.backups, self.trash,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def migrate_renamed_product(self, source_root: str | Path | None = None) -> Path | None:
        """Copy non-database state from the former product name without deleting it."""

        source = Path(source_root).expanduser().resolve() if source_root else legacy_product_root()
        if source == self.root or not source.is_dir():
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        for filename in ("settings.json", "providers.json"):
            old_file = source / filename
            new_file = self.root / filename
            if old_file.is_file() and not new_file.exists():
                shutil.copy2(old_file, new_file)
                if filename == "providers.json":
                    try:
                        new_file.chmod(0o600)
                    except OSError:
                        pass
        for directory in ("archive", "notes", "assets", "transcripts", "cache"):
            old_directory = source / directory
            if not old_directory.is_dir():
                continue
            for old_file in old_directory.rglob("*"):
                if not old_file.is_file():
                    continue
                new_file = self.root / directory / old_file.relative_to(old_directory)
                if new_file.exists():
                    continue
                new_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_file, new_file)
        return source

    def migrate_legacy_providers(self, legacy: str | Path | None = None) -> bool:
        if self.providers.exists():
            return False
        candidates = (
            [Path(legacy).expanduser().resolve()]
            if legacy
            else [legacy_product_root() / "providers.json", Path.cwd() / ".knowledge" / "providers.json"]
        )
        candidate = next((path for path in candidates if path.is_file()), None)
        if candidate is None:
            return False
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, self.providers)
        try:
            self.providers.chmod(0o600)
        except OSError:
            pass
        return True

    def migrate_legacy_database(self, candidates: list[str | Path] | None = None) -> Path | None:
        """Safely copy the richest legacy database with SQLite's online backup API."""

        if self.database.exists():
            return None
        raw_candidates = candidates or [
            legacy_product_root() / "knowledge.db",
            Path.cwd() / ".knowledge" / "preview.db",
            Path.cwd() / ".knowledge" / "knowledge.db",
        ]
        ranked: list[tuple[int, Path]] = []
        for raw in raw_candidates:
            candidate = Path(raw).expanduser().resolve()
            if not candidate.is_file():
                continue
            try:
                source = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True)
                try:
                    count = int(source.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
                finally:
                    source.close()
            except sqlite3.Error:
                continue
            ranked.append((count, candidate))
        if not ranked:
            return None
        _, selected = max(ranked, key=lambda item: (item[0], item[1].stat().st_mtime))
        self.root.mkdir(parents=True, exist_ok=True)
        source = sqlite3.connect(f"file:{selected}?mode=ro", uri=True)
        destination = sqlite3.connect(self.database)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        return selected


@dataclass(slots=True)
class DesktopSettings:
    default_model: str = DEFAULT_ANSWER_MODEL
    answer_language: str = "zh-CN"
    archive_originals: bool = True
    create_source_notes: bool = True
    auto_synthesize_notes: bool = True
    enable_cloud_vision: bool = True
    vision_max_images: int = 12
    whisper_model: str = "small"
    embedding_provider: str = "fastembed"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    obsidian_vault: str | None = None
    watched_folders_enabled: bool = True
    watched_scan_minutes: int = 10
    update_manifest_url: str | None = None
    recent_imports: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "DesktopSettings":
        target = Path(path)
        if not target.is_file():
            return cls()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        allowed = cls.__dataclass_fields__
        values = {key: raw[key] for key in allowed if key in raw}
        try:
            settings = cls(**values)
        except (TypeError, ValueError):
            return cls()
        settings.vision_max_images = min(50, max(0, int(settings.vision_max_images)))
        settings.watched_scan_minutes = min(1440, max(1, int(settings.watched_scan_minutes)))
        if settings.embedding_provider not in {"fastembed", "hash"}:
            settings.embedding_provider = "fastembed"
        return settings

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(asdict(self), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
