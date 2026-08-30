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
    def models(self) -> Path:
        """Application-owned model directory.

        Model weights are intentionally kept outside the packaged application so
        they can be installed, verified and removed independently.
        """

        return self.root / "models"

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
            self.cache, self.models, self.backups, self.trash,
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
    ocr_engine: str = "auto"
    ocr_complex_layout_enabled: bool = True
    ocr_low_confidence_threshold: float = 0.65
    whisper_model: str = "small"
    transcription_engine: str = "auto"
    transcription_allow_cpu_fallback: bool = True
    transcription_profile: str = "chinese-accuracy"
    asr_provider: str = "qwen3-mlx"
    asr_model: str = "Qwen3-ASR-1.7B"
    asr_model_path: str | None = None
    asr_model_sha256: str | None = None
    asr_whisper_fallback_model_path: str | None = None
    asr_whisper_fallback_model_sha256: str | None = None
    transcription_language: str = "zh"
    asr_knowledge_space_id: str = "本地知识库"
    asr_context_terms: list[str] = field(default_factory=list)
    word_timestamps: bool = True
    diarization_enabled: bool = False
    diarization_provider: str = "auto"
    diarization_model_path: str | None = None
    diarization_model_sha256: str | None = None
    diarization_min_speakers: int = 1
    diarization_max_speakers: int = 8
    transcript_quality_gate: bool = True
    deep_correction_enabled: bool = False
    deep_correction_model: str = "compatible::deepseek::deepseek-v4-flash"
    deep_correction_retranscribe_anomalies: bool = True
    deep_correction_web_verification: bool = False
    deep_correction_generate_knowledge_cards: bool = True
    deep_correction_generate_mermaid: bool = True
    deep_correction_auto_apply_high_confidence: bool = False
    deep_correction_confidence_threshold: float = 0.92
    deep_correction_chunk_seconds: int = 300
    deep_correction_overlap_seconds: int = 30
    deep_correction_max_external_queries: int = 12
    model_idle_timeout_seconds: int = 300
    embedding_provider: str = "hash"
    embedding_model: str = "hash-384-v1"
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
        try:
            settings.ocr_low_confidence_threshold = min(
                1.0, max(0.0, float(settings.ocr_low_confidence_threshold))
            )
        except (TypeError, ValueError, OverflowError):
            settings.ocr_low_confidence_threshold = cls().ocr_low_confidence_threshold
        settings.watched_scan_minutes = min(1440, max(1, int(settings.watched_scan_minutes)))
        settings.ocr_engine = str(settings.ocr_engine or "auto").strip().casefold()
        settings.transcription_engine = str(settings.transcription_engine or "auto").strip().casefold()
        settings.transcription_profile = str(
            settings.transcription_profile or "compatibility"
        ).strip().casefold()
        settings.asr_provider = str(settings.asr_provider or "auto").strip().casefold()
        settings.asr_model = str(settings.asr_model or settings.whisper_model or "small").strip()
        settings.transcription_language = str(
            settings.transcription_language or "auto"
        ).strip()
        settings.diarization_provider = str(
            settings.diarization_provider or "auto"
        ).strip().casefold()
        settings.deep_correction_model = str(
            settings.deep_correction_model
            or "compatible::deepseek::deepseek-v4-flash"
        ).strip()
        if settings.ocr_engine not in {"auto", "rapidocr", "paddleocr"}:
            settings.ocr_engine = "auto"
        if settings.transcription_engine not in {"auto", "mlx", "cuda", "cpu", "faster-whisper"}:
            settings.transcription_engine = "auto"
        if settings.transcription_profile not in {
            "chinese-accuracy", "fast-preview", "compatibility", "custom",
        }:
            settings.transcription_profile = "compatibility"
        if settings.asr_provider == "qwen3-asr-mlx":
            settings.asr_provider = "qwen3-mlx"
        if settings.asr_provider not in {
            "auto", "qwen3-mlx", "mlx-whisper", "faster-whisper",
        }:
            settings.asr_provider = "auto"
        if settings.diarization_provider not in {"auto", "pyannote", "sherpa", "none"}:
            settings.diarization_provider = "auto"
        if settings.transcription_language.casefold() not in {"auto", "zh", "en"}:
            settings.transcription_language = "auto"
        settings.asr_knowledge_space_id = str(
            settings.asr_knowledge_space_id or ""
        ).strip() or cls().asr_knowledge_space_id
        if not isinstance(settings.asr_context_terms, list):
            settings.asr_context_terms = []
        settings.asr_context_terms = list(dict.fromkeys(
            str(term).strip() for term in settings.asr_context_terms
            if isinstance(term, str) and term.strip()
        ))[:200]
        for field_name in (
            "asr_model_sha256",
            "asr_whisper_fallback_model_sha256",
            "diarization_model_sha256",
        ):
            checksum = str(getattr(settings, field_name) or "").strip().casefold()
            setattr(
                settings,
                field_name,
                checksum
                if len(checksum) == 64
                and all(character in "0123456789abcdef" for character in checksum)
                else None,
            )
        try:
            settings.diarization_min_speakers = max(
                1, min(20, int(settings.diarization_min_speakers))
            )
            settings.diarization_max_speakers = max(
                settings.diarization_min_speakers,
                min(20, int(settings.diarization_max_speakers)),
            )
        except (TypeError, ValueError, OverflowError):
            settings.diarization_min_speakers = 1
            settings.diarization_max_speakers = 8
        try:
            settings.deep_correction_confidence_threshold = min(
                1.0,
                max(0.5, float(settings.deep_correction_confidence_threshold)),
            )
            settings.deep_correction_chunk_seconds = min(
                900, max(60, int(settings.deep_correction_chunk_seconds))
            )
            settings.deep_correction_overlap_seconds = min(
                120,
                max(0, int(settings.deep_correction_overlap_seconds)),
            )
            settings.deep_correction_overlap_seconds = min(
                settings.deep_correction_overlap_seconds,
                max(0, settings.deep_correction_chunk_seconds // 3),
            )
            settings.deep_correction_max_external_queries = min(
                50, max(0, int(settings.deep_correction_max_external_queries))
            )
        except (TypeError, ValueError, OverflowError):
            defaults = cls()
            settings.deep_correction_confidence_threshold = (
                defaults.deep_correction_confidence_threshold
            )
            settings.deep_correction_chunk_seconds = defaults.deep_correction_chunk_seconds
            settings.deep_correction_overlap_seconds = defaults.deep_correction_overlap_seconds
            settings.deep_correction_max_external_queries = (
                defaults.deep_correction_max_external_queries
            )
        try:
            settings.model_idle_timeout_seconds = max(
                30, min(3600, int(settings.model_idle_timeout_seconds))
            )
        except (TypeError, ValueError, OverflowError):
            settings.model_idle_timeout_seconds = 300
        if settings.embedding_provider not in {"fastembed", "hash"}:
            settings.embedding_provider = "hash"
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
