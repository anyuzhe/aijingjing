from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


CHECKPOINT_FORMAT = "ai-jingjing-media-checkpoint-v1"
MANIFEST_FORMAT = "ai-jingjing-media-checkpoint-manifest-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def configuration_hash(value: Mapping[str, object]) -> str:
    """Return the stable identity of settings that affect derived media facts."""

    return hashlib.sha256(_canonical_json(dict(value))).hexdigest()


def glossary_version(terms: object) -> str:
    values = terms if isinstance(terms, (list, tuple)) else ()
    normalized = [" ".join(str(item or "").split()) for item in values]
    return hashlib.sha256(
        _canonical_json([item for item in normalized if item])
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    source_sha256: str
    config_hash: str
    asr_provider: str
    asr_model: str
    speaker_provider: str
    glossary_version: str
    asr_model_sha256: str = ""
    speaker_model_sha256: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class MediaCheckpointStore:
    """Crash-safe persistent stage artifacts for one source/configuration pair."""

    def __init__(
        self,
        cache_directory: str | Path,
        identity: CheckpointIdentity,
    ) -> None:
        for label, digest in (
            ("源 SHA256", identity.source_sha256),
            ("配置哈希", identity.config_hash),
            ("词库版本", identity.glossary_version),
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{label} 必须是小写十六进制 SHA256")
        for label, digest in (
            ("ASR 模型 SHA256", identity.asr_model_sha256),
            ("说话人模型 SHA256", identity.speaker_model_sha256),
        ):
            if digest and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label} 必须为空或小写十六进制 SHA256")
        self.identity = identity
        cache = Path(cache_directory).expanduser().resolve()
        self.directory = (
            cache
            / "media-checkpoints"
            / identity.source_sha256[:2]
            / identity.source_sha256
            / identity.config_hash
        )
        self.manifest_path = self.directory / "manifest.json"

    def path(self, name: str) -> Path:
        candidate = Path(name)
        if candidate.name != name or name in {"", ".", ".."}:
            raise ValueError("检查点文件名不安全")
        return self.directory / name

    def checkpoint_metadata(
        self,
        stage: str,
        *,
        runtime: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "format": CHECKPOINT_FORMAT,
            "stage": stage,
            **self.identity.to_dict(),
            "runtime": dict(runtime or {}),
        }

    def _identity_matches(self, value: Mapping[str, object]) -> bool:
        return all(
            str(value.get(key) or "") == expected
            for key, expected in self.identity.to_dict().items()
        )

    def _manifest(self) -> dict[str, object]:
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            value = {}
        if not isinstance(value, dict) or not self._identity_matches(value):
            value = {
                "format": MANIFEST_FORMAT,
                **self.identity.to_dict(),
                "artifacts": {},
            }
        if not isinstance(value.get("artifacts"), dict):
            value["artifacts"] = {}
        return value

    def record_file(
        self,
        name: str,
        stage: str,
        *,
        runtime: Mapping[str, object] | None = None,
    ) -> Path:
        target = self.path(name)
        if not target.is_file() or target.is_symlink():
            raise FileNotFoundError(target)
        manifest = self._manifest()
        artifacts = manifest["artifacts"]
        assert isinstance(artifacts, dict)
        artifacts[name] = {
            **self.identity.to_dict(),
            "stage": stage,
            "status": "complete",
            "sha256": _sha256_file(target),
            "size": target.stat().st_size,
            "runtime": dict(runtime or {}),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.manifest_path, manifest)
        return target

    def is_file_valid(self, name: str) -> bool:
        target = self.path(name)
        if not target.is_file() or target.is_symlink():
            return False
        manifest = self._manifest()
        artifacts = manifest.get("artifacts")
        record = artifacts.get(name) if isinstance(artifacts, dict) else None
        if not isinstance(record, dict) or not self._identity_matches(record):
            return False
        try:
            return (
                record.get("status") == "complete"
                and int(record.get("size", -1)) == target.stat().st_size
                and str(record.get("sha256") or "") == _sha256_file(target)
            )
        except (OSError, TypeError, ValueError):
            return False

    def write_json(
        self,
        name: str,
        stage: str,
        payload: object,
        *,
        runtime: Mapping[str, object] | None = None,
    ) -> Path:
        target = self.path(name)
        _atomic_json(target, {
            **self.checkpoint_metadata(stage, runtime=runtime),
            "payload": payload,
        })
        return self.record_file(name, stage, runtime=runtime)

    def read_json(self, name: str, stage: str) -> object | None:
        if not self.is_file_valid(name):
            return None
        try:
            value = json.loads(self.path(name).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        if value.get("format") != CHECKPOINT_FORMAT or value.get("stage") != stage:
            return None
        if not self._identity_matches(value):
            return None
        return value.get("payload")


__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointIdentity",
    "MediaCheckpointStore",
    "configuration_hash",
    "glossary_version",
]
