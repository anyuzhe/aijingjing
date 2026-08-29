from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_REGISTRY_FILENAME = ".temporary-cleanup-registry.json"
_MARKER_FILENAME = ".ai-jingjing-temporary-owner.json"
_FORMAT = "ai-jingjing-temporary-cleanup-v1"
_MAX_REGISTRY_BYTES = 1024 * 1024


class TemporaryCleanupRegistryError(OSError):
    """A cleanup registry operation could not be completed safely."""


@dataclass(frozen=True, slots=True)
class CleanupRetryReport:
    removed: int = 0
    pending: int = 0
    rejected: int = 0

    @property
    def warning(self) -> str | None:
        if not self.pending and not self.rejected:
            return None
        count = self.pending + self.rejected
        return f"临时清理失败：{count} 个缓存目录仍在安全登记中，将在后续自动重试"


class TemporaryCleanupRegistry:
    """Durably records disposable cache directories and retries safe deletion.

    A registry row alone is never authority to delete a path.  Deletion requires an
    exact cache-relative path, the same directory device/inode, and a regular marker
    file containing the same random ownership token.  Invalid rows remain visible but
    are never followed or deleted.
    """

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root.expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.cache_root / _REGISTRY_FILENAME
        self._lock = threading.RLock()

    def register(self, path: Path) -> str:
        raw = path.expanduser()
        if raw.is_symlink():
            raise TemporaryCleanupRegistryError("拒绝登记符号链接临时目录")
        candidate = raw.resolve()
        relative = self._relative_candidate(candidate)
        try:
            metadata = candidate.stat(follow_symlinks=False)
        except OSError as error:
            raise TemporaryCleanupRegistryError("待登记临时目录不存在") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise TemporaryCleanupRegistryError("仅允许登记应用创建的临时目录")

        token = secrets.token_urlsafe(24)
        relative_text = relative.as_posix()
        marker_payload = {
            "format": _FORMAT,
            "path": relative_text,
            "token": token,
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
        }
        marker = candidate / _MARKER_FILENAME
        self._write_marker_exclusive(marker, marker_payload)
        entry = {
            **marker_payload,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with self._lock:
                entries = self._load_entries()
                if any(value.get("path") == relative_text for value in entries):
                    raise TemporaryCleanupRegistryError("临时目录已经登记")
                entries.append(entry)
                self._save_entries(entries)
        except BaseException:
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return token

    def verify(self, path: Path, token: str) -> bool:
        try:
            candidate = path.expanduser().resolve()
            relative = self._relative_candidate(candidate).as_posix()
            with self._lock:
                entry = next(
                    (
                        value
                        for value in self._load_entries()
                        if value.get("path") == relative and value.get("token") == token
                    ),
                    None,
                )
            return entry is not None and self._validated_candidate(entry) == candidate
        except (OSError, ValueError, TypeError):
            return False

    def forget(self, path: Path, token: str) -> None:
        candidate = path.expanduser().resolve(strict=False)
        relative = self._relative_candidate(candidate).as_posix()
        with self._lock:
            entries = self._load_entries()
            retained = [
                value
                for value in entries
                if not (value.get("path") == relative and value.get("token") == token)
            ]
            if len(retained) != len(entries):
                self._save_entries(retained)

    def retry_pending(self, *, attempts: int = 3) -> CleanupRetryReport:
        removed = 0
        rejected = 0
        with self._lock:
            entries = self._load_entries()
            retained: list[dict[str, object]] = []
            for entry in entries:
                state, candidate = self._candidate_state(entry)
                if state == "missing":
                    removed += 1
                    continue
                if state != "valid" or candidate is None:
                    rejected += 1
                    retained.append(entry)
                    continue
                deleted = False
                for _attempt in range(max(1, attempts)):
                    try:
                        shutil.rmtree(candidate)
                    except OSError:
                        continue
                    deleted = True
                    removed += 1
                    break
                if not deleted:
                    retained.append(entry)
            if retained != entries:
                self._save_entries(retained)
        return CleanupRetryReport(
            removed=removed,
            pending=max(0, len(retained) - rejected),
            rejected=rejected,
        )

    def pending_count(self) -> int:
        with self._lock:
            return len(self._load_entries())

    def _candidate_state(self, entry: dict[str, object]) -> tuple[str, Path | None]:
        try:
            raw_path = entry.get("path")
            if not isinstance(raw_path, str):
                return "invalid", None
            relative = self._validated_relative_path(raw_path)
            lexical = self.cache_root.joinpath(relative)
            if not lexical.exists() and not lexical.is_symlink():
                return "missing", lexical
            candidate = self._validated_candidate(entry)
            return ("valid", candidate) if candidate is not None else ("invalid", lexical)
        except (OSError, RuntimeError, TypeError, ValueError):
            return "invalid", None

    def _validated_candidate(self, entry: dict[str, object]) -> Path | None:
        raw_path = entry.get("path")
        token = entry.get("token")
        device = entry.get("device")
        inode = entry.get("inode")
        if (
            not isinstance(raw_path, str)
            or not isinstance(token, str)
            or not token
            or not isinstance(device, int)
            or isinstance(device, bool)
            or not isinstance(inode, int)
            or isinstance(inode, bool)
        ):
            return None
        relative = self._validated_relative_path(raw_path)
        lexical = self.cache_root.joinpath(relative)
        if lexical.is_symlink():
            return None
        candidate = lexical.resolve(strict=True)
        if self._relative_candidate(candidate).as_posix() != raw_path:
            return None
        metadata = candidate.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or int(metadata.st_dev) != device
            or int(metadata.st_ino) != inode
        ):
            return None
        marker = candidate / _MARKER_FILENAME
        if marker.is_symlink():
            return None
        marker_metadata = marker.stat(follow_symlinks=False)
        if not stat.S_ISREG(marker_metadata.st_mode) or marker_metadata.st_size > 16 * 1024:
            return None
        payload = json.loads(marker.read_text(encoding="utf-8"))
        expected = {
            "format": _FORMAT,
            "path": raw_path,
            "token": token,
            "device": device,
            "inode": inode,
        }
        return candidate if payload == expected else None

    def _relative_candidate(self, candidate: Path) -> Path:
        if candidate == self.cache_root or not candidate.is_relative_to(self.cache_root):
            raise TemporaryCleanupRegistryError("临时目录必须是产品缓存目录的后代")
        return candidate.relative_to(self.cache_root)

    @staticmethod
    def _validated_relative_path(value: str) -> Path:
        relative = Path(value)
        if (
            not value
            or relative.is_absolute()
            or relative == Path(".")
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise TemporaryCleanupRegistryError("临时清理登记包含无效路径")
        return relative

    def _load_entries(self) -> list[dict[str, object]]:
        if not self.registry_path.exists():
            return []
        if self.registry_path.is_symlink():
            raise TemporaryCleanupRegistryError("临时清理登记文件不安全")
        metadata = self.registry_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_REGISTRY_BYTES:
            raise TemporaryCleanupRegistryError("临时清理登记文件无效")
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TemporaryCleanupRegistryError("临时清理登记文件无法读取") from error
        if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
            raise TemporaryCleanupRegistryError("临时清理登记格式无效")
        entries = payload.get("entries")
        if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
            raise TemporaryCleanupRegistryError("临时清理登记内容无效")
        return list(entries)

    def _save_entries(self, entries: list[dict[str, object]]) -> None:
        if not entries:
            if self.registry_path.is_symlink():
                raise TemporaryCleanupRegistryError("拒绝删除不安全的临时清理登记")
            self.registry_path.unlink(missing_ok=True)
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.registry_path.name}.", suffix=".tmp", dir=self.cache_root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {"format": _FORMAT, "entries": entries},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.registry_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _write_marker_exclusive(path: Path, payload: dict[str, object]) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            raise TemporaryCleanupRegistryError("无法建立临时目录所有权标记") from error
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise


__all__ = [
    "CleanupRetryReport",
    "TemporaryCleanupRegistry",
    "TemporaryCleanupRegistryError",
]
