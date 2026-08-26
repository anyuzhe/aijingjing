from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SUPPORTED_SUFFIXES = {
    ".md", ".markdown", ".txt", ".csv", ".json", ".yaml", ".yml",
    ".pdf", ".docx", ".pptx",
    ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif",
    ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
}


@dataclass(slots=True)
class FolderScan:
    root: str
    current: dict[str, dict[str, int]] = field(default_factory=dict)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "current": self.current,
            "changed": self.changed,
            "removed": self.removed,
            "unchanged": self.unchanged,
        }


def scan_folder(
    root: str | Path,
    previous: dict[str, dict[str, int]] | None = None,
    *,
    recursive: bool = True,
    suffixes: Iterable[str] = SUPPORTED_SUFFIXES,
) -> FolderScan:
    """Return a cheap deterministic incremental scan without opening file contents."""

    folder = Path(root).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"监听目录不存在：{folder}")
    allowed = {suffix.casefold() for suffix in suffixes}
    old = previous or {}
    result = FolderScan(root=str(folder))
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    for path in iterator:
        if not path.is_file() or path.suffix.casefold() not in allowed:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        relative = path.relative_to(folder).as_posix()
        fingerprint = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        result.current[relative] = fingerprint
        prior = old.get(relative) or {}
        if prior.get("size") == fingerprint["size"] and prior.get("mtime_ns") == fingerprint["mtime_ns"]:
            result.unchanged += 1
        else:
            result.changed.append(str(path))
    result.removed = sorted(set(old) - set(result.current))
    result.changed.sort()
    return result
