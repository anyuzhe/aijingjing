from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from ..models import utcnow_iso
from ..product import PRODUCT_NAME, DesktopSettings, ProductPaths


BACKUP_FORMAT_V1 = "ai-jingjing-backup-v1"
BACKUP_FORMAT_V2 = "ai-jingjing-backup-v2"
BACKUP_CONTENT_ROOTS = ("notes", "archive", "assets", "transcripts")

# Backups may legitimately contain large videos. These limits are deliberately
# generous, but still prevent a small, hostile archive from expanding without
# bound on a user's machine.
MAX_BACKUP_ENTRIES = 100_000
MAX_ENTRY_UNCOMPRESSED_BYTES = 8 * 1024**3
MAX_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024**3
MAX_COMPRESSION_RATIO = 2_000
MAX_MANIFEST_BYTES = 2 * 1024**2
COPY_BUFFER_BYTES = 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{value}" for value in range(1, 10)),
    *(f"LPT{value}" for value in range(1, 10)),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(COPY_BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_and_hash(source: BinaryIO, destination: BinaryIO, *, limit: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while block := source.read(COPY_BUFFER_BYTES):
        total += len(block)
        if total > limit:
            raise ValueError("备份条目解压后超过安全上限")
        destination.write(block)
        digest.update(block)
    return total, digest.hexdigest()


def _snapshot_database(source_path: Path, destination_path: Path) -> None:
    """Create a consistent SQLite snapshot without copying WAL files."""

    if source_path.is_file():
        source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    else:
        # A fresh installation can be backed up before its first database use.
        source = sqlite3.connect(":memory:")
    destination = sqlite3.connect(destination_path)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()


def _iter_backup_files(paths: ProductPaths, database_snapshot: Path) -> list[tuple[Path, str]]:
    values: list[tuple[Path, str]] = [(database_snapshot, "knowledge.db")]
    for root_name in BACKUP_CONTENT_ROOTS:
        base = getattr(paths, root_name)
        if not base.is_dir():
            continue
        for candidate in sorted(base.rglob("*")):
            # Never follow a symlink out of the product data directory.
            if candidate.is_symlink() or not candidate.is_file():
                continue
            relative = candidate.relative_to(base).as_posix()
            archive_name = f"{root_name}/{relative}"
            _safe_archive_name(archive_name)
            values.append((candidate, archive_name))
    return values


def create_backup(paths: ProductPaths) -> Path:
    """Create a credential-free V2 backup with per-file integrity metadata."""

    paths.ensure()
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    target = paths.backups / f"AI静静备份-{stamp}.aijjbackup"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".backup-", suffix=".tmp", dir=paths.backups
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    descriptor, database_name = tempfile.mkstemp(
        prefix=".knowledge-", suffix=".db", dir=paths.backups
    )
    os.close(descriptor)
    database_snapshot = Path(database_name)
    try:
        database_snapshot.unlink(missing_ok=True)
        _snapshot_database(paths.database, database_snapshot)
        files = _iter_backup_files(paths, database_snapshot)
        manifest_files: list[dict[str, object]] = []
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for file_path, archive_name in files:
                # Hash the exact byte stream written into the ZIP. Reading once
                # avoids a race where an active import changes a file between a
                # separate digest pass and archive.write().
                with file_path.open("rb") as source, archive.open(
                    archive_name, "w", force_zip64=True
                ) as destination:
                    size, checksum = _copy_and_hash(
                        source, destination, limit=MAX_ENTRY_UNCOMPRESSED_BYTES
                    )
                manifest_files.append(
                    {"path": archive_name, "size": size, "sha256": checksum}
                )

            # Re-serialize only the typed settings schema so manually inserted
            # fields such as api_key/token cannot leak into a backup.
            settings_bytes = (
                json.dumps(
                    asdict(DesktopSettings.load(paths.settings)),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            archive.writestr("settings.json", settings_bytes)
            manifest_files.append(
                {
                    "path": "settings.json",
                    "size": len(settings_bytes),
                    "sha256": hashlib.sha256(settings_bytes).hexdigest(),
                }
            )

            manifest = {
                "format": BACKUP_FORMAT_V2,
                "created_at": utcnow_iso(),
                "product": PRODUCT_NAME,
                "includes_credentials": False,
                "excluded": ["providers.json", "system-keyring"],
                "content_roots": list(BACKUP_CONTENT_ROOTS),
                "files": sorted(manifest_files, key=lambda item: str(item["path"])),
            }
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
        database_snapshot.unlink(missing_ok=True)
    return target


def _safe_archive_name(raw_name: str, *, directory: bool = False) -> str:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise ValueError("备份包含不安全路径")
    normalized = raw_name[:-1] if directory and raw_name.endswith("/") else raw_name
    path = PurePosixPath(normalized)
    unsafe_component = any(
        ":" in part
        or any(ord(character) < 32 for character in part)
        or part.rstrip(" .") != part
        or part.upper().split(".", 1)[0] in _WINDOWS_RESERVED_NAMES
        for part in path.parts
    )
    if (
        not normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or unsafe_component
        or path.as_posix() != normalized
    ):
        raise ValueError("备份包含不安全路径")
    return normalized


def _validate_zip_structure(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > MAX_BACKUP_ENTRIES:
        raise ValueError("备份文件条目过多，可能是压缩炸弹")
    values: dict[str, zipfile.ZipInfo] = {}
    total_size = 0
    for info in infos:
        name = _safe_archive_name(info.filename, directory=info.is_dir())
        if name in values:
            raise ValueError("备份包含重复路径")
        values[name] = info
        if info.flag_bits & 0x1:
            raise ValueError("不支持加密备份条目")
        mode = (info.external_attr >> 16) & 0xFFFF
        if mode and stat.S_ISLNK(mode):
            raise ValueError("备份包含不安全的符号链接")
        if info.is_dir():
            continue
        if info.file_size < 0 or info.file_size > MAX_ENTRY_UNCOMPRESSED_BYTES:
            raise ValueError("备份包含超大条目")
        total_size += info.file_size
        if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("备份解压总大小超过安全上限，可能是压缩炸弹")
        if info.file_size and (
            info.compress_size <= 0
            or info.file_size / max(1, info.compress_size) > MAX_COMPRESSION_RATIO
        ):
            raise ValueError("备份压缩比异常，可能是压缩炸弹")
    return values


def _read_manifest(archive: zipfile.ZipFile, infos: dict[str, zipfile.ZipInfo]) -> dict[str, object]:
    info = infos.get("manifest.json")
    if info is None or info.is_dir() or info.file_size > MAX_MANIFEST_BYTES:
        raise ValueError("不是有效的 AI静静备份")
    try:
        raw = archive.read(info)
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError, zipfile.BadZipFile) as error:
        raise ValueError("备份清单损坏") from error
    if not isinstance(manifest, dict):
        raise ValueError("备份清单格式错误")
    return manifest


def _validate_v2_manifest(
    manifest: dict[str, object], infos: dict[str, zipfile.ZipInfo]
) -> dict[str, dict[str, object]]:
    if manifest.get("includes_credentials") is not False:
        raise ValueError("备份清单未声明凭据已排除")
    if manifest.get("content_roots") != list(BACKUP_CONTENT_ROOTS):
        raise ValueError("备份清单缺少完整知识目录")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("备份清单缺少文件校验信息")
    values: dict[str, dict[str, object]] = {}
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ValueError("备份清单文件项格式错误")
        path = _safe_archive_name(str(raw.get("path") or ""))
        if path == "manifest.json" or path in values:
            raise ValueError("备份清单包含重复或非法文件项")
        size = raw.get("size")
        checksum = str(raw.get("sha256") or "").lower()
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("备份清单包含无效文件大小")
        if not _SHA256_RE.fullmatch(checksum):
            raise ValueError("备份清单包含无效 SHA-256")
        info = infos.get(path)
        if info is None or info.is_dir() or info.file_size != size:
            raise ValueError("备份内容与清单不一致")
        values[path] = {"size": size, "sha256": checksum}
    payload_names = {name for name, info in infos.items() if name != "manifest.json" and not info.is_dir()}
    if set(values) != payload_names:
        raise ValueError("备份内容与清单文件列表不一致")
    if not {"knowledge.db", "settings.json"}.issubset(values):
        raise ValueError("备份缺少数据库或设置")
    for name in values:
        if name in {"knowledge.db", "settings.json"}:
            continue
        if not any(name.startswith(f"{root}/") for root in BACKUP_CONTENT_ROOTS):
            raise ValueError("备份包含未授权的数据路径")
    return values


def _v1_files(infos: dict[str, zipfile.ZipInfo]) -> dict[str, dict[str, object] | None]:
    values: dict[str, dict[str, object] | None] = {}
    for name, info in infos.items():
        if info.is_dir() or name == "manifest.json":
            continue
        if name in {"knowledge.db", "settings.json"} or name.startswith(("notes/", "archive/")):
            values[name] = None
        else:
            raise ValueError("旧版备份包含未知数据路径")
    if "knowledge.db" not in values:
        raise ValueError("备份缺少数据库")
    return values


def _validate_database(path: Path) -> None:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            result = connection.execute("PRAGMA integrity_check").fetchone()
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise ValueError("备份数据库无法打开") from error
    if not result or result[0] != "ok":
        raise ValueError("备份数据库完整性检查失败")


def _validate_settings(path: Path) -> None:
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("备份设置文件损坏") from error
    if not isinstance(payload, dict):
        raise ValueError("备份设置文件格式错误")


def _extract_and_validate(
    archive: zipfile.ZipFile,
    infos: dict[str, zipfile.ZipInfo],
    files: dict[str, dict[str, object] | None],
    staging: Path,
) -> None:
    expected_total = sum(infos[name].file_size for name in files)
    free_space = shutil.disk_usage(staging).free
    if expected_total > max(0, free_space - 64 * 1024**2):
        raise ValueError("磁盘空间不足，无法安全验证备份")
    for name, metadata in files.items():
        info = infos[name]
        target = staging.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with archive.open(info, "r") as source, target.open("wb") as destination:
                size, checksum = _copy_and_hash(
                    source, destination, limit=min(MAX_ENTRY_UNCOMPRESSED_BYTES, info.file_size)
                )
        except (OSError, RuntimeError, EOFError, zipfile.BadZipFile) as error:
            raise ValueError(f"备份条目损坏：{name}") from error
        if size != info.file_size:
            raise ValueError(f"备份条目大小不一致：{name}")
        if metadata is not None and (
            size != metadata["size"] or checksum != metadata["sha256"]
        ):
            raise ValueError(f"备份文件哈希校验失败：{name}")


def _backup_database_for_rollback(database: Path, destination: Path) -> bool:
    if not database.is_file():
        return False
    _snapshot_database(database, destination)
    return True


def _restore_database(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def _apply_staged_restore(paths: ProductPaths, staging: Path, roots: tuple[str, ...]) -> None:
    rollback = staging / ".rollback"
    rollback.mkdir()
    moved_targets: list[tuple[Path, Path | None]] = []
    database_rollback = rollback / "knowledge.db"
    had_database = _backup_database_for_rollback(paths.database, database_rollback)
    try:
        for name in roots:
            incoming = staging / name
            incoming.mkdir(parents=True, exist_ok=True)
            target = getattr(paths, name)
            previous = rollback / name
            if target.exists():
                os.replace(target, previous)
                moved_targets.append((target, previous))
            else:
                moved_targets.append((target, None))
            os.replace(incoming, target)

        incoming_settings = staging / "settings.json"
        if incoming_settings.is_file():
            previous_settings = rollback / "settings.json"
            if paths.settings.exists():
                os.replace(paths.settings, previous_settings)
                moved_targets.append((paths.settings, previous_settings))
            else:
                moved_targets.append((paths.settings, None))
            os.replace(incoming_settings, paths.settings)

        _restore_database(staging / "knowledge.db", paths.database)
    except Exception:
        if had_database:
            _restore_database(database_rollback, paths.database)
        elif paths.database.exists():
            paths.database.unlink(missing_ok=True)
        for target, previous in reversed(moved_targets):
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            if previous is not None and previous.exists():
                os.replace(previous, target)
        raise


def restore_backup(paths: ProductPaths, backup: str | Path) -> dict[str, object]:
    """Validate a V1/V2 archive completely before changing live product data."""

    source = Path(backup).expanduser().resolve()
    if not source.is_file():
        raise ValueError("备份文件不存在")
    paths.ensure()
    try:
        with zipfile.ZipFile(source) as archive, tempfile.TemporaryDirectory(
            prefix=".restore-stage-", dir=paths.root
        ) as temporary:
            infos = _validate_zip_structure(archive)
            manifest = _read_manifest(archive, infos)
            format_name = str(manifest.get("format") or "")
            if format_name == BACKUP_FORMAT_V2:
                files = _validate_v2_manifest(manifest, infos)
                roots = BACKUP_CONTENT_ROOTS
            elif format_name == BACKUP_FORMAT_V1:
                files = _v1_files(infos)
                roots = ("notes", "archive")
            else:
                raise ValueError("不支持的备份版本")

            staging = Path(temporary)
            _extract_and_validate(archive, infos, files, staging)
            _validate_database(staging / "knowledge.db")
            _validate_settings(staging / "settings.json")

            # This is intentionally after every untrusted byte has been checked.
            safety = create_backup(paths)
            _apply_staged_restore(paths, staging, roots)
    except zipfile.BadZipFile as error:
        raise ValueError("不是有效的 AI静静备份") from error
    return {
        "status": "complete",
        "format": format_name,
        "safety_backup": str(safety),
        "restored": str(source),
    }
