from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


MAX_UPDATE_MANIFEST_BYTES = 256 * 1024
MAX_UPDATE_PACKAGE_BYTES = 8 * 1024**3
DOWNLOAD_BUFFER_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SEMVER_RE = re.compile(
    r"^(?:v)?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


@dataclass(frozen=True, slots=True)
class SemanticVersion:
    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()

    def newer_than(self, other: "SemanticVersion") -> bool:
        left = (self.major, self.minor, self.patch)
        right = (other.major, other.minor, other.patch)
        if left != right:
            return left > right
        if not self.prerelease:
            return bool(other.prerelease)
        if not other.prerelease:
            return False
        for left_part, right_part in zip(self.prerelease, other.prerelease):
            if left_part == right_part:
                continue
            left_numeric = left_part.isdigit()
            right_numeric = right_part.isdigit()
            if left_numeric and right_numeric:
                return int(left_part) > int(right_part)
            if left_numeric != right_numeric:
                return not left_numeric
            return left_part > right_part
        return len(self.prerelease) > len(other.prerelease)


@dataclass(slots=True)
class UpdateReport:
    status: str
    current_version: str
    latest_version: str | None = None
    download_url: str | None = None
    sha256: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "download_url": self.download_url,
            "sha256": self.sha256,
            "notes": self.notes,
        }


def _version(value: str) -> SemanticVersion:
    """Parse a strict SemVer 2.0 version (with an optional conventional `v`)."""

    match = _SEMVER_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"无效的版本号：{value!r}")
    prerelease = tuple((match.group(4) or "").split(".")) if match.group(4) else ()
    if any(part.isdigit() and len(part) > 1 and part.startswith("0") for part in prerelease):
        raise ValueError(f"无效的版本号：{value!r}")
    return SemanticVersion(
        int(match.group(1)), int(match.group(2)), int(match.group(3)), prerelease
    )


def _https_url(value: str, *, label: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{label}必须使用有效的 HTTPS 地址")
    return value


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject every redirect hop that leaves HTTPS, not only the final URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        try:
            _https_url(str(newurl), label="更新重定向地址")
        except ValueError as error:
            raise urllib.error.HTTPError(
                str(newurl), code, str(error), headers, fp
            ) from error
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _secure_urlopen(request: urllib.request.Request, *, timeout: float):
    return urllib.request.build_opener(_HTTPSOnlyRedirectHandler()).open(
        request, timeout=timeout
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_download_sha256(path: str | Path, expected_sha256: str) -> Path:
    """Verify a downloaded installer before it is opened or installed."""

    expected = expected_sha256.strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ValueError("更新包 SHA-256 格式无效")
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise ValueError("更新包不存在")
    actual = sha256_file(target)
    if not hmac.compare_digest(actual, expected):
        raise ValueError("更新包完整性校验失败：SHA-256 不匹配")
    return target


def _download_filename(url: str) -> str:
    raw = urllib.parse.unquote(PurePosixPath(urllib.parse.urlsplit(url).path).name)
    name = re.sub(r"[\x00-\x1f<>:\"/\\|?*]", "-", raw).strip(" .")[:180]
    if not name or name in {".", ".."}:
        return "AI-Jingjing-update.bin"
    if name.upper().split(".", 1)[0] in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{value}" for value in range(1, 10)),
        *(f"LPT{value}" for value in range(1, 10)),
    }:
        name = f"AI-Jingjing-{name}"
    return name


def download_verified_update(
    download_url: str,
    expected_sha256: str,
    destination_dir: str | Path,
    *,
    max_bytes: int = MAX_UPDATE_PACKAGE_BYTES,
) -> Path:
    """Download an installer to a temporary file, verify it, then atomically publish it."""

    download_url = _https_url(download_url.strip(), label="更新包下载地址")
    expected = expected_sha256.strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ValueError("更新包 SHA-256 格式无效")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("更新包大小上限无效")
    destination = Path(destination_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".AI-Jingjing-update-", suffix=".download", dir=destination
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    total = 0
    final_url = download_url
    expected_length = 0
    try:
        request = urllib.request.Request(
            download_url,
            headers={
                "User-Agent": "AI-Jingjing-Updater/1",
                "Accept": "application/octet-stream, application/x-apple-diskimage, */*",
            },
        )
        with os.fdopen(descriptor, "wb") as output, _secure_urlopen(
            request, timeout=60
        ) as response:
            if getattr(response, "status", 200) != 200:
                raise RuntimeError(f"更新服务器返回 HTTP {response.status}")
            final_url = str(getattr(response, "geturl", lambda: download_url)())
            try:
                _https_url(final_url, label="更新包重定向地址")
            except ValueError as error:
                raise RuntimeError(str(error)) from error
            try:
                expected_length = int(response.headers.get("Content-Length") or 0)
            except (TypeError, ValueError) as error:
                raise RuntimeError("更新服务器返回无效的文件大小") from error
            if expected_length < 0 or expected_length > max_bytes:
                raise RuntimeError("更新包超过允许的大小上限")
            while block := response.read(DOWNLOAD_BUFFER_BYTES):
                total += len(block)
                if total > max_bytes:
                    raise RuntimeError("更新包超过允许的大小上限")
                output.write(block)
                digest.update(block)
            if expected_length and total != expected_length:
                raise RuntimeError("更新包下载不完整")
            output.flush()
            os.fsync(output.fileno())
        if not hmac.compare_digest(digest.hexdigest(), expected):
            raise ValueError("更新包完整性校验失败：SHA-256 不匹配")
        target = destination / _download_filename(final_url)
        os.replace(temporary, target)
        return target
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def check_for_update(current_version: str, manifest_url: str | None) -> UpdateReport:
    current = _version(current_version)
    if not manifest_url:
        return UpdateReport(
            "manual",
            current_version,
            notes="当前安装包未配置发布服务器；应用支持校验 HTTPS 更新清单和安装包 SHA-256。",
        )
    manifest_url = _https_url(manifest_url, label="更新清单")
    request = urllib.request.Request(
        manifest_url,
        headers={"User-Agent": f"AI-Jingjing/{current_version}", "Accept": "application/json"},
    )
    with _secure_urlopen(request, timeout=8) as response:
        if getattr(response, "status", 200) != 200:
            raise RuntimeError(f"更新服务器返回 HTTP {response.status}")
        final_url = str(getattr(response, "geturl", lambda: manifest_url)())
        try:
            _https_url(final_url, label="更新清单重定向地址")
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        content_length = int(response.headers.get("Content-Length") or 0)
        if content_length > MAX_UPDATE_MANIFEST_BYTES:
            raise RuntimeError("更新清单异常过大")
        raw = response.read(MAX_UPDATE_MANIFEST_BYTES + 1)
        if len(raw) > MAX_UPDATE_MANIFEST_BYTES:
            raise RuntimeError("更新清单异常过大")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("更新清单不是有效的 UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("更新清单格式错误")

    latest_text = str(payload.get("version") or "").strip()
    download = str(payload.get("download_url") or "").strip()
    checksum = str(payload.get("sha256") or "").strip().lower()
    try:
        latest = _version(latest_text)
        download = _https_url(download, label="更新包下载地址")
    except ValueError as error:
        raise RuntimeError(f"更新清单无效：{error}") from error
    if not _SHA256_RE.fullmatch(checksum):
        raise RuntimeError("更新清单缺少有效的 SHA-256")
    status = "available" if latest.newer_than(current) else "current"
    return UpdateReport(
        status=status,
        current_version=current_version,
        latest_version=latest_text,
        download_url=download,
        sha256=checksum,
        notes=str(payload.get("notes") or ""),
    )
