from __future__ import annotations

"""Local privacy scanning and safe, non-delivering share-copy creation.

This module deliberately has no network client.  It prepares a local directory
that a person may review and share later; it never uploads, sends, or publishes
anything.  OCR is optional and must be supplied as a *local* callback by the
caller.

The scanner is intentionally conservative.  A clean result means that none of
the patterns supported here were found; it is not a proof that a file contains
no private information.  Coverage limitations are always returned explicitly.
"""

import bisect
import errno
import hashlib
import html
import json
import os
import re
import shutil
import stat
import tempfile
import sys
import warnings
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence
from xml.etree import ElementTree

from ..models import utcnow_iso
from ..product import PRODUCT_NAME


SHARE_FORMAT = "ai-jingjing-share-v1"
COPY_BUFFER_BYTES = 1024 * 1024
MAX_SCAN_FILES = 100_000
MAX_TEXT_BYTES = 8 * 1024**2
MAX_OCR_CHARACTERS = 2_000_000
MAX_SHARE_FILES = 50_000
MAX_SHARE_FILE_BYTES = 4 * 1024**3
MAX_SHARE_TOTAL_BYTES = 16 * 1024**3
MAX_DOCUMENT_MEMBERS = 20_000
MAX_DOCUMENT_PAGES = 5_000
MAX_PDF_RENDER_PIXELS = 40_000_000
MAX_RAW_DOCUMENT_BYTES = 64 * 1024**2
MAX_DECODED_CONTAINER_BYTES = 16 * 1024**2

_MAX_REPORTED_LINE_NUMBERS = 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{value}" for value in range(1, 10)),
    *(f"LPT{value}" for value in range(1, 10)),
}

_TEXT_EXTENSIONS = {
    ".cfg",
    ".conf",
    ".css",
    ".csv",
    ".htm",
    ".html",
    ".ini",
    ".ipynb",
    ".java",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".log",
    ".md",
    ".properties",
    ".ps1",
    ".py",
    ".rb",
    ".rst",
    ".sh",
    ".sql",
    ".srt",
    ".tex",
    ".toml",
    ".ts",
    ".tsx",
    ".tsv",
    ".txt",
    ".vtt",
    ".xml",
    ".yaml",
    ".yml",
}
_IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_PDF_EXTENSIONS = {".pdf"}
_AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".amr",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
_VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}
_OFFICE_EXTENSIONS = {
    ".doc",
    ".docm",
    ".docx",
    ".odp",
    ".ods",
    ".odt",
    ".ppt",
    ".pptm",
    ".pptx",
    ".xls",
    ".xlsb",
    ".xlsm",
    ".xlsx",
}
_ARCHIVE_EXTENSIONS = {
    ".7z",
    ".bz2",
    ".gz",
    ".rar",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
}
_DATABASE_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}

_GENERAL_LIMITATION = (
    "本地规则扫描只能发现已知模式，不能证明资料完全不含隐私；本模块不会联网或自动外发。"
)
_PDF_LIMITATION = (
    "PDF 会扫描可提取文字和元数据，但原始 PDF 容器可能藏有解析器不可达的数据；"
    "当前版本不会把原始 PDF 加入分享副本，未来需生成渲染净化副本。"
)
_IMAGE_LIMITATION_NO_OCR = (
    "图片像素中的文字未执行本地 OCR；只检查了可读取的 EXIF/图片元数据。"
)
_IMAGE_LIMITATION_WITH_OCR = (
    "本地 OCR 可能漏字、错字或遗漏遮挡/低清文字，原始图片还可能携带解析器不可达的数据；"
    "当前版本不会把原始图片加入分享副本，未来需生成像素重编码的净化副本。"
)
_MEDIA_LIMITATION = (
    "音频和视频未转写，也未分析语音、字幕、画面、人脸或位置信息。"
)
_OFFICE_LIMITATION = (
    "Office 会扫描开放文档包中的 XML 正文和元数据，但原始 ZIP 容器可能藏有目录外数据；"
    "当前版本不会把原始 Office/ODF 容器加入分享副本，请先导出为脱敏文本。"
)
_ARCHIVE_LIMITATION = "压缩包及加密容器的内部文件未展开扫描。"
_BINARY_LIMITATION = "未知二进制文件的内部内容未解析。"

# There is deliberately no review-only bypass.  A file whose contents were not
# actually inspected must not become publishable merely because the caller set
# an "expert" flag.  ``require_clean_scan`` remains in the public data model for
# backwards compatibility, but credentials and uninspected content are never
# overridable.
_EXPERT_REVIEW_ALLOWLIST: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class _ContentRule:
    category: str
    severity: str
    summary: str
    pattern: re.Pattern[str]


_CONTENT_RULES = (
    _ContentRule(
        "private_key",
        "blocked",
        "检测到私钥结构，具体内容已隐藏。",
        re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----", re.IGNORECASE),
    ),
    _ContentRule(
        "provider_api_key",
        "blocked",
        "检测到云服务 API 密钥格式，具体内容已隐藏。",
        re.compile(r"(?<![A-Za-z0-9])sk-(?:ant-)?[A-Za-z0-9_-]{16,}(?![A-Za-z0-9])"),
    ),
    _ContentRule(
        "github_token",
        "blocked",
        "检测到 GitHub 访问令牌格式，具体内容已隐藏。",
        re.compile(r"(?<![A-Za-z0-9])gh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
    ),
    _ContentRule(
        "gitlab_token",
        "blocked",
        "检测到 GitLab 访问令牌格式，具体内容已隐藏。",
        re.compile(r"(?<![A-Za-z0-9])glpat-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9])"),
    ),
    _ContentRule(
        "huggingface_token",
        "blocked",
        "检测到模型服务访问令牌格式，具体内容已隐藏。",
        re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
    ),
    _ContentRule(
        "slack_token",
        "blocked",
        "检测到协作服务访问令牌格式，具体内容已隐藏。",
        re.compile(r"(?<![A-Za-z0-9])xox(?:a|b|p|r|s)-[A-Za-z0-9-]{16,}(?![A-Za-z0-9])"),
    ),
    _ContentRule(
        "aws_access_key",
        "blocked",
        "检测到云平台访问密钥格式，具体内容已隐藏。",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    _ContentRule(
        "google_api_key",
        "blocked",
        "检测到 Google API 密钥格式，具体内容已隐藏。",
        re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{30,}(?![A-Za-z0-9])"),
    ),
    _ContentRule(
        "jwt_token",
        "blocked",
        "检测到 JWT 令牌格式，具体内容已隐藏。",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
        ),
    ),
    _ContentRule(
        "bearer_token",
        "blocked",
        "检测到 Bearer 认证令牌，具体内容已隐藏。",
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}", re.IGNORECASE),
    ),
    _ContentRule(
        "credential_assignment",
        "blocked",
        "检测到疑似凭据赋值，键名和值均已隐藏。",
        re.compile(
            r"(?im)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|secret)\b"
            r"\s*(?::|=)\s*[\"']?[^\s\"'#,;]{8,}"
        ),
    ),
    _ContentRule(
        "email_address",
        "blocked",
        "检测到电子邮箱地址，具体地址已隐藏。",
        re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9.-])"),
    ),
    _ContentRule(
        "phone_number",
        "blocked",
        "检测到疑似中国大陆手机号码，具体号码已隐藏。",
        re.compile(r"(?<!\d)(?:\+?86[ -]?)?1[3-9]\d{9}(?!\d)"),
    ),
    _ContentRule(
        "absolute_user_path",
        "blocked",
        "检测到包含用户名的绝对本机路径，具体路径已隐藏。",
        re.compile(
            r"(?:/(?:Users|home)/[^/\s\"'<>]+(?:/[^\s\"'<>]*)?|[A-Za-z]:\\Users\\[^\\\s\"'<>]+(?:\\[^\s\"'<>]*)?)",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class PrivacyFinding:
    severity: str
    category: str
    redacted_path: str
    count: int
    line_numbers: tuple[int, ...] = ()
    summary: str = "检测到需要复核的隐私风险，具体内容已隐藏。"

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "category": self.category,
            "redacted_path": self.redacted_path,
            "count": self.count,
            "line_numbers": list(self.line_numbers),
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class PrivacyScanReport:
    status: str
    root_name: str
    scanned_files: int
    text_files_scanned: int
    image_files_checked: int
    ocr_images_scanned: int
    skipped_files: int
    findings: tuple[PrivacyFinding, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def has_blockers(self) -> bool:
        return any(item.severity == "blocked" for item in self.findings)

    @property
    def category_counts(self) -> dict[str, int]:
        totals: Counter[str] = Counter()
        for item in self.findings:
            totals[item.category] += item.count
        return dict(sorted(totals.items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "root_name": self.root_name,
            "scanned_files": self.scanned_files,
            "text_files_scanned": self.text_files_scanned,
            "image_files_checked": self.image_files_checked,
            "ocr_images_scanned": self.ocr_images_scanned,
            "skipped_files": self.skipped_files,
            "has_blockers": self.has_blockers,
            "category_counts": self.category_counts,
            "findings": [item.to_dict() for item in self.findings],
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class ShareCopyOptions:
    """Explicit selection for a local share copy.

    ``public_sources`` must contain paths relative to the product data root.
    Nothing is selected by default.  ``require_clean_scan`` is retained for
    compatibility, but setting it to ``False`` does not bypass credentials,
    personal data, parser failures, truncated scans or uninspected binary
    content.  Only files whose selected bytes were actually inspected can be
    included.
    """

    include_notes: bool = False
    public_sources: tuple[str, ...] = ()
    scan_images_with_ocr: bool = False
    require_clean_scan: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_sources", tuple(str(item) for item in self.public_sources))


@dataclass(frozen=True, slots=True)
class ShareCopyReport:
    status: str
    destination: Path
    file_count: int
    total_bytes: int
    manifest_path: Path
    manifest_sha256: str
    privacy_report: PrivacyScanReport

    def to_dict(self, *, include_destination: bool = False) -> dict[str, object]:
        value: dict[str, object] = {
            "status": self.status,
            "destination_name": _safe_root_name(self.destination),
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "manifest_name": self.manifest_path.name,
            "manifest_sha256": self.manifest_sha256,
            "privacy_report": self.privacy_report.to_dict(),
        }
        if include_destination:
            value["destination"] = str(self.destination)
            value["manifest_path"] = str(self.manifest_path)
        return value


class PrivacyViolationError(ValueError):
    """Raised when a default-safe share is blocked by the redacted scan."""

    def __init__(self, report: PrivacyScanReport):
        self.report = report
        super().__init__("隐私扫描未达到安全分享标准；请查看脱敏报告后调整所选资料")


def _share_scan_is_acceptable(report: PrivacyScanReport, *, require_clean: bool) -> bool:
    if report.has_blockers:
        return False
    if report.status == "clean":
        return True
    if require_clean or not report.findings:
        return False
    return all(
        item.severity == "review" and item.category in _EXPERT_REVIEW_ALLOWLIST
        for item in report.findings
    )


@dataclass(slots=True)
class _ScanAccumulator:
    root_name: str
    scanned_files: int = 0
    text_files_scanned: int = 0
    image_files_checked: int = 0
    ocr_images_scanned: int = 0
    skipped_files: int = 0
    findings: list[PrivacyFinding] = field(default_factory=list)
    limitations: set[str] = field(default_factory=lambda: {_GENERAL_LIMITATION})

    def add(
        self,
        *,
        severity: str,
        category: str,
        relative_path: PurePosixPath,
        count: int = 1,
        line_numbers: Iterable[int] = (),
        summary: str,
    ) -> None:
        self.findings.append(
            PrivacyFinding(
                severity=severity,
                category=category,
                redacted_path=_redacted_report_path(relative_path),
                count=max(1, int(count)),
                line_numbers=tuple(line_numbers)[:_MAX_REPORTED_LINE_NUMBERS],
                summary=summary,
            )
        )

    def finish(self) -> PrivacyScanReport:
        findings = tuple(
            sorted(
                self.findings,
                key=lambda item: (
                    0 if item.severity == "blocked" else 1,
                    item.redacted_path,
                    item.category,
                ),
            )
        )
        if any(item.severity == "blocked" for item in findings):
            status = "blocked"
        elif findings:
            status = "review"
        else:
            status = "clean"
        return PrivacyScanReport(
            status=status,
            root_name=self.root_name,
            scanned_files=self.scanned_files,
            text_files_scanned=self.text_files_scanned,
            image_files_checked=self.image_files_checked,
            ocr_images_scanned=self.ocr_images_scanned,
            skipped_files=self.skipped_files,
            findings=findings,
            limitations=tuple(sorted(self.limitations)),
        )


def _is_secret_like_filename(name: str) -> bool:
    lower = name.casefold()
    suffix = Path(lower).suffix
    if lower == ".env" or lower.startswith(".env."):
        return True
    if suffix in {".key", ".keystore", ".jks", ".p12", ".pem", ".pfx"}:
        return True
    if lower in {
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "id_dsa",
        "id_ed25519",
        "id_ecdsa",
        "id_rsa",
        "providers.json",
    }:
        return True
    stem = Path(lower).stem.replace("-", "_")
    config_suffixes = {"", ".cfg", ".conf", ".ini", ".json", ".toml", ".yaml", ".yml"}
    if suffix in config_suffixes and stem in {
        "api_key",
        "api_keys",
        "credential",
        "credentials",
        "secret",
        "secrets",
        "service_account",
        "settings",
        "token",
        "tokens",
    }:
        return True
    if lower.startswith("service-account") and suffix == ".json":
        return True
    if lower.startswith("cookies") and suffix in {"", ".db", ".json", ".sqlite", ".txt"}:
        return True
    return False


def _is_link_or_reparse(info: os.stat_result) -> bool:
    """Treat Windows junctions/reparse points like symbolic links."""

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(reparse_flag and attributes & reparse_flag)


def _component_contains_sensitive_literal(component: str) -> bool:
    if _is_secret_like_filename(component):
        return True
    return any(
        rule.category
        in {
            "provider_api_key",
            "github_token",
            "gitlab_token",
            "huggingface_token",
            "slack_token",
            "aws_access_key",
            "google_api_key",
            "jwt_token",
            "email_address",
            "phone_number",
        }
        and rule.pattern.search(component)
        for rule in _CONTENT_RULES
    )


def _path_contains_sensitive_literal(path: PurePosixPath) -> bool:
    return any(_component_contains_sensitive_literal(part) for part in path.parts)


def _redacted_report_path(path: PurePosixPath) -> str:
    raw = path.as_posix()
    if _path_contains_sensitive_literal(path):
        digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"<敏感路径-{digest}>"
    safe_parts: list[str] = []
    for part in path.parts:
        cleaned = "".join(character for character in part if character.isprintable())
        safe_parts.append(cleaned[:120] or "<空名称>")
    return "/".join(safe_parts) or "<根目录>"


def _safe_root_name(path: Path) -> str:
    name = path.name or "share-copy"
    pure = PurePosixPath(name)
    if _path_contains_sensitive_literal(pure):
        return "<已脱敏目录>"
    cleaned = "".join(character for character in name if character.isprintable())
    return cleaned[:120] or "share-copy"


def _line_numbers(text: str, starts: Sequence[int]) -> tuple[int, ...]:
    newlines = [index for index, character in enumerate(text) if character == "\n"]
    values = sorted(
        {
            bisect.bisect_right(newlines, start) + 1
            for start in starts[:_MAX_REPORTED_LINE_NUMBERS]
        }
    )
    return tuple(values[:_MAX_REPORTED_LINE_NUMBERS])


def _scan_text(
    text: str,
    relative_path: PurePosixPath,
    accumulator: _ScanAccumulator,
    *,
    category_prefix: str = "",
) -> None:
    for rule in _CONTENT_RULES:
        count = 0
        starts: list[int] = []
        for match in rule.pattern.finditer(text):
            count += 1
            if len(starts) < _MAX_REPORTED_LINE_NUMBERS:
                starts.append(match.start())
        if not count:
            continue
        summary = rule.summary
        if category_prefix == "image_ocr_":
            summary = "图片 OCR 文字中检测到敏感信息，具体内容已隐藏。"
        elif category_prefix == "image_metadata_":
            summary = "图片元数据中检测到敏感信息，具体内容已隐藏。"
        accumulator.add(
            severity=rule.severity,
            category=f"{category_prefix}{rule.category}",
            relative_path=relative_path,
            count=count,
            line_numbers=_line_numbers(text, starts),
            summary=summary,
        )


def _read_regular_file(path: Path, *, limit: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY  # type: ignore[attr-defined]
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("不是普通文件")
        if info.st_size > limit:
            raise OverflowError("文件过大")
        values: list[bytes] = []
        total = 0
        while block := os.read(descriptor, min(COPY_BUFFER_BYTES, limit + 1 - total)):
            total += len(block)
            if total > limit:
                raise OverflowError("文件过大")
            values.append(block)
        return b"".join(values)
    finally:
        os.close(descriptor)


def _looks_textual(payload: bytes) -> bool:
    if not payload:
        return True
    encodings = ["utf-8-sig"]
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings.insert(0, "utf-16")
    text: str | None = None
    for encoding in encodings:
        try:
            text = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None or "\x00" in text:
        return False
    # Inspect the complete bounded payload, not only its prefix.  This rejects
    # a valid Markdown/JSON prefix followed by opaque compressed or binary data.
    printable = sum(character.isprintable() or character in "\r\n\t" for character in text)
    return not text or printable / len(text) >= 0.95


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _coerce_ocr_text(value: Any) -> tuple[str, bool]:
    pieces: list[str] = []
    characters = 0
    truncated = False

    def visit(item: Any, depth: int = 0) -> None:
        nonlocal characters, truncated
        if truncated or depth > 8 or item is None:
            return
        if isinstance(item, str):
            remaining = MAX_OCR_CHARACTERS - characters
            if remaining <= 0:
                truncated = True
                return
            pieces.append(item[:remaining])
            characters += min(len(item), remaining)
            if len(item) > remaining:
                truncated = True
            return
        if isinstance(item, Mapping):
            preferred = ("text", "markdown", "content", "transcript", "lines")
            selected = [item[key] for key in preferred if key in item]
            for child in selected or list(item.values()):
                visit(child, depth + 1)
            return
        if isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray, memoryview)):
            for child in item:
                visit(child, depth + 1)

    visit(value)
    return "\n".join(pieces), truncated


def _scan_image(
    path: Path,
    relative_path: PurePosixPath,
    accumulator: _ScanAccumulator,
    *,
    enable_image_ocr: bool,
    ocr_reader: Callable[[Path], Any] | None,
) -> None:
    accumulator.image_files_checked += 1
    accumulator.add(
        severity="review",
        category="image_original_container_not_shareable",
        relative_path=relative_path,
        summary="原始图片容器可能携带无法完整验证的数据，当前版本不会将其加入分享副本。",
    )
    metadata_text: list[str] = []
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        accumulator.add(
            severity="review",
            category="image_metadata_unavailable",
            relative_path=relative_path,
            summary="未安装图片元数据读取组件，EXIF 未检查。",
        )
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(path) as image:
                    exif = image.getexif()
                    if exif:
                        accumulator.add(
                            severity="review",
                            category="image_exif",
                            relative_path=relative_path,
                            count=len(exif),
                            summary="图片包含 EXIF 元数据；具体字段和值已隐藏。",
                        )
                        if 34853 in exif:
                            accumulator.add(
                                severity="blocked",
                                category="image_gps_exif",
                                relative_path=relative_path,
                                summary="图片包含 GPS EXIF 字段；具体位置已隐藏。",
                            )
                        for value in exif.values():
                            if isinstance(value, str):
                                metadata_text.append(value)
                    for value in image.info.values():
                        if isinstance(value, str):
                            metadata_text.append(value)
        except (OSError, ValueError, RuntimeError, Warning):
            accumulator.add(
                severity="review",
                category="image_metadata_unreadable",
                relative_path=relative_path,
                summary="图片无法安全读取，EXIF/元数据检查未完成。",
            )
    if metadata_text:
        _scan_text(
            "\n".join(metadata_text)[:MAX_OCR_CHARACTERS],
            relative_path,
            accumulator,
            category_prefix="image_metadata_",
        )

    if not enable_image_ocr:
        accumulator.limitations.add(_IMAGE_LIMITATION_NO_OCR)
        accumulator.add(
            severity="review",
            category="image_text_unscanned",
            relative_path=relative_path,
            summary="图片像素文字未做 OCR；需要本地 OCR 后再安全分享。",
        )
        accumulator.skipped_files += 1
        return

    accumulator.limitations.add(_IMAGE_LIMITATION_WITH_OCR)
    if ocr_reader is None:
        accumulator.add(
            severity="review",
            category="image_ocr_unavailable",
            relative_path=relative_path,
            summary="已请求图片 OCR，但没有可用的本地 OCR 读取器。",
        )
        accumulator.skipped_files += 1
        return
    try:
        raw_value = ocr_reader(path)
        text, truncated = _coerce_ocr_text(raw_value)
    except Exception:
        # Callback exceptions may contain OCR text or absolute paths.  Never
        # place their message in a report or log payload.
        accumulator.add(
            severity="review",
            category="image_ocr_failed",
            relative_path=relative_path,
            summary="本地 OCR 执行失败；错误细节已隐藏。",
        )
        accumulator.skipped_files += 1
        return
    if isinstance(raw_value, Mapping):
        engine = str(raw_value.get("engine") or "").strip().casefold()
        if engine in {"none", "unavailable", "failed", "error"}:
            accumulator.add(
                severity="review",
                category="image_ocr_unavailable",
                relative_path=relative_path,
                summary="本地 OCR 没有可用识别引擎，图片文字未完成扫描。",
            )
            accumulator.skipped_files += 1
            return
    accumulator.ocr_images_scanned += 1
    if truncated:
        accumulator.add(
            severity="review",
            category="image_ocr_truncated",
            relative_path=relative_path,
            summary="OCR 返回内容超过安全扫描上限，结果不完整。",
        )
    if text:
        _scan_text(text, relative_path, accumulator, category_prefix="image_ocr_")
    else:
        # An empty OCR response is ambiguous: it may be a truly blank image,
        # an unsupported image, or a silent engine failure.  A privacy scanner
        # must never convert that ambiguity into a clean sharing decision.
        accumulator.add(
            severity="review",
            category="image_ocr_empty",
            relative_path=relative_path,
            summary="本地 OCR 未返回文字，无法确认图片像素中不存在隐私内容。",
        )
        accumulator.skipped_files += 1


def _add_document_review(
    accumulator: _ScanAccumulator,
    relative_path: PurePosixPath,
    *,
    category: str,
    summary: str,
    count: int = 1,
) -> None:
    """Record an uninspected document surface without exposing parser details."""

    accumulator.add(
        severity="review",
        category=category,
        relative_path=relative_path,
        count=count,
        summary=summary,
    )


def _scan_document_raw_bytes(
    path: Path,
    relative_path: PurePosixPath,
    accumulator: _ScanAccumulator,
    *,
    document_kind: str,
    decode_pdf_strings: bool = False,
) -> bool:
    """Scan bounded container bytes, including comments and unreferenced tails.

    Structured parsers intentionally ignore bytes that are not reachable from a
    PDF cross-reference table or ZIP central directory.  Those bytes are still
    copied into a share, so they must be scanned independently.  The views here
    cover ordinary byte strings, UTF-16 LE/BE text and common PDF hex/octal
    string encodings.  Returning ``False`` leaves a non-overridable finding.
    """

    kind_label = "PDF" if document_kind == "pdf" else "Office"
    try:
        payload = _read_regular_file(path, limit=MAX_RAW_DOCUMENT_BYTES)
    except OverflowError:
        accumulator.skipped_files += 1
        _add_document_review(
            accumulator,
            relative_path,
            category=f"{document_kind}_raw_scan_limit_exceeded",
            summary=f"{kind_label} 原始容器超过本地逐字节扫描上限，原件未加入分享副本。",
        )
        return False
    except (OSError, ValueError):
        accumulator.skipped_files += 1
        _add_document_review(
            accumulator,
            relative_path,
            category=f"{document_kind}_raw_content_unreadable",
            summary=f"{kind_label} 原始容器无法安全读取，原件未加入分享副本。",
        )
        return False

    def scan_byte_views(value: bytes) -> None:
        # Latin-1 is a lossless one-byte view and therefore catches every ASCII
        # credential in comments, unused objects and bytes appended after EOF.
        _scan_text(value.decode("latin-1"), relative_path, accumulator)
        # A UTF-16 value can begin on either byte parity inside a container.
        # Scan both alignments instead of assuming that the file offset is even.
        for encoding in ("utf-16-le", "utf-16-be"):
            for offset in (0, 1):
                if len(value) - offset >= 2:
                    _scan_text(value[offset:].decode(encoding, errors="ignore"), relative_path, accumulator)

    scan_byte_views(payload)

    if not decode_pdf_strings:
        return True

    decoded_bytes = 0
    encoded_scan_incomplete = False

    # PDF hex strings may store both ordinary and UTF-16 text.  Decode only
    # bracketed strings and cap aggregate output to avoid adversarial expansion.
    for match in re.finditer(rb"<([0-9A-Fa-f\x09\x0a\x0c\x0d\x20]{16,})>", payload):
        compact = re.sub(rb"\s+", b"", match.group(1))
        if len(compact) % 2:
            compact += b"0"
        try:
            decoded = bytes.fromhex(compact.decode("ascii"))
        except (UnicodeError, ValueError):
            continue
        decoded_bytes += len(decoded)
        if decoded_bytes > MAX_DECODED_CONTAINER_BYTES:
            encoded_scan_incomplete = True
            break
        scan_byte_views(decoded)

    # Literal PDF strings can octal-escape every credential byte.  Applying the
    # bounded substitution to the raw payload also covers unused objects.
    def decode_octal(match: re.Match[bytes]) -> bytes:
        try:
            return bytes((int(match.group(1), 8) & 0xFF,))
        except (TypeError, ValueError):
            return match.group(0)

    unescaped = re.sub(rb"\\([0-7]{1,3})", decode_octal, payload)
    unescaped = re.sub(rb"\\([\\()])", rb"\1", unescaped)
    _scan_text(unescaped.decode("latin-1"), relative_path, accumulator)
    if encoded_scan_incomplete:
        _add_document_review(
            accumulator,
            relative_path,
            category="pdf_encoded_text_scan_limit_exceeded",
            summary="PDF 编码字符串超过本地隐私扫描上限，原件未加入分享副本。",
        )
        return False
    return True


def _scan_pdf(
    path: Path,
    relative_path: PurePosixPath,
    accumulator: _ScanAccumulator,
    *,
    enable_image_ocr: bool,
    ocr_reader: Callable[[Path], Any] | None,
    max_text_bytes: int,
) -> None:
    """Scan extractable PDF text, metadata and optionally rendered visual pages."""

    accumulator.limitations.add(_PDF_LIMITATION)
    # A PDF can retain unreachable objects, incremental-update history and
    # arbitrarily filtered streams that neither the structured parser nor a
    # bounded raw-text view can prove safe.  We therefore inspect what we can
    # for a useful redacted report, but never authorize copying the original
    # container.  A future sharing path must render a new sanitized artifact.
    _add_document_review(
        accumulator,
        relative_path,
        category="pdf_original_container_not_shareable",
        summary="原始 PDF 容器可能包含解析器不可达的数据；请先导出为脱敏文本，未来版本将支持渲染净化副本。",
    )
    _scan_document_raw_bytes(
        path,
        relative_path,
        accumulator,
        document_kind="pdf",
        decode_pdf_strings=True,
    )
    try:
        import pymupdf  # type: ignore
    except ImportError:
        accumulator.skipped_files += 1
        _add_document_review(
            accumulator,
            relative_path,
            category="pdf_parser_unavailable",
            summary="本地 PDF 解析组件不可用，原件未加入分享副本。",
        )
        return

    scanned_bytes = 0
    truncated = False

    def scan_piece(value: object) -> None:
        nonlocal scanned_bytes, truncated
        if value is None or truncated:
            return
        text = str(value)
        if not text:
            return
        encoded = text.encode("utf-8", errors="replace")
        remaining = max_text_bytes - scanned_bytes
        if remaining <= 0:
            truncated = True
            return
        if len(encoded) > remaining:
            encoded = encoded[:remaining]
            text = encoded.decode("utf-8", errors="ignore")
            truncated = True
        scanned_bytes += len(encoded)
        if text:
            _scan_text(text, relative_path, accumulator)

    try:
        with pymupdf.open(path) as document:
            if bool(getattr(document, "needs_pass", False)):
                accumulator.skipped_files += 1
                _add_document_review(
                    accumulator,
                    relative_path,
                    category="pdf_encrypted_content",
                    summary="PDF 已加密，无法验证内部是否包含凭据或私人内容。",
                )
                return
            page_count = int(getattr(document, "page_count", 0))
            if page_count > MAX_DOCUMENT_PAGES:
                accumulator.skipped_files += 1
                _add_document_review(
                    accumulator,
                    relative_path,
                    category="pdf_page_limit_exceeded",
                    summary="PDF 页数超过本地隐私扫描上限，内容未完整检查。",
                )
                return

            metadata = getattr(document, "metadata", None)
            if isinstance(metadata, Mapping):
                scan_piece("\n".join(f"{key}: {value}" for key, value in metadata.items()))
            try:
                scan_piece(document.get_xml_metadata())
            except Exception:
                _add_document_review(
                    accumulator,
                    relative_path,
                    category="pdf_metadata_unreadable",
                    summary="PDF 的部分元数据无法读取，原件未加入分享副本。",
                )
            try:
                table_of_contents = document.get_toc(simple=True)
            except Exception:
                table_of_contents = ()
                _add_document_review(
                    accumulator,
                    relative_path,
                    category="pdf_outline_unreadable",
                    summary="PDF 目录无法读取，原件未加入分享副本。",
                )
            scan_piece(table_of_contents)

            visual_pages: list[int] = []
            for page_number in range(page_count):
                page = document.load_page(page_number)
                page_text = page.get_text("text")
                scan_piece(page_text)
                # A compact word stream catches secrets split over adjacent PDF
                # text spans without including the secret itself in diagnostics.
                words = page.get_text("words")
                scan_piece("".join(str(word[4]) for word in words if len(word) > 4))

                try:
                    links = page.get_links()
                except Exception:
                    links = ()
                    _add_document_review(
                        accumulator,
                        relative_path,
                        category="pdf_links_unreadable",
                        summary="PDF 的部分链接无法读取，原件未加入分享副本。",
                    )
                scan_piece(links)

                try:
                    annotation = page.first_annot
                    annotation_count = 0
                    while annotation is not None and annotation_count < MAX_DOCUMENT_MEMBERS:
                        scan_piece(getattr(annotation, "info", None))
                        annotation = annotation.next
                        annotation_count += 1
                    if annotation is not None:
                        _add_document_review(
                            accumulator,
                            relative_path,
                            category="pdf_annotation_limit_exceeded",
                            summary="PDF 批注数量超过隐私扫描上限。",
                        )
                except Exception:
                    _add_document_review(
                        accumulator,
                        relative_path,
                        category="pdf_annotations_unreadable",
                        summary="PDF 的部分批注无法读取，原件未加入分享副本。",
                    )

                try:
                    widget = page.first_widget
                    widget_count = 0
                    while widget is not None and widget_count < MAX_DOCUMENT_MEMBERS:
                        scan_piece(
                            {
                                "name": getattr(widget, "field_name", None),
                                "label": getattr(widget, "field_label", None),
                                "value": getattr(widget, "field_value", None),
                            }
                        )
                        widget = widget.next
                        widget_count += 1
                    if widget is not None:
                        _add_document_review(
                            accumulator,
                            relative_path,
                            category="pdf_form_limit_exceeded",
                            summary="PDF 表单字段数量超过隐私扫描上限。",
                        )
                except Exception:
                    _add_document_review(
                        accumulator,
                        relative_path,
                        category="pdf_forms_unreadable",
                        summary="PDF 的部分表单字段无法读取，原件未加入分享副本。",
                    )

                try:
                    has_visual_content = bool(page.get_images(full=True) or page.get_drawings())
                except Exception:
                    has_visual_content = True
                if has_visual_content or not str(page_text).strip():
                    visual_pages.append(page_number)

            try:
                embedded_names = tuple(document.embfile_names())
            except Exception:
                embedded_names = ()
                _add_document_review(
                    accumulator,
                    relative_path,
                    category="pdf_attachments_unreadable",
                    summary="PDF 附件清单无法读取，原件未加入分享副本。",
                )
            if len(embedded_names) > MAX_DOCUMENT_MEMBERS:
                _add_document_review(
                    accumulator,
                    relative_path,
                    category="pdf_attachment_limit_exceeded",
                    summary="PDF 附件数量超过隐私扫描上限。",
                )
                embedded_names = embedded_names[:MAX_DOCUMENT_MEMBERS]
            for name in embedded_names:
                scan_piece(name)
                try:
                    payload = document.embfile_get(name)
                except Exception:
                    _add_document_review(
                        accumulator,
                        relative_path,
                        category="pdf_attachment_unreadable",
                        summary="PDF 中存在无法读取的附件。",
                    )
                    continue
                if len(payload) > max_text_bytes or not _looks_textual(payload):
                    _add_document_review(
                        accumulator,
                        relative_path,
                        category="pdf_attachment_unscanned",
                        summary="PDF 中存在无法验证内容的附件。",
                    )
                else:
                    scan_piece(_decode_text(payload))

            if visual_pages:
                if not enable_image_ocr or ocr_reader is None:
                    accumulator.skipped_files += 1
                    _add_document_review(
                        accumulator,
                        relative_path,
                        category="pdf_visual_content_unscanned",
                        count=len(visual_pages),
                        summary="PDF 含扫描页、图片或矢量图，未完成本地 OCR，原件未加入分享副本。",
                    )
                else:
                    with tempfile.TemporaryDirectory(prefix="ai-jingjing-pdf-privacy-") as temporary:
                        temporary_root = Path(temporary)
                        for page_number in visual_pages:
                            page = document.load_page(page_number)
                            scale = 1.5
                            rect = page.rect
                            estimated_pixels = float(rect.width) * scale * float(rect.height) * scale
                            if estimated_pixels > MAX_PDF_RENDER_PIXELS:
                                _add_document_review(
                                    accumulator,
                                    relative_path,
                                    category="pdf_page_render_limit_exceeded",
                                    summary="PDF 页面尺寸超过安全 OCR 上限。",
                                )
                                continue
                            try:
                                pixmap = page.get_pixmap(
                                    matrix=pymupdf.Matrix(scale, scale),
                                    alpha=False,
                                )
                                rendered = temporary_root / f"page-{page_number + 1}.png"
                                pixmap.save(rendered)
                                _scan_image(
                                    rendered,
                                    relative_path,
                                    accumulator,
                                    enable_image_ocr=True,
                                    ocr_reader=ocr_reader,
                                )
                            except Exception:
                                _add_document_review(
                                    accumulator,
                                    relative_path,
                                    category="pdf_visual_content_unreadable",
                                    summary="PDF 的部分视觉内容无法完成本地 OCR。",
                                )
            accumulator.text_files_scanned += 1
    except Exception:
        accumulator.skipped_files += 1
        _add_document_review(
            accumulator,
            relative_path,
            category="pdf_content_unscanned",
            summary="PDF 无法安全解析，原件未加入分享副本。",
        )
        return

    if truncated:
        accumulator.skipped_files += 1
        _add_document_review(
            accumulator,
            relative_path,
            category="pdf_text_too_large",
            summary="PDF 可提取文字超过隐私扫描上限，内容未完整检查。",
        )


_OPEN_OFFICE_EXTENSIONS = {
    ".docm",
    ".docx",
    ".odp",
    ".ods",
    ".odt",
    ".pptm",
    ".pptx",
    ".xlsm",
    ".xlsx",
}
_OFFICE_TEXT_MEMBER_EXTENSIONS = {".csv", ".html", ".json", ".rels", ".txt", ".vml", ".xhtml", ".xml"}


def _office_member_path_is_safe(name: str) -> bool:
    if not name or "\x00" in name or "\\" in name:
        return False
    member = PurePosixPath(name)
    return not member.is_absolute() and all(part not in {"", ".", ".."} for part in member.parts)


def _scan_office_xml(
    payload: bytes,
    relative_path: PurePosixPath,
    accumulator: _ScanAccumulator,
) -> bool:
    raw = _decode_text(payload)
    _scan_text(raw, relative_path, accumulator)
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", raw, re.IGNORECASE):
        _add_document_review(
            accumulator,
            relative_path,
            category="office_xml_unsafe",
            summary="Office 文档包含不安全的 XML 声明，无法完成可靠扫描。",
        )
        return False
    try:
        root = ElementTree.fromstring(payload)
    except (ElementTree.ParseError, ValueError, UnicodeError):
        _add_document_review(
            accumulator,
            relative_path,
            category="office_xml_unreadable",
            summary="Office 文档中的部分 XML 无法解析，原件未加入分享副本。",
        )
        return False
    text_nodes = [html.unescape(str(value)) for value in root.itertext() if str(value)]
    attributes = [
        html.unescape(str(value))
        for element in root.iter()
        for value in element.attrib.values()
        if str(value)
    ]
    # Scan both delimited and contiguous forms so a key split across adjacent
    # Office runs cannot evade the detector.
    values = text_nodes + attributes
    if values:
        _scan_text("\n".join(values) + "\n" + "".join(values), relative_path, accumulator)
    return True


def _scan_office(
    path: Path,
    relative_path: PurePosixPath,
    accumulator: _ScanAccumulator,
    *,
    enable_image_ocr: bool,
    ocr_reader: Callable[[Path], Any] | None,
    max_text_bytes: int,
) -> None:
    """Safely inspect OOXML/ODF text and metadata without extracting the ZIP."""

    accumulator.limitations.add(_OFFICE_LIMITATION)
    # ZIP readers ignore bytes after the central directory and compressed
    # payloads can use encodings outside this scanner's bounded decoder set.
    # Continue parsing for useful findings, but never copy the original Office
    # container into a share until a sanitized rebuild/export path exists.
    _add_document_review(
        accumulator,
        relative_path,
        category="office_original_container_not_shareable",
        summary="原始 Office/ODF 容器可能包含目录外或无法验证的数据；请先导出为脱敏文本。",
    )
    _scan_document_raw_bytes(
        path,
        relative_path,
        accumulator,
        document_kind="office",
    )
    suffix = path.suffix.casefold()
    if suffix not in _OPEN_OFFICE_EXTENSIONS:
        accumulator.skipped_files += 1
        _add_document_review(
            accumulator,
            relative_path,
            category="office_binary_unscanned",
            summary="此 Office 格式不是可安全检查的开放文档包；请先导出为脱敏文本或新版 Office 格式。",
        )
        return

    parsed_text_members = 0
    scanned_payload_bytes = 0
    binary_members = 0
    try:
        with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory(
            prefix="ai-jingjing-office-privacy-"
        ) as temporary:
            members = archive.infolist()
            if len(members) > MAX_DOCUMENT_MEMBERS:
                accumulator.skipped_files += 1
                _add_document_review(
                    accumulator,
                    relative_path,
                    category="office_member_limit_exceeded",
                    summary="Office 文档包的内部文件数量超过隐私扫描上限。",
                )
                return
            temporary_root = Path(temporary)
            for index, info in enumerate(members):
                if info.is_dir():
                    continue
                if not _office_member_path_is_safe(info.filename):
                    _add_document_review(
                        accumulator,
                        relative_path,
                        category="office_package_unsafe",
                        summary="Office 文档包包含不安全的内部路径。",
                    )
                    continue
                _scan_text(info.filename, relative_path, accumulator)
                if info.flag_bits & 0x1:
                    _add_document_review(
                        accumulator,
                        relative_path,
                        category="office_encrypted_content",
                        summary="Office 文档包含加密内容，无法验证是否含有凭据或私人信息。",
                    )
                    continue
                member = PurePosixPath(info.filename)
                member_suffix = member.suffix.casefold()
                member_name = member.name.casefold()
                is_text_member = (
                    member_name in {"mimetype", ".rels"}
                    or member_suffix in _OFFICE_TEXT_MEMBER_EXTENSIONS
                )
                if is_text_member:
                    remaining = max_text_bytes - scanned_payload_bytes
                    if info.file_size > remaining or remaining <= 0:
                        _add_document_review(
                            accumulator,
                            relative_path,
                            category="office_text_too_large",
                            summary="Office 文档可提取内容超过隐私扫描上限。",
                        )
                        continue
                    with archive.open(info, "r") as stream:
                        payload = stream.read(remaining + 1)
                    if len(payload) > remaining:
                        _add_document_review(
                            accumulator,
                            relative_path,
                            category="office_text_too_large",
                            summary="Office 文档可提取内容超过隐私扫描上限。",
                        )
                        continue
                    scanned_payload_bytes += len(payload)
                    parsed_text_members += 1
                    if member_suffix in {".xml", ".rels", ".vml"}:
                        _scan_office_xml(payload, relative_path, accumulator)
                    else:
                        _scan_text(_decode_text(payload), relative_path, accumulator)
                    continue

                if member_suffix in _IMAGE_EXTENSIONS:
                    if not enable_image_ocr or ocr_reader is None:
                        binary_members += 1
                        continue
                    if info.file_size > max_text_bytes:
                        _add_document_review(
                            accumulator,
                            relative_path,
                            category="office_media_too_large",
                            summary="Office 文档中的图片超过本地 OCR 安全上限。",
                        )
                        continue
                    with archive.open(info, "r") as stream:
                        payload = stream.read(max_text_bytes + 1)
                    if len(payload) > max_text_bytes:
                        _add_document_review(
                            accumulator,
                            relative_path,
                            category="office_media_too_large",
                            summary="Office 文档中的图片超过本地 OCR 安全上限。",
                        )
                        continue
                    rendered = temporary_root / f"member-{index}{member_suffix}"
                    rendered.write_bytes(payload)
                    _scan_image(
                        rendered,
                        relative_path,
                        accumulator,
                        enable_image_ocr=True,
                        ocr_reader=ocr_reader,
                    )
                    continue

                # Macros, OLE embeddings, fonts and every other opaque package
                # member remain non-overridable because their bytes were not
                # semantically inspected.
                binary_members += 1
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile):
        accumulator.skipped_files += 1
        _add_document_review(
            accumulator,
            relative_path,
            category="office_content_unscanned",
            summary="Office 文档无法安全解析，原件未加入分享副本。",
        )
        return

    if parsed_text_members == 0:
        accumulator.skipped_files += 1
        _add_document_review(
            accumulator,
            relative_path,
            category="office_content_unscanned",
            summary="Office 文档包未包含可验证的正文或元数据。",
        )
    else:
        accumulator.text_files_scanned += 1
    if binary_members:
        accumulator.skipped_files += 1
        _add_document_review(
            accumulator,
            relative_path,
            category="office_binary_content_unscanned",
            count=binary_members,
            summary="Office 文档包含未检查的宏、嵌入对象、媒体或其他二进制内容。",
        )


def _scan_regular_file(
    path: Path,
    relative_path: PurePosixPath,
    accumulator: _ScanAccumulator,
    *,
    enable_image_ocr: bool,
    ocr_reader: Callable[[Path], Any] | None,
    max_text_bytes: int,
) -> None:
    accumulator.scanned_files += 1
    if _path_contains_sensitive_literal(relative_path):
        accumulator.add(
            severity="blocked",
            category="sensitive_path",
            relative_path=relative_path,
            summary="文件路径本身包含疑似敏感信息，具体路径已脱敏。",
        )
    if _is_secret_like_filename(path.name):
        accumulator.add(
            severity="blocked",
            category="secret_like_file",
            relative_path=relative_path,
            summary="检测到凭据或密钥类文件名，具体路径已脱敏。",
        )

    suffix = path.suffix.casefold()
    if suffix in _IMAGE_EXTENSIONS:
        _scan_image(
            path,
            relative_path,
            accumulator,
            enable_image_ocr=enable_image_ocr,
            ocr_reader=ocr_reader,
        )
        return
    if suffix in _PDF_EXTENSIONS:
        _scan_pdf(
            path,
            relative_path,
            accumulator,
            enable_image_ocr=enable_image_ocr,
            ocr_reader=ocr_reader,
            max_text_bytes=max_text_bytes,
        )
        return
    if suffix in _AUDIO_EXTENSIONS or suffix in _VIDEO_EXTENSIONS:
        accumulator.limitations.add(_MEDIA_LIMITATION)
        accumulator.skipped_files += 1
        accumulator.add(
            severity="review",
            category="media_content_unscanned",
            relative_path=relative_path,
            summary="音视频内容未转写或逐帧扫描。",
        )
        return
    if suffix in _OFFICE_EXTENSIONS:
        _scan_office(
            path,
            relative_path,
            accumulator,
            enable_image_ocr=enable_image_ocr,
            ocr_reader=ocr_reader,
            max_text_bytes=max_text_bytes,
        )
        return
    if suffix in _ARCHIVE_EXTENSIONS:
        accumulator.limitations.add(_ARCHIVE_LIMITATION)
        accumulator.skipped_files += 1
        accumulator.add(
            severity="review",
            category="archive_content_unscanned",
            relative_path=relative_path,
            summary="压缩包或容器内部内容未扫描。",
        )
        return

    try:
        payload = _read_regular_file(path, limit=max_text_bytes)
    except OverflowError:
        accumulator.skipped_files += 1
        accumulator.add(
            severity="review",
            category="text_too_large",
            relative_path=relative_path,
            summary="文件超过文本隐私扫描大小上限，内容未完整检查。",
        )
        return
    except (OSError, ValueError):
        accumulator.skipped_files += 1
        accumulator.add(
            severity="review",
            category="file_unreadable",
            relative_path=relative_path,
            summary="文件无法安全读取，错误细节已隐藏。",
        )
        return

    if not _looks_textual(payload):
        accumulator.limitations.add(_BINARY_LIMITATION)
        accumulator.skipped_files += 1
        accumulator.add(
            severity="review",
            category="binary_content_unscanned",
            relative_path=relative_path,
            summary="未知二进制文件的内部内容未解析。",
        )
        return
    accumulator.text_files_scanned += 1
    _scan_text(_decode_text(payload), relative_path, accumulator)


def _discover_scan_files(root: Path) -> tuple[list[tuple[Path, PurePosixPath]], list[PurePosixPath], int]:
    files: list[tuple[Path, PurePosixPath]] = []
    symlinks: list[PurePosixPath] = []
    errors = 0
    try:
        root_info = root.lstat()
    except OSError:
        raise ValueError("隐私扫描目标不存在或不可访问") from None
    if _is_link_or_reparse(root_info):
        return [], [PurePosixPath(root.name or "<根目录>")], 0
    if stat.S_ISREG(root_info.st_mode):
        return [(root, PurePosixPath(root.name))], [], 0
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("隐私扫描目标必须是普通文件或目录")

    def onerror(_error: OSError) -> None:
        nonlocal errors
        errors += 1

    exhausted = False
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False, onerror=onerror):
        directory_path = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = directory_path / name
            relative = PurePosixPath(candidate.relative_to(root).as_posix())
            try:
                info = candidate.lstat()
            except OSError:
                errors += 1
                continue
            if _is_link_or_reparse(info):
                symlinks.append(relative)
            elif stat.S_ISDIR(info.st_mode):
                kept_directories.append(name)
            else:
                errors += 1
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            candidate = directory_path / name
            relative = PurePosixPath(candidate.relative_to(root).as_posix())
            try:
                info = candidate.lstat()
            except OSError:
                errors += 1
                continue
            if _is_link_or_reparse(info):
                symlinks.append(relative)
            elif stat.S_ISREG(info.st_mode):
                files.append((candidate, relative))
            else:
                errors += 1
            if len(files) + len(symlinks) >= MAX_SCAN_FILES:
                exhausted = True
                break
        if exhausted:
            directory_names[:] = []
            break
    if exhausted:
        errors += 1
    return files, symlinks, errors


def _scan_known_files(
    root: Path,
    files: Sequence[tuple[Path, PurePosixPath]],
    *,
    symlinks: Sequence[PurePosixPath] = (),
    discovery_errors: int = 0,
    enable_image_ocr: bool,
    ocr_reader: Callable[[Path], Any] | None,
    max_text_bytes: int,
) -> PrivacyScanReport:
    accumulator = _ScanAccumulator(root_name=_safe_root_name(root))
    for relative in symlinks:
        accumulator.skipped_files += 1
        accumulator.add(
            severity="blocked",
            category="symbolic_link",
            relative_path=relative,
            summary="检测到符号链接；为防止路径越界，未跟随该链接。",
        )
    if discovery_errors:
        accumulator.skipped_files += discovery_errors
        accumulator.add(
            severity="review",
            category="scan_incomplete",
            relative_path=PurePosixPath("<扫描范围>"),
            count=discovery_errors,
            summary="部分目录或文件无法安全枚举，扫描结果不完整。",
        )
    for path, relative in files:
        _scan_regular_file(
            path,
            relative,
            accumulator,
            enable_image_ocr=enable_image_ocr,
            ocr_reader=ocr_reader,
            max_text_bytes=max_text_bytes,
        )
    return accumulator.finish()


def scan_privacy(
    root: str | Path,
    *,
    enable_image_ocr: bool = False,
    ocr_reader: Callable[[Path], Any] | None = None,
    max_text_bytes: int = MAX_TEXT_BYTES,
) -> PrivacyScanReport:
    """Scan a local file/directory and return only redacted findings.

    ``ocr_reader``, when supplied, must perform local OCR and return text (or a
    nested structure containing text).  The module never transmits the image.
    PDF and open Office packages are parsed locally for extractable text and
    metadata.  Encrypted, malformed, truncated or otherwise uninspected content
    receives a non-overridable review finding.  Audio and video are not deeply
    parsed and therefore cannot be placed in a share copy.
    """

    if not isinstance(max_text_bytes, int) or not 1024 <= max_text_bytes <= 64 * 1024**2:
        raise ValueError("文本扫描大小上限必须在 1 KiB 到 64 MiB 之间")
    target = Path(root).expanduser()
    files, symlinks, errors = _discover_scan_files(target)
    return _scan_known_files(
        target,
        files,
        symlinks=symlinks,
        discovery_errors=errors,
        enable_image_ocr=enable_image_ocr,
        ocr_reader=ocr_reader,
        max_text_bytes=max_text_bytes,
    )


def _validate_manifest_path(raw: str) -> PurePosixPath:
    if not raw or "\x00" in raw or "\\" in raw:
        raise ValueError("所选公开资料路径不安全")
    path = PurePosixPath(raw)
    unsafe_component = any(
        not part
        or part in {".", ".."}
        or ":" in part
        or any(ord(character) < 32 for character in part)
        or part.rstrip(" .") != part
        or part.upper().split(".", 1)[0] in _WINDOWS_RESERVED_NAMES
        for part in path.parts
    )
    if path.is_absolute() or unsafe_component or path.as_posix() != raw:
        raise ValueError("所选公开资料路径不安全")
    return path


def _is_forbidden_share_path(relative: PurePosixPath) -> bool:
    if _path_contains_sensitive_literal(relative):
        return True
    lowered = [part.casefold() for part in relative.parts]
    forbidden_directories = {
        ".cache",
        ".git",
        ".obsidian",
        "backups",
        "cache",
        "caches",
        "chat",
        "chat_history",
        "chats",
        "conversation",
        "conversations",
        "keyring",
        "providers",
        "trash",
    }
    if any(part in forbidden_directories for part in lowered):
        return True
    name = lowered[-1]
    suffix = PurePosixPath(name).suffix
    if name in {".ds_store", "knowledge.db", "providers.json", "settings.json"}:
        return True
    if PurePosixPath(name).stem.replace("-", "_") in {
        "chat_history",
        "conversation",
        "conversations",
    }:
        return True
    if suffix in _DATABASE_EXTENSIONS:
        return True
    return _is_secret_like_filename(name)


def _ensure_source_component_safety(root: Path, relative: PurePosixPath) -> Path:
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        try:
            info = candidate.lstat()
        except OSError:
            raise ValueError("所选公开资料不存在或不可访问") from None
        if _is_link_or_reparse(info):
            raise ValueError("所选公开资料包含符号链接，已拒绝")
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        raise ValueError("所选公开资料路径越界，已拒绝") from None
    return candidate


def _walk_share_directory(
    root: Path,
    directory: Path,
    selected: dict[str, Path],
    *,
    strict_forbidden: bool,
) -> None:
    walk_error = False

    def onerror(_error: OSError) -> None:
        nonlocal walk_error
        walk_error = True

    for current, directory_names, file_names in os.walk(
        directory, topdown=True, followlinks=False, onerror=onerror
    ):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            relative = _validate_manifest_path(candidate.relative_to(root).as_posix())
            try:
                info = candidate.lstat()
            except OSError:
                raise ValueError("所选公开资料无法安全枚举") from None
            if _is_link_or_reparse(info):
                raise ValueError("所选公开资料包含符号链接，已拒绝")
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("所选公开资料包含特殊文件，已拒绝")
            if _is_forbidden_share_path(relative):
                if strict_forbidden:
                    raise ValueError("所选公开资料包含禁止分享的数据类型")
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            candidate = current_path / name
            relative = _validate_manifest_path(candidate.relative_to(root).as_posix())
            try:
                info = candidate.lstat()
            except OSError:
                raise ValueError("所选公开资料无法安全枚举") from None
            if _is_link_or_reparse(info):
                raise ValueError("所选公开资料包含符号链接，已拒绝")
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("所选公开资料包含特殊文件，已拒绝")
            if _is_forbidden_share_path(relative):
                if strict_forbidden:
                    raise ValueError("所选公开资料包含禁止分享的数据类型")
                continue
            selected[relative.as_posix()] = candidate
            if len(selected) > MAX_SHARE_FILES:
                raise ValueError("分享副本文件数量超过安全上限")
    if walk_error:
        raise ValueError("所选公开资料无法安全枚举")


def _collect_share_files(root: Path, options: ShareCopyOptions) -> list[tuple[Path, PurePosixPath]]:
    try:
        root_info = root.lstat()
    except OSError:
        raise ValueError("知识数据目录不存在或不可访问") from None
    if _is_link_or_reparse(root_info) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("知识数据目录必须是非符号链接的普通目录")

    selected: dict[str, Path] = {}
    if options.include_notes:
        notes_relative = _validate_manifest_path("notes")
        notes = root / "notes"
        if notes.exists():
            notes = _ensure_source_component_safety(root, notes_relative)
            try:
                info = notes.lstat()
            except OSError:
                raise ValueError("笔记目录不可访问") from None
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("笔记路径不是普通目录")
            _walk_share_directory(root, notes, selected, strict_forbidden=False)

    for raw in options.public_sources:
        relative = _validate_manifest_path(raw)
        if _is_forbidden_share_path(relative):
            raise ValueError("所选公开资料属于禁止分享的数据类型")
        candidate = _ensure_source_component_safety(root, relative)
        info = candidate.lstat()
        if stat.S_ISREG(info.st_mode):
            selected[relative.as_posix()] = candidate
        elif stat.S_ISDIR(info.st_mode):
            _walk_share_directory(root, candidate, selected, strict_forbidden=True)
        else:
            raise ValueError("所选公开资料包含特殊文件，已拒绝")
        if len(selected) > MAX_SHARE_FILES:
            raise ValueError("分享副本文件数量超过安全上限")

    casefolded: dict[str, str] = {}
    values: list[tuple[Path, PurePosixPath]] = []
    total_size = 0
    for relative_string, source in sorted(selected.items()):
        relative = _validate_manifest_path(relative_string)
        folded = relative_string.casefold()
        if folded in casefolded and casefolded[folded] != relative_string:
            raise ValueError("所选资料存在跨平台重名文件，已拒绝")
        casefolded[folded] = relative_string
        try:
            info = source.lstat()
        except OSError:
            raise ValueError("所选公开资料不可访问") from None
        if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
            raise ValueError("所选公开资料包含不安全文件类型")
        if info.st_size > MAX_SHARE_FILE_BYTES:
            raise ValueError("单个分享文件超过安全大小上限")
        total_size += info.st_size
        if total_size > MAX_SHARE_TOTAL_BYTES:
            raise ValueError("分享副本总大小超过安全上限")
        values.append((source, relative))
    return values


def _copy_file_and_hash(source: Path, destination: Path) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY  # type: ignore[attr-defined]
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    total = 0
    try:
        source_info = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_info.st_mode):
            raise ValueError("源文件不是普通文件")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_BINARY"):
            destination_flags |= os.O_BINARY  # type: ignore[attr-defined]
        destination_descriptor = os.open(destination, destination_flags, 0o600)
        try:
            while block := os.read(source_descriptor, COPY_BUFFER_BYTES):
                total += len(block)
                if total > MAX_SHARE_FILE_BYTES:
                    raise ValueError("单个分享文件超过安全大小上限")
                view = memoryview(block)
                while view:
                    written = os.write(destination_descriptor, view)
                    view = view[written:]
                digest.update(block)
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
        destination.chmod(0o600)
    finally:
        os.close(source_descriptor)
    return total, digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(COPY_BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_destination(source_root: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError("分享副本目标已存在；为防覆盖，请选择新目录")
    try:
        source_resolved = source_root.resolve(strict=True)
        destination_resolved = destination.resolve(strict=False)
    except OSError:
        raise ValueError("分享副本目标路径不可用") from None
    if destination_resolved == source_resolved or destination_resolved.is_relative_to(source_resolved):
        raise ValueError("分享副本目标不能位于知识数据目录内部")


def _publish_directory_no_replace(stage: Path, target: Path) -> None:
    """Atomically rename a directory while refusing a raced-in destination."""

    if os.name == "nt":
        # Windows rename is already no-replace for an existing destination.
        try:
            os.rename(stage, target)
        except FileExistsError:
            raise ValueError("分享副本目标已存在；未覆盖任何内容") from None
        except OSError:
            raise RuntimeError("分享副本无法安全发布") from None
        return

    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
            rename_exclusive = 0x00000004  # RENAME_EXCL from <stdio.h>
            operation = libc.renamex_np
            operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            operation.restype = ctypes.c_int
            result = operation(os.fsencode(stage), os.fsencode(target), rename_exclusive)
        elif hasattr(libc, "renameat2"):
            at_fdcwd = -100
            rename_no_replace = 1
            operation = libc.renameat2
            operation.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            operation.restype = ctypes.c_int
            result = operation(
                at_fdcwd,
                os.fsencode(stage),
                at_fdcwd,
                os.fsencode(target),
                rename_no_replace,
            )
        else:
            # This fallback retains atomic rename but cannot close the tiny
            # check/rename race on an uncommon POSIX runtime lacking both APIs.
            if target.exists() or target.is_symlink():
                raise ValueError("分享副本目标已存在；未覆盖任何内容")
            os.rename(stage, target)
            return
    except ValueError:
        raise
    except Exception:
        raise RuntimeError("分享副本无法安全发布") from None

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY, errno.EISDIR, errno.ENOTDIR}:
        raise ValueError("分享副本目标已存在；未覆盖任何内容")
    raise RuntimeError("分享副本无法安全发布")


def _ensure_destination_parent_is_real(directory: Path) -> None:
    """Reject an output parent reached through a symbolic-link component."""

    absolute = directory.absolute()
    existing: list[Path] = []
    cursor = absolute
    while True:
        if cursor.exists() or cursor.is_symlink():
            existing.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    for candidate in reversed(existing):
        try:
            info = candidate.lstat()
        except OSError:
            raise ValueError("分享副本目标路径不可安全访问") from None
        if _is_link_or_reparse(info):
            # macOS exposes a few root-owned compatibility aliases such as
            # /var -> /private/var; TemporaryDirectory commonly uses them.
            # They are fixed OS paths rather than a user-selected redirect.
            allowed_system_aliases = {Path("/etc"), Path("/tmp"), Path("/var")}
            if candidate not in allowed_system_aliases or getattr(info, "st_uid", -1) != 0:
                raise ValueError("分享副本目标路径包含符号链接，已拒绝")


def _verify_staged_files(stage: Path, records: Sequence[Mapping[str, object]]) -> None:
    expected = {str(item["path"]) for item in records}
    actual: set[str] = set()
    walk_error = False

    def onerror(_error: OSError) -> None:
        nonlocal walk_error
        walk_error = True

    for current, directory_names, file_names in os.walk(
        stage, topdown=True, followlinks=False, onerror=onerror
    ):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in directory_names:
            candidate = current_path / name
            try:
                info = candidate.lstat()
            except OSError:
                raise RuntimeError("分享副本校验无法安全枚举") from None
            if _is_link_or_reparse(info):
                raise RuntimeError("分享副本校验发现符号链接")
            if not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("分享副本校验发现特殊文件")
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in file_names:
            candidate = current_path / name
            relative = candidate.relative_to(stage).as_posix()
            try:
                info = candidate.lstat()
            except OSError:
                raise RuntimeError("分享副本校验无法安全枚举") from None
            if _is_link_or_reparse(info):
                raise RuntimeError("分享副本校验发现符号链接")
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError("分享副本校验发现特殊文件")
            if relative != "share_manifest.json":
                actual.add(relative)
    if walk_error:
        raise RuntimeError("分享副本校验无法安全枚举")
    if actual != expected:
        raise RuntimeError("分享副本文件清单不一致")
    for item in records:
        relative = _validate_manifest_path(str(item["path"]))
        path = stage.joinpath(*relative.parts)
        expected_size = item.get("size")
        expected_hash = item.get("sha256")
        try:
            path_info = path.lstat()
        except OSError:
            raise RuntimeError("分享副本文件完整性校验失败") from None
        if (
            not stat.S_ISREG(path_info.st_mode)
            or _is_link_or_reparse(path_info)
            or not isinstance(expected_size, int)
            or not isinstance(expected_hash, str)
            or not _SHA256_RE.fullmatch(expected_hash)
            or path.stat().st_size != expected_size
            or _sha256_file(path) != expected_hash
        ):
            raise RuntimeError("分享副本文件完整性校验失败")


def create_share_copy(
    source_root: str | Path,
    destination: str | Path,
    *,
    options: ShareCopyOptions | None = None,
    ocr_reader: Callable[[Path], Any] | None = None,
) -> ShareCopyReport:
    """Create an atomic, local-only share directory after a redacted scan.

    The destination must not exist.  Credentials, settings, databases, caches,
    conversations, keyrings and similar paths are forbidden even when named in
    ``public_sources``.  No upload or external delivery is performed.
    """

    selected_options = options or ShareCopyOptions()
    root = Path(source_root).expanduser()
    target = Path(destination).expanduser()
    _validate_destination(root, target)
    files = _collect_share_files(root, selected_options)
    privacy_report = _scan_known_files(
        root,
        files,
        enable_image_ocr=selected_options.scan_images_with_ocr,
        ocr_reader=ocr_reader,
        max_text_bytes=MAX_TEXT_BYTES,
    )
    if not _share_scan_is_acceptable(
        privacy_report,
        require_clean=selected_options.require_clean_scan,
    ):
        raise PrivacyViolationError(privacy_report)

    stage: Path | None = None
    try:
        _ensure_destination_parent_is_real(target.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        _ensure_destination_parent_is_real(target.parent)
        # A generic prefix avoids echoing a potentially sensitive destination
        # name into transient filesystem entries or diagnostic listings.
        stage = Path(tempfile.mkdtemp(prefix=".ai-jingjing-share-", dir=target.parent))
        records: list[dict[str, object]] = []
        total_bytes = 0
        contains_raw_media = False
        for source, relative in files:
            destination_file = stage.joinpath(*relative.parts)
            size, checksum = _copy_file_and_hash(source, destination_file)
            total_bytes += size
            if total_bytes > MAX_SHARE_TOTAL_BYTES:
                raise ValueError("分享副本总大小超过安全上限")
            contains_raw_media = contains_raw_media or source.suffix.casefold() in (
                _IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS
            )
            records.append({"path": relative.as_posix(), "size": size, "sha256": checksum})

        # Re-scan the exact bytes that will be published.  This closes the gap
        # where an actively edited source changes between the initial scan and
        # the copy.  A failed second scan is cleaned with the staging directory.
        staged_files = [
            (stage.joinpath(*PurePosixPath(str(item["path"])).parts), PurePosixPath(str(item["path"])))
            for item in records
        ]
        privacy_report = _scan_known_files(
            target,
            staged_files,
            enable_image_ocr=selected_options.scan_images_with_ocr,
            ocr_reader=ocr_reader,
            max_text_bytes=MAX_TEXT_BYTES,
        )
        if not _share_scan_is_acceptable(
            privacy_report,
            require_clean=selected_options.require_clean_scan,
        ):
            raise PrivacyViolationError(privacy_report)

        manifest: dict[str, object] = {
            "format": SHARE_FORMAT,
            "created_at": utcnow_iso(),
            "product": PRODUCT_NAME,
            "external_delivery": False,
            "contains_conversations": False,
            "contains_credentials": False if privacy_report.status == "clean" else None,
            "contains_raw_media": contains_raw_media,
            "selection": {
                "include_notes": selected_options.include_notes,
                "public_sources_count": len(selected_options.public_sources),
            },
            "privacy": privacy_report.to_dict(),
            "file_count": len(records),
            "total_bytes": total_bytes,
            "files": records,
        }
        manifest_path = stage / "share_manifest.json"
        _write_json_atomic(manifest_path, manifest)
        _verify_staged_files(stage, records)
        manifest_sha256 = _sha256_file(manifest_path)
        _publish_directory_no_replace(stage, target)
        stage = None
        return ShareCopyReport(
            status="created",
            destination=target.resolve(),
            file_count=len(records),
            total_bytes=total_bytes,
            manifest_path=(target / "share_manifest.json").resolve(),
            manifest_sha256=manifest_sha256,
            privacy_report=privacy_report,
        )
    except (PrivacyViolationError, ValueError):
        raise
    except Exception:
        # Avoid leaking absolute paths or source content through exception text.
        raise RuntimeError("安全分享副本生成失败，未保留未完成输出") from None
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def verify_share_copy(directory: str | Path) -> dict[str, object]:
    """Verify a generated share directory without trusting its manifest paths."""

    root = Path(directory).expanduser()
    try:
        info = root.lstat()
    except OSError:
        raise ValueError("分享副本不存在或不可访问") from None
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("分享副本必须是非符号链接目录")
    manifest_path = root / "share_manifest.json"
    try:
        raw = _read_regular_file(manifest_path, limit=4 * 1024**2)
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, OverflowError, UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("分享副本清单无效") from None
    if not isinstance(manifest, dict) or manifest.get("format") != SHARE_FORMAT:
        raise ValueError("分享副本清单格式无效")
    raw_records = manifest.get("files")
    if not isinstance(raw_records, list) or len(raw_records) > MAX_SHARE_FILES:
        raise ValueError("分享副本文件清单无效")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    total = 0
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValueError("分享副本文件清单无效")
        relative = _validate_manifest_path(str(raw_record.get("path", "")))
        name = relative.as_posix()
        size = raw_record.get("size")
        checksum = raw_record.get("sha256")
        if (
            name.casefold() in seen
            or _is_forbidden_share_path(relative)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > MAX_SHARE_FILE_BYTES
            or not isinstance(checksum, str)
            or not _SHA256_RE.fullmatch(checksum)
        ):
            raise ValueError("分享副本文件清单无效")
        seen.add(name.casefold())
        total += size
        if total > MAX_SHARE_TOTAL_BYTES:
            raise ValueError("分享副本总大小超过安全上限")
        records.append({"path": name, "size": size, "sha256": checksum})
    if manifest.get("file_count") != len(records) or manifest.get("total_bytes") != total:
        raise ValueError("分享副本文件统计与清单不一致")
    _verify_staged_files(root, records)
    return {
        "status": "verified",
        "format": SHARE_FORMAT,
        "file_count": len(records),
        "total_bytes": total,
        "manifest_sha256": _sha256_file(manifest_path),
    }


__all__ = [
    "PrivacyFinding",
    "PrivacyScanReport",
    "PrivacyViolationError",
    "ShareCopyOptions",
    "ShareCopyReport",
    "create_share_copy",
    "scan_privacy",
    "verify_share_copy",
]
