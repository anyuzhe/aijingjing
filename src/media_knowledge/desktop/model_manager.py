from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ..product import ProductPaths


@dataclass(frozen=True, slots=True)
class LocalModelSpec:
    model_id: str
    label: str
    provider: str
    repo_id: str | None
    kind: str
    approximate_size_gb: float
    license_name: str
    description: str
    gated: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocalModelStatus:
    spec: LocalModelSpec
    installed: bool
    path: str | None
    size_bytes: int
    verified: bool
    source: str | None = None
    error: str | None = None
    content_sha256: str | None = None
    content_verified: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            **self.spec.to_dict(),
            "installed": self.installed,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "verified": self.verified,
            "source": self.source,
            "error": self.error,
            "content_sha256": self.content_sha256,
            "content_verified": self.content_verified,
        }


MODEL_SPECS: tuple[LocalModelSpec, ...] = (
    LocalModelSpec(
        "qwen3-asr-1.7b-mlx", "Qwen3-ASR 1.7B · 中文高精度", "qwen3-mlx",
        "mlx-community/Qwen3-ASR-1.7B-8bit", "asr", 2.2, "Apache-2.0",
        "中文会议、课程和专业内容的高精度档；支持上下文词与时间戳。",
    ),
    LocalModelSpec(
        "qwen3-asr-0.6b-mlx", "Qwen3-ASR 0.6B · 快速预览", "qwen3-mlx",
        "mlx-community/Qwen3-ASR-0.6B-8bit", "asr", 0.9, "Apache-2.0",
        "速度和占用优先，适合先预览后决定是否精转。",
    ),
    LocalModelSpec(
        "whisper-large-v3-mlx", "Whisper Large v3 · MLX", "mlx-whisper",
        "mlx-community/whisper-large-v3-mlx", "asr", 3.1, "MIT",
        "Apple Silicon 兼容高精度模型，也是 Qwen 不可用时的首选回退。",
    ),
    LocalModelSpec(
        "whisper-medium-mlx", "Whisper Medium · MLX", "mlx-whisper",
        "mlx-community/whisper-medium-mlx", "asr", 1.6, "MIT",
        "精度和模型体积之间的兼容档。",
    ),
    LocalModelSpec(
        "whisper-small-mlx", "Whisper Small · MLX", "mlx-whisper",
        "mlx-community/whisper-small-mlx", "asr", 0.5, "MIT",
        "占用较低的兼容模型。",
    ),
    LocalModelSpec(
        "whisper-base-mlx", "Whisper Base · MLX", "mlx-whisper",
        "mlx-community/whisper-base-mlx", "asr", 0.2, "MIT",
        "轻量兼容模型，适合短录音预览。",
    ),
    LocalModelSpec(
        "whisper-tiny-mlx", "Whisper Tiny · MLX", "mlx-whisper",
        "mlx-community/whisper-tiny-mlx", "asr", 0.1, "MIT",
        "速度优先的最小兼容模型。",
    ),
    LocalModelSpec(
        "faster-whisper-large-v3", "Whisper Large v3 · CTranslate2", "faster-whisper",
        "Systran/faster-whisper-large-v3", "asr", 3.1, "MIT",
        "面向 NVIDIA CUDA 或 CPU 的高精度兼容模型。",
    ),
    LocalModelSpec(
        "faster-whisper-medium", "Whisper Medium · CTranslate2", "faster-whisper",
        "Systran/faster-whisper-medium", "asr", 1.6, "MIT",
        "面向 NVIDIA CUDA 或 CPU 的中型兼容模型。",
    ),
    LocalModelSpec(
        "faster-whisper-small", "Whisper Small · CTranslate2", "faster-whisper",
        "Systran/faster-whisper-small", "asr", 0.5, "MIT",
        "面向 NVIDIA CUDA 或 CPU 的轻量兼容模型。",
    ),
    LocalModelSpec(
        "faster-whisper-base", "Whisper Base · CTranslate2", "faster-whisper",
        "Systran/faster-whisper-base", "asr", 0.2, "MIT",
        "面向 NVIDIA CUDA 或 CPU 的轻量预览模型。",
    ),
    LocalModelSpec(
        "faster-whisper-tiny", "Whisper Tiny · CTranslate2", "faster-whisper",
        "Systran/faster-whisper-tiny", "asr", 0.1, "MIT",
        "面向 NVIDIA CUDA 或 CPU 的最小预览模型。",
    ),
    LocalModelSpec(
        "pyannote-community-1", "pyannote Community-1 · 说话人识别", "pyannote",
        "pyannote/speaker-diarization-community-1", "diarization", 2.5, "CC-BY-4.0",
        "离线区分多位说话人；首次下载需先在 Hugging Face 接受使用条件。", True,
    ),
)


def _directory_size(path: Path) -> int:
    total = 0
    seen_files: set[tuple[int, int]] = set()
    try:
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            stat = child.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen_files:
                continue
            seen_files.add(identity)
            total += stat.st_size
    except OSError:
        return 0
    return total


def _has_model_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {child.name for child in path.iterdir() if child.is_file()}
    return bool(
        names.intersection({"config.json", "config.yaml", "model.safetensors", "weights.npz"})
        or any(path.glob("*.safetensors"))
    )


def _content_sha256(
    path: Path,
    *,
    progress: Callable[[str], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> str:
    """Hash model file names and bytes, following snapshot symlinks safely."""

    if not path.is_dir():
        raise ValueError("模型目录不存在")
    digest = hashlib.sha256()
    file_count = 0
    candidates = sorted(path.rglob("*"), key=lambda item: item.as_posix())
    total_bytes = sum(
        candidate.stat().st_size
        for candidate in candidates
        if candidate.is_file()
        and candidate.relative_to(path).parts
        and candidate.relative_to(path).parts[0] not in {".cache", ".locks"}
    )
    hashed_bytes = 0
    for candidate in candidates:
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(path)
        if relative.parts and relative.parts[0] in {".cache", ".locks"}:
            continue
        try:
            resolved = candidate.resolve(strict=True)
            if not resolved.is_file():
                continue
            size = resolved.stat().st_size
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\0")
            with resolved.open("rb") as handle:
                while block := handle.read(4 * 1024 * 1024):
                    if check_cancelled:
                        check_cancelled()
                    digest.update(block)
                    hashed_bytes += len(block)
                    if progress and total_bytes:
                        progress(
                            "正在校验模型内容 "
                            f"{min(100, round(hashed_bytes * 100 / total_bytes))}%"
                        )
            file_count += 1
        except OSError as exc:
            raise RuntimeError(f"无法读取模型文件：{relative}") from exc
    if file_count == 0:
        raise ValueError("模型目录中没有可校验文件")
    return digest.hexdigest()


class LocalModelManager:
    """Explicit, local-first model lifecycle manager.

    Merely constructing this class or asking for status never performs a network
    request. Downloads only happen through :meth:`download`, which is designed to
    be called by an explicit user action.
    """

    def __init__(self, paths: ProductPaths) -> None:
        self.paths = paths.ensure()
        self.root = paths.models
        self.registry_path = self.root / "registry.json"
        self._specs = {item.model_id: item for item in MODEL_SPECS}

    def specs(self, *, kind: str | None = None) -> list[LocalModelSpec]:
        return [item for item in MODEL_SPECS if kind is None or item.kind == kind]

    def spec(self, model_id: str) -> LocalModelSpec:
        try:
            return self._specs[model_id]
        except KeyError as exc:
            raise ValueError(f"未知本地模型：{model_id}") from exc

    def _load_registry(self) -> dict[str, dict[str, object]]:
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        models = payload.get("models") if isinstance(payload, dict) else None
        return {
            str(key): dict(value)
            for key, value in (models or {}).items()
            if isinstance(value, dict)
        }

    def _save_registry(self, values: dict[str, dict[str, object]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".registry.", suffix=".tmp", dir=self.root
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "models": values}, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.registry_path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def _default_path(self, model_id: str) -> Path:
        return self.root / model_id

    def _registered_path(self, model_id: str) -> tuple[Path | None, str | None]:
        value = self._load_registry().get(model_id) or {}
        raw = value.get("path")
        if raw:
            return Path(str(raw)).expanduser().resolve(), str(value.get("source") or "registered")
        default = self._default_path(model_id)
        if default.is_dir():
            return default, "managed"
        return None, None

    def _cached_huggingface_path(self, spec: LocalModelSpec) -> Path | None:
        if not spec.repo_id:
            return None
        # This lookup is deliberately local-only. It recognizes weights already
        # present in the standard cache without initiating a download.
        try:
            from huggingface_hub import snapshot_download  # type: ignore

            result = snapshot_download(spec.repo_id, local_files_only=True)
            candidate = Path(result).resolve()
            return candidate if _has_model_files(candidate) else None
        except Exception:
            return None

    def resolve(self, model_id: str, *, include_huggingface_cache: bool = True) -> Path | None:
        spec = self.spec(model_id)
        candidate, _source = self._registered_path(model_id)
        if candidate and _has_model_files(candidate):
            return candidate
        return self._cached_huggingface_path(spec) if include_huggingface_cache else None

    def status(self, model_id: str) -> LocalModelStatus:
        spec = self.spec(model_id)
        registry_value = self._load_registry().get(model_id) or {}
        candidate, source = self._registered_path(model_id)
        if not candidate or not _has_model_files(candidate):
            candidate = self._cached_huggingface_path(spec)
            source = "huggingface-cache" if candidate else None
        if not candidate:
            return LocalModelStatus(spec, False, None, 0, False, None)
        size = _directory_size(candidate)
        content_sha256 = str(registry_value.get("sha256") or "") or None
        verification_error = str(registry_value.get("verification_error") or "") or None
        structurally_valid = _has_model_files(candidate)
        return LocalModelStatus(
            spec,
            True,
            str(candidate),
            size,
            structurally_valid and not verification_error,
            source,
            verification_error,
            content_sha256,
            bool(content_sha256 and not verification_error),
        )

    def statuses(self, *, kind: str | None = None) -> list[LocalModelStatus]:
        return [self.status(spec.model_id) for spec in self.specs(kind=kind)]

    def verified_content_sha256_for_path(self, source: str | Path | None) -> str | None:
        """Return a stored, verified model identity without hashing or networking."""

        if not source:
            return None
        try:
            target = Path(source).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        for value in self._load_registry().values():
            raw_path = str(value.get("path") or "").strip()
            checksum = str(value.get("sha256") or "").strip().casefold()
            if not raw_path or value.get("verification_error"):
                continue
            try:
                registered = Path(raw_path).expanduser().resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if (
                registered == target
                and len(checksum) == 64
                and all(character in "0123456789abcdef" for character in checksum)
            ):
                return checksum
        return None

    def register_path(
        self,
        model_id: str,
        source: str | Path,
        *,
        progress: Callable[[str], None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> LocalModelStatus:
        path = Path(source).expanduser().resolve()
        if not _has_model_files(path):
            raise ValueError("所选目录不是可识别的模型目录（缺少 config 或权重文件）")
        if progress:
            progress("正在读取并校验已有模型目录")
        checksum = _content_sha256(
            path, progress=progress, check_cancelled=check_cancelled
        )
        values = self._load_registry()
        values[model_id] = {
            "path": str(path),
            "source": "external",
            "registered_at": int(time.time()),
            "sha256": checksum,
            "verified_at": int(time.time()),
        }
        self._save_registry(values)
        return self.status(model_id)

    def import_model(
        self,
        model_id: str,
        source: str | Path,
        *,
        progress: Callable[[str], None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> LocalModelStatus:
        source_path = Path(source).expanduser().resolve()
        if not _has_model_files(source_path):
            raise ValueError("所选目录不是可识别的模型目录（缺少 config 或权重文件）")
        destination = self._default_path(model_id)
        if destination.exists():
            raise FileExistsError("该模型已存在；请先移除后再导入")
        staging = self.root / ".imports" / f"{model_id}-{uuid.uuid4().hex}"
        staging.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress("正在复制并校验本地模型")
        try:
            shutil.copytree(source_path, staging)
            if check_cancelled:
                check_cancelled()
            if not _has_model_files(staging):
                raise ValueError("复制后的模型校验失败")
            checksum = _content_sha256(
                staging, progress=progress, check_cancelled=check_cancelled
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
        finally:
            if staging.is_dir():
                shutil.rmtree(staging, ignore_errors=True)
        values = self._load_registry()
        values[model_id] = {
            "path": str(destination),
            "source": "imported",
            "registered_at": int(time.time()),
            "sha256": checksum,
            "verified_at": int(time.time()),
        }
        self._save_registry(values)
        return self.status(model_id)

    def download(
        self,
        model_id: str,
        *,
        token: str | None = None,
        progress: Callable[[str], None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> LocalModelStatus:
        """Download a model only after the caller's explicit user action."""

        spec = self.spec(model_id)
        if not spec.repo_id:
            raise ValueError("该模型不支持在线安装，请导入本地目录")
        destination = self._default_path(model_id)
        if destination.exists():
            raise FileExistsError("该模型已存在；请先移除后再重新安装")
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError as exc:
            raise RuntimeError("模型下载组件 huggingface-hub 未安装") from exc
        staging = self.root / ".downloads" / f"{model_id}-{uuid.uuid4().hex}"
        staging.parent.mkdir(parents=True, exist_ok=True)
        if check_cancelled:
            check_cancelled()
        if progress:
            progress(f"正在下载 {spec.label}，请保持网络连接")
        try:
            snapshot_download(
                spec.repo_id,
                local_dir=str(staging),
                token=token or None,
            )
            if check_cancelled:
                check_cancelled()
            if not _has_model_files(staging):
                raise RuntimeError("模型下载完成但校验失败")
            if progress:
                progress("下载完成，正在计算模型内容 SHA-256")
            checksum = _content_sha256(
                staging, progress=progress, check_cancelled=check_cancelled
            )
            os.replace(staging, destination)
        finally:
            if staging.is_dir():
                shutil.rmtree(staging, ignore_errors=True)
        values = self._load_registry()
        values[model_id] = {
            "path": str(destination),
            "source": "downloaded",
            "repo_id": spec.repo_id,
            "registered_at": int(time.time()),
            "sha256": checksum,
            "verified_at": int(time.time()),
        }
        self._save_registry(values)
        return self.status(model_id)

    def remove(self, model_id: str) -> bool:
        """Remove only an app-managed model; external/cache weights stay untouched."""

        values = self._load_registry()
        value = values.get(model_id) or {}
        raw_path = value.get("path")
        candidate = Path(str(raw_path)).resolve() if raw_path else self._default_path(model_id).resolve()
        managed_root = self.root.resolve()
        removed = False
        if candidate == self._default_path(model_id).resolve() and candidate.parent == managed_root:
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate)
                removed = True
        values.pop(model_id, None)
        self._save_registry(values)
        return removed

    def fingerprint(self, model_id: str) -> str | None:
        path = self.resolve(model_id)
        if path is None:
            return None
        return _content_sha256(path)

    def verify(
        self,
        model_id: str,
        *,
        progress: Callable[[str], None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> LocalModelStatus:
        """Explicitly recompute and compare the installed model content hash."""

        status = self.status(model_id)
        if not status.path or not status.installed:
            raise ValueError("该模型尚未安装")
        if check_cancelled:
            check_cancelled()
        if progress:
            progress(f"正在校验 {status.spec.label} 的全部模型文件")
        actual = _content_sha256(
            Path(status.path), progress=progress, check_cancelled=check_cancelled
        )
        if check_cancelled:
            check_cancelled()
        values = self._load_registry()
        value = values.get(model_id) or {
            "path": status.path,
            "source": status.source or "external",
            "registered_at": int(time.time()),
        }
        expected = str(value.get("sha256") or "")
        if expected and expected != actual:
            value["verification_error"] = (
                f"模型内容 SHA-256 不匹配（期望 {expected[:12]}…，"
                f"实际 {actual[:12]}…）"
            )
        else:
            value["sha256"] = actual
            value["verified_at"] = int(time.time())
            value.pop("verification_error", None)
        values[model_id] = value
        self._save_registry(values)
        result = self.status(model_id)
        if result.error:
            raise RuntimeError(result.error)
        return result
