from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sqlite3
import sys
from pathlib import Path

from ..ingestion.transcription import (
    TranscriptionUnavailable,
    hardware_capabilities,
    select_transcription_plan,
)
from ..ingestion.asr import AsrProviderRegistry, AsrRouter, TranscriptionRequest
from ..ingestion.diarization.sherpa_provider import find_sherpa_onnx_models
from ..resource_scheduler import LOCAL_HEAVY_TASKS
from .controller import DesktopController


COMPONENTS = {
    "PySide6": ("PySide6", "桌面界面"),
    "PySide6 QtMultimedia": ("PySide6.QtMultimedia", "内置音视频播放器"),
    "PyMuPDF": ("fitz", "PDF 解析与页面预览"),
    "python-docx": ("docx", "Word 解析"),
    "python-pptx": ("pptx", "PPTX 结构解析"),
    "Pillow": ("PIL", "图片与 PPT 页面渲染"),
    "RapidOCR": ("rapidocr", "本地图片 OCR"),
    "PaddleOCR PP-StructureV3": ("paddleocr", "复杂扫描件、表格、公式与多栏版面 OCR（可选）"),
    "mlx-whisper": ("mlx_whisper", "Apple Silicon 加速音视频转写（可选）"),
    "mlx-audio Qwen3-ASR": ("mlx_audio", "Apple Silicon 中文高精度转写（可选）"),
    "faster-whisper": ("faster_whisper", "本地音视频转写"),
    "pyannote.audio": ("pyannote.audio", "本地多说话人识别（可选）"),
    "sherpa-onnx": ("sherpa_onnx", "轻量说话人识别后端（可选）"),
    "imageio-ffmpeg": ("imageio_ffmpeg", "内置 FFmpeg"),
    "yt-dlp": ("yt_dlp", "公开视频平台字幕与媒体连接器"),
}

OPTIONAL_COMPONENTS = {
    "PaddleOCR PP-StructureV3", "mlx-whisper", "mlx-audio Qwen3-ASR",
    "pyannote.audio", "sherpa-onnx", "yt-dlp",
}


def _module_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _available(module: str) -> bool:
    if module == "rapidocr":
        return _module_present("rapidocr") or _module_present("rapidocr_onnxruntime")
    if module == "paddleocr":
        return all(_module_present(name) for name in ("paddleocr", "paddle", "paddlex"))
    return _module_present(module)


def _machine_info() -> dict[str, object]:
    memory_bytes = 0
    try:
        memory_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    machine = platform.machine() or "unknown"
    processor = platform.processor() or machine
    return {
        "system": platform.system() or sys.platform,
        "machine": machine,
        "processor": processor,
        "apple_silicon": platform.system() == "Darwin" and machine.casefold() in {"arm64", "aarch64"},
        "memory_bytes": max(0, memory_bytes),
        "memory_gb": round(memory_bytes / (1024 ** 3), 1) if memory_bytes else None,
    }


def _model_directory_valid(value: str | Path | None) -> bool:
    if not value:
        return False
    path = Path(value).expanduser()
    if not path.is_dir():
        return False
    try:
        names = {item.name for item in path.iterdir() if item.is_file()}
    except OSError:
        return False
    return bool(
        "config.json" in names
        and (
            "model.safetensors" in names
            or any(path.glob("*.safetensors"))
            or "weights.npz" in names
        )
    )


def _sherpa_model_directory_valid(value: str | Path | None) -> bool:
    if not value:
        return False
    path = Path(value).expanduser()
    if not path.is_dir():
        return False
    try:
        segmentation, embedding = find_sherpa_onnx_models(path)
    except OSError:
        return False
    return segmentation is not None and embedding is not None


def _embedding_local_state(controller: DesktopController) -> dict[str, object]:
    provider = controller.settings.embedding_provider
    if provider == "hash":
        return {"provider": provider, "local_ready": True, "reason": "内置算法，不需要模型权重"}
    cache = controller.paths.cache / "models"
    ready = bool(
        cache.is_dir()
        and any(cache.rglob("files_metadata.json"))
        and any(cache.rglob("*.onnx"))
    )
    return {
        "provider": provider,
        "model": controller.settings.embedding_model,
        "cache": str(cache),
        "local_ready": ready,
        "reason": (
            "FastEmbed 本地模型已发现"
            if ready else "FastEmbed 本地模型未准备；正式任务不会自动下载"
        ),
    }


def run_diagnostics(controller: DesktopController) -> dict[str, object]:
    components = [
        {
            "name": name,
            "available": _available(module),
            "purpose": purpose,
            "optional": name in OPTIONAL_COMPONENTS,
        }
        for name, (module, purpose) in COMPONENTS.items()
    ]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg and _available("imageio_ffmpeg"):
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            pass
    sqlite_fts = False
    try:
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE VIRTUAL TABLE check_fts USING fts5(content)")
        sqlite_fts = True
        connection.close()
    except sqlite3.Error:
        pass
    writable = os.access(controller.paths.root, os.W_OK)
    capabilities = hardware_capabilities()
    route: dict[str, object]
    managed_ids = {
        "Qwen3-ASR-1.7B": "qwen3-asr-1.7b-mlx",
        "Qwen3-ASR-0.6B": "qwen3-asr-0.6b-mlx",
        "large-v3": "whisper-large-v3-mlx",
        "medium": "whisper-medium-mlx",
        "small": "whisper-small-mlx",
        "base": "whisper-base-mlx",
        "tiny": "whisper-tiny-mlx",
    }
    selected_model = controller.settings.asr_model or controller.settings.whisper_model
    selected_provider = controller.settings.asr_provider
    managed_id = (
        f"faster-whisper-{selected_model}"
        if selected_provider == "faster-whisper"
        and selected_model in {"large-v3", "medium", "small", "base", "tiny"}
        else managed_ids.get(selected_model)
    )
    local_path = (
        controller.settings.asr_model_path
        or (controller.resolve_transcription_model(managed_id) if managed_id else None)
    )
    model_statuses = controller.transcription_model_statuses()
    status_by_id = {str(item.get("model_id")): item for item in model_statuses}
    try:
        request = TranscriptionRequest(
            audio_path=controller.paths.cache / "diagnostic-placeholder.wav",
            profile=controller.settings.transcription_profile,
            provider=(
                None if controller.settings.asr_provider == "auto"
                else controller.settings.asr_provider
            ),
            model=selected_model,
            model_path=Path(local_path) if local_path else None,
            whisper_fallback_model_path=(
                Path(value)
                if (value := controller.resolve_transcription_model("whisper-small-mlx"))
                else None
            ),
            language=controller.settings.transcription_language,
            context_terms=tuple(controller.settings.asr_context_terms),
            word_timestamps=controller.settings.word_timestamps,
            allow_fallback=controller.settings.transcription_allow_cpu_fallback,
        )
        attempts = AsrRouter(AsrProviderRegistry()).resolve_attempts(request)
        first = attempts[0]
        if first.provider == "qwen3-mlx":
            available = bool(capabilities.get("mlx_audio")) and _model_directory_valid(local_path)
            error = None if available else "Qwen3-ASR 运行组件或所选本地权重尚未安装"
        elif first.provider == "mlx-whisper":
            available = bool(capabilities.get("mlx_whisper")) and _model_directory_valid(
                first.model_path or local_path
            )
            error = None if available else "mlx-whisper 运行组件或所选本地权重尚未安装"
        else:
            legacy = select_transcription_plan(
                controller.settings.whisper_model,
                preferred_engine=controller.settings.transcription_engine,
                allow_cpu_fallback=controller.settings.transcription_allow_cpu_fallback,
                capabilities=capabilities,
            )
            runtime_available = bool(capabilities.get("faster_whisper"))
            weights_available = _model_directory_valid(first.model_path or local_path)
            available = runtime_available and weights_available
            if not runtime_available:
                error = "faster-whisper 运行组件尚未安装"
            elif not weights_available:
                error = "faster-whisper 所选 CTranslate2 本地权重尚未安装"
            else:
                error = None
        route = {
            "available": available,
            "error": error,
            "profile": request.profile,
            "requested_provider": request.provider or "auto",
            "requested_model": request.model,
            "model_path": local_path,
            "attempts": [
                {
                    "provider": item.provider,
                    "model": item.model,
                    "device": item.device,
                    "compute_type": item.compute_type,
                    "local_model_path": str(item.model_path) if item.model_path else None,
                }
                for item in attempts
            ],
        }
        if first.provider == "faster-whisper":
            route["legacy_plan"] = legacy.to_dict()
    except (TranscriptionUnavailable, ValueError, OSError) as exc:
        route = {"available": False, "error": str(exc)}
    pyannote_model = status_by_id.get("pyannote-community-1") or {}
    sherpa_model = status_by_id.get("sherpa-speaker-diarization-zh") or {}
    diarization_provider = controller.settings.diarization_provider
    pyannote_path = str(pyannote_model.get("path") or "") or None
    sherpa_path = str(sherpa_model.get("path") or "") or None
    pyannote_ready = bool(
        _available("pyannote.audio")
        and pyannote_model.get("verified")
        and pyannote_path
        and Path(pyannote_path).is_dir()
    )
    sherpa_ready = bool(
        _available("sherpa_onnx")
        and sherpa_model.get("verified")
        and _sherpa_model_directory_valid(sherpa_path)
    )
    diarization_model_path: str | None = None
    if not controller.settings.diarization_enabled or diarization_provider == "none":
        diarization_ready = True
        diarization_reason = "未启用说话人识别"
    elif diarization_provider == "pyannote":
        diarization_ready = pyannote_ready
        diarization_model_path = pyannote_path
        diarization_reason = (
            "pyannote 本地运行组件和权重已就绪"
            if diarization_ready else "pyannote 组件或本地权重未安装"
        )
    elif diarization_provider == "sherpa":
        diarization_ready = sherpa_ready
        diarization_model_path = sherpa_path
        diarization_reason = (
            "Sherpa-ONNX 本地组件与双 ONNX 模型已就绪"
            if diarization_ready else "Sherpa-ONNX 组件或 segmentation/embedding 模型未就绪"
        )
    else:
        # Match the runtime router: pyannote is the first automatic choice,
        # while an installed Sherpa bundle is a complete local fallback.
        diarization_ready = pyannote_ready or sherpa_ready
        if pyannote_ready:
            diarization_model_path = pyannote_path
            diarization_reason = "自动路线将使用 pyannote 本地模型"
        elif sherpa_ready:
            diarization_model_path = sherpa_path
            diarization_reason = "自动路线将使用 Sherpa-ONNX 本地模型"
        else:
            diarization_reason = "未找到完整的 pyannote 或 Sherpa-ONNX 本地路线"
    embedding = _embedding_local_state(controller)
    cloud_features = []
    if controller.settings.auto_synthesize_notes:
        cloud_features.append("AI 知识提炼")
    if controller.settings.enable_cloud_vision:
        cloud_features.append("云端视觉理解")
    offline_risks: list[str] = []
    if not route.get("available"):
        offline_risks.append("当前 ASR 路线没有完整的本地运行组件和权重")
    if not embedding.get("local_ready"):
        offline_risks.append("当前向量模型不在本地；运行时会报错而不会下载")
    if controller.settings.diarization_enabled and not diarization_ready:
        offline_risks.append("已启用说话人识别，但本地组件或权重未就绪")
    if cloud_features:
        offline_risks.append("已显式启用云功能：" + "、".join(cloud_features))
    strict_offline = not offline_risks
    resource_state = LOCAL_HEAVY_TASKS.snapshot()
    required_names = {
        "PyMuPDF", "python-docx", "python-pptx", "Pillow",
        "RapidOCR",
    }
    return {
        "product": "AI知识库-AI静静",
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "machine": _machine_info(),
        "packaged": bool(getattr(sys, "frozen", False)),
        "data_root": str(controller.paths.root),
        "data_root_writable": writable,
        "sqlite_fts5": sqlite_fts,
        "ffmpeg": str(ffmpeg) if ffmpeg else None,
        "providers": controller.providers.status(),
        "components": components,
        "ocr": {
            "requested_engine": controller.settings.ocr_engine,
            "complex_layout_enabled": controller.settings.ocr_complex_layout_enabled,
            "low_confidence_threshold": controller.settings.ocr_low_confidence_threshold,
            "rapidocr_available": _available("rapidocr"),
            "paddleocr_available": _available("paddleocr"),
            "paddleocr_package_available": _module_present("paddleocr") and _module_present("paddlex"),
            "paddlepaddle_available": _available("paddle"),
        },
        "transcription": {
            "capabilities": capabilities,
            "route": route,
            "models": model_statuses,
            "diarization": {
                "enabled": controller.settings.diarization_enabled,
                "provider": diarization_provider,
                "model_path": diarization_model_path,
                "ready": diarization_ready,
                "reason": diarization_reason,
            },
        },
        "embedding": embedding,
        "offline": {
            "strict_ready": strict_offline,
            "hidden_model_downloads_blocked": True,
            "cloud_features": cloud_features,
            "risks": offline_risks,
        },
        "resource_scheduler": {
            "concurrency_limit": resource_state.concurrency_limit,
            "active_task": resource_state.active_task,
            "waiting_tasks": resource_state.waiting_tasks,
            "release_policy": "任务结束立即释放 Python/MLX 缓存",
        },
        "ready_for_text_qa": writable and sqlite_fts,
        "ready_for_all_media": writable and sqlite_fts and bool(ffmpeg) and bool(route.get("available")) and all(
            item["available"] for item in components if item["name"] in required_names
        ),
    }
