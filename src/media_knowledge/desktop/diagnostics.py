from __future__ import annotations

import importlib.util
import os
import shutil
import sqlite3
import sys

from ..ingestion.transcription import (
    TranscriptionUnavailable,
    hardware_capabilities,
    select_transcription_plan,
)
from .controller import DesktopController


COMPONENTS = {
    "PySide6": ("PySide6", "桌面界面"),
    "PyMuPDF": ("fitz", "PDF 解析与页面预览"),
    "python-docx": ("docx", "Word 解析"),
    "python-pptx": ("pptx", "PPTX 结构解析"),
    "Pillow": ("PIL", "图片与 PPT 页面渲染"),
    "RapidOCR": ("rapidocr", "本地图片 OCR"),
    "PaddleOCR PP-StructureV3": ("paddleocr", "复杂扫描件、表格、公式与多栏版面 OCR（可选）"),
    "mlx-whisper": ("mlx_whisper", "Apple Silicon 加速音视频转写（可选）"),
    "faster-whisper": ("faster_whisper", "本地音视频转写"),
    "imageio-ffmpeg": ("imageio_ffmpeg", "内置 FFmpeg"),
    "yt-dlp": ("yt_dlp", "公开视频平台字幕与媒体连接器"),
}

OPTIONAL_COMPONENTS = {"PaddleOCR PP-StructureV3", "mlx-whisper", "yt-dlp"}


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
    try:
        route = select_transcription_plan(
            controller.settings.whisper_model,
            preferred_engine=controller.settings.transcription_engine,
            allow_cpu_fallback=controller.settings.transcription_allow_cpu_fallback,
            capabilities=capabilities,
        ).to_dict()
        route["available"] = True
    except TranscriptionUnavailable as exc:
        route = {"available": False, "error": str(exc)}
    required_names = {
        "PyMuPDF", "python-docx", "python-pptx", "Pillow",
        "RapidOCR",
    }
    return {
        "product": "AI知识库-AI静静",
        "python": sys.version.split()[0],
        "platform": sys.platform,
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
        "transcription": {"capabilities": capabilities, "route": route},
        "ready_for_text_qa": writable and sqlite_fts,
        "ready_for_all_media": writable and sqlite_fts and bool(ffmpeg) and bool(route.get("available")) and all(
            item["available"] for item in components if item["name"] in required_names
        ),
    }
