from __future__ import annotations

import importlib.util
import os
import shutil
import sqlite3
import sys

from .controller import DesktopController


COMPONENTS = {
    "PySide6": ("PySide6", "桌面界面"),
    "PyMuPDF": ("fitz", "PDF 解析与页面预览"),
    "python-docx": ("docx", "Word 解析"),
    "python-pptx": ("pptx", "PPTX 结构解析"),
    "Pillow": ("PIL", "图片与 PPT 页面渲染"),
    "RapidOCR": ("rapidocr", "本地图片 OCR"),
    "faster-whisper": ("faster_whisper", "本地音视频转写"),
    "imageio-ffmpeg": ("imageio_ffmpeg", "内置 FFmpeg"),
}


def run_diagnostics(controller: DesktopController) -> dict[str, object]:
    components = [
        {
            "name": name,
            "available": importlib.util.find_spec(module) is not None,
            "purpose": purpose,
        }
        for name, (module, purpose) in COMPONENTS.items()
    ]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg and importlib.util.find_spec("imageio_ffmpeg"):
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
        "ready_for_text_qa": writable and sqlite_fts,
        "ready_for_all_media": writable and sqlite_fts and all(
            item["available"] for item in components if item["name"] != "PySide6"
        ),
    }
