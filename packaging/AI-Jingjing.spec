# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import importlib.util
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


project_root = Path(SPECPATH).parent
datas = [
    (str(project_root / "src" / "media_knowledge" / "desktop" / "assets"), "media_knowledge/desktop/assets"),
]
binaries = []


def module_available(name):
    return importlib.util.find_spec(name) is not None


# PyInstaller discovers Python imports, but the OCR/ASR/platform packages also
# load models, VAD assets, native libraries and extractor modules dynamically.
# Collect them explicitly so a clean desktop build behaves like the source
# environment instead of failing only after the user imports media.
for data_package in ("rapidocr", "faster_whisper"):
    if module_available(data_package):
        datas += collect_data_files(data_package)

if module_available("ctranslate2"):
    binaries += collect_dynamic_libs("ctranslate2")
platform_keyring = (
    ["keyring.backends.macOS"] if sys.platform == "darwin"
    else ["keyring.backends.Windows"] if sys.platform == "win32"
    else []
)
hiddenimports = collect_submodules("media_knowledge") + platform_keyring
for dynamic_package in ("yt_dlp", "mlx_whisper"):
    if module_available(dynamic_package):
        hiddenimports += collect_submodules(dynamic_package)
        datas += collect_data_files(dynamic_package)
icon_candidate = project_root / "packaging" / (
    "AI-Jingjing.icns" if sys.platform == "darwin" else "AI-Jingjing.ico"
)
icon_file = str(icon_candidate) if icon_candidate.is_file() else None

a = Analysis(
    [str(project_root / "packaging" / "launcher.py")],
    pathex=[str(project_root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter", "matplotlib", "pytest", "rapidocr.inference_engine.tensorrt",
        "onnxruntime.tools", "onnxruntime.transformers", "keyring.testing",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="AI知识库-AI静静",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=False, icon=icon_file,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name="AI知识库-AI静静")
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="AI知识库-AI静静.app",
        icon=icon_file,
        bundle_identifier="com.aijingjing.knowledge",
        info_plist={
            "CFBundleDisplayName": "AI知识库-AI静静",
            "CFBundleName": "AI静静",
            "CFBundleShortVersionString": "2.2.0",
            "CFBundleVersion": "2.2.0",
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": "用于导入和转写用户选择的音频资料。",
        },
    )
