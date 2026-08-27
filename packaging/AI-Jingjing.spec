# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_submodules


project_root = Path(SPECPATH).parent
datas = [
    (str(project_root / "src" / "media_knowledge" / "desktop" / "assets"), "media_knowledge/desktop/assets"),
]
binaries = []
hiddenimports = collect_submodules("media_knowledge") + ["keyring.backends.macOS"]
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
app = BUNDLE(
    coll,
    name="AI知识库-AI静静.app",
    icon=icon_file,
    bundle_identifier="com.aijingjing.knowledge",
    info_plist={
        "CFBundleDisplayName": "AI知识库-AI静静",
        "CFBundleName": "AI静静",
        "CFBundleShortVersionString": "2.0.3",
        "CFBundleVersion": "2.0.3",
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": "用于导入和转写用户选择的音频资料。",
    },
)
