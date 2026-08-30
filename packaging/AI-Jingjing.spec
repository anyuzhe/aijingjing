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
if module_available("yt_dlp"):
    hiddenimports += collect_submodules("yt_dlp")
    datas += collect_data_files("yt_dlp")
if module_available("mlx_whisper"):
    # Importing the package recursively discovers every inference module. Do
    # not collect torch_whisper: it is a conversion helper, is never used at
    # runtime, and would add the entire PyTorch distribution to the app.
    hiddenimports += ["mlx_whisper"]
    datas += collect_data_files("mlx_whisper")
if module_available("mlx_audio"):
    # Qwen3-ASR is loaded dynamically from the STT registry. We ship the
    # inference code, not model weights; users install weights explicitly from
    # the in-app model manager.
    hiddenimports += collect_submodules("mlx_audio.stt")
    datas += collect_data_files("mlx_audio")
    binaries += collect_dynamic_libs("mlx_audio")
if module_available("mlx"):
    # MLX loads its Metal kernel library and native dylibs at runtime; normal
    # Python import analysis alone does not reliably preserve these files.
    datas += collect_data_files("mlx")
    binaries += collect_dynamic_libs("mlx")
if module_available("sherpa_onnx"):
    # Speaker diarization is loaded behind the provider registry.  Preserve
    # both its Python API and native ONNX Runtime/C++ libraries in the frozen
    # application; model weights remain user-managed and are never bundled.
    hiddenimports += collect_submodules("sherpa_onnx")
    datas += collect_data_files("sherpa_onnx")
    binaries += collect_dynamic_libs("sherpa_onnx")
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
        "torch", "mlx_whisper.torch_whisper",
        # The product intentionally has no microphone capture or realtime
        # recording path.  mlx-audio may expose sounddevice as an unrelated
        # optional dependency, so keep it out of the frozen desktop bundle.
        "sounddevice",
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
            "CFBundleShortVersionString": "2.4.0",
            "CFBundleVersion": "2.4.0",
            "NSHighResolutionCapable": True,
        },
    )
