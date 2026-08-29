#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
runtime_dir="$project_root/.venv-desktop"

python3 -m venv "$runtime_dir"
"$runtime_dir/bin/python" -m pip install --upgrade pip setuptools wheel
install_extras="full"
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  # Ship the Apple Silicon build with MLX transcription instead of silently
  # falling back to the much slower CPU int8 runtime.
  install_extras="full,apple-media"
fi
"$runtime_dir/bin/pip" install -e "$project_root[$install_extras]"
"$runtime_dir/bin/python" -m unittest discover -s "$project_root/tests" -v
QT_QPA_PLATFORM=offscreen "$runtime_dir/bin/python" "$project_root/scripts/make_icons.py"
"$runtime_dir/bin/pyinstaller" --noconfirm --clean "$project_root/packaging/AI-Jingjing.spec"

echo "构建完成：$project_root/dist/AI知识库-AI静静.app"
