#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
runtime_dir="$project_root/.venv-desktop"

python3 -m venv "$runtime_dir"
"$runtime_dir/bin/python" -m pip install --upgrade pip setuptools wheel
platform_name="$(uname -s)"
platform_arch="$(uname -m)"
install_extras="full"

# The Apple Silicon package must include both the portable media stack and the
# native MLX/Qwen runtime. Other platforms intentionally stay on the portable
# `full` extra: MLX wheels are macOS arm64 specific and would make those builds
# fail dependency resolution.
if [[ "$platform_name" == "Darwin" && "$platform_arch" == "arm64" ]]; then
  install_extras="full,apple-media"
fi

echo "构建平台：${platform_name}/${platform_arch}；安装可选依赖：${install_extras}"
"$runtime_dir/bin/pip" install -e "$project_root[$install_extras]"
# The suite is unittest-compatible.  On macOS, pytest can occasionally finish
# every assertion successfully and then abort while its plugin layer tears down
# Qt multimedia objects.  Direct unittest discovery owns a simpler Qt lifetime
# and provides a stable release gate (the same test modules still run).
QT_QPA_PLATFORM=offscreen "$runtime_dir/bin/python" -m unittest discover \
  -s "$project_root/tests" -q
QT_QPA_PLATFORM=offscreen "$runtime_dir/bin/python" "$project_root/scripts/make_icons.py"
"$runtime_dir/bin/pyinstaller" --noconfirm --clean "$project_root/packaging/AI-Jingjing.spec"

echo "构建完成：$project_root/dist/AI知识库-AI静静.app"
