$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RuntimeDir = Join-Path $ProjectRoot ".venv-desktop"

py -3.11 -m venv $RuntimeDir
& "$RuntimeDir\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& "$RuntimeDir\Scripts\pip.exe" install -e "$ProjectRoot[full]"
& "$RuntimeDir\Scripts\python.exe" -m unittest discover -s "$ProjectRoot\tests" -v
& "$RuntimeDir\Scripts\python.exe" "$ProjectRoot\scripts\make_icons.py"
& "$RuntimeDir\Scripts\pyinstaller.exe" --noconfirm --clean "$ProjectRoot\packaging\AI-Jingjing.spec"

Write-Host "构建完成：$ProjectRoot\dist\AI知识库-AI静静"
