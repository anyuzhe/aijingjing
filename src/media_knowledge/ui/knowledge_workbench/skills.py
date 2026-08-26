from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ...codex_cli import codex_http_transport_args


SKILL_NAME = "knowledge-ingestor"
DEFAULT_SKILL_ROOT = Path.home() / ".codex" / "skills" / SKILL_NAME


def _cli_error_detail(stderr: str, stdout: str) -> str:
    """Return the actionable Codex CLI error instead of its trailing help hint."""

    lines = [line.strip() for line in (stderr or stdout or "未知错误").splitlines() if line.strip()]
    for line in lines:
        if line.casefold().startswith("error:"):
            return line
    useful = [
        line
        for line in lines
        if not line.startswith("Usage:")
        and "For more information, try '--help'." not in line
    ]
    return (useful[-1] if useful else "未知错误")[:500]


@dataclass(frozen=True, slots=True)
class SkillRunResult:
    skill: str
    markdown: str
    sources: list[str]


class KnowledgeIngestorBridge:
    """A narrow local bridge to the installed knowledge-ingestor Codex Skill."""

    def __init__(
        self,
        *,
        workspace_root: str | Path | None = None,
        skill_root: str | Path | None = None,
        codex_executable: str | None = None,
        timeout_seconds: int = 30 * 60,
    ) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd()).expanduser().resolve()
        self.skill_root = Path(skill_root or DEFAULT_SKILL_ROOT).expanduser().resolve()
        self.codex_executable = codex_executable or shutil.which("codex")
        self.timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return bool(self.codex_executable and (self.skill_root / "SKILL.md").is_file())

    def status(self) -> dict:
        return {
            "name": SKILL_NAME,
            "label": "知识摄取",
            "description": "理解并归档多模态资料，同时更新 Obsidian 知识笔记。",
            "available": self.available,
        }

    def _configured_write_roots(self) -> list[Path]:
        config_path = self.skill_root / "config.yaml"
        if not config_path.is_file():
            return []
        roots: list[Path] = []
        section: str | None = None
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            if not raw_line.startswith((" ", "\t")):
                match = re.match(r"([A-Za-z_][\w-]*):\s*$", raw_line)
                section = match.group(1) if match else None
                continue
            match = re.match(r"\s+(root|vault_root):\s*[\"']?(.+?)[\"']?\s*$", raw_line)
            if not match or section not in {"archive", "obsidian"}:
                continue
            value = match.group(2).strip().strip("\"'")
            if not value or value.lower() in {"null", "none", "~"}:
                continue
            target = Path(value).expanduser().resolve()
            allowed = target if target.exists() else target.parent
            if allowed.is_dir() and allowed not in roots:
                roots.append(allowed)
        return roots

    @staticmethod
    def validate_sources(values: object) -> list[str]:
        if not isinstance(values, list):
            return []
        if len(values) > 64:
            raise ValueError("一次最多选择 64 个本地文件")
        resolved: list[str] = []
        for value in values:
            path = Path(str(value)).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"所选文件不存在：{path.name}")
            item = str(path)
            if item not in resolved:
                resolved.append(item)
        return resolved

    def pick_files(self) -> list[str]:
        if sys.platform != "darwin":
            raise ValueError("当前系统暂不支持原生文件选择器")
        script = """
        try
          set selectedFiles to choose file with prompt "选择要交给 knowledge-ingestor 的资料" with multiple selections allowed
          set outputText to ""
          repeat with selectedFile in selectedFiles
            set outputText to outputText & POSIX path of selectedFile & linefeed
          end repeat
          return outputText
        on error number -128
          return ""
        end try
        """
        try:
            completed = subprocess.run(
                ["osascript", "-e", script],
                check=False,
                capture_output=True,
                text=True,
                timeout=5 * 60,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("文件选择超时") from exc
        if completed.returncode != 0:
            raise ValueError("无法打开系统文件选择器")
        return self.validate_sources([line for line in completed.stdout.splitlines() if line.strip()])

    def build_prompt(self, instruction: str, sources: list[str]) -> str:
        source_block = json.dumps(sources, ensure_ascii=False, indent=2) if sources else "[]"
        return f"""$knowledge-ingestor

请严格使用 knowledge-ingestor Skill 完成下面的任务。
用户已在本地知识工作台中明确选择并运行此 Skill，本次操作获准按照 Skill 的操作约定写入已配置的 KnowledgeArchive 与 Obsidian 目标。
只处理用户任务中明确提供的内容、URL 和下列本地文件，不要扩大范围。

用户任务：
{instruction}

本地文件：
{source_block}

请使用中文返回实际完成情况，并明确报告归档、笔记、链接、跳过项和问题数量。不要把计划描述成已完成的结果。
"""

    def run(self, instruction: str, sources: object = None) -> SkillRunResult:
        if not self.available or not self.codex_executable:
            raise ValueError("knowledge-ingestor Skill 或 Codex 执行环境不可用")
        task = str(instruction or "").strip()
        if not task:
            raise ValueError("请填写要交给 Skill 的任务")
        if len(task) > 20_000:
            raise ValueError("Skill 任务内容过长")
        selected_sources = self.validate_sources(sources)
        descriptor, output_name = tempfile.mkstemp(prefix="knowledge-ingestor-", suffix=".md")
        os.close(descriptor)
        output_path = Path(output_name)
        command = [
            self.codex_executable,
            "--search",
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--sandbox",
            "workspace-write",
            "--approve-for-me",
            *codex_http_transport_args(),
            "-C",
            str(self.workspace_root),
        ]
        for root in self._configured_write_roots():
            command.extend(["--add-dir", str(root)])
        command.extend(["--output-last-message", str(output_path), "-"])
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = process.communicate(
                    self.build_prompt(task, selected_sources), timeout=self.timeout_seconds
                )
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.communicate()
                raise ValueError("Skill 运行超时，任务已终止") from exc
            if process.returncode != 0:
                raise ValueError(f"Skill 运行失败：{_cli_error_detail(stderr, stdout)}")
            markdown = output_path.read_text(encoding="utf-8").strip() if output_path.is_file() else ""
            markdown = markdown or stdout.strip()
            if not markdown:
                raise ValueError("Skill 已结束，但没有返回结果")
            return SkillRunResult(SKILL_NAME, markdown, selected_sources)
        finally:
            output_path.unlink(missing_ok=True)
