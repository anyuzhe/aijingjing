from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from ..codex_cli import codex_http_transport_args
from ..models import estimate_tokens
from ..qa.models import AnswerRequest, AnswerResponse, Evidence, TokenUsage


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


class AnswerProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def generate(self, request: AnswerRequest) -> AnswerResponse:
        raise NotImplementedError


def _terms(value: str) -> set[str]:
    lowered = value.casefold()
    words = set(re.findall(r"[a-z0-9_-]+", lowered))
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    cjk = {char for run in cjk_runs for char in run}
    cjk.update(run[index : index + 2] for run in cjk_runs for index in range(len(run) - 1))
    return words | cjk


def _best_excerpt(question: str, evidence: Evidence, limit: int = 360) -> str:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[。！？.!?])\s+|\n+", evidence.content)
        if part.strip()
    ]
    if not sentences:
        return evidence.content[:limit].strip()
    query_terms = _terms(question)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (-len(query_terms & _terms(item[1])), item[0]),
    )
    excerpt = ranked[0][1]
    return excerpt[:limit].rstrip() + ("…" if len(excerpt) > limit else "")


class ExtractiveGroundedProvider(AnswerProvider):
    """Offline fallback that can only quote or compact the supplied evidence."""

    name = "local-extractive"
    model = "grounded-extractive-v1"

    def generate(self, request: AnswerRequest) -> AnswerResponse:
        if not request.evidence:
            markdown = "知识库中没有足够资料回答这个问题。"
        else:
            lines = ["根据当前检索到的资料："]
            for evidence in request.evidence[:4]:
                excerpt = _best_excerpt(request.question, evidence)
                lines.append(f"- {excerpt} [{evidence.evidence_id}]")
            markdown = "\n".join(lines)
        usage = TokenUsage(
            input_tokens=estimate_tokens(request.user_prompt) + estimate_tokens(request.system_prompt),
            output_tokens=estimate_tokens(markdown),
        )
        return AnswerResponse(markdown, self.model, self.name, usage)


class OpenAICompatibleAnswerProvider(AnswerProvider):
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 90.0,
        temperature: float | None = 0.1,
    ):
        if not base_url or not api_key or not model:
            raise ValueError("base_url, api_key, and model are required")
        normalized = base_url.rstrip("/")
        self.endpoint = (
            normalized if normalized.endswith("/chat/completions") else normalized + "/chat/completions"
        )
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self._prefer_curl = False

    def _request_with_curl(self, payload: dict[str, object]) -> dict[str, object]:
        """Use curl when the system Python TLS stack cannot reach a compatible API."""

        curl = shutil.which("curl")
        if not curl:
            raise RuntimeError("answer provider request failed: curl is unavailable")
        header_descriptor, header_name = tempfile.mkstemp(prefix="knowledge-headers-", suffix=".txt")
        payload_descriptor, payload_name = tempfile.mkstemp(prefix="knowledge-payload-", suffix=".json")
        header_path = Path(header_name)
        payload_path = Path(payload_name)
        try:
            os.fchmod(header_descriptor, 0o600)
            os.fchmod(payload_descriptor, 0o600)
            with os.fdopen(header_descriptor, "w", encoding="utf-8") as header_file:
                header_file.write(f"Authorization: Bearer {self.api_key}\nContent-Type: application/json\n")
            with os.fdopen(payload_descriptor, "w", encoding="utf-8") as payload_file:
                json.dump(payload, payload_file, ensure_ascii=False)
            process = subprocess.run(
                [
                    curl,
                    "--silent",
                    "--show-error",
                    "--max-time",
                    str(max(1, int(self.timeout))),
                    "--header",
                    f"@{header_path}",
                    "--data-binary",
                    f"@{payload_path}",
                    "--write-out",
                    "\n%{http_code}",
                    self.endpoint,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout + 5,
            )
            if process.returncode != 0:
                detail = (process.stderr or "curl request failed").strip().splitlines()[-1]
                raise RuntimeError(f"answer provider request failed: {detail[:300]}")
            raw_body, separator, status = process.stdout.rpartition("\n")
            if not separator or not status.isdigit():
                raise RuntimeError("answer provider response has no HTTP status")
            if not 200 <= int(status) < 300:
                raise RuntimeError(f"answer provider request failed: HTTP {status}")
            body = json.loads(raw_body)
            if not isinstance(body, dict):
                raise RuntimeError("answer provider returned invalid JSON")
            return body
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("answer provider request timed out") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("answer provider returned invalid JSON") from exc
        finally:
            header_path.unlink(missing_ok=True)
            payload_path.unlink(missing_ok=True)

    def generate(self, request: AnswerRequest) -> AnswerResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        body = self.request_json(payload)
        return self._answer_response(body)

    def request_json(self, payload: dict[str, object]) -> dict[str, object]:
        """Send a raw OpenAI-compatible Chat Completions payload.

        The desktop ingestion pipeline uses this for multimodal message content while the
        grounded QA path continues to call :meth:`generate`.
        """

        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        if self._prefer_curl:
            body = self._request_with_curl(payload)
        else:
            try:
                with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f"answer provider request failed: HTTP {exc.code}") from exc
            except urllib.error.URLError:
                body = self._request_with_curl(payload)
                self._prefer_curl = True
            except json.JSONDecodeError as exc:
                raise RuntimeError("answer provider returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise RuntimeError("answer provider returned invalid JSON")
        return body

    def _answer_response(self, body: dict[str, object]) -> AnswerResponse:
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("answer provider response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        markdown = message.get("content") if isinstance(message, dict) else None
        if not isinstance(markdown, str) or not markdown.strip():
            raise RuntimeError("answer provider returned empty content")
        raw_usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        usage = TokenUsage(
            input_tokens=int(raw_usage.get("prompt_tokens", 0)),
            output_tokens=int(raw_usage.get("completion_tokens", 0)),
            total_tokens=int(raw_usage.get("total_tokens", 0)),
        )
        return AnswerResponse(markdown.strip(), self.model, self.name, usage)


class CodexAnswerProvider(AnswerProvider):
    """Use the authenticated local Codex CLI as a grounded answer synthesizer."""

    name = "codex-local"
    model = "codex-auto"

    def __init__(
        self,
        codex_executable: str | None = None,
        *,
        workspace_root: str | Path | None = None,
        timeout: float = 180.0,
        reasoning_effort: str = "low",
        model: str = "gpt-5.6-luna",
    ) -> None:
        self.codex_executable = codex_executable or shutil.which("codex")
        if not self.codex_executable:
            raise ValueError("本地 Codex 执行环境不可用")
        self.workspace_root = Path(workspace_root or Path.cwd()).expanduser().resolve()
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort.strip().casefold()
        self.model = model

    def generate(self, request: AnswerRequest) -> AnswerResponse:
        descriptor, output_name = tempfile.mkstemp(prefix="knowledge-answer-", suffix=".md")
        os.close(descriptor)
        output_path = Path(output_name)
        command = [
            self.codex_executable,
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--sandbox",
            "read-only",
            "--model",
            self.model,
            "--config",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            *codex_http_transport_args(),
            "-C",
            str(self.workspace_root),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        prompt = (
            "请直接完成下面的知识库问答，不要调用工具、不要读取本地文件、不要搜索网络。\n\n"
            f"系统要求：\n{request.system_prompt}\n\n"
            f"用户与证据：\n{request.user_prompt}\n"
        )
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = process.communicate(prompt, timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.communicate()
                raise RuntimeError("中文回答生成超时") from exc
            if process.returncode != 0:
                raise RuntimeError(f"中文回答生成失败：{_cli_error_detail(stderr, stdout)}")
            markdown = output_path.read_text(encoding="utf-8").strip() if output_path.is_file() else ""
            markdown = markdown or stdout.strip()
            if not markdown:
                raise RuntimeError("回答模型没有返回内容")
            usage = TokenUsage(
                input_tokens=estimate_tokens(prompt),
                output_tokens=estimate_tokens(markdown),
            )
            return AnswerResponse(markdown, self.model, self.name, usage)
        finally:
            output_path.unlink(missing_ok=True)
