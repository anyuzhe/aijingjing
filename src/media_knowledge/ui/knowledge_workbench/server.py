from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from ... import __version__
from ...answer_models import available_answer_models
from ...config import AppConfig
from ...indexing import IndexingService
from ...qa.engine import KnowledgeQAEngine
from ...retrieval import KnowledgeRetriever
from ...runtime import build_answer_provider, build_embedding_provider, build_rerank_provider
from ...storage import ConversationRepository, KnowledgeDatabase
from ...sync import ObsidianMarkdownSync
from .obsidian import ObsidianAnswerWriter
from .repository import WorkbenchRepository
from .skills import KnowledgeIngestorBridge, SKILL_NAME


ASSET_ROOT = Path(__file__).with_name("assets")


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "KnowledgeWorkbench/0.6"
    protocol_version = "HTTP/1.1"

    @property
    def config(self) -> AppConfig:
        return self.server.config  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("请求长度无效") from exc
        if length <= 0 or length > 1_000_000:
            raise ValueError("请求内容为空或过大")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求内容不是有效的 UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON 请求内容必须是对象")
        return payload

    def _ndjson(self, payload: dict) -> None:
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        self.wfile.flush()

    @staticmethod
    def _scope(payload: dict) -> dict:
        filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
        return {
            "collections": list(filters.get("collections") or []),
            "tags": list(filters.get("tags") or []),
            "media_types": list(filters.get("media_types") or []),
            "folders": list(filters.get("folders") or []),
            "document_ids": list(filters.get("document_ids") or []),
            "date_range": (
                filters.get("date_from"),
                filters.get("date_to"),
            ),
        }

    def _retriever(self, database: KnowledgeDatabase) -> KnowledgeRetriever:
        embedding = build_embedding_provider(self.config)
        return KnowledgeRetriever(
            database,
            embedding,
            rerank_provider=build_rerank_provider(self.config),
        )

    def _services(
        self,
        database: KnowledgeDatabase,
        *,
        model_id: str | None = None,
        deep_analysis: bool = False,
    ) -> tuple[KnowledgeRetriever, KnowledgeQAEngine]:
        retriever = self._retriever(database)
        return retriever, KnowledgeQAEngine(
            database,
            retriever,
            answer_provider=build_answer_provider(
                self.config,
                model_id=model_id,
                deep_analysis=deep_analysis,
            ),
        )

    def _asset(self, name: str) -> None:
        path = (ASSET_ROOT / name).resolve()
        if ASSET_ROOT.resolve() not in path.parents or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", media_type + ("; charset=utf-8" if media_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/api/health":
            self._json({"status": "ok", "version": __version__})
            return
        if route == "/api/bootstrap":
            with KnowledgeDatabase(self.config.database_path) as database:
                payload = WorkbenchRepository(database).bootstrap()
            payload["capabilities"] = {
                "web_search": False,
                "obsidian": bool(
                    self.config.obsidian_vault_root and self.config.obsidian_vault_root.is_dir()
                ),
                "obsidian_vault": (
                    self.config.obsidian_vault_root.name if self.config.obsidian_vault_root else None
                ),
                "obsidian_sync": {
                    "available": bool(
                        self.config.obsidian_vault_root and self.config.obsidian_vault_root.is_dir()
                    ),
                    "running": self.server.sync_running,  # type: ignore[attr-defined]
                    "last_result": self.server.last_sync,  # type: ignore[attr-defined]
                },
                "models": [model.to_dict() for model in available_answer_models(self.config)],
                "default_model": next(
                    model.id for model in available_answer_models(self.config) if model.default
                ),
                "answer_language": "zh-CN",
                "qa_provider": self.config.qa_provider,
                "skills": [KnowledgeIngestorBridge().status()],
            }
            self._json(payload)
            return
        if route.startswith("/api/conversations/"):
            conversation_id = unquote(route.removeprefix("/api/conversations/"))
            try:
                with KnowledgeDatabase(self.config.database_path) as database:
                    payload = ConversationRepository(database).conversation_record(conversation_id)
                self._json(payload)
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        if route == "/api/source/content":
            chunk_id = (parse_qs(parsed.query).get("chunk_id") or [""])[0]
            self._serve_source_content(chunk_id)
            return
        if route == "/":
            self._asset("index.html")
            return
        if route.startswith("/assets/"):
            self._asset(route.removeprefix("/assets/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _source_reference(self, chunk_id: str):
        if not chunk_id:
            return None
        with KnowledgeDatabase(self.config.database_path) as database:
            row = database.fetch_candidates([chunk_id]).get(chunk_id)
        return row["source_reference"] if row else None

    def _serve_source_content(self, chunk_id: str) -> None:
        reference = self._source_reference(chunk_id)
        path = Path(reference.local_path).expanduser() if reference and reference.local_path else None
        if not path or not path.is_file():
            self._json({"error": "本地原始文件不可用"}, HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start, end = 0, max(0, size - 1)
        range_header = self.headers.get("Range")
        status = HTTPStatus.OK
        if range_header and range_header.startswith("bytes="):
            try:
                raw_start, raw_end = range_header.removeprefix("bytes=").split("-", 1)
                start = int(raw_start) if raw_start else 0
                end = min(int(raw_end), size - 1) if raw_end else size - 1
                if start < 0 or end < start or start >= size:
                    raise ValueError
                status = HTTPStatus.PARTIAL_CONTENT
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        length = end - start + 1
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Disposition", f"inline; filename*=UTF-8''{quote(path.name)}")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                block = handle.read(min(64 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
            if route == "/api/search":
                self._handle_search(payload)
                return
            if route == "/api/ask/stream":
                self._handle_ask_stream(payload)
                return
            if route == "/api/obsidian/save":
                self._handle_obsidian_save(payload)
                return
            if route == "/api/obsidian/sync":
                self._handle_obsidian_sync()
                return
            if route == "/api/skills/pick-files":
                self._handle_skill_file_picker()
                return
            if route == "/api/skills/invoke/stream":
                self._handle_skill_stream(payload)
                return
            if route == "/api/source/open-native":
                self._handle_open_native(payload)
                return
            self._json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json({"error": f"请求失败：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_search(self, payload: dict) -> None:
        query = str(payload.get("query") or "").strip()
        if not query:
            raise ValueError("搜索内容不能为空")
        top_k = min(30, max(1, int(payload.get("top_k", 16))))
        with KnowledgeDatabase(self.config.database_path) as database:
            retriever = self._retriever(database)
            results = retriever.search_knowledge(query, top_k=top_k, **self._scope(payload))
        self._json({"query": query, "count": len(results), "results": [item.to_dict() for item in results]})

    def _handle_ask_stream(self, payload: dict) -> None:
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self._ndjson({"type": "status", "stage": "retrieving", "message": "正在检索选定知识"})
            with KnowledgeDatabase(self.config.database_path) as database:
                deep_analysis = bool(payload.get("deep_analysis"))
                _, engine = self._services(
                    database,
                    model_id=str(payload.get("model") or "") or None,
                    deep_analysis=deep_analysis,
                )
                answer = engine.ask(
                    question,
                    conversation_id=str(payload.get("conversation_id") or "") or None,
                    mode=str(payload.get("mode") or "knowledge"),
                    top_k=12 if deep_analysis else min(12, max(1, int(payload.get("top_k", 10)))),
                    response_language=str(payload.get("response_language") or "zh-CN"),
                    progress_callback=lambda stage, message: self._ndjson(
                        {"type": "status", "stage": stage, "message": message}
                    ),
                    **self._scope(payload),
                )
            self._ndjson({"type": "status", "stage": "finalizing", "message": "回答已生成，正在整理引用"})
            markdown = answer.markdown
            for index in range(0, len(markdown), 36):
                self._ndjson({"type": "delta", "text": markdown[index : index + 36]})
            self._ndjson({"type": "final", "answer": answer.to_dict()})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                self._ndjson({"type": "error", "error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _handle_obsidian_save(self, payload: dict) -> None:
        answer_id = str(payload.get("answer_id") or "").strip()
        if not answer_id:
            raise ValueError("缺少回答编号")
        with KnowledgeDatabase(self.config.database_path) as database:
            answer = ConversationRepository(database).answer_record(answer_id)
        tags = payload.get("tags") if isinstance(payload.get("tags"), list) else None
        result = ObsidianAnswerWriter(self.config.obsidian_vault_root).save(
            answer,
            title=str(payload.get("title") or "").strip() or None,
            tags=[str(tag) for tag in tags] if tags else None,
        )
        self._json(result, HTTPStatus.CREATED)

    def _handle_obsidian_sync(self) -> None:
        result = self.server.sync_obsidian()  # type: ignore[attr-defined]
        self._json(result)

    def _handle_skill_file_picker(self) -> None:
        files = KnowledgeIngestorBridge().pick_files()
        self._json({"files": files})

    def _handle_skill_stream(self, payload: dict) -> None:
        skill = str(payload.get("skill") or "").strip()
        if skill != SKILL_NAME:
            raise ValueError("不支持该 Skill")
        instruction = str(payload.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("请填写要交给 Skill 的任务")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self._ndjson(
                {
                    "type": "status",
                    "stage": "starting",
                    "message": "正在启动 knowledge-ingestor Skill",
                }
            )
            result = KnowledgeIngestorBridge().run(instruction, payload.get("sources"))
            self._ndjson(
                {
                    "type": "status",
                    "stage": "syncing",
                    "message": "正在把 Obsidian 新笔记同步到知识索引",
                }
            )
            try:
                sync_result = self.server.sync_obsidian()  # type: ignore[attr-defined]
            except Exception as exc:
                sync_result = {"status": "error", "error": str(exc)}
            self._ndjson(
                {
                    "type": "status",
                    "stage": "saving",
                    "message": "正在把 Skill 结果保存到对话",
                }
            )
            with KnowledgeDatabase(self.config.database_path) as database:
                repository = ConversationRepository(database)
                conversation_id = repository.ensure_conversation(
                    str(payload.get("conversation_id") or "") or None,
                    title=f"Skill · {instruction[:64]}",
                )
                repository.add_message(
                    conversation_id,
                    "user",
                    instruction,
                    {"skill": skill, "sources": result.sources},
                )
                repository.add_message(
                    conversation_id,
                    "assistant",
                    result.markdown,
                    {"skill": skill, "sources": result.sources, "sync": sync_result},
                )
            for index in range(0, len(result.markdown), 48):
                self._ndjson({"type": "delta", "text": result.markdown[index : index + 48]})
            self._ndjson(
                {
                    "type": "final",
                    "result": {
                        "skill": result.skill,
                        "markdown": result.markdown,
                        "sources": result.sources,
                        "conversation_id": conversation_id,
                        "sync": sync_result,
                    },
                }
            )
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                self._ndjson({"type": "error", "error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _handle_open_native(self, payload: dict) -> None:
        chunk_id = str(payload.get("chunk_id") or "")
        reference = self._source_reference(chunk_id)
        path = Path(reference.local_path).expanduser() if reference and reference.local_path else None
        if not path or not path.is_file():
            raise ValueError("本地原始文件不可用")
        if sys.platform == "darwin":
            command = ["open", str(path)]
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
            self._json({"opened": True, "path": str(path)})
            return
        else:
            command = ["xdg-open", str(path)]
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._json({"opened": True, "path": str(path)})


class WorkbenchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: AppConfig):
        self.config = config
        self._sync_lock = threading.Lock()
        self.last_sync: dict[str, object] | None = None
        super().__init__(address, WorkbenchHandler)

    @property
    def sync_running(self) -> bool:
        return self._sync_lock.locked()

    def sync_obsidian(self) -> dict[str, object]:
        vault = self.config.obsidian_vault_root
        if not vault or not vault.is_dir():
            raise ValueError("尚未配置可用的 Obsidian 仓库")
        if not self._sync_lock.acquire(blocking=False):
            raise ValueError("Obsidian 知识库正在同步，请稍后再试")
        try:
            with KnowledgeDatabase(self.config.database_path) as database:
                indexing = IndexingService(database, build_embedding_provider(self.config))
                result = ObsidianMarkdownSync(database, indexing, vault).sync().to_dict()
            self.last_sync = result
            return result
        finally:
            self._sync_lock.release()


def serve(
    config: AppConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    server = WorkbenchHTTPServer((host, port), config)
    if config.obsidian_vault_root and config.obsidian_vault_root.is_dir():
        try:
            server.sync_obsidian()
        except Exception as exc:
            print(f"Obsidian sync failed: {exc}", file=sys.stderr)
    url = f"http://{host}:{server.server_port}"
    print(f"Knowledge Workbench: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
