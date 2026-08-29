from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .config import AppConfig
from .documents import load_documents
from .indexing import IndexingService
from .qa.engine import KnowledgeQAEngine
from .retrieval import KnowledgeRetriever
from .runtime import build_answer_provider, build_embedding_provider, build_rerank_provider
from .storage import ConversationRepository, KnowledgeDatabase


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knowledge", description="AI知识库-AI静静")
    parser.add_argument("--db", help="SQLite database path (default: KNOWLEDGE_DB or .knowledge/knowledge.db)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Index one or more UCB JSON, Markdown, or text files")
    index.add_argument("paths", nargs="+")
    index.add_argument("--title", help="Override title when indexing one Markdown/text file")
    index.add_argument("--media-type", help="Override media type for Markdown/text")
    index.add_argument("--collection", action="append", default=[])
    index.add_argument("--tag", action="append", default=[])

    subparsers.add_parser("reindex", help="Rebuild embeddings and FTS from stored chunks")

    search = subparsers.add_parser("search", help="Hybrid vector + BM25 search")
    search.add_argument("query")
    search.add_argument("--collection", action="append", default=[])
    search.add_argument("--tag", action="append", default=[])
    search.add_argument("--media-type", action="append", default=[])
    search.add_argument("--folder", action="append", default=[])
    search.add_argument("--document-id", action="append", default=[])
    search.add_argument("--date-from")
    search.add_argument("--date-to")
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--hide-debug", action="store_true")

    ask = subparsers.add_parser("ask", help="Grounded knowledge question answering with citations")
    ask.add_argument("question")
    ask.add_argument("--conversation-id")
    ask.add_argument("--mode", choices=("knowledge", "knowledge+web"), default="knowledge")
    ask.add_argument("--collection", action="append", default=[])
    ask.add_argument("--tag", action="append", default=[])
    ask.add_argument("--media-type", action="append", default=[])
    ask.add_argument("--folder", action="append", default=[])
    ask.add_argument("--document-id", action="append", default=[])
    ask.add_argument("--date-from")
    ask.add_argument("--date-to")
    ask.add_argument("--top-k", type=int, default=10)
    ask.add_argument("--hide-evidence", action="store_true")

    evaluate = subparsers.add_parser(
        "eval", help="Evaluate retrieval and citations against a local golden JSON dataset"
    )
    evaluate.add_argument("dataset", help="Path to the golden evaluation JSON file")
    evaluate.add_argument("--top-k", type=int, default=10)
    evaluate.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip deterministic local answer/citation evaluation",
    )

    conversation = subparsers.add_parser("conversation-show", help="Show persisted conversation and summary")
    conversation.add_argument("conversation_id")

    workbench = subparsers.add_parser("workbench", help="Start the local three-column Knowledge Workbench")
    workbench.add_argument("--host", default="127.0.0.1")
    workbench.add_argument("--port", type=int, default=8765)
    workbench.add_argument("--obsidian-vault")
    workbench.add_argument("--no-open", action="store_true")

    desktop = subparsers.add_parser("desktop", help="启动 AI知识库-AI静静 桌面应用")
    desktop.add_argument("--data-dir", help="自定义知识数据目录")

    ingest = subparsers.add_parser("ingest", help="使用桌面版内置摄取服务批量入库")
    ingest.add_argument("sources", nargs="+")
    ingest.add_argument("--data-dir", help="自定义知识数据目录")

    doctor = subparsers.add_parser("doctor", help="检查桌面版与多媒体组件")
    doctor.add_argument("--data-dir", help="自定义知识数据目录")

    subparsers.add_parser("index-status", help="Show document, chunk, embedding, and facet counts")
    delete = subparsers.add_parser("delete", help="Delete a document and cascade chunks/embeddings")
    delete.add_argument("document_id")
    return parser


def _dump(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "desktop":
        from .desktop.app import create_application

        application, window = create_application(args.data_dir)
        window.show()
        return int(application.exec())
    if args.command == "ingest":
        from .desktop import DesktopController

        summary = DesktopController(args.data_dir).ingest(args.sources)
        _dump(summary.to_dict())
        return 2 if summary.failed else 0
    if args.command == "doctor":
        from .desktop import DesktopController
        from .desktop.diagnostics import run_diagnostics

        _dump(run_diagnostics(DesktopController(args.data_dir)))
        return 0
    config = AppConfig.from_env(args.db)
    if args.command == "workbench":
        if "KNOWLEDGE_QA_PROVIDER" not in os.environ:
            deepseek_default = next(
                (
                    provider
                    for provider in config.qa_compatible_providers
                    if provider.id == "deepseek"
                    and "deepseek-v4-flash-vision-exp" in provider.models
                ),
                None,
            )
            if deepseek_default:
                config.qa_provider = deepseek_default.id
                config.qa_model = "deepseek-v4-flash-vision-exp"
        if args.obsidian_vault:
            config.obsidian_vault_root = Path(args.obsidian_vault).expanduser().resolve()
        from .ui.knowledge_workbench import serve

        serve(config, host=args.host, port=args.port, open_browser=not args.no_open)
        return 0
    embedding = build_embedding_provider(config)
    with KnowledgeDatabase(config.database_path) as database:
        indexing = IndexingService(database, embedding)
        if args.command == "index":
            if args.title and len(args.paths) != 1:
                raise ValueError("--title can only be used with one input path")
            reports = []
            for raw_path in args.paths:
                documents = load_documents(
                    Path(raw_path),
                    title=args.title,
                    media_type=args.media_type,
                    collections=args.collection,
                    tags=args.tag,
                )
                reports.extend(indexing.index_document(document).to_dict() for document in documents)
            _dump(reports)
            return 0
        if args.command == "reindex":
            _dump(indexing.reindex())
            return 0
        if args.command == "search":
            retriever = KnowledgeRetriever(
                database,
                embedding,
                rerank_provider=build_rerank_provider(config),
            )
            results = retriever.search_knowledge(
                args.query,
                collections=args.collection,
                tags=args.tag,
                media_types=args.media_type,
                folders=args.folder,
                document_ids=args.document_id,
                date_range=(args.date_from, args.date_to),
                top_k=args.top_k,
            )
            payload = [result.to_dict() for result in results]
            if args.hide_debug:
                for item in payload:
                    item.pop("debug", None)
            _dump(payload)
            return 0
        if args.command == "ask":
            retriever = KnowledgeRetriever(
                database,
                embedding,
                rerank_provider=build_rerank_provider(config),
            )
            engine = KnowledgeQAEngine(
                database,
                retriever,
                answer_provider=build_answer_provider(config),
            )
            payload = engine.ask(
                args.question,
                conversation_id=args.conversation_id,
                mode=args.mode,
                collections=args.collection,
                tags=args.tag,
                media_types=args.media_type,
                folders=args.folder,
                document_ids=args.document_id,
                date_range=(args.date_from, args.date_to),
                top_k=args.top_k,
            ).to_dict()
            if args.hide_evidence:
                payload.pop("evidence", None)
            _dump(payload)
            return 0
        if args.command == "eval":
            from .evaluation import GoldenEvaluator, load_golden_dataset

            retriever = KnowledgeRetriever(
                database,
                embedding,
                rerank_provider=build_rerank_provider(config),
            )
            engine = None if args.retrieval_only else KnowledgeQAEngine(database, retriever)
            report = GoldenEvaluator(retriever, qa_engine=engine).evaluate(
                load_golden_dataset(args.dataset),
                top_k=args.top_k,
                evaluate_citations=not args.retrieval_only,
            )
            _dump(report)
            return 0
        if args.command == "conversation-show":
            _dump(ConversationRepository(database).conversation_record(args.conversation_id))
            return 0
        if args.command == "index-status":
            _dump(database.status())
            return 0
        if args.command == "delete":
            deleted = indexing.delete_document(args.document_id)
            _dump({"document_id": args.document_id, "deleted": deleted})
            return 0 if deleted else 2
    return 1


def main() -> None:
    try:
        raise SystemExit(run())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"knowledge: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
