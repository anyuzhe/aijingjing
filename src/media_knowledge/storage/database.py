from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..models import KnowledgeChunk, KnowledgeDocument, SearchFilters, SourceReference, utcnow_iso


BASE_SCHEMA_VERSION = 8
SCHEMA_VERSION = 11


class KnowledgeDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "KnowledgeDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def migrate(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    checksum TEXT,
                    original_uri TEXT,
                    local_path TEXT,
                    obsidian_path TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
                CREATE INDEX IF NOT EXISTS idx_documents_checksum ON documents(checksum);
                CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at DESC);

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_key TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    heading_path_json TEXT NOT NULL DEFAULT '[]',
                    token_count INTEGER NOT NULL,
                    source_reference_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding_status TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(document_id, chunk_key)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);
                CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

                CREATE TABLE IF NOT EXISTS source_references (
                    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                    source_id TEXT NOT NULL,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    media_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    original_uri TEXT,
                    local_path TEXT,
                    obsidian_path TEXT,
                    page_number INTEGER,
                    slide_number INTEGER,
                    timestamp_start REAL,
                    timestamp_end REAL,
                    section TEXT,
                    image_path TEXT,
                    text_start INTEGER,
                    text_end INTEGER,
                    checksum TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_source_references_source ON source_references(source_id);
                CREATE INDEX IF NOT EXISTS idx_source_references_document ON source_references(document_id);

                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS collections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS document_collections (
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                    PRIMARY KEY(document_id, collection_id)
                );
                CREATE INDEX IF NOT EXISTS idx_document_collections_collection
                    ON document_collections(collection_id, document_id);
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS document_tags (
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
                    PRIMARY KEY(document_id, tag_id)
                );
                CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag_id, document_id);

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_through_ordinal INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(conversation_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, ordinal);

                CREATE TABLE IF NOT EXISTS answers (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    question_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    answer_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    markdown TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    token_usage_json TEXT NOT NULL DEFAULT '{}',
                    retrieval_info_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_answers_conversation ON answers(conversation_id, created_at);

                CREATE TABLE IF NOT EXISTS answer_evidence (
                    answer_id TEXT NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
                    evidence_id TEXT NOT NULL,
                    source_kind TEXT NOT NULL CHECK(source_kind IN ('knowledge', 'web')),
                    document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
                    chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    score REAL NOT NULL,
                    source_reference_json TEXT NOT NULL,
                    PRIMARY KEY(answer_id, evidence_id)
                );
                CREATE INDEX IF NOT EXISTS idx_answer_evidence_chunk ON answer_evidence(chunk_id);

                CREATE TABLE IF NOT EXISTS citations (
                    answer_id TEXT NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
                    citation_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    source_kind TEXT NOT NULL CHECK(source_kind IN ('knowledge', 'web')),
                    document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
                    chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
                    media_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    original_uri TEXT,
                    local_path TEXT,
                    obsidian_path TEXT,
                    page_number INTEGER,
                    slide_number INTEGER,
                    timestamp_start REAL,
                    timestamp_end REAL,
                    section TEXT,
                    PRIMARY KEY(answer_id, citation_id),
                    FOREIGN KEY(answer_id, evidence_id)
                        REFERENCES answer_evidence(answer_id, evidence_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_citations_chunk ON citations(chunk_id);

                CREATE TABLE IF NOT EXISTS annotations (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    locator_json TEXT NOT NULL DEFAULT '{}',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_annotations_document ON annotations(document_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS watched_folders (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    collection TEXT NOT NULL DEFAULT '自动同步',
                    recursive INTEGER NOT NULL DEFAULT 1,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_scan_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    source_document_ids_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_updated ON artifacts(updated_at DESC);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    title,
                    content,
                    tokenize='unicode61'
                );
                """
            )
            document_columns = {
                row["name"] for row in self.connection.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "enabled" not in document_columns:
                self.connection.execute(
                    "ALTER TABLE documents ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1"
                )
            # Versions 1-8 were historically maintained by the idempotent base schema
            # above.  Keep that compatibility baseline, then apply every later change as
            # an explicit, recorded migration so existing user databases are upgraded
            # without relying on the final CREATE TABLE declarations alone.
            self.connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (BASE_SCHEMA_VERSION, utcnow_iso()),
            )
            self._apply_incremental_migrations()
            missing_references = self.connection.execute(
                """SELECT c.id, c.document_id, c.source_reference_json
                   FROM chunks c LEFT JOIN source_references sr ON sr.chunk_id=c.id
                   WHERE sr.chunk_id IS NULL"""
            ).fetchall()
            for row in missing_references:
                self._upsert_source_reference(
                    row["id"],
                    row["document_id"],
                    SourceReference.from_dict(json.loads(row["source_reference_json"])),
                )
            self.connection.execute("PRAGMA optimize")

    def _apply_incremental_migrations(self) -> None:
        applied = {
            int(row["version"])
            for row in self.connection.execute("SELECT version FROM schema_migrations").fetchall()
        }
        migrations = (
            (9, self._migrate_answer_feedback),
            (10, self._migrate_ingestion_jobs),
            (11, self._migrate_knowledge_governance),
        )
        for version, migration in migrations:
            if version in applied:
                continue
            migration()
            self.connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (version, utcnow_iso()),
            )

    def _migrate_answer_feedback(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS answer_feedback (
                   answer_id TEXT PRIMARY KEY REFERENCES answers(id) ON DELETE CASCADE,
                   rating TEXT NOT NULL CHECK(rating IN ('up', 'down')),
                   reason TEXT NOT NULL DEFAULT '',
                   created_at TEXT NOT NULL,
                   updated_at TEXT NOT NULL
               )"""
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_answer_feedback_rating ON answer_feedback(rating, updated_at DESC)"
        )

    def _migrate_ingestion_jobs(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS ingestion_jobs (
                   id TEXT PRIMARY KEY,
                   status TEXT NOT NULL CHECK(status IN (
                       'queued', 'running', 'completed', 'failed', 'cancelled'
                   )),
                   total_items INTEGER NOT NULL DEFAULT 0 CHECK(total_items >= 0),
                   completed_items INTEGER NOT NULL DEFAULT 0 CHECK(completed_items >= 0),
                   succeeded_items INTEGER NOT NULL DEFAULT 0 CHECK(succeeded_items >= 0),
                   failed_items INTEGER NOT NULL DEFAULT 0 CHECK(failed_items >= 0),
                   cancelled_items INTEGER NOT NULL DEFAULT 0 CHECK(cancelled_items >= 0),
                   progress_percent INTEGER NOT NULL DEFAULT 0 CHECK(
                       progress_percent >= 0 AND progress_percent <= 100
                   ),
                   current_item TEXT,
                   current_stage TEXT,
                   message TEXT NOT NULL DEFAULT '',
                   error TEXT,
                   metadata_json TEXT NOT NULL DEFAULT '{}',
                   created_at TEXT NOT NULL,
                   started_at TEXT,
                   completed_at TEXT,
                   updated_at TEXT NOT NULL
               )"""
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status_updated "
            "ON ingestion_jobs(status, updated_at DESC)"
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS ingestion_job_items (
                   id TEXT PRIMARY KEY,
                   job_id TEXT NOT NULL REFERENCES ingestion_jobs(id) ON DELETE CASCADE,
                   ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                   source TEXT NOT NULL,
                   status TEXT NOT NULL CHECK(status IN (
                       'queued', 'running', 'completed', 'failed', 'cancelled'
                   )),
                   progress_percent INTEGER NOT NULL DEFAULT 0 CHECK(
                       progress_percent >= 0 AND progress_percent <= 100
                   ),
                   stage TEXT NOT NULL DEFAULT 'queued',
                   message TEXT NOT NULL DEFAULT '',
                   result_json TEXT NOT NULL DEFAULT '{}',
                   error TEXT,
                   created_at TEXT NOT NULL,
                   started_at TEXT,
                   completed_at TEXT,
                   updated_at TEXT NOT NULL,
                   UNIQUE(job_id, ordinal)
               )"""
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestion_job_items_job_status "
            "ON ingestion_job_items(job_id, status, ordinal)"
        )

    def _migrate_knowledge_governance(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL CHECK(item_type IN (
                    'source', 'topic', 'entity', 'analysis', 'decision', 'output'
                )),
                status TEXT NOT NULL CHECK(status IN (
                    'draft', 'current', 'needs-review', 'stale', 'archived'
                )),
                maturity TEXT NOT NULL CHECK(maturity IN (
                    'unreviewed', 'indexed', 'summarized', 'compiled', 'low-value'
                )),
                title TEXT NOT NULL CHECK(length(trim(title)) > 0),
                summary TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL DEFAULT '',
                document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
                artifact_id TEXT REFERENCES artifacts(id) ON DELETE SET NULL,
                high_value INTEGER NOT NULL DEFAULT 0 CHECK(high_value IN (0, 1)),
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_knowledge_items_type_status
                ON knowledge_items(item_type, status, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_knowledge_items_maturity
                ON knowledge_items(maturity, high_value, updated_at DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_items_source_document
                ON knowledge_items(document_id)
                WHERE document_id IS NOT NULL AND item_type = 'source';
            CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_items_output_artifact
                ON knowledge_items(artifact_id)
                WHERE artifact_id IS NOT NULL AND item_type = 'output';

            CREATE TABLE IF NOT EXISTS knowledge_aliases (
                item_id TEXT NOT NULL REFERENCES knowledge_items(id) ON DELETE CASCADE,
                alias TEXT NOT NULL CHECK(length(trim(alias)) > 0),
                normalized_alias TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(item_id, normalized_alias)
            );
            CREATE INDEX IF NOT EXISTS idx_knowledge_aliases_normalized
                ON knowledge_aliases(normalized_alias, item_id);

            CREATE TABLE IF NOT EXISTS knowledge_item_tags (
                item_id TEXT NOT NULL REFERENCES knowledge_items(id) ON DELETE CASCADE,
                tag TEXT NOT NULL CHECK(length(trim(tag)) > 0),
                normalized_tag TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(item_id, normalized_tag)
            );
            CREATE INDEX IF NOT EXISTS idx_knowledge_item_tags_normalized
                ON knowledge_item_tags(normalized_tag, item_id);

            CREATE TABLE IF NOT EXISTS knowledge_relations (
                id TEXT PRIMARY KEY,
                source_item_id TEXT NOT NULL REFERENCES knowledge_items(id) ON DELETE CASCADE,
                target_item_id TEXT NOT NULL REFERENCES knowledge_items(id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL CHECK(relation_type IN (
                    'supports', 'extends', 'contradicts', 'supersedes', 'opens'
                )),
                summary TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK(source_item_id <> target_item_id),
                UNIQUE(source_item_id, target_item_id, relation_type)
            );
            CREATE INDEX IF NOT EXISTS idx_knowledge_relations_source
                ON knowledge_relations(source_item_id, relation_type, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_knowledge_relations_target
                ON knowledge_relations(target_item_id, relation_type, updated_at DESC);
            """
        )
        now = utcnow_iso()
        self.connection.execute(
            """INSERT OR IGNORE INTO knowledge_items(
                   id, item_type, status, maturity, title, summary, body,
                   document_id, artifact_id, high_value, metadata_json, created_at, updated_at
               )
               SELECT 'kg:document:' || id, 'source', 'current', 'indexed', title, '', '',
                      id, NULL, 0, '{"managed_from":"documents"}', created_at, updated_at
               FROM documents"""
        )
        self.connection.execute(
            """INSERT OR IGNORE INTO knowledge_items(
                   id, item_type, status, maturity, title, summary, body,
                   document_id, artifact_id, high_value, metadata_json, created_at, updated_at
               )
               SELECT 'kg:artifact:' || id, 'output', 'draft', 'compiled', title, '', markdown,
                      NULL, id, 0, '{"managed_from":"artifacts"}', created_at, updated_at
               FROM artifacts"""
        )
        artifact_rows = self.connection.execute(
            "SELECT id, source_document_ids_json FROM artifacts"
        ).fetchall()
        for row in artifact_rows:
            try:
                document_ids = json.loads(row["source_document_ids_json"] or "[]")
            except json.JSONDecodeError:
                document_ids = []
            for document_id in document_ids if isinstance(document_ids, list) else []:
                source_item_id = f"kg:document:{document_id}"
                target_item_id = f"kg:artifact:{row['id']}"
                exists = self.connection.execute(
                    "SELECT 1 FROM knowledge_items WHERE id=?", (source_item_id,)
                ).fetchone()
                if not exists:
                    continue
                digest = hashlib.sha256(
                    f"{source_item_id}|{target_item_id}|supports".encode("utf-8")
                ).hexdigest()[:24]
                self.connection.execute(
                    """INSERT OR IGNORE INTO knowledge_relations(
                           id, source_item_id, target_item_id, relation_type, summary,
                           metadata_json, created_at, updated_at
                       ) VALUES (?, ?, ?, 'supports', '来源资料支持该输出',
                                 '{"managed_from":"artifacts"}', ?, ?)""",
                    (f"rel-{digest}", source_item_id, target_item_id, now, now),
                )

    def get_document_by_source_id(self, source_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM documents WHERE source_id = ?", (source_id,)
        ).fetchone()

    def get_document_by_content_hash(self, content_hash: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM documents WHERE content_hash = ? ORDER BY created_at LIMIT 1",
            (content_hash,),
        ).fetchone()

    def get_document_by_checksum(self, checksum: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM documents WHERE checksum = ? ORDER BY created_at LIMIT 1", (checksum,)
        ).fetchone()

    def get_document(self, document_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()

    def duplicate_groups(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT COALESCE(NULLIF(checksum, ''), content_hash) AS fingerprint,
                      COUNT(*) AS document_count,
                      GROUP_CONCAT(id, CHAR(31)) AS document_ids,
                      GROUP_CONCAT(title, CHAR(31)) AS titles
               FROM documents
               GROUP BY COALESCE(NULLIF(checksum, ''), content_hash)
               HAVING COUNT(*) > 1
               ORDER BY document_count DESC"""
        ).fetchall()
        return [
            {
                "fingerprint": row["fingerprint"],
                "document_count": int(row["document_count"]),
                "document_ids": str(row["document_ids"]).split(chr(31)),
                "titles": str(row["titles"]).split(chr(31)),
            }
            for row in rows
        ]

    def upsert_document(self, document: KnowledgeDocument, content_hash: str) -> None:
        source = document.source
        self.connection.execute(
            """
            INSERT INTO documents(
                id, source_id, title, media_type, content_hash, checksum, original_uri,
                local_path, obsidian_path, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_id=excluded.source_id, title=excluded.title, media_type=excluded.media_type,
                content_hash=excluded.content_hash, checksum=excluded.checksum,
                original_uri=excluded.original_uri, local_path=excluded.local_path,
                obsidian_path=excluded.obsidian_path, metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                document.document_id,
                document.source_id,
                document.title,
                document.media_type,
                content_hash,
                source.checksum,
                source.original_uri,
                source.local_path,
                source.obsidian_path,
                json.dumps(document.metadata, ensure_ascii=False, sort_keys=True),
                document.created_at,
                document.updated_at,
            ),
        )
        self.connection.execute(
            """UPDATE knowledge_items SET title=?, updated_at=?
               WHERE document_id=? AND item_type='source' AND metadata_json=?""",
            (
                document.title,
                document.updated_at,
                document.document_id,
                '{"managed_from":"documents"}',
            ),
        )
        self.connection.execute(
            """INSERT OR IGNORE INTO knowledge_items(
                   id, item_type, status, maturity, title, summary, body,
                   document_id, artifact_id, high_value, metadata_json, created_at, updated_at
               ) VALUES (?, 'source', 'current', 'indexed', ?, '', '', ?, NULL, 0,
                         '{"managed_from":"documents"}', ?, ?)""",
            (
                f"kg:document:{document.document_id}",
                document.title,
                document.document_id,
                document.created_at,
                document.updated_at,
            ),
        )

    def replace_facets(self, document_id: str, collections: Sequence[str], tags: Sequence[str]) -> None:
        self.connection.execute("DELETE FROM document_collections WHERE document_id = ?", (document_id,))
        self.connection.execute("DELETE FROM document_tags WHERE document_id = ?", (document_id,))
        for name in sorted({item.strip() for item in collections if item.strip()}):
            self.connection.execute("INSERT OR IGNORE INTO collections(name) VALUES (?)", (name,))
            self.connection.execute(
                """INSERT INTO document_collections(document_id, collection_id)
                   SELECT ?, id FROM collections WHERE name = ?""",
                (document_id, name),
            )
        for name in sorted({item.strip() for item in tags if item.strip()}):
            self.connection.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (name,))
            self.connection.execute(
                """INSERT INTO document_tags(document_id, tag_id)
                   SELECT ?, id FROM tags WHERE name = ?""",
                (document_id, name),
            )

    def update_document_facets(
        self,
        document_id: str,
        collections: Sequence[str],
        tags: Sequence[str],
    ) -> None:
        with self.connection:
            self.replace_facets(document_id, collections, tags)

    def list_facets(self) -> dict[str, list[dict[str, Any]]]:
        collections = self.connection.execute(
            """SELECT co.name, COUNT(dc.document_id) AS document_count
               FROM collections co LEFT JOIN document_collections dc ON dc.collection_id=co.id
               GROUP BY co.id ORDER BY co.name"""
        ).fetchall()
        tags = self.connection.execute(
            """SELECT t.name, COUNT(dt.document_id) AS document_count
               FROM tags t LEFT JOIN document_tags dt ON dt.tag_id=t.id
               GROUP BY t.id ORDER BY t.name"""
        ).fetchall()
        return {
            "collections": [dict(row) for row in collections],
            "tags": [dict(row) for row in tags],
        }

    def document_facets(self, document_id: str) -> dict[str, list[str]]:
        collections = self.connection.execute(
            """SELECT co.name FROM collections co JOIN document_collections dc ON dc.collection_id=co.id
               WHERE dc.document_id=? ORDER BY co.name""",
            (document_id,),
        ).fetchall()
        tags = self.connection.execute(
            """SELECT t.name FROM tags t JOIN document_tags dt ON dt.tag_id=t.id
               WHERE dt.document_id=? ORDER BY t.name""",
            (document_id,),
        ).fetchall()
        return {
            "collections": [str(row["name"]) for row in collections],
            "tags": [str(row["name"]) for row in tags],
        }

    def get_chunks(self, document_id: str) -> dict[str, sqlite3.Row]:
        rows = self.connection.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY ordinal", (document_id,)
        ).fetchall()
        return {row["chunk_key"]: row for row in rows}

    def list_chunks(self, document_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT c.*, sr.page_number, sr.slide_number, sr.timestamp_start,
                      sr.timestamp_end, sr.section, sr.image_path
               FROM chunks c LEFT JOIN source_references sr ON sr.chunk_id=c.id
               WHERE c.document_id=? ORDER BY c.ordinal""",
            (document_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def all_chunks(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            """SELECT c.*, d.title FROM chunks c JOIN documents d ON d.id = c.document_id
               ORDER BY c.document_id, c.ordinal"""
        ).fetchall()

    def upsert_chunk(self, chunk: KnowledgeChunk, title: str) -> None:
        self.connection.execute(
            """
            INSERT INTO chunks(
                id, document_id, chunk_key, ordinal, content, heading_path_json, token_count,
                source_reference_json, metadata_json, embedding_status, content_hash,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                ordinal=excluded.ordinal, content=excluded.content,
                heading_path_json=excluded.heading_path_json, token_count=excluded.token_count,
                source_reference_json=excluded.source_reference_json,
                metadata_json=excluded.metadata_json, embedding_status=excluded.embedding_status,
                content_hash=excluded.content_hash, updated_at=excluded.updated_at
            """,
            (
                chunk.id,
                chunk.document_id,
                chunk.chunk_key,
                chunk.ordinal,
                chunk.content,
                json.dumps(chunk.heading_path, ensure_ascii=False),
                chunk.token_count,
                json.dumps(chunk.source_reference.to_dict(), ensure_ascii=False, sort_keys=True),
                json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                chunk.embedding_status,
                chunk.content_hash,
                chunk.created_at,
                chunk.updated_at,
            ),
        )
        self._upsert_source_reference(chunk.id, chunk.document_id, chunk.source_reference)
        self.connection.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk.id,))
        self.connection.execute(
            "INSERT INTO chunks_fts(chunk_id, document_id, title, content) VALUES (?, ?, ?, ?)",
            (chunk.id, chunk.document_id, title, chunk.content),
        )

    def _upsert_source_reference(
        self, chunk_id: str, document_id: str, reference: SourceReference
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO source_references(
                chunk_id, source_id, document_id, media_type, title, original_uri, local_path,
                obsidian_path, page_number, slide_number, timestamp_start, timestamp_end,
                section, image_path, text_start, text_end, checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                source_id=excluded.source_id, document_id=excluded.document_id,
                media_type=excluded.media_type, title=excluded.title,
                original_uri=excluded.original_uri, local_path=excluded.local_path,
                obsidian_path=excluded.obsidian_path, page_number=excluded.page_number,
                slide_number=excluded.slide_number, timestamp_start=excluded.timestamp_start,
                timestamp_end=excluded.timestamp_end, section=excluded.section,
                image_path=excluded.image_path, text_start=excluded.text_start,
                text_end=excluded.text_end, checksum=excluded.checksum
            """,
            (
                chunk_id,
                reference.source_id,
                document_id,
                reference.media_type,
                reference.title,
                reference.original_uri,
                reference.local_path,
                reference.obsidian_path,
                reference.page_number,
                reference.slide_number,
                reference.timestamp_start,
                reference.timestamp_end,
                reference.section,
                reference.image_path,
                reference.text_start,
                reference.text_end,
                reference.checksum,
            ),
        )

    def set_embedding_status(self, chunk_id: str, status: str) -> None:
        self.connection.execute(
            "UPDATE chunks SET embedding_status = ?, updated_at = ? WHERE id = ?",
            (status, utcnow_iso(), chunk_id),
        )

    def delete_chunks(self, chunk_ids: Iterable[str]) -> int:
        ids = list(chunk_ids)
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        self.connection.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", ids)
        cursor = self.connection.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
        return cursor.rowcount

    def upsert_embedding(
        self,
        chunk_id: str,
        provider: str,
        model: str,
        vector: Sequence[float],
        content_hash: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO embeddings(chunk_id, provider, model, dimensions, vector_json, content_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                provider=excluded.provider, model=excluded.model, dimensions=excluded.dimensions,
                vector_json=excluded.vector_json, content_hash=excluded.content_hash,
                updated_at=excluded.updated_at
            """,
            (
                chunk_id,
                provider,
                model,
                len(vector),
                json.dumps(list(vector), separators=(",", ":")),
                content_hash,
                utcnow_iso(),
            ),
        )
        self.set_embedding_status(chunk_id, "ready")

    @staticmethod
    def _filter_sql(filters: SearchFilters, alias: str = "d") -> tuple[list[str], list[Any]]:
        clauses: list[str] = [f"{alias}.enabled = 1"]
        params: list[Any] = []
        if filters.media_types:
            placeholders = ",".join("?" for _ in filters.media_types)
            clauses.append(f"{alias}.media_type IN ({placeholders})")
            params.extend(filters.media_types)
        if filters.collections:
            placeholders = ",".join("?" for _ in filters.collections)
            clauses.append(
                f"EXISTS (SELECT 1 FROM document_collections dc JOIN collections co ON co.id=dc.collection_id "
                f"WHERE dc.document_id={alias}.id AND co.name IN ({placeholders}))"
            )
            params.extend(filters.collections)
        if filters.tags:
            placeholders = ",".join("?" for _ in filters.tags)
            clauses.append(
                f"EXISTS (SELECT 1 FROM document_tags dt JOIN tags t ON t.id=dt.tag_id "
                f"WHERE dt.document_id={alias}.id AND t.name IN ({placeholders}))"
            )
            params.extend(filters.tags)
        if filters.document_ids:
            placeholders = ",".join("?" for _ in filters.document_ids)
            clauses.append(f"{alias}.id IN ({placeholders})")
            params.extend(filters.document_ids)
        if filters.folders:
            folder_clauses = []
            for folder in filters.folders:
                escaped = folder.rstrip("/\\").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                folder_clauses.append(
                    f"({alias}.local_path LIKE ? ESCAPE '\\' OR {alias}.obsidian_path LIKE ? ESCAPE '\\')"
                )
                params.extend((escaped + "/%", escaped + "/%"))
            clauses.append("(" + " OR ".join(folder_clauses) + ")")
        if filters.date_from:
            clauses.append(f"{alias}.updated_at >= ?")
            params.append(filters.date_from)
        if filters.date_to:
            clauses.append(f"{alias}.updated_at <= ?")
            params.append(filters.date_to)
        return clauses, params

    def iter_embeddings(
        self,
        filters: SearchFilters,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[sqlite3.Row]:
        clauses, params = self._filter_sql(filters)
        if provider:
            clauses.append("e.provider = ?")
            params.append(provider)
        if model:
            clauses.append("e.model = ?")
            params.append(model)
        where = " AND ".join(["c.embedding_status = 'ready'", *clauses])
        return self.connection.execute(
            f"""SELECT e.chunk_id, e.vector_json FROM embeddings e
                 JOIN chunks c ON c.id=e.chunk_id JOIN documents d ON d.id=c.document_id
                 WHERE {where}""",
            params,
        ).fetchall()

    def embedding_profile(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT provider, model, dimensions, COUNT(*) AS chunk_count
               FROM embeddings GROUP BY provider, model, dimensions
               ORDER BY chunk_count DESC, provider, model"""
        ).fetchall()
        return [dict(row) for row in rows]

    def embeddings_need_reindex(self, provider: str, model: str) -> bool:
        row = self.connection.execute(
            """SELECT
                   (SELECT COUNT(*) FROM chunks) AS total,
                   (SELECT COUNT(*) FROM embeddings e JOIN chunks c ON c.id=e.chunk_id
                    WHERE e.provider=? AND e.model=? AND c.embedding_status='ready') AS matching""",
            (provider, model),
        ).fetchone()
        return bool(row and int(row["total"]) != int(row["matching"]))

    def keyword_search(self, fts_query: str, limit: int, filters: SearchFilters) -> list[sqlite3.Row]:
        clauses, params = self._filter_sql(filters)
        where = " AND ".join(["chunks_fts MATCH ?", *clauses])
        return self.connection.execute(
            f"""SELECT f.chunk_id, bm25(chunks_fts, 0.0, 0.0, 1.5, 1.0) AS rank
                 FROM chunks_fts f JOIN documents d ON d.id=f.document_id
                 WHERE {where} ORDER BY rank ASC LIMIT ?""",
            [fts_query, *params, limit],
        ).fetchall()

    def fetch_candidates(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.connection.execute(
            f"""SELECT c.id AS chunk_id, c.content, d.title AS document_title,
                        sr.source_id, sr.document_id, sr.media_type, sr.title AS reference_title,
                        sr.original_uri, sr.local_path, sr.obsidian_path, sr.page_number,
                        sr.slide_number, sr.timestamp_start, sr.timestamp_end, sr.section,
                        sr.image_path, sr.text_start, sr.text_end, sr.checksum
                 FROM chunks c JOIN documents d ON d.id=c.document_id
                 JOIN source_references sr ON sr.chunk_id=c.id
                 WHERE c.id IN ({placeholders})""",
            list(chunk_ids),
        ).fetchall()
        return {
            row["chunk_id"]: {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "title": row["document_title"],
                "source_reference": SourceReference(
                    source_id=row["source_id"],
                    document_id=row["document_id"],
                    chunk_id=row["chunk_id"],
                    media_type=row["media_type"],
                    title=row["reference_title"],
                    original_uri=row["original_uri"],
                    local_path=row["local_path"],
                    obsidian_path=row["obsidian_path"],
                    page_number=row["page_number"],
                    slide_number=row["slide_number"],
                    timestamp_start=row["timestamp_start"],
                    timestamp_end=row["timestamp_end"],
                    section=row["section"],
                    image_path=row["image_path"],
                    text_start=row["text_start"],
                    text_end=row["text_end"],
                    checksum=row["checksum"],
                ),
            }
            for row in rows
        }

    def delete_document(self, document_id: str) -> bool:
        with self.connection:
            self.connection.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
            cursor = self.connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0

    def rename_document(self, document_id: str, title: str) -> bool:
        clean = title.strip()
        if not clean:
            raise ValueError("资料标题不能为空")
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE documents SET title=?, updated_at=? WHERE id=?",
                (clean, utcnow_iso(), document_id),
            )
            self.connection.execute(
                "UPDATE source_references SET title=? WHERE document_id=?",
                (clean, document_id),
            )
            self.connection.execute(
                "UPDATE chunks_fts SET title=? WHERE document_id=?",
                (clean, document_id),
            )
            self.connection.execute(
                """UPDATE knowledge_items SET title=?, updated_at=?
                   WHERE document_id=? AND item_type='source'""",
                (clean, utcnow_iso(), document_id),
            )
        return cursor.rowcount > 0

    def set_document_enabled(self, document_id: str, enabled: bool) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE documents SET enabled=?, updated_at=? WHERE id=?",
                (1 if enabled else 0, utcnow_iso(), document_id),
            )
        return cursor.rowcount > 0

    def save_annotation(
        self,
        annotation_id: str,
        document_id: str,
        content: str,
        *,
        chunk_id: str | None = None,
        kind: str = "note",
        locator: dict[str, Any] | None = None,
        tags: Sequence[str] = (),
    ) -> None:
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """INSERT INTO annotations(
                       id, document_id, chunk_id, kind, content, locator_json, tags_json,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET content=excluded.content, kind=excluded.kind,
                       locator_json=excluded.locator_json, tags_json=excluded.tags_json,
                       updated_at=excluded.updated_at""",
                (
                    annotation_id,
                    document_id,
                    chunk_id,
                    kind,
                    content.strip(),
                    json.dumps(locator or {}, ensure_ascii=False),
                    json.dumps(list(tags), ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def list_annotations(self, document_id: str | None = None) -> list[dict[str, Any]]:
        if document_id:
            rows = self.connection.execute(
                "SELECT * FROM annotations WHERE document_id=? ORDER BY updated_at DESC",
                (document_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM annotations ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def save_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        title: str,
        markdown: str,
        source_document_ids: Sequence[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """INSERT INTO artifacts(
                       id, artifact_type, title, markdown, source_document_ids_json,
                       metadata_json, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET artifact_type=excluded.artifact_type,
                       title=excluded.title, markdown=excluded.markdown,
                       source_document_ids_json=excluded.source_document_ids_json,
                       metadata_json=excluded.metadata_json, updated_at=excluded.updated_at""",
                (
                    artifact_id,
                    artifact_type,
                    title,
                    markdown,
                    json.dumps(list(source_document_ids)),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            output_item_id = f"kg:artifact:{artifact_id}"
            self.connection.execute(
                """INSERT OR IGNORE INTO knowledge_items(
                       id, item_type, status, maturity, title, summary, body,
                       document_id, artifact_id, high_value, metadata_json, created_at, updated_at
                   ) VALUES (?, 'output', 'draft', 'compiled', ?, '', ?, NULL, ?, 0,
                             '{"managed_from":"artifacts"}', ?, ?)""",
                (output_item_id, title, markdown, artifact_id, now, now),
            )
            self.connection.execute(
                """UPDATE knowledge_items SET title=?, body=?, updated_at=?
                   WHERE artifact_id=? AND item_type='output' AND metadata_json=?""",
                (title, markdown, now, artifact_id, '{"managed_from":"artifacts"}'),
            )
            self.connection.execute(
                "DELETE FROM knowledge_relations "
                "WHERE target_item_id=? AND metadata_json=?",
                (output_item_id, '{"managed_from":"artifacts"}'),
            )
            for document_id in source_document_ids:
                source_item_id = f"kg:document:{document_id}"
                document = self.connection.execute(
                    "SELECT title, created_at, updated_at FROM documents WHERE id=?",
                    (document_id,),
                ).fetchone()
                if document is None:
                    continue
                self.connection.execute(
                    """INSERT OR IGNORE INTO knowledge_items(
                           id, item_type, status, maturity, title, summary, body,
                           document_id, artifact_id, high_value, metadata_json, created_at, updated_at
                       ) VALUES (?, 'source', 'current', 'indexed', ?, '', '', ?, NULL, 0,
                                 '{"managed_from":"documents"}', ?, ?)""",
                    (
                        source_item_id,
                        document["title"],
                        document_id,
                        document["created_at"],
                        document["updated_at"],
                    ),
                )
                digest = hashlib.sha256(
                    f"{source_item_id}|{output_item_id}|supports".encode("utf-8")
                ).hexdigest()[:24]
                self.connection.execute(
                    """INSERT OR IGNORE INTO knowledge_relations(
                           id, source_item_id, target_item_id, relation_type, summary,
                           metadata_json, created_at, updated_at
                       ) VALUES (?, ?, ?, 'supports', '来源资料支持该输出',
                                 '{"managed_from":"artifacts"}', ?, ?)""",
                    (f"rel-{digest}", source_item_id, output_item_id, now, now),
                )

    def list_artifacts(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM artifacts ORDER BY updated_at DESC"
            ).fetchall()
        ]

    def add_watched_folder(
        self,
        watcher_id: str,
        path: str,
        *,
        collection: str = "自动同步",
        recursive: bool = True,
    ) -> None:
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """INSERT INTO watched_folders(
                       id, path, collection, recursive, enabled, metadata_json,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, 1, '{}', ?, ?)
                   ON CONFLICT(path) DO UPDATE SET collection=excluded.collection,
                       recursive=excluded.recursive, enabled=1, updated_at=excluded.updated_at""",
                (watcher_id, path, collection.strip() or "自动同步", 1 if recursive else 0, now, now),
            )

    def list_watched_folders(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM watched_folders ORDER BY enabled DESC, path"
        ).fetchall()
        values = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            item["enabled"] = bool(item["enabled"])
            item["recursive"] = bool(item["recursive"])
            values.append(item)
        return values

    def update_watched_folder(
        self,
        watcher_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        enabled: bool | None = None,
        last_scan_at: str | None = None,
    ) -> bool:
        assignments = ["updated_at=?"]
        params: list[Any] = [utcnow_iso()]
        if metadata is not None:
            assignments.append("metadata_json=?")
            params.append(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
        if enabled is not None:
            assignments.append("enabled=?")
            params.append(1 if enabled else 0)
        if last_scan_at is not None:
            assignments.append("last_scan_at=?")
            params.append(last_scan_at)
        params.append(watcher_id)
        with self.connection:
            cursor = self.connection.execute(
                f"UPDATE watched_folders SET {', '.join(assignments)} WHERE id=?", params
            )
        return cursor.rowcount > 0

    def delete_watched_folder(self, watcher_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM watched_folders WHERE id=?", (watcher_id,))
        return cursor.rowcount > 0

    def integrity_report(self) -> dict[str, Any]:
        integrity = [str(row[0]) for row in self.connection.execute("PRAGMA integrity_check").fetchall()]
        foreign_keys = [dict(row) for row in self.connection.execute("PRAGMA foreign_key_check").fetchall()]
        orphan_fts = int(
            self.connection.execute(
                """SELECT COUNT(*) FROM chunks_fts f
                   LEFT JOIN chunks c ON c.id=f.chunk_id WHERE c.id IS NULL"""
            ).fetchone()[0]
        )
        missing_fts = int(
            self.connection.execute(
                """SELECT COUNT(*) FROM chunks c
                   LEFT JOIN chunks_fts f ON f.chunk_id=c.id WHERE f.chunk_id IS NULL"""
            ).fetchone()[0]
        )
        return {
            "ok": integrity == ["ok"] and not foreign_keys and not orphan_fts and not missing_fts,
            "integrity": integrity,
            "foreign_key_errors": foreign_keys,
            "orphan_fts": orphan_fts,
            "missing_fts": missing_fts,
        }

    def rebuild_fts(self) -> int:
        with self.connection:
            self.connection.execute("DELETE FROM chunks_fts")
            self.connection.execute(
                """INSERT INTO chunks_fts(chunk_id, document_id, title, content)
                   SELECT c.id, c.document_id, d.title, c.content
                   FROM chunks c JOIN documents d ON d.id=c.document_id"""
            )
        return self.connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    def status(self) -> dict[str, Any]:
        counts = {}
        for table in (
            "documents", "chunks", "source_references", "embeddings", "collections", "tags",
            "conversations", "messages", "answers", "answer_evidence", "citations",
            "answer_feedback", "annotations", "watched_folders", "artifacts",
            "ingestion_jobs", "ingestion_job_items",
            "knowledge_items", "knowledge_aliases", "knowledge_item_tags",
            "knowledge_relations",
        ):
            counts[table] = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        states = {
            row["embedding_status"]: row["count"]
            for row in self.connection.execute(
                "SELECT embedding_status, COUNT(*) AS count FROM chunks GROUP BY embedding_status"
            ).fetchall()
        }
        return {
            "database": str(self.path),
            "schema_version": SCHEMA_VERSION,
            **counts,
            "embedding_status": states,
        }

    @staticmethod
    def source_reference(row: sqlite3.Row) -> SourceReference:
        return SourceReference.from_dict(json.loads(row["source_reference_json"]))
