from __future__ import annotations

from pathlib import Path
from typing import Any

from ...storage import KnowledgeDatabase


class WorkbenchRepository:
    def __init__(self, database: KnowledgeDatabase):
        self.database = database

    def bootstrap(self) -> dict[str, Any]:
        connection = self.database.connection
        collections = [
            {"name": row["name"], "count": row["count"]}
            for row in connection.execute(
                """SELECT co.name, COUNT(dc.document_id) AS count
                   FROM collections co LEFT JOIN document_collections dc ON dc.collection_id=co.id
                   GROUP BY co.id ORDER BY co.name COLLATE NOCASE"""
            ).fetchall()
        ]
        tags = [
            {"name": row["name"], "count": row["count"]}
            for row in connection.execute(
                """SELECT t.name, COUNT(dt.document_id) AS count
                   FROM tags t LEFT JOIN document_tags dt ON dt.tag_id=t.id
                   GROUP BY t.id ORDER BY count DESC, t.name COLLATE NOCASE LIMIT 40"""
            ).fetchall()
        ]
        media_types = [
            {"name": row["media_type"], "count": row["count"]}
            for row in connection.execute(
                "SELECT media_type, COUNT(*) AS count FROM documents GROUP BY media_type ORDER BY count DESC"
            ).fetchall()
        ]
        document_rows = connection.execute(
            """SELECT id, title, media_type, local_path, obsidian_path, updated_at
               FROM documents ORDER BY updated_at DESC LIMIT 12"""
        ).fetchall()
        recent_documents = [dict(row) for row in document_rows]
        folder_counts: dict[str, int] = {}
        for row in connection.execute("SELECT local_path, obsidian_path FROM documents").fetchall():
            for raw_path in (row["local_path"], row["obsidian_path"]):
                if not raw_path:
                    continue
                parent = str(Path(raw_path).parent)
                if parent and parent != ".":
                    folder_counts[parent] = folder_counts.get(parent, 0) + 1
        folders = [
            {"name": Path(path).name or path, "value": path, "count": count}
            for path, count in sorted(folder_counts.items(), key=lambda item: (-item[1], item[0].casefold()))
        ][:30]
        conversations = [
            dict(row)
            for row in connection.execute(
                """SELECT c.id, c.title, c.updated_at,
                          (SELECT content FROM messages m WHERE m.conversation_id=c.id
                           ORDER BY m.ordinal DESC LIMIT 1) AS preview
                   FROM conversations c ORDER BY c.updated_at DESC LIMIT 10"""
            ).fetchall()
        ]
        return {
            "stats": self.database.status(),
            "collections": collections,
            "tags": tags,
            "media_types": media_types,
            "folders": folders,
            "recent_documents": recent_documents,
            "conversations": conversations,
        }
