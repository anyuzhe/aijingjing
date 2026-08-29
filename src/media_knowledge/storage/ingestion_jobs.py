from __future__ import annotations

import json
import uuid
from typing import Any, Iterable

from ..models import utcnow_iso
from .database import KnowledgeDatabase


TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}
TERMINAL_ITEM_STATUSES = {"completed", "failed", "cancelled"}
VALID_JOB_STATUSES = {"queued", "running", *TERMINAL_JOB_STATUSES}
_UNSET = object()


class IngestionJobRepository:
    """Durable import batches and per-source progress for desktop task recovery."""

    MAX_PAGE_SIZE = 500

    def __init__(self, database: KnowledgeDatabase):
        self.database = database

    @classmethod
    def _page(cls, limit: int, offset: int) -> tuple[int, int]:
        try:
            safe_limit = int(limit)
            safe_offset = int(offset)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit 和 offset 必须是整数") from exc
        return max(1, min(cls.MAX_PAGE_SIZE, safe_limit)), max(0, safe_offset)

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def create_job(
        self,
        items: Iterable[str],
        *,
        metadata: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        sources = [str(item).strip() for item in items if str(item).strip()]
        if not sources:
            raise ValueError("导入任务至少需要一个资料来源")
        identifier = str(job_id or f"ingest-{uuid.uuid4().hex}")
        now = utcnow_iso()
        with self.database.connection:
            self.database.connection.execute(
                """INSERT INTO ingestion_jobs(
                       id, status, total_items, metadata_json, created_at, updated_at
                   ) VALUES (?, 'queued', ?, ?, ?, ?)""",
                (identifier, len(sources), self._json(metadata or {}), now, now),
            )
            for ordinal, source in enumerate(sources):
                self.database.connection.execute(
                    """INSERT INTO ingestion_job_items(
                           id, job_id, ordinal, source, status, progress_percent, stage,
                           created_at, updated_at
                       ) VALUES (?, ?, ?, ?, 'queued', 0, 'queued', ?, ?)""",
                    (f"{identifier}-item-{ordinal:06d}", identifier, ordinal, source, now, now),
                )
        return self.job_record(identifier)

    def list_jobs(
        self,
        *,
        statuses: Iterable[str] | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        safe_limit, safe_offset = self._page(limit, offset)
        raw_statuses = [statuses] if isinstance(statuses, str) else (statuses or [])
        normalized = list(
            dict.fromkeys(
                str(value).strip() for value in raw_statuses if str(value).strip()
            )
        )
        unknown = set(normalized) - VALID_JOB_STATUSES
        if unknown:
            raise ValueError(f"不支持的任务状态：{', '.join(sorted(unknown))}")
        where = ""
        parameters: list[Any] = []
        if normalized:
            placeholders = ",".join("?" for _ in normalized)
            where = f"WHERE status IN ({placeholders})"
            parameters.extend(normalized)
        rows = self.database.connection.execute(
            f"""SELECT * FROM ingestion_jobs {where}
                ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?""",
            (*parameters, safe_limit, safe_offset),
        ).fetchall()
        return [self._job_dict(row) for row in rows]

    def job_record(self, job_id: str) -> dict[str, Any]:
        row = self.database.connection.execute(
            "SELECT * FROM ingestion_jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"ingestion job does not exist: {job_id}")
        items = self.database.connection.execute(
            "SELECT * FROM ingestion_job_items WHERE job_id=? ORDER BY ordinal", (job_id,)
        ).fetchall()
        result = self._job_dict(row)
        result["items"] = [self._item_dict(item) for item in items]
        return result

    @staticmethod
    def _job_dict(row: Any) -> dict[str, Any]:
        value = dict(row)
        try:
            value["metadata"] = json.loads(value.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            value["metadata"] = {}
        return value

    @staticmethod
    def _item_dict(row: Any) -> dict[str, Any]:
        value = dict(row)
        try:
            value["result"] = json.loads(value.pop("result_json") or "{}")
        except json.JSONDecodeError:
            value["result"] = {}
        return value

    def begin_job(self, job_id: str) -> None:
        now = utcnow_iso()
        with self.database.connection:
            cursor = self.database.connection.execute(
                """UPDATE ingestion_jobs
                   SET status='running', started_at=COALESCE(started_at, ?),
                       completed_at=NULL, error=NULL, message='正在导入', updated_at=?
                   WHERE id=? AND status != 'completed'""",
                (now, now, job_id),
            )
        if cursor.rowcount == 0 and self.database.connection.execute(
            "SELECT 1 FROM ingestion_jobs WHERE id=?", (job_id,)
        ).fetchone() is None:
            raise ValueError(f"ingestion job does not exist: {job_id}")

    def record_progress(
        self,
        job_id: str,
        source: str,
        stage: str,
        percent: int,
        message: str,
    ) -> None:
        now = utcnow_iso()
        progress = max(0, min(100, int(percent)))
        stage_name = str(stage or "running")
        # The service emits its final progress event immediately before returning the
        # structured result. Keep it running until record_result persists that payload.
        status = "failed" if stage_name == "failed" else "running"
        item = self.database.connection.execute(
            """SELECT id FROM ingestion_job_items
               WHERE job_id=? AND source=? AND status IN ('queued', 'running')
               ORDER BY ordinal LIMIT 1""",
            (job_id, source),
        ).fetchone()
        if item is None:
            return
        completed_at = now if status in TERMINAL_ITEM_STATUSES else None
        with self.database.connection:
            self.database.connection.execute(
                """UPDATE ingestion_job_items
                   SET status=?, progress_percent=?, stage=?, message=?,
                       started_at=COALESCE(started_at, ?), completed_at=?, updated_at=?
                   WHERE id=?""",
                (status, progress, stage_name, str(message or ""), now, completed_at, now, item["id"]),
            )
            self.database.connection.execute(
                """UPDATE ingestion_jobs
                   SET status='running', current_item=?, current_stage=?, message=?, updated_at=?
                   WHERE id=?""",
                (source, stage_name, str(message or ""), now, job_id),
            )
            self._refresh_counts(job_id)

    def record_result(self, job_id: str, source: str, result: dict[str, Any]) -> None:
        raw_status = str(result.get("status") or "failed")
        status = "cancelled" if raw_status == "cancelled" else "failed" if raw_status == "failed" else "completed"
        error = str(result.get("error") or "").strip() or None
        message = (
            "任务已取消" if status == "cancelled"
            else error or (f"已完成：{result.get('title')}" if result.get("title") else "导入完成")
        )
        item = self.database.connection.execute(
            """SELECT id FROM ingestion_job_items
               WHERE job_id=? AND source=? AND status != 'completed'
               ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,
                        ordinal LIMIT 1""",
            (job_id, source),
        ).fetchone()
        if item is None:
            return
        now = utcnow_iso()
        with self.database.connection:
            self.database.connection.execute(
                """UPDATE ingestion_job_items
                   SET status=?, progress_percent=100, stage=?, message=?, result_json=?,
                       error=?, started_at=COALESCE(started_at, ?), completed_at=?, updated_at=?
                   WHERE id=?""",
                (
                    status, status, message, self._json(result), error,
                    now, now, now, item["id"],
                ),
            )
            self._refresh_counts(job_id)

    def finalize_job(self, job_id: str) -> dict[str, Any]:
        now = utcnow_iso()
        with self.database.connection:
            counts = self._item_counts(job_id)
            if counts["queued"] or counts["running"]:
                status = "running"
                completed_at = None
                message = "仍有待处理资料"
            elif counts["cancelled"]:
                status = "cancelled"
                completed_at = now
                message = f"已取消，完成 {counts['completed']}/{counts['total']} 项"
            elif counts["failed"]:
                status = "failed"
                completed_at = now
                message = f"完成 {counts['completed']} 项，失败 {counts['failed']} 项"
            else:
                status = "completed"
                completed_at = now
                message = f"已完成 {counts['completed']} 项"
            self._write_counts(
                job_id, counts, status=status, message=message,
                completed_at=completed_at, updated_at=now,
            )
        return self.job_record(job_id)

    def fail_job(self, job_id: str, error: str) -> dict[str, Any]:
        now = utcnow_iso()
        reason = str(error or "导入任务异常终止")[:2000]
        with self.database.connection:
            self.database.connection.execute(
                """UPDATE ingestion_job_items
                   SET status='failed', stage='failed', message=?, error=?,
                       progress_percent=100, completed_at=?, updated_at=?
                   WHERE job_id=? AND status IN ('queued', 'running')""",
                (reason, reason, now, now, job_id),
            )
            counts = self._item_counts(job_id)
            self._write_counts(
                job_id, counts, status="failed", message="导入任务异常终止",
                error=reason, completed_at=now, updated_at=now,
            )
        return self.job_record(job_id)

    def cancel_job(self, job_id: str) -> bool:
        now = utcnow_iso()
        with self.database.connection:
            cursor = self.database.connection.execute(
                """UPDATE ingestion_jobs SET status='cancelled', message='任务已取消',
                       completed_at=?, updated_at=?
                   WHERE id=? AND status IN ('queued', 'running', 'failed')""",
                (now, now, job_id),
            )
            self.database.connection.execute(
                """UPDATE ingestion_job_items
                   SET status='cancelled', stage='cancelled', message='任务已取消',
                       progress_percent=100, completed_at=?, updated_at=?
                   WHERE job_id=? AND status IN ('queued', 'running')""",
                (now, now, job_id),
            )
            if cursor.rowcount:
                self._refresh_counts(job_id)
        return cursor.rowcount > 0

    def recover_interrupted_jobs(self) -> int:
        """Make jobs left running by a process exit resumable without losing results."""

        rows = self.database.connection.execute(
            "SELECT id FROM ingestion_jobs WHERE status='running'"
        ).fetchall()
        if not rows:
            return 0
        now = utcnow_iso()
        with self.database.connection:
            for row in rows:
                job_id = str(row["id"])
                self.database.connection.execute(
                    """UPDATE ingestion_job_items
                       SET status='queued', stage='queued', message='等待恢复',
                           started_at=NULL, completed_at=NULL, error=NULL, updated_at=?
                       WHERE job_id=? AND status='running'""",
                    (now, job_id),
                )
                counts = self._item_counts(job_id)
                self._write_counts(
                    job_id, counts, status="queued", message="程序上次退出，任务可继续",
                    error=None, completed_at=None, updated_at=now,
                )
        return len(rows)

    def reset_failed_items(self, job_id: str) -> int:
        now = utcnow_iso()
        with self.database.connection:
            cursor = self.database.connection.execute(
                """UPDATE ingestion_job_items
                   SET status='queued', progress_percent=0, stage='queued', message='等待重试',
                       result_json='{}', error=NULL, started_at=NULL, completed_at=NULL, updated_at=?
                   WHERE job_id=? AND status IN ('failed', 'cancelled')""",
                (now, job_id),
            )
            if cursor.rowcount:
                counts = self._item_counts(job_id)
                self._write_counts(
                    job_id, counts, status="queued", message="等待重试",
                    error=None, completed_at=None, updated_at=now,
                )
        return cursor.rowcount

    def pending_sources(self, job_id: str) -> list[str]:
        if self.database.connection.execute(
            "SELECT 1 FROM ingestion_jobs WHERE id=?", (job_id,)
        ).fetchone() is None:
            raise ValueError(f"ingestion job does not exist: {job_id}")
        rows = self.database.connection.execute(
            """SELECT source FROM ingestion_job_items
               WHERE job_id=? AND status='queued' ORDER BY ordinal""",
            (job_id,),
        ).fetchall()
        return [str(row["source"]) for row in rows]

    def _refresh_counts(self, job_id: str) -> None:
        counts = self._item_counts(job_id)
        self._write_counts(job_id, counts, updated_at=utcnow_iso())

    def _item_counts(self, job_id: str) -> dict[str, int]:
        rows = self.database.connection.execute(
            """SELECT status, COUNT(*) AS count, COALESCE(SUM(progress_percent), 0) AS progress
               FROM ingestion_job_items WHERE job_id=? GROUP BY status""",
            (job_id,),
        ).fetchall()
        values = {status: 0 for status in VALID_JOB_STATUSES}
        progress = 0
        for row in rows:
            values[str(row["status"])] = int(row["count"])
            progress += int(row["progress"] or 0)
        values["total"] = sum(values[status] for status in VALID_JOB_STATUSES)
        values["progress"] = round(progress / values["total"]) if values["total"] else 0
        return values

    def _write_counts(
        self,
        job_id: str,
        counts: dict[str, int],
        *,
        status: str | None = None,
        message: str | None = None,
        error: str | None | object = _UNSET,
        completed_at: str | None | object = _UNSET,
        updated_at: str,
    ) -> None:
        assignments = [
            "total_items=?", "completed_items=?", "succeeded_items=?",
            "failed_items=?", "cancelled_items=?", "progress_percent=?", "updated_at=?",
        ]
        parameters: list[Any] = [
            counts["total"],
            counts["completed"] + counts["failed"] + counts["cancelled"],
            counts["completed"], counts["failed"], counts["cancelled"],
            counts["progress"], updated_at,
        ]
        if status is not None:
            assignments.append("status=?")
            parameters.append(status)
        if message is not None:
            assignments.append("message=?")
            parameters.append(message)
        if error is not _UNSET:
            assignments.append("error=?")
            parameters.append(error)
        if completed_at is not _UNSET:
            assignments.append("completed_at=?")
            parameters.append(completed_at)
        parameters.append(job_id)
        self.database.connection.execute(
            f"UPDATE ingestion_jobs SET {', '.join(assignments)} WHERE id=?", parameters
        )
