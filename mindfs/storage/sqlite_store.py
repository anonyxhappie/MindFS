"""SQLite persistent metadata store for MindFS."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from mindfs.artifacts.models import ChunkItem, SemanticArtifact
from mindfs.identification.models import FileCategory, FileInfo, ProcessingStatus
from mindfs.resources.manager import OperationDiagnostic


class SQLiteStore:
    """Manages persistent metadata, artifacts, chunks, errors, and run records in SQLite."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        with self._get_connection() as conn:
            conn.executescript(schema_sql)
            # Safe migration for new chat_messages columns
            try:
                conn.execute("ALTER TABLE chat_messages ADD COLUMN explored_files TEXT")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE chat_messages ADD COLUMN subagents TEXT")
            except Exception:
                pass

    # ---------------- File Operations ----------------

    def upsert_file(self, file_info: FileInfo) -> None:
        """Inserts or updates a file record."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO files (
                    file_id, path, relative_path, size_bytes, mtime, mtime_ns,
                    sha256, mime_type, category, processor, processor_version,
                    status, status_reason, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    file_id = excluded.file_id,
                    relative_path = excluded.relative_path,
                    size_bytes = excluded.size_bytes,
                    mtime = excluded.mtime,
                    mtime_ns = excluded.mtime_ns,
                    sha256 = excluded.sha256,
                    mime_type = excluded.mime_type,
                    category = excluded.category,
                    processor = excluded.processor,
                    processor_version = excluded.processor_version,
                    status = excluded.status,
                    status_reason = excluded.status_reason,
                    indexed_at = excluded.indexed_at
                """,
                (
                    file_info.file_id,
                    file_info.canonical_path,
                    file_info.relative_path,
                    file_info.size_bytes,
                    file_info.mtime,
                    file_info.mtime_ns,
                    file_info.sha256,
                    file_info.mime_type,
                    file_info.category.value if isinstance(file_info.category, FileCategory) else str(file_info.category),
                    file_info.processor,
                    "1.0.0",
                    file_info.processing_status.value if isinstance(file_info.processing_status, ProcessingStatus) else str(file_info.processing_status),
                    file_info.status_reason,
                    now,
                ),
            )

    def get_file_by_path(self, canonical_path: str) -> Optional[FileInfo]:
        """Retrieves a file record by canonical path."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM files WHERE path = ?", (canonical_path,)).fetchone()
            if not row:
                return None
            return FileInfo(
                file_id=row["file_id"],
                canonical_path=row["path"],
                relative_path=row["relative_path"],
                filename=Path(row["path"]).name,
                extension=Path(row["path"]).suffix.lower(),
                mime_type=row["mime_type"],
                category=FileCategory(row["category"]),
                size_bytes=row["size_bytes"],
                mtime=row["mtime"],
                mtime_ns=row["mtime_ns"],
                sha256=row["sha256"],
                processor=row["processor"],
                processing_status=ProcessingStatus(row["status"]),
                status_reason=row["status_reason"],
            )

    def get_file_by_id(self, file_id: str) -> Optional[FileInfo]:
        """Retrieves a file record by file_id."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
            if not row:
                return None
            return FileInfo(
                file_id=row["file_id"],
                canonical_path=row["path"],
                relative_path=row["relative_path"],
                filename=Path(row["path"]).name,
                extension=Path(row["path"]).suffix.lower(),
                mime_type=row["mime_type"],
                category=FileCategory(row["category"]),
                size_bytes=row["size_bytes"],
                mtime=row["mtime"],
                mtime_ns=row["mtime_ns"],
                sha256=row["sha256"],
                processor=row["processor"],
                processing_status=ProcessingStatus(row["status"]),
                status_reason=row["status_reason"],
            )

    def list_all_files(self) -> List[FileInfo]:
        """Lists all indexed file records."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM files ORDER BY relative_path").fetchall()
            return [
                FileInfo(
                    file_id=row["file_id"],
                    canonical_path=row["path"],
                    relative_path=row["relative_path"],
                    filename=Path(row["path"]).name,
                    extension=Path(row["path"]).suffix.lower(),
                    mime_type=row["mime_type"],
                    category=FileCategory(row["category"]),
                    size_bytes=row["size_bytes"],
                    mtime=row["mtime"],
                    mtime_ns=row["mtime_ns"],
                    sha256=row["sha256"],
                    processor=row["processor"],
                    processing_status=ProcessingStatus(row["status"]),
                    status_reason=row["status_reason"],
                )
                for row in rows
            ]

    def delete_file(self, canonical_path: str) -> List[int]:
        """
        Deletes a file and all associated artifacts/chunks via CASCADE.
        Returns the list of deleted vector_ids to prune from FAISS.
        """
        vector_ids: List[int] = []
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT vector_id FROM chunks WHERE source_path = ? AND vector_id IS NOT NULL",
                (canonical_path,)
            ).fetchall()
            vector_ids = [r["vector_id"] for r in rows if r["vector_id"] is not None]

            conn.execute("DELETE FROM files WHERE path = ?", (canonical_path,))
        return vector_ids

    def delete_directory_files(self, canonical_dir_prefix: str) -> List[int]:
        """
        Deletes all files under a directory prefix.
        Returns the list of deleted vector_ids.
        """
        prefix = canonical_dir_prefix.rstrip("/") + "/%"
        exact = canonical_dir_prefix.rstrip("/")
        vector_ids: List[int] = []
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT vector_id FROM chunks WHERE (source_path LIKE ? OR source_path = ?) AND vector_id IS NOT NULL",
                (prefix, exact)
            ).fetchall()
            vector_ids = [r["vector_id"] for r in rows if r["vector_id"] is not None]

            conn.execute("DELETE FROM files WHERE path LIKE ? OR path = ?", (prefix, exact))
        return vector_ids

    # ---------------- Artifact Operations ----------------

    def save_artifacts(self, artifacts: List[SemanticArtifact]) -> None:
        """Inserts semantic artifacts."""
        if not artifacts:
            return
        with self._get_connection() as conn:
            for art in artifacts:
                offset_json = json.dumps(art.source_offset) if art.source_offset is not None else None
                meta_json = json.dumps(art.metadata)
                entities_json = json.dumps(art.entities)
                conn.execute(
                    """
                    INSERT INTO artifacts (
                        artifact_id, file_id, artifact_type, source_path,
                        source_offset, text, summary, metadata, entities,
                        created_at, processor, processor_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(artifact_id) DO UPDATE SET
                        text = excluded.text,
                        summary = excluded.summary,
                        metadata = excluded.metadata,
                        entities = excluded.entities
                    """,
                    (
                        art.artifact_id,
                        art.file_id,
                        art.artifact_type,
                        art.source_path,
                        offset_json,
                        art.text,
                        art.summary,
                        meta_json,
                        entities_json,
                        art.created_at,
                        art.processor,
                        art.processor_version,
                    ),
                )

    def get_artifacts_by_file(self, file_id: str) -> List[SemanticArtifact]:
        """Retrieves all artifacts for a given file_id."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM artifacts WHERE file_id = ?", (file_id,)).fetchall()
            results = []
            for r in rows:
                results.append(
                    SemanticArtifact(
                        artifact_id=r["artifact_id"],
                        file_id=r["file_id"],
                        artifact_type=r["artifact_type"],
                        source_path=r["source_path"],
                        source_offset=json.loads(r["source_offset"]) if r["source_offset"] else None,
                        text=r["text"],
                        summary=r["summary"],
                        metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                        entities=json.loads(r["entities"]) if r["entities"] else [],
                        created_at=r["created_at"],
                        processor=r["processor"],
                        processor_version=r["processor_version"],
                    )
                )
            return results

    # ---------------- Chunk Operations ----------------

    def save_chunks(self, chunks: List[Tuple[ChunkItem, Optional[int]]]) -> None:
        """Inserts chunk items with their corresponding vector IDs."""
        if not chunks:
            return
        with self._get_connection() as conn:
            for item, vector_id in chunks:
                offset_json = json.dumps(item.source_offset) if item.source_offset is not None else None
                meta_json = json.dumps(item.metadata)
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, artifact_id, file_id, vector_id,
                        source_path, source_offset, chunk_index,
                        text, char_start, char_end, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        vector_id = excluded.vector_id,
                        text = excluded.text,
                        metadata = excluded.metadata
                    """,
                    (
                        item.chunk_id,
                        item.artifact_id,
                        item.file_id,
                        vector_id,
                        item.source_path,
                        offset_json,
                        item.chunk_index,
                        item.text,
                        item.char_start,
                        item.char_end,
                        meta_json,
                    ),
                )

    def get_chunk_by_vector_id(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves chunk and associated metadata by FAISS vector_id."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT c.*, a.artifact_type, f.mime_type, f.category
                FROM chunks c
                JOIN artifacts a ON c.artifact_id = a.artifact_id
                JOIN files f ON c.file_id = f.file_id
                WHERE c.vector_id = ?
                """,
                (vector_id,)
            ).fetchone()
            if not row:
                return None
            return {
                "chunk_id": row["chunk_id"],
                "artifact_id": row["artifact_id"],
                "file_id": row["file_id"],
                "vector_id": row["vector_id"],
                "source_path": row["source_path"],
                "source_offset": json.loads(row["source_offset"]) if row["source_offset"] else None,
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "artifact_type": row["artifact_type"],
                "mime_type": row["mime_type"],
                "category": row["category"],
            }

    def get_all_vector_mappings(self) -> List[Tuple[int, str]]:
        """Returns all (vector_id, chunk_id) mappings ordered by vector_id."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT vector_id, chunk_id FROM chunks WHERE vector_id IS NOT NULL ORDER BY vector_id"
            ).fetchall()
            return [(r["vector_id"], r["chunk_id"]) for r in rows]

    def get_max_vector_id(self) -> int:
        """Returns the highest vector_id in SQLite (or -1 if empty)."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT MAX(vector_id) AS max_id FROM chunks").fetchone()
            if row and row["max_id"] is not None:
                return int(row["max_id"])
            return -1

    # ---------------- Runs & Errors ----------------

    def record_run(self, run_id: str, start_time: str, status: str = "RUNNING") -> None:
        """Records the start of an indexing run."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO index_runs (run_id, start_time, status) VALUES (?, ?, ?)",
                (run_id, start_time, status)
            )

    def update_run(
        self,
        run_id: str,
        end_time: str,
        files_scanned: int,
        files_indexed: int,
        files_skipped: int,
        files_unsupported: int,
        files_failed: int,
        artifacts_created: int,
        chunks_created: int,
        peak_rss_mb: float,
        duration_seconds: float,
        status: str = "COMPLETED"
    ) -> None:
        """Updates run statistics at completion."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE index_runs SET
                    end_time = ?,
                    files_scanned = ?,
                    files_indexed = ?,
                    files_skipped = ?,
                    files_unsupported = ?,
                    files_failed = ?,
                    artifacts_created = ?,
                    chunks_created = ?,
                    peak_rss_mb = ?,
                    duration_seconds = ?,
                    status = ?
                WHERE run_id = ?
                """,
                (
                    end_time,
                    files_scanned,
                    files_indexed,
                    files_skipped,
                    files_unsupported,
                    files_failed,
                    artifacts_created,
                    chunks_created,
                    peak_rss_mb,
                    duration_seconds,
                    status,
                    run_id
                )
            )

    def record_error(self, file_path: str, processor: str, error_message: str) -> None:
        """Records an error encountered during processing."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO errors (file_path, processor, error_message, timestamp) VALUES (?, ?, ?, ?)",
                (file_path, processor, error_message, now)
            )

    def save_diagnostic(self, diag: OperationDiagnostic) -> None:
        """Persists an operation diagnostic."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO diagnostics (
                    operation, peak_rss_mb, current_rss_mb, duration_seconds,
                    files_processed, bytes_processed, chunks_processed, errors,
                    status, details, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    diag.operation,
                    diag.peak_rss_mb,
                    diag.current_rss_mb,
                    diag.duration_seconds,
                    diag.files_processed,
                    diag.bytes_processed,
                    diag.chunks_processed,
                    diag.errors,
                    diag.status,
                    json.dumps(diag.details),
                    diag.timestamp,
                )
            )

    def clear_all(self) -> None:
        """Wipes all data from tables."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM artifacts")
            conn.execute("DELETE FROM files")
            conn.execute("DELETE FROM errors")
            conn.execute("DELETE FROM index_runs")
            conn.execute("DELETE FROM diagnostics")
            conn.execute("DELETE FROM indexed_folders")

    def get_status_summary(self) -> Dict[str, Any]:
        """Returns comprehensive status dictionary."""
        with self._get_connection() as conn:
            total_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            completed = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'COMPLETED'").fetchone()[0]
            partial = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'PARTIAL'").fetchone()[0]
            skipped = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'SKIPPED'").fetchone()[0]
            unsupported = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'UNSUPPORTED'").fetchone()[0]
            failed = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'FAILED'").fetchone()[0]
            total_artifacts = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            total_vectors = conn.execute("SELECT COUNT(*) FROM chunks WHERE vector_id IS NOT NULL").fetchone()[0]
            last_run = conn.execute("SELECT * FROM index_runs ORDER BY start_time DESC LIMIT 1").fetchone()
            recent_errors = conn.execute("SELECT * FROM errors ORDER BY timestamp DESC LIMIT 5").fetchall()

            db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "files_indexed": completed + partial,
                "files_total": total_files,
                "files_completed": completed,
                "files_partial": partial,
                "files_skipped": skipped,
                "files_unsupported": unsupported,
                "files_failed": failed,
                "artifacts_count": total_artifacts,
                "chunks_count": total_chunks,
                "vectors_count": total_vectors,
                "db_size_bytes": db_size_bytes,
                "db_size_mb": round(db_size_bytes / (1024 * 1024), 2),
                "last_run": dict(last_run) if last_run else None,
                "recent_errors": [dict(e) for e in recent_errors],
            }

    # ---------------- Indexed Folders Operations ----------------

    def add_indexed_folder(self, folder_path: str, folder_name: Optional[str] = None) -> None:
        """Records a folder as an active indexed root."""
        p = str(Path(folder_path).expanduser().resolve())
        name = folder_name or Path(p).name or p
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO indexed_folders (folder_path, folder_name, added_at, last_indexed_at, files_count)
                VALUES (?, ?, ?, ?, 0)
                ON CONFLICT(folder_path) DO UPDATE SET
                    folder_name = excluded.folder_name
                """,
                (p, name, now, now),
            )

    def list_indexed_folders(self) -> List[Dict[str, Any]]:
        """Lists all registered indexed folders with live counts."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM indexed_folders ORDER BY added_at ASC").fetchall()
            folders = []
            for r in rows:
                folder_p = r["folder_path"]
                # Count files under this folder path
                cnt = conn.execute("SELECT COUNT(*) FROM files WHERE path LIKE ?", (f"{folder_p}%",)).fetchone()[0]
                folders.append({
                    "folder_path": folder_p,
                    "folder_name": r["folder_name"],
                    "added_at": r["added_at"],
                    "last_indexed_at": r["last_indexed_at"],
                    "files_count": cnt,
                })
            return folders

    def update_indexed_folder(self, folder_path: str, files_count: int) -> None:
        """Updates last indexed time and file count for an indexed folder."""
        p = str(Path(folder_path).expanduser().resolve())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE indexed_folders
                SET last_indexed_at = ?, files_count = ?
                WHERE folder_path = ?
                """,
                (now, files_count, p),
            )

    def remove_indexed_folder(self, folder_path: str) -> List[int]:
        """
        Removes an indexed folder and deletes all its files, artifacts, and chunks.
        Returns deleted vector_ids to prune from FAISS.
        """
        p = str(Path(folder_path).expanduser().resolve())
        deleted_vids: List[int] = []
        with self._get_connection() as conn:
            # Find all files belonging to this folder
            rows = conn.execute("SELECT path FROM files WHERE path LIKE ?", (f"{p}%",)).fetchall()
            for r in rows:
                file_path = r["path"]
                vids = conn.execute(
                    "SELECT vector_id FROM chunks WHERE file_id = (SELECT file_id FROM files WHERE path = ?) AND vector_id IS NOT NULL",
                    (file_path,)
                ).fetchall()
                deleted_vids.extend([v[0] for v in vids])
                conn.execute("DELETE FROM files WHERE path = ?", (file_path,))

            conn.execute("DELETE FROM indexed_folders WHERE folder_path = ?", (p,))
            return deleted_vids

    # ---------------- Audit Logs & Rollback Operations ----------------

    def record_audit_log(
        self,
        log_id: str,
        plan_id: Optional[str],
        action_type: str,
        source_path: Optional[str] = None,
        destination_path: Optional[str] = None,
        backup_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status: str = "COMPLETED",
    ) -> None:
        """Records a mutating filesystem operation for audit and rollback."""
        now = datetime.now(timezone.utc).isoformat()
        details_json = json.dumps(details or {})
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (
                    log_id, plan_id, action_type, source_path, destination_path,
                    backup_path, details, timestamp, status, undone
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (log_id, plan_id, action_type, source_path, destination_path, backup_path, details_json, now, status),
            )

    def get_audit_log(self, log_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single audit log entry by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM audit_logs WHERE log_id = ?", (log_id,)).fetchone()
            if not row:
                return None
            res = dict(row)
            if res.get("details"):
                try:
                    res["details"] = json.loads(res["details"])
                except Exception:
                    pass
            return res

    def list_audit_logs(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Lists recent audit log entries ordered newest first."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            logs = []
            for r in rows:
                item = dict(r)
                if item.get("details"):
                    try:
                        item["details"] = json.loads(item["details"])
                    except Exception:
                        pass
                logs.append(item)
            return logs

    def mark_audit_undone(self, log_id: str) -> None:
        """Marks an audit entry as undone / rolled back."""
        with self._get_connection() as conn:
            conn.execute("UPDATE audit_logs SET undone = 1, status = 'ROLLED_BACK' WHERE log_id = ?", (log_id,))

    # ---------------- Chat Session & Message Persistence ----------------

    def create_chat_session(self, session_id: str, title: str = "New Chat", model_name: Optional[str] = None) -> Dict[str, Any]:
        """Creates a new persistent chat session."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (session_id, title, model_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (session_id, title, model_name, now, now),
            )
        return {
            "session_id": session_id,
            "title": title,
            "model_name": model_name,
            "created_at": now,
            "updated_at": now,
        }

    def list_chat_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists all chat sessions ordered by most recently active."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single chat session metadata."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
            return dict(row) if row else None

    def update_chat_session(self, session_id: str, title: Optional[str] = None) -> None:
        """Updates chat session title and timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            if title:
                conn.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title, now, session_id),
                )
            else:
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )

    def delete_chat_session(self, session_id: str) -> None:
        """Deletes a chat session and all its messages."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))

    def add_chat_message(
        self,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        thoughts: Optional[List[Any]] = None,
        tool_calls: Optional[List[Any]] = None,
        explored_files: Optional[List[str]] = None,
        subagents: Optional[List[Any]] = None,
        plan_id: Optional[str] = None,
        plan_data: Optional[Dict[str, Any]] = None,
        status: str = "COMPLETED",
        can_undo: bool = False,
        undo_log_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Appends a new message (with optional thoughts, tool calls, and plan) to session."""
        now = datetime.now(timezone.utc).isoformat()
        thoughts_json = json.dumps(thoughts or [])
        tools_json = json.dumps(tool_calls or [])
        explored_json = json.dumps(explored_files or [])
        subagents_json = json.dumps(subagents or [])
        plan_json = json.dumps(plan_data) if plan_data else None
        undo_json = json.dumps(undo_log_ids or [])

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (
                    message_id, session_id, role, content, thoughts, tool_calls,
                    explored_files, subagents, plan_id, plan_data, status, can_undo, undo_log_ids, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    thoughts_json,
                    tools_json,
                    explored_json,
                    subagents_json,
                    plan_id,
                    plan_json,
                    status,
                    1 if can_undo else 0,
                    undo_json,
                    now,
                ),
            )
            conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))

        return {
            "message_id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "thoughts": thoughts or [],
            "tool_calls": tool_calls or [],
            "explored_files": explored_files or [],
            "subagents": subagents or [],
            "plan_id": plan_id,
            "plan_data": plan_data,
            "status": status,
            "can_undo": can_undo,
            "undo_log_ids": undo_log_ids or [],
            "created_at": now,
        }

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves chronological messages for a conversation session."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
            messages = []
            for r in rows:
                m = dict(r)
                m["thoughts"] = json.loads(m["thoughts"]) if m.get("thoughts") else []
                m["tool_calls"] = json.loads(m["tool_calls"]) if m.get("tool_calls") else []
                m["explored_files"] = json.loads(m["explored_files"]) if m.get("explored_files") else []
                m["subagents"] = json.loads(m["subagents"]) if m.get("subagents") else []
                m["plan_data"] = json.loads(m["plan_data"]) if m.get("plan_data") else None
                m["undo_log_ids"] = json.loads(m["undo_log_ids"]) if m.get("undo_log_ids") else []
                m["can_undo"] = bool(m.get("can_undo", 0))
                messages.append(m)
            return messages

    def update_plan_status(self, plan_id: str, new_status: str) -> None:
        """Updates the status of a message containing a plan."""
        with self._get_connection() as conn:
            conn.execute("UPDATE chat_messages SET status = ? WHERE plan_id = ?", (new_status, plan_id))

    def update_chat_message_plan(
        self,
        message_id: str,
        status: str,
        plan_data: Optional[Dict[str, Any]] = None,
        can_undo: bool = False,
        undo_log_ids: Optional[List[str]] = None,
    ) -> None:
        """Updates plan state on a chat message post-approval/rejection/undo."""
        with self._get_connection() as conn:
            plan_json = json.dumps(plan_data) if plan_data else None
            undo_json = json.dumps(undo_log_ids or [])
            conn.execute(
                """
                UPDATE chat_messages
                SET status = ?, plan_data = COALESCE(?, plan_data), can_undo = ?, undo_log_ids = ?
                WHERE message_id = ?
                """,
                (status, plan_json, 1 if can_undo else 0, undo_json, message_id),
            )

    # ---------------- Recent Workspaces Persistence ----------------

    def record_recent_workspace(self, path: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Records or updates a workspace path in the recent workspaces history."""
        resolved = str(Path(path).expanduser().resolve())
        folder_name = name or Path(resolved).name or resolved
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO recent_workspaces (path, name, last_used_at)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name,
                    last_used_at = excluded.last_used_at
                """,
                (resolved, folder_name, now),
            )
        return {"path": resolved, "name": folder_name, "last_used_at": now}

    def list_recent_workspaces(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Lists recent workspaces ordered newest first."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM recent_workspaces ORDER BY last_used_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def remove_recent_workspace(self, path: str) -> None:
        """Removes a workspace from the recent history."""
        resolved = str(Path(path).expanduser().resolve())
        with self._get_connection() as conn:
            conn.execute("DELETE FROM recent_workspaces WHERE path = ?", (resolved,))




