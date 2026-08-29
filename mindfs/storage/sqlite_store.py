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

