"""Incremental Indexer for MindFS with bounded streaming and change detection."""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from mindfs.artifacts.models import ChunkItem, SemanticArtifact
from mindfs.config.settings import MindFSConfig
from mindfs.filesystem.sandbox import FilesystemSandbox, SandboxSecurityError
from mindfs.identification.detector import FileDetector
from mindfs.identification.models import FileCategory, FileInfo, ProcessingStatus
from mindfs.indexing.chunker import Chunker
from mindfs.indexing.embeddings import EmbeddingPipeline
from mindfs.indexing.vector_store import VectorStore
from mindfs.processors.base import FileProcessor
from mindfs.processors.registry import ProcessorRegistry
from mindfs.resources.manager import ResourceManager
from mindfs.storage.sqlite_store import SQLiteStore


class Indexer:
    """Orchestrates incremental file scanning, modular processing, chunking, embedding, and dual persistence."""

    def __init__(
        self,
        config: MindFSConfig,
        sandbox: FilesystemSandbox,
        store: SQLiteStore,
        registry: ProcessorRegistry,
        vector_store: VectorStore,
        embedding_pipeline: EmbeddingPipeline,
        resource_manager: Optional[ResourceManager] = None,
    ):
        self.config = config
        self.sandbox = sandbox
        self.store = store
        self.registry = registry
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline
        self.detector = FileDetector(sandbox)
        self.chunker = Chunker(config)
        self.resources = resource_manager or ResourceManager(config)

        # Restore all previously indexed folders into sandbox allowed roots
        try:
            for f_info in self.store.list_indexed_folders():
                self.sandbox.add_allowed_root(f_info["folder_path"])
        except Exception:
            pass

    def _should_skip_file(self, existing: Optional[FileInfo], current: FileInfo) -> bool:
        """Determines whether a file is completely unchanged and already indexed."""
        if not existing:
            return False
        # Compare size and mtime_ns (allowing for sub-millisecond precision differences)
        if existing.size_bytes == current.size_bytes:
            if existing.mtime_ns == current.mtime_ns or abs(existing.mtime_ns - current.mtime_ns) < 1_000_000:
                if existing.processing_status in (ProcessingStatus.COMPLETED, ProcessingStatus.SKIPPED, ProcessingStatus.UNSUPPORTED):
                    return True
        return False

    def index_single_file(self, target_path: Path | str, is_reindex: bool = False) -> Tuple[str, Optional[FileInfo]]:
        """Indexes an individual file within the sandbox."""
        resolved = self.sandbox.validate_and_resolve(target_path, must_exist=True)
        if resolved.is_dir():
            return "skipped_dir", None

        existing_record = self.store.get_file_by_path(str(resolved))
        file_info = self.detector.identify(resolved, compute_hash=False)

        if not is_reindex and self._should_skip_file(existing_record, file_info):
            return "skipped_unchanged", existing_record

        # If modified/reindexing, clean up old records & vectors first
        if existing_record:
            old_vector_ids = self.store.delete_file(str(resolved))
            self.vector_store.remove_vectors(old_vector_ids)

        processor = self.registry.get_processor(file_info)
        file_info.processor = processor.name

        # Check file size limit for deep processing
        max_bytes = int(self.config.index.max_file_size_mb * 1024 * 1024)
        is_oversized = file_info.size_bytes > max_bytes and file_info.category not in (FileCategory.IMAGE, FileCategory.AUDIO, FileCategory.VIDEO, FileCategory.ARCHIVE)

        artifacts: List[SemanticArtifact] = []

        try:
            if is_oversized:
                # Lightweight inspection for oversized files
                file_info.sha256 = self.detector.compute_sha256(resolved, max_bytes=1048576)
                inspection = processor.inspect(file_info)
                file_info.processing_status = ProcessingStatus.SKIPPED
                file_info.status_reason = f"Oversized file ({round(file_info.size_bytes/(1024*1024), 2)} MB > {self.config.index.max_file_size_mb} MB limit)"
                
                # Still create a metadata artifact
                artifact = SemanticArtifact(
                    file_id=file_info.file_id,
                    artifact_type="oversized_file_metadata",
                    source_path=file_info.canonical_path,
                    source_offset=None,
                    text=f"Oversized File: {file_info.filename}\nSize: {file_info.size_bytes} bytes\nInspection: {inspection}",
                    summary=f"Oversized file '{file_info.filename}' ({round(file_info.size_bytes/(1024*1024), 2)} MB) deep indexing skipped.",
                    metadata=inspection,
                    processor=processor.name,
                    processor_version=processor.version,
                )
                artifacts.append(artifact)
            else:
                # Deep processing
                artifacts = processor.extract(file_info)
                file_info.processing_status = ProcessingStatus.COMPLETED

            # Save file record and artifacts to SQLite
            self.store.upsert_file(file_info)
            self.store.save_artifacts(artifacts)

            # Chunk and embed
            all_chunks: List[ChunkItem] = []
            for art in artifacts:
                c_items = self.chunker.chunk_artifact(art)
                all_chunks.extend(c_items)

            if all_chunks:
                texts_to_embed = [c.text for c in all_chunks]
                embeddings = self.embedding_pipeline.embed_texts(texts_to_embed)

                # Allocate consecutive vector IDs starting from max_vector_id + 1
                start_vid = self.store.get_max_vector_id() + 1
                vector_ids = [start_vid + idx for idx in range(len(all_chunks))]

                # Persist chunks to SQLite with assigned vector IDs
                chunk_tuples = [(chunk, vid) for chunk, vid in zip(all_chunks, vector_ids)]
                self.store.save_chunks(chunk_tuples)

                # Add to FAISS index
                self.vector_store.add_vectors(embeddings, vector_ids)

            return "indexed", file_info

        except Exception as exc:
            self.store.record_error(file_info.canonical_path, processor.name, str(exc))
            file_info.processing_status = ProcessingStatus.FAILED
            file_info.status_reason = str(exc)
            self.store.upsert_file(file_info)
            return "failed", file_info

    def scan_directory(self, target_dir: Path | str, recursive: bool = True) -> List[Path]:
        """Safely scans a directory for files without escaping sandbox."""
        resolved_dir = self.sandbox.validate_and_resolve(target_dir, must_exist=True)
        if not resolved_dir.is_dir():
            return [resolved_dir]

        discovered_files: List[Path] = []
        
        if recursive:
            for root, dirs, files in os.walk(resolved_dir):
                # Filter out internal or hidden directories
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "build", "dist")]
                for fname in sorted(files):
                    if fname.startswith("."):
                        continue
                    fpath = Path(root) / fname
                    try:
                        valid_p = self.sandbox.validate_and_resolve(fpath)
                        discovered_files.append(valid_p)
                    except SandboxSecurityError:
                        continue
        else:
            for entry in sorted(os.scandir(resolved_dir), key=lambda e: e.name):
                if entry.name.startswith("."):
                    continue
                if entry.is_file():
                    try:
                        valid_p = self.sandbox.validate_and_resolve(entry.path)
                        discovered_files.append(valid_p)
                    except SandboxSecurityError:
                        continue

        return discovered_files

    def index_path(
        self,
        target_path: Optional[Path | str] = None,
        recursive: Optional[bool] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Incrementally indexes a file or directory path.
        Returns detailed summary and records an index_run.
        """
        run_id = uuid.uuid4().hex[:12]
        start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.store.record_run(run_id, start_iso, status="RUNNING")

        is_recursive = self.config.index.recursive_default if recursive is None else recursive
        target = target_path or ""
        
        # If absolute path is provided, automatically add to sandbox allowed roots
        if target and Path(target).is_absolute():
            target_resolved = Path(target).expanduser().resolve()
            if target_resolved.is_dir():
                self.sandbox.add_allowed_root(target_resolved)
            elif target_resolved.parent.is_dir():
                self.sandbox.add_allowed_root(target_resolved.parent)

        resolved = self.sandbox.validate_and_resolve(target, must_exist=True)

        with self.resources.track_operation(f"index_path:{self.sandbox.relative_path(resolved)}") as diag:
            files_to_process = self.scan_directory(resolved, recursive=is_recursive)
            
            scanned = len(files_to_process)
            indexed = 0
            skipped = 0
            unsupported = 0
            failed = 0

            # Prune records for files that were deleted from disk under target directory
            if resolved.is_dir():
                self.store.add_indexed_folder(str(resolved), resolved.name)
                current_file_paths = {str(p) for p in files_to_process}
                all_stored = self.store.list_all_files()
                prefix = str(resolved)
                for rec in all_stored:
                    if rec.canonical_path.startswith(prefix) and rec.canonical_path not in current_file_paths:
                        # File was deleted on disk, prune from SQLite and FAISS
                        del_vids = self.store.delete_file(rec.canonical_path)
                        self.vector_store.remove_vectors(del_vids)

            for idx, fpath in enumerate(files_to_process, 1):
                rel_p = self.sandbox.relative_path(fpath)
                if progress_callback:
                    try:
                        progress_callback(idx, scanned, rel_p, "processing", indexed, skipped, failed)
                    except Exception:
                        pass

                status, info = self.index_single_file(fpath)
                if status == "indexed":
                    indexed += 1
                elif status == "skipped_unchanged" or status == "skipped_dir":
                    skipped += 1
                elif status == "failed":
                    failed += 1
                elif status == "unsupported":
                    unsupported += 1

                if info:
                    diag.bytes_processed += info.size_bytes

                if progress_callback:
                    try:
                        progress_callback(idx, scanned, rel_p, status, indexed, skipped, failed)
                    except Exception:
                        pass

            diag.files_processed = indexed

            if resolved.is_dir():
                self.store.update_indexed_folder(str(resolved), scanned)

            end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            status_summary = self.store.get_status_summary()

            self.store.update_run(
                run_id=run_id,
                end_time=end_iso,
                files_scanned=scanned,
                files_indexed=indexed,
                files_skipped=skipped,
                files_unsupported=unsupported,
                files_failed=failed,
                artifacts_created=status_summary["artifacts_count"],
                chunks_created=status_summary["chunks_count"],
                peak_rss_mb=diag.peak_rss_mb,
                duration_seconds=diag.duration_seconds,
                status="COMPLETED"
            )
            self.store.save_diagnostic(diag)

            return {
                "run_id": run_id,
                "target_path": str(resolved),
                "files_scanned": scanned,
                "files_indexed": indexed,
                "files_skipped": skipped,
                "files_failed": failed,
                "total_artifacts": status_summary["artifacts_count"],
                "total_vectors": status_summary["vectors_count"],
                "peak_rss_mb": diag.peak_rss_mb,
                "duration_seconds": diag.duration_seconds,
            }

    def remove_file(self, target_path: Path | str) -> bool:
        """Removes a file and associated vectors from the index."""
        resolved = self.sandbox.validate_and_resolve(target_path)
        vids = self.store.delete_file(str(resolved))
        self.vector_store.remove_vectors(vids)
        return True

    def remove_directory(self, target_dir: Path | str) -> int:
        """Removes all indexed files under a directory."""
        resolved = self.sandbox.validate_and_resolve(target_dir)
        vids = self.store.delete_directory_files(str(resolved))
        self.vector_store.remove_vectors(vids)
        return len(vids)

    def rebuild_index(self) -> Dict[str, Any]:
        """Clears all index tables and vector stores, then reindexes the entire workspace."""
        self.clear_index()
        return self.index_path("", recursive=True)

    def clear_index(self) -> None:
        """Clears SQLite metadata and FAISS vector index."""
        self.store.clear_all()
        self.vector_store.clear()

