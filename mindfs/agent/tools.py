"""Deterministic filesystem and indexing tools for the MindFS agent."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union
import uuid
from pydantic import BaseModel, Field

from mindfs.config.settings import MindFSConfig
from mindfs.filesystem.sandbox import FilesystemSandbox, PathNotFoundError, SandboxSecurityError
from mindfs.identification.detector import FileDetector
from mindfs.indexing.indexer import Indexer
from mindfs.processors.registry import ProcessorRegistry
from mindfs.retrieval.search import SearchEngine
from mindfs.storage.sqlite_store import SQLiteStore


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: Any
    error: Optional[str] = None
    audit_id: Optional[str] = None


class FilesystemTools:
    """Deterministic, validated tools for interacting with the sandboxed filesystem and index."""

    def __init__(
        self,
        config: MindFSConfig,
        sandbox: FilesystemSandbox,
        store: SQLiteStore,
        indexer: Indexer,
        search_engine: SearchEngine,
        registry: ProcessorRegistry,
    ):
        self.config = config
        self.sandbox = sandbox
        self.store = store
        self.indexer = indexer
        self.search_engine = search_engine
        self.registry = registry
        self.detector = FileDetector(sandbox)

    def _get_trash_dir(self) -> Path:
        """Returns the internal trash directory inside the workspace."""
        trash = Path(self.config.resolved_workspace_root) / ".mindfs" / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        return trash

    # ---------------- Read-Only Tools ----------------

    def list_directory(self, path: Optional[str] = None) -> ToolResult:
        """Lists files and folders inside the sandboxed workspace directory."""
        try:
            entries = self.sandbox.list_dir(path)
            data = [
                {"name": name, "type": entry_type, "size_bytes": size}
                for name, entry_type, size in entries
            ]
            return ToolResult(tool_name="list_directory", success=True, data=data)
        except (SandboxSecurityError, PathNotFoundError, Exception) as exc:
            return ToolResult(tool_name="list_directory", success=False, data=None, error=str(exc))

    def read_file_metadata(self, file_path: str) -> ToolResult:
        """Retrieves authoritative filesystem and index metadata for a file."""
        try:
            resolved = self.sandbox.validate_and_resolve(file_path, must_exist=True)
            stored = self.store.get_file_by_path(str(resolved))
            if stored:
                return ToolResult(tool_name="read_file_metadata", success=True, data=stored.model_dump())
            
            info = self.detector.identify(resolved, compute_hash=False)
            return ToolResult(tool_name="read_file_metadata", success=True, data=info.model_dump())
        except (SandboxSecurityError, PathNotFoundError, Exception) as exc:
            return ToolResult(tool_name="read_file_metadata", success=False, data=None, error=str(exc))

    def inspect_file(self, file_path: str) -> ToolResult:
        """Performs fast technical inspection of a file using its specialized processor."""
        try:
            resolved = self.sandbox.validate_and_resolve(file_path, must_exist=True)
            info = self.detector.identify(resolved, compute_hash=False)
            proc = self.registry.get_processor(info)
            inspection_data = proc.inspect(info)
            result = {
                "file": self.sandbox.relative_path(resolved),
                "category": info.category.value,
                "mime_type": info.mime_type,
                "size_bytes": info.size_bytes,
                "processor": proc.name,
                "inspection": inspection_data,
            }
            return ToolResult(tool_name="inspect_file", success=True, data=result)
        except (SandboxSecurityError, PathNotFoundError, Exception) as exc:
            return ToolResult(tool_name="inspect_file", success=False, data=None, error=str(exc))

    def index_path(
        self,
        path: Optional[str] = None,
        recursive: Optional[bool] = None,
        progress_callback: Optional[Any] = None,
    ) -> ToolResult:
        """Indexes files in a path incrementally."""
        try:
            summary = self.indexer.index_path(path, recursive=recursive, progress_callback=progress_callback)
            return ToolResult(tool_name="index_path", success=True, data=summary)
        except (SandboxSecurityError, PathNotFoundError, Exception) as exc:
            return ToolResult(tool_name="index_path", success=False, data=None, error=str(exc))

    def rag_search(
        self,
        query: str,
        path_filter: Optional[str] = None,
        file_type_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> ToolResult:
        """Performs semantic search across the indexed filesystem artifacts."""
        try:
            res = self.search_engine.search(
                query=query,
                path_filter=path_filter,
                file_type_filter=file_type_filter,
                limit=limit,
            )
            evidence_data = [
                {
                    "source": e.relative_path,
                    "location": e.source_location,
                    "artifact_type": e.artifact_type,
                    "similarity": round(e.similarity_score, 3),
                    "text": e.text,
                    "citation": e.formatted_citation(),
                }
                for e in res.evidence
            ]
            return ToolResult(
                tool_name="rag_search",
                success=True,
                data={
                    "query": query,
                    "has_sufficient_evidence": res.has_sufficient_evidence,
                    "evidence_count": len(evidence_data),
                    "evidence": evidence_data,
                },
            )
        except Exception as exc:
            return ToolResult(tool_name="rag_search", success=False, data=None, error=str(exc))

    def get_index_status(self) -> ToolResult:
        """Retrieves comprehensive statistics about the persistent index."""
        try:
            summary = self.store.get_status_summary()
            summary["vector_count_faiss"] = self.indexer.vector_store.total_vectors()
            return ToolResult(tool_name="get_index_status", success=True, data=summary)
        except Exception as exc:
            return ToolResult(tool_name="get_index_status", success=False, data=None, error=str(exc))

    # ---------------- Consequential / Mutating Tools ----------------

    def create_file(self, path: str, content: str = "", plan_id: Optional[str] = None) -> ToolResult:
        """Creates a new file inside the sandboxed workspace."""
        try:
            resolved = self.sandbox.validate_and_resolve(path, must_exist=False)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            
            # Incremental index
            self.indexer.index_single_file(resolved)

            rel_p = self.sandbox.relative_path(resolved)
            log_id = uuid.uuid4().hex[:12]
            self.store.record_audit_log(
                log_id=log_id,
                plan_id=plan_id,
                action_type="create_file",
                source_path=rel_p,
                details={"size_bytes": len(content.encode("utf-8"))},
            )
            return ToolResult(
                tool_name="create_file",
                success=True,
                data={"path": rel_p, "size_bytes": len(content.encode("utf-8"))},
                audit_id=log_id,
            )
        except Exception as exc:
            return ToolResult(tool_name="create_file", success=False, data=None, error=str(exc))

    def create_directory(self, path: str, plan_id: Optional[str] = None) -> ToolResult:
        """Creates a new directory inside the sandboxed workspace."""
        try:
            resolved = self.sandbox.validate_and_resolve(path, must_exist=False)
            resolved.mkdir(parents=True, exist_ok=True)
            rel_p = self.sandbox.relative_path(resolved)
            log_id = uuid.uuid4().hex[:12]
            self.store.record_audit_log(
                log_id=log_id,
                plan_id=plan_id,
                action_type="create_directory",
                source_path=rel_p,
            )
            return ToolResult(
                tool_name="create_directory",
                success=True,
                data={"path": rel_p},
                audit_id=log_id,
            )
        except Exception as exc:
            return ToolResult(tool_name="create_directory", success=False, data=None, error=str(exc))

    def move_path(self, source_path: str, destination_path: str, plan_id: Optional[str] = None) -> ToolResult:
        """Moves a file or directory within the sandboxed workspace."""
        try:
            src_res = self.sandbox.validate_and_resolve(source_path, must_exist=True)
            dst_res = self.sandbox.validate_and_resolve(destination_path, must_exist=False)
            
            if dst_res.is_dir() or destination_path.endswith("/") or destination_path.endswith("\\"):
                dst_res.mkdir(parents=True, exist_ok=True)
                target_dest = dst_res / src_res.name
            else:
                dst_res.parent.mkdir(parents=True, exist_ok=True)
                target_dest = dst_res

            # Clean old index entries
            if src_res.is_file():
                del_vids = self.store.delete_file(str(src_res))
                self.indexer.vector_store.remove_vectors(del_vids)
            elif src_res.is_dir():
                prefix = str(src_res)
                all_stored = self.store.list_all_files()
                for rec in all_stored:
                    if rec.canonical_path.startswith(prefix):
                        del_vids = self.store.delete_file(rec.canonical_path)
                        self.indexer.vector_store.remove_vectors(del_vids)

            shutil.move(str(src_res), str(target_dest))

            # Re-index moved file/dir
            if target_dest.is_file():
                self.indexer.index_single_file(target_dest)
            elif target_dest.is_dir():
                self.indexer.index_path(str(target_dest), recursive=True)

            src_rel = self.sandbox.relative_path(src_res)
            dst_rel = self.sandbox.relative_path(target_dest)
            log_id = uuid.uuid4().hex[:12]
            self.store.record_audit_log(
                log_id=log_id,
                plan_id=plan_id,
                action_type="move_path",
                source_path=src_rel,
                destination_path=dst_rel,
            )
            return ToolResult(
                tool_name="move_path",
                success=True,
                data={"source": src_rel, "destination": dst_rel},
                audit_id=log_id,
            )
        except Exception as exc:
            return ToolResult(tool_name="move_path", success=False, data=None, error=str(exc))

    def copy_path(self, source_path: str, destination_path: str, plan_id: Optional[str] = None) -> ToolResult:
        """Copies a file or directory within the sandboxed workspace."""
        try:
            src_res = self.sandbox.validate_and_resolve(source_path, must_exist=True)
            dst_res = self.sandbox.validate_and_resolve(destination_path, must_exist=False)

            if dst_res.is_dir() or destination_path.endswith("/") or destination_path.endswith("\\"):
                dst_res.mkdir(parents=True, exist_ok=True)
                target_dest = dst_res / src_res.name
            else:
                dst_res.parent.mkdir(parents=True, exist_ok=True)
                target_dest = dst_res

            if src_res.is_file():
                shutil.copy2(str(src_res), str(target_dest))
                self.indexer.index_single_file(target_dest)
            else:
                shutil.copytree(str(src_res), str(target_dest), dirs_exist_ok=True)
                self.indexer.index_path(str(target_dest), recursive=True)

            src_rel = self.sandbox.relative_path(src_res)
            dst_rel = self.sandbox.relative_path(target_dest)
            log_id = uuid.uuid4().hex[:12]
            self.store.record_audit_log(
                log_id=log_id,
                plan_id=plan_id,
                action_type="copy_path",
                source_path=src_rel,
                destination_path=dst_rel,
            )
            return ToolResult(
                tool_name="copy_path",
                success=True,
                data={"source": src_rel, "destination": dst_rel},
                audit_id=log_id,
            )
        except Exception as exc:
            return ToolResult(tool_name="copy_path", success=False, data=None, error=str(exc))

    def rename_path(self, source_path: str, new_name: str, plan_id: Optional[str] = None) -> ToolResult:
        """Renames a file or folder in place."""
        try:
            src_res = self.sandbox.validate_and_resolve(source_path, must_exist=True)
            target_dest = src_res.parent / new_name
            return self.move_path(source_path, str(target_dest), plan_id=plan_id)
        except Exception as exc:
            return ToolResult(tool_name="rename_path", success=False, data=None, error=str(exc))

    def delete_file(self, path: str, plan_id: Optional[str] = None) -> ToolResult:
        """Safely deletes a file by backing it up to the trash folder."""
        try:
            resolved = self.sandbox.validate_and_resolve(path, must_exist=True)
            if not resolved.is_file():
                return ToolResult(tool_name="delete_file", success=False, data=None, error=f"Path '{path}' is not a file")

            # Backup to trash
            trash_dir = self._get_trash_dir()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_fn = f"{ts}_{uuid.uuid4().hex[:6]}_{resolved.name}"
            backup_path = trash_dir / backup_fn
            shutil.copy2(str(resolved), str(backup_path))

            # Delete file & vectors
            del_vids = self.store.delete_file(str(resolved))
            self.indexer.vector_store.remove_vectors(del_vids)
            resolved.unlink()

            rel_p = self.sandbox.relative_path(resolved)
            log_id = uuid.uuid4().hex[:12]
            self.store.record_audit_log(
                log_id=log_id,
                plan_id=plan_id,
                action_type="delete_file",
                source_path=rel_p,
                backup_path=str(backup_path),
            )
            return ToolResult(
                tool_name="delete_file",
                success=True,
                data={"deleted_path": rel_p, "backup_path": str(backup_path)},
                audit_id=log_id,
            )
        except Exception as exc:
            return ToolResult(tool_name="delete_file", success=False, data=None, error=str(exc))

    def delete_directory(self, path: str, recursive: bool = False, plan_id: Optional[str] = None) -> ToolResult:
        """Safely deletes a directory by backing it up to the trash folder."""
        try:
            resolved = self.sandbox.validate_and_resolve(path, must_exist=True)
            if not resolved.is_dir():
                return ToolResult(tool_name="delete_directory", success=False, data=None, error=f"Path '{path}' is not a directory")

            # Backup to trash
            trash_dir = self._get_trash_dir()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{ts}_{uuid.uuid4().hex[:6]}_{resolved.name}"
            backup_path = trash_dir / backup_name
            shutil.copytree(str(resolved), str(backup_path))

            # Remove records from SQLite & FAISS
            prefix = str(resolved)
            all_stored = self.store.list_all_files()
            for rec in all_stored:
                if rec.canonical_path.startswith(prefix):
                    del_vids = self.store.delete_file(rec.canonical_path)
                    self.indexer.vector_store.remove_vectors(del_vids)

            shutil.rmtree(str(resolved))

            rel_p = self.sandbox.relative_path(resolved)
            log_id = uuid.uuid4().hex[:12]
            self.store.record_audit_log(
                log_id=log_id,
                plan_id=plan_id,
                action_type="delete_directory",
                source_path=rel_p,
                backup_path=str(backup_path),
            )
            return ToolResult(
                tool_name="delete_directory",
                success=True,
                data={"deleted_path": rel_p, "backup_path": str(backup_path)},
                audit_id=log_id,
            )
        except Exception as exc:
            return ToolResult(tool_name="delete_directory", success=False, data=None, error=str(exc))

    def organize_files(
        self,
        source_dir: str = "",
        rules: Optional[Dict[str, str]] = None,
        plan_id: Optional[str] = None,
    ) -> ToolResult:
        """Batch categorizes and organizes files into clean subdirectories."""
        try:
            resolved = self.sandbox.validate_and_resolve(source_dir, must_exist=True)
            if not resolved.is_dir():
                return ToolResult(tool_name="organize_files", success=False, data=None, error="Target must be a directory")

            default_rules = {
                ".pdf": "documents",
                ".docx": "documents",
                ".txt": "documents",
                ".md": "documents",
                ".png": "images",
                ".jpg": "images",
                ".jpeg": "images",
                ".svg": "images",
                ".mp4": "media",
                ".mp3": "media",
                ".wav": "media",
                ".csv": "data",
                ".json": "data",
                ".yaml": "data",
                ".yml": "data",
                ".zip": "archives",
                ".tar": "archives",
                ".gz": "archives",
            }
            active_rules = rules or default_rules
            moved_items = []

            for entry in list(resolved.iterdir()):
                if entry.is_file() and not entry.name.startswith("."):
                    ext = entry.suffix.lower()
                    target_folder = active_rules.get(ext)
                    if target_folder:
                        dest_folder = resolved / target_folder
                        dest_folder.mkdir(parents=True, exist_ok=True)
                        dest_file = dest_folder / entry.name

                        if dest_file.exists():
                            # Don't overwrite if existing
                            continue

                        # Move file
                        move_res = self.move_path(
                            self.sandbox.relative_path(entry),
                            self.sandbox.relative_path(dest_file),
                            plan_id=plan_id,
                        )
                        if move_res.success:
                            moved_items.append(move_res.data)

            log_id = uuid.uuid4().hex[:12]
            self.store.record_audit_log(
                log_id=log_id,
                plan_id=plan_id,
                action_type="organize_files",
                source_path=self.sandbox.relative_path(resolved),
                details={"moved_count": len(moved_items), "moved_items": moved_items},
            )
            return ToolResult(
                tool_name="organize_files",
                success=True,
                data={"organized_count": len(moved_items), "moved": moved_items},
                audit_id=log_id,
            )
        except Exception as exc:
            return ToolResult(tool_name="organize_files", success=False, data=None, error=str(exc))

    def undo_action(self, log_id: str) -> ToolResult:
        """Rolls back an executed mutating tool operation using its audit log record."""
        try:
            log_entry = self.store.get_audit_log(log_id)
            if not log_entry:
                return ToolResult(tool_name="undo_action", success=False, data=None, error=f"Audit log ID '{log_id}' not found")

            if log_entry.get("undone"):
                return ToolResult(tool_name="undo_action", success=False, data=None, error=f"Action '{log_id}' has already been undone")

            action_type = log_entry["action_type"]
            src_p = log_entry.get("source_path")
            dst_p = log_entry.get("destination_path")
            backup_p = log_entry.get("backup_path")

            if action_type == "create_file":
                if src_p:
                    res_file = self.sandbox.validate_and_resolve(src_p, must_exist=True)
                    del_vids = self.store.delete_file(str(res_file))
                    self.indexer.vector_store.remove_vectors(del_vids)
                    res_file.unlink(missing_ok=True)

            elif action_type == "create_directory":
                if src_p:
                    res_dir = self.sandbox.validate_and_resolve(src_p, must_exist=True)
                    shutil.rmtree(str(res_dir), ignore_errors=True)

            elif action_type in ("move_path", "rename_path"):
                if src_p and dst_p:
                    # Move back from destination to source
                    res_dst = self.sandbox.validate_and_resolve(dst_p, must_exist=True)
                    res_src = self.sandbox.validate_and_resolve(src_p, must_exist=False)
                    res_src.parent.mkdir(parents=True, exist_ok=True)
                    
                    if res_dst.is_file():
                        del_vids = self.store.delete_file(str(res_dst))
                        self.indexer.vector_store.remove_vectors(del_vids)
                        shutil.move(str(res_dst), str(res_src))
                        self.indexer.index_single_file(res_src)
                    elif res_dst.is_dir():
                        shutil.move(str(res_dst), str(res_src))
                        self.indexer.index_path(str(res_src), recursive=True)

            elif action_type == "copy_path":
                if dst_p:
                    res_dst = self.sandbox.validate_and_resolve(dst_p, must_exist=True)
                    if res_dst.is_file():
                        del_vids = self.store.delete_file(str(res_dst))
                        self.indexer.vector_store.remove_vectors(del_vids)
                        res_dst.unlink()
                    elif res_dst.is_dir():
                        shutil.rmtree(str(res_dst))

            elif action_type in ("delete_file", "delete_directory"):
                if backup_p and src_p:
                    b_path = Path(backup_p)
                    if not b_path.exists():
                        return ToolResult(tool_name="undo_action", success=False, data=None, error=f"Backup file not found at {backup_p}")
                    
                    res_src = self.sandbox.validate_and_resolve(src_p, must_exist=False)
                    res_src.parent.mkdir(parents=True, exist_ok=True)
                    
                    if b_path.is_file():
                        shutil.copy2(str(b_path), str(res_src))
                        self.indexer.index_single_file(res_src)
                    elif b_path.is_dir():
                        shutil.copytree(str(b_path), str(res_src), dirs_exist_ok=True)
                        self.indexer.index_path(str(res_src), recursive=True)

            # Mark audit undone
            self.store.mark_audit_undone(log_id)

            return ToolResult(
                tool_name="undo_action",
                success=True,
                data={"undone_log_id": log_id, "action_type": action_type, "restored_path": src_p},
            )
        except Exception as exc:
            return ToolResult(tool_name="undo_action", success=False, data=None, error=str(exc))

    def list_operation_history(self, limit: int = 30) -> ToolResult:
        """Lists recent mutating operations from persistent audit history."""
        try:
            logs = self.store.list_audit_logs(limit=limit)
            return ToolResult(tool_name="list_operation_history", success=True, data={"operations": logs, "total_count": len(logs)})
        except Exception as exc:
            return ToolResult(tool_name="list_operation_history", success=False, data=None, error=str(exc))

    def revert_last_operation(self) -> ToolResult:
        """Finds and rolls back the most recent active mutating operation."""
        try:
            recent_logs = self.store.list_audit_logs(limit=20)
            active_logs = [l for l in recent_logs if not l.get("undone")]
            if not active_logs:
                return ToolResult(
                    tool_name="revert_last_operation",
                    success=False,
                    data=None,
                    error="No active reversible operations found in audit history.",
                )
            target_log = active_logs[0]
            res = self.undo_action(target_log["log_id"])
            if res.success:
                res.tool_name = "revert_last_operation"
            return res
        except Exception as exc:
            return ToolResult(tool_name="revert_last_operation", success=False, data=None, error=str(exc))

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Validates and executes a tool call deterministically."""
        name = tool_call.name
        args = tool_call.arguments or {}

        if name == "list_directory":
            return self.list_directory(path=args.get("path"))
        elif name == "read_file_metadata":
            if "file_path" not in args:
                return ToolResult(tool_name=name, success=False, data=None, error="Missing required argument 'file_path'")
            return self.read_file_metadata(file_path=args["file_path"])
        elif name == "inspect_file":
            if "file_path" not in args:
                return ToolResult(tool_name=name, success=False, data=None, error="Missing required argument 'file_path'")
            return self.inspect_file(file_path=args["file_path"])
        elif name == "index_path":
            return self.index_path(path=args.get("path"), recursive=args.get("recursive"))
        elif name == "rag_search":
            if "query" not in args:
                return ToolResult(tool_name=name, success=False, data=None, error="Missing required argument 'query'")
            return self.rag_search(
                query=args["query"],
                path_filter=args.get("path_filter"),
                file_type_filter=args.get("file_type_filter"),
                limit=args.get("limit"),
            )
        elif name == "get_index_status":
            return self.get_index_status()
        elif name == "create_file":
            return self.create_file(path=args.get("path", ""), content=args.get("content", ""), plan_id=args.get("plan_id"))
        elif name == "create_directory":
            return self.create_directory(path=args.get("path", ""), plan_id=args.get("plan_id"))
        elif name == "move_path":
            return self.move_path(source_path=args.get("source_path", ""), destination_path=args.get("destination_path", ""), plan_id=args.get("plan_id"))
        elif name == "copy_path":
            return self.copy_path(source_path=args.get("source_path", ""), destination_path=args.get("destination_path", ""), plan_id=args.get("plan_id"))
        elif name == "rename_path":
            return self.rename_path(source_path=args.get("source_path", ""), new_name=args.get("new_name", ""), plan_id=args.get("plan_id"))
        elif name == "delete_file":
            return self.delete_file(path=args.get("path", ""), plan_id=args.get("plan_id"))
        elif name == "delete_directory":
            return self.delete_directory(path=args.get("path", ""), recursive=args.get("recursive", False), plan_id=args.get("plan_id"))
        elif name == "organize_files":
            return self.organize_files(source_dir=args.get("source_dir", ""), rules=args.get("rules"), plan_id=args.get("plan_id"))
        elif name == "undo_action":
            return self.undo_action(log_id=args.get("log_id", ""))
        elif name in ("revert_last_operation", "undo_last_operation"):
            return self.revert_last_operation()
        elif name == "list_operation_history":
            return self.list_operation_history(limit=int(args.get("limit", 30)))

        return ToolResult(tool_name=name, success=False, data=None, error=f"Unknown tool '{name}'")
