"""Lightweight, offline HTTP server and REST API for the MindFS Web UI."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import urllib.parse
import threading
import uuid

from mindfs.agent.llm import LLMEngine
from mindfs.agent.loop import MindFSAgent
from mindfs.agent.tools import FilesystemTools
from mindfs.config.settings import MindFSConfig, load_config
from mindfs.filesystem.sandbox import FilesystemSandbox
from mindfs.indexing.embeddings import EmbeddingPipeline
from mindfs.indexing.indexer import Indexer
from mindfs.indexing.vector_store import VectorStore
from mindfs.models.manager import ModelManager
from mindfs.processors import create_default_registry
from mindfs.resources.manager import ResourceManager
from mindfs.retrieval.search import SearchEngine
from mindfs.storage.sqlite_store import SQLiteStore


class MindFSServer:
    """Manages the MindFS application instance for the Web UI."""

    # Global history DB (shared across all workspaces) at ~/.mindfs/history.db
    _GLOBAL_HISTORY_DB = Path.home() / ".mindfs" / "history.db"

    def __init__(self, config_path: Optional[str] = None, workspace_root: Optional[str] = None):
        self.config_path = config_path
        self.indexing_lock = threading.Lock()
        self.indexing_progress: Dict[str, Any] = {
            "is_indexing": False,
            "target_path": "",
            "current_file": "",
            "processed": 0,
            "total": 0,
            "indexed": 0,
            "skipped": 0,
            "failed": 0,
            "percent": 0,
            "status": "idle",
            "last_result": None,
            "error": None,
        }
        # Initialize persistent global history store once (shared across workspaces)
        self._GLOBAL_HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
        self.global_store = SQLiteStore(self._GLOBAL_HISTORY_DB)
        self._init_system(workspace_root=workspace_root)

    def update_indexing_progress(self, idx: int, total: int, filename: str, status: str, indexed: int = 0, skipped: int = 0, failed: int = 0):
        with self.indexing_lock:
            percent = int((idx / max(total, 1)) * 100) if total > 0 else 0
            self.indexing_progress.update({
                "is_indexing": True,
                "current_file": filename,
                "processed": idx,
                "total": total,
                "indexed": indexed,
                "skipped": skipped,
                "failed": failed,
                "percent": percent,
                "status": status,
            })

    def _init_system(self, workspace_root: Optional[str] = None):
        self.config = load_config(config_path=self.config_path, workspace_root=workspace_root)
        self.store = SQLiteStore(self.config.resolved_db_path)
        
        # Load any existing indexed folders
        existing_folders = []
        try:
            existing_folders = [f["folder_path"] for f in self.store.list_indexed_folders()]
        except Exception:
            pass

        self.sandbox = FilesystemSandbox(self.config.resolved_workspace_root, allowed_roots=existing_folders)
        self.resources = ResourceManager(self.config)
        self.registry = create_default_registry(self.config)
        self.vector_store = VectorStore(
            embedding_dim=self.config.embedding.embedding_dim,
            index_path=self.config.resolved_faiss_path,
        )
        self.embedding_pipeline = EmbeddingPipeline(self.config)
        self.indexer = Indexer(
            config=self.config,
            sandbox=self.sandbox,
            store=self.store,
            registry=self.registry,
            vector_store=self.vector_store,
            embedding_pipeline=self.embedding_pipeline,
            resource_manager=self.resources,
        )
        self.search_engine = SearchEngine(
            config=self.config,
            sandbox=self.sandbox,
            store=self.store,
            vector_store=self.vector_store,
            embedding_pipeline=self.embedding_pipeline,
            resource_manager=self.resources,
        )
        self.tools = FilesystemTools(
            config=self.config,
            sandbox=self.sandbox,
            store=self.store,
            indexer=self.indexer,
            search_engine=self.search_engine,
            registry=self.registry,
        )
        self.llm_engine = LLMEngine(self.config)
        self.model_manager = ModelManager(config=self.config, llm_engine=self.llm_engine)
        self.agent = MindFSAgent(config=self.config, tools=self.tools, llm_engine=self.llm_engine)

        # Auto-activate first compatible generative LLM on startup
        try:
            discovered = self.model_manager.discover_models()
            compatible_llms = [
                m for m in discovered 
                if m.compatibility == "compatible" and not any(emb in m.model_name.lower() for emb in ("sentence-transformers", "bge-", "e5-", "minilm"))
            ]
            if compatible_llms:
                self.model_manager.switch_model(compatible_llms[0].file_path, context_tokens=self.config.llm.context_tokens or 2048)
        except Exception:
            pass

        try:
            self.global_store.record_recent_workspace(str(self.config.resolved_workspace_root))
        except Exception:
            pass

    def set_workspace(self, new_root: str) -> Dict[str, Any]:
        """Dynamically switches active workspace root."""
        p = Path(new_root).expanduser().resolve()
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        self._init_system(workspace_root=str(p))
        # _init_system also calls global_store.record_recent_workspace; duplicate call not needed
        return {
            "workspace_root": str(p),
            "status": "switched",
        }


def create_request_handler(app: MindFSServer):
    """Creates a request handler bound to the MindFSServer instance."""

    class MindFSHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send_json(self, status_code: int, data: Any):
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html_content: str):
            body = html_content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len == 0:
                return {}
            raw = self.rfile.read(content_len).decode("utf-8")
            return json.loads(raw)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = urllib.parse.parse_qs(parsed.query)

            if path == "/" or path == "/index.html":
                html_path = Path(__file__).parent / "static" / "index.html"
                if html_path.exists():
                    with open(html_path, "r", encoding="utf-8") as f:
                        self._send_html(f.read())
                    return
                else:
                    self._send_json(404, {"error": "UI index.html not found"})
                    return

            elif path.startswith("/static/"):
                filename = path[len("/static/"):]
                static_path = Path(__file__).parent / "static" / filename
                if static_path.exists() and static_path.is_file():
                    ext = static_path.suffix.lower()
                    mime = {
                        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".gif": "image/gif",
                        ".svg": "image/svg+xml", ".ico": "image/x-icon",
                        ".css": "text/css", ".js": "application/javascript",
                    }.get(ext, "application/octet-stream")
                    data = static_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.end_headers()
                    self.wfile.write(data)
                    return
                else:
                    self._send_json(404, {"error": f"Static file '{filename}' not found"})
                    return

            elif path == "/api/status":
                status = app.tools.get_index_status()
                res = status.data if status.success else {"error": status.error}
                res["workspace_root"] = str(app.config.resolved_workspace_root)
                res["indexed_folders"] = app.store.list_indexed_folders()
                res["peak_rss_mb"] = app.resources.get_peak_rss_mb()
                res["current_rss_mb"] = app.resources.get_current_rss_mb()
                res["budget_max_rss_mb"] = app.resources.max_rss_mb
                self._send_json(200, res)

            elif path == "/api/chat/sessions":
                sessions = app.store.list_chat_sessions()
                self._send_json(200, {"sessions": sessions})

            elif path.startswith("/api/chat/sessions/") and path.endswith("/messages"):
                parts = path.split("/")
                session_id = parts[4] if len(parts) > 4 else ""
                messages = app.store.get_session_messages(session_id)
                self._send_json(200, {"session_id": session_id, "messages": messages})

            elif path == "/api/folders":
                folders = app.store.list_indexed_folders()
                self._send_json(200, {"folders": folders})

            elif path == "/api/workspaces/recent":
                recent = app.global_store.list_recent_workspaces(limit=15)
                self._send_json(200, {
                    "current_workspace": str(app.config.resolved_workspace_root),
                    "recent_workspaces": recent,
                })

            elif path == "/api/operations/history":
                limit = int(params.get("limit", ["40"])[0])
                logs = app.store.list_audit_logs(limit=limit)
                self._send_json(200, {"operations": logs, "count": len(logs)})

            elif path == "/api/operations/revert-last":
                undo_res = app.tools.revert_last_operation()
                if undo_res.success:
                    self._send_json(200, undo_res.data)
                else:
                    self._send_json(400, {"error": undo_res.error})

            elif path == "/api/diagnostics":
                summary = app.resources.get_summary()
                self._send_json(200, summary)

            elif path == "/api/files":
                target_sub = params.get("path", [""])[0]
                dir_res = app.tools.list_directory(target_sub)
                if dir_res.success:
                    parent = str(Path(target_sub).parent) if target_sub and target_sub != "." else None
                    if parent == ".":
                        parent = ""
                    self._send_json(200, {
                        "entries": dir_res.data,
                        "current_path": target_sub,
                        "parent_path": parent,
                        "workspace_root": str(app.config.resolved_workspace_root),
                    })
                else:
                    self._send_json(400, {"error": dir_res.error})

            elif path == "/api/browse_directories":
                req_path = params.get("path", [""])[0]
                if not req_path:
                    target_dir = Path(app.config.resolved_workspace_root)
                else:
                    target_dir = Path(req_path).expanduser().resolve()

                if not target_dir.exists() or not target_dir.is_dir():
                    target_dir = Path.home()

                subdirs = []
                files_list = []
                try:
                    for entry in sorted(os.scandir(target_dir), key=lambda e: e.name.lower()):
                        if entry.name.startswith(".") or entry.name in ("__pycache__", "node_modules", ".git"):
                            continue
                        try:
                            is_dir = entry.is_dir(follow_symlinks=True)
                        except OSError:
                            is_dir = False

                        if is_dir:
                            subdirs.append(entry.name)
                        else:
                            try:
                                sz = entry.stat().st_size
                            except OSError:
                                sz = 0
                            files_list.append({"name": entry.name, "size_bytes": sz})
                except Exception:
                    pass

                parent = str(target_dir.parent) if target_dir != target_dir.parent else None
                shortcuts = [
                    {"label": "Home", "path": str(Path.home())},
                    {"label": "Desktop", "path": str(Path.home() / "Desktop")},
                    {"label": "Documents", "path": str(Path.home() / "Documents")},
                    {"label": "Downloads", "path": str(Path.home() / "Downloads")},
                ]
                if (Path.home() / "Desktop" / "code").exists():
                    shortcuts.append({"label": "Code", "path": str(Path.home() / "Desktop" / "code")})
                if (Path("/") / "Users").exists():
                    shortcuts.append({"label": "/Users", "path": "/Users"})

                self._send_json(200, {
                    "current_path": str(target_dir),
                    "parent_path": parent,
                    "directories": subdirs,
                    "files": files_list,
                    "shortcuts": shortcuts,
                })

            elif path == "/api/models":
                models = app.model_manager.scan_directories()
                active = app.model_manager.get_active_model()
                self._send_json(200, {
                    "models": [m.model_dump() for m in models],
                    "active_model": active.model_dump() if active else None,
                })

            elif path == "/api/agent/audit":
                logs = app.store.list_audit_logs(limit=30)
                self._send_json(200, {"audit_logs": logs})

            elif path == "/api/index/progress":
                with app.indexing_lock:
                    self._send_json(200, app.indexing_progress)

            elif path == "/api/inspect":
                target_file = params.get("file_path", [""])[0]
                if not target_file:
                    self._send_json(400, {"error": "Missing 'file_path' parameter"})
                    return
                insp_res = app.tools.inspect_file(target_file)
                if insp_res.success:
                    self._send_json(200, insp_res.data)
                else:
                    self._send_json(400, {"error": insp_res.error})

            else:
                self._send_json(404, {"error": f"Endpoint '{path}' not found"})

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/api/set_workspace":
                payload = self._read_json()
                new_root = payload.get("workspace_root", "").strip()
                if not new_root:
                    self._send_json(400, {"error": "Missing 'workspace_root'"})
                    return
                try:
                    result = app.set_workspace(new_root)
                    self._send_json(200, result)
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})

            elif path == "/api/folders/add" or path == "/api/index":
                payload = self._read_json()
                target_p = payload.get("folder_path") or payload.get("path", "").strip()
                recursive = payload.get("recursive", True)

                # Add to sandbox allowed roots
                if target_p and Path(target_p).is_absolute():
                    tp = Path(target_p).expanduser().resolve()
                    if tp.is_dir():
                        app.sandbox.add_allowed_root(tp)

                with app.indexing_lock:
                    app.indexing_progress.update({
                        "is_indexing": True,
                        "target_path": str(target_p),
                        "current_file": "Scanning directory...",
                        "processed": 0,
                        "total": 0,
                        "indexed": 0,
                        "skipped": 0,
                        "failed": 0,
                        "percent": 0,
                        "status": "scanning",
                        "last_result": None,
                        "error": None,
                    })

                idx_res = app.tools.index_path(
                    target_p,
                    recursive=recursive,
                    progress_callback=app.update_indexing_progress,
                )

                with app.indexing_lock:
                    app.indexing_progress["is_indexing"] = False
                    if idx_res.success:
                        app.indexing_progress["status"] = "completed"
                        app.indexing_progress["last_result"] = idx_res.data
                        app.indexing_progress["percent"] = 100
                    else:
                        app.indexing_progress["status"] = "failed"
                        app.indexing_progress["error"] = idx_res.error

                if idx_res.success:
                    self._send_json(200, idx_res.data)
                else:
                    self._send_json(400, {"error": idx_res.error})

            elif path == "/api/folders/remove":
                payload = self._read_json()
                folder_p = payload.get("folder_path", "").strip()
                if not folder_p:
                    self._send_json(400, {"error": "Missing 'folder_path'"})
                    return
                try:
                    deleted_vids = app.store.remove_indexed_folder(folder_p)
                    if deleted_vids:
                        app.vector_store.remove_vectors(deleted_vids)
                    app.sandbox.remove_allowed_root(folder_p)
                    self._send_json(200, {
                        "status": "removed",
                        "folder_path": folder_p,
                        "deleted_vectors": len(deleted_vids)
                    })
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})

            elif path == "/api/rebuild":
                with app.indexing_lock:
                    app.indexing_progress.update({
                        "is_indexing": True,
                        "target_path": "rebuild",
                        "current_file": "Clearing indexes...",
                        "processed": 0,
                        "total": 0,
                        "indexed": 0,
                        "skipped": 0,
                        "failed": 0,
                        "percent": 0,
                        "status": "rebuilding",
                        "last_result": None,
                        "error": None,
                    })

                app.indexer.clear_index()
                idx_res = app.tools.index_path(
                    "",
                    recursive=True,
                    progress_callback=app.update_indexing_progress,
                )

                with app.indexing_lock:
                    app.indexing_progress["is_indexing"] = False
                    if idx_res.success:
                        app.indexing_progress["status"] = "completed"
                        app.indexing_progress["last_result"] = idx_res.data
                        app.indexing_progress["percent"] = 100
                    else:
                        app.indexing_progress["status"] = "failed"
                        app.indexing_progress["error"] = idx_res.error

                if idx_res.success:
                    self._send_json(200, idx_res.data)
                else:
                    self._send_json(400, {"error": idx_res.error})

            elif path == "/api/search":
                payload = self._read_json()
                query = payload.get("query", "").strip()
                path_filter = payload.get("path_filter")
                type_filter = payload.get("type_filter")
                limit = payload.get("limit", 5)

                res = app.search_engine.search(
                    query=query,
                    path_filter=path_filter,
                    file_type_filter=type_filter,
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
                self._send_json(200, {
                    "query": query,
                    "has_sufficient_evidence": res.has_sufficient_evidence,
                    "evidence_count": len(evidence_data),
                    "evidence": evidence_data,
                    "diagnostic_info": res.diagnostic_info,
                })

            elif path == "/api/models/switch":
                payload = self._read_json()
                model_p = payload.get("model_path", "").strip()
                ctx_tokens = int(payload.get("context_tokens", 2048))
                if not model_p:
                    self._send_json(400, {"error": "Missing 'model_path' parameter"})
                    return
                try:
                    res = app.model_manager.switch_model(model_p, context_tokens=ctx_tokens)
                    self._send_json(200, res)
                except Exception as exc:
                    self._send_json(400, {"error": str(exc)})

            elif path == "/api/models/scan":
                payload = self._read_json()
                custom_paths = payload.get("custom_paths", [])
                models = app.model_manager.scan_directories(custom_paths)
                active = app.model_manager.get_active_model()
                self._send_json(200, {
                    "models": [m.model_dump() for m in models],
                    "active_model": active.model_dump() if active else None,
                })

            elif path == "/api/agent/approve" or path == "/api/agent/plan/approve":
                payload = self._read_json()
                plan_id = payload.get("plan_id", "").strip()
                session_id = payload.get("session_id", "").strip()
                approved = bool(payload.get("approved", True))
                if not plan_id:
                    self._send_json(400, {"error": "Missing 'plan_id' parameter"})
                    return
                resp = app.agent.execute_plan(plan_id, approved=approved)
                if session_id:
                    asst_msg_id = uuid.uuid4().hex[:12]
                    app.store.add_chat_message(
                        asst_msg_id,
                        session_id,
                        "assistant",
                        resp.answer,
                        thoughts=[t.model_dump() for t in resp.thoughts],
                        tool_calls=[o.model_dump() for o in resp.operations],
                        explored_files=resp.explored_files,
                        subagents=[s.model_dump() for s in resp.subagents],
                        plan_id=plan_id,
                        status=resp.status,
                        can_undo=resp.can_undo,
                        undo_log_ids=resp.undo_log_ids,
                    )
                self._send_json(200, resp.model_dump())

            elif path == "/api/agent/undo" or path == "/api/operations/revert":
                payload = self._read_json()
                log_id = payload.get("log_id", "").strip()
                session_id = payload.get("session_id", "").strip()
                if not log_id:
                    self._send_json(400, {"error": "Missing 'log_id' parameter"})
                    return
                undo_res = app.tools.undo_action(log_id)
                if undo_res.success:
                    data = undo_res.data or {}
                    answer_text = f"↩️ **Action Undone**: Restored `{data.get('restored_path')}` (Action: `{data.get('action_type')}`)."
                    if session_id:
                        msg_id = uuid.uuid4().hex[:12]
                        app.store.add_chat_message(
                            msg_id,
                            session_id,
                            "assistant",
                            answer_text,
                            thoughts=[{"title": "Audit Log Rollback Execution", "detail": f"Reverted action '{data.get('action_type')}' (ID: {log_id}). Restored '{data.get('restored_path')}'."}],
                            tool_calls=[{"type": "command", "title": "Undo Action", "command_or_tool": "undo_action", "args": {"log_id": log_id}, "summary": f"Restored {data.get('restored_path')}", "status": "COMPLETED"}],
                            subagents=[{"role": "Undo & Rollback Subagent", "name": "Audit Reversal Engine", "task": f"Reverted action '{data.get('action_type')}'", "status": "COMPLETED"}],
                            status="COMPLETED",
                        )
                    resp_data = {
                        **data,
                        "answer": answer_text,
                        "content": answer_text,
                        "role": "assistant",
                        "status": "COMPLETED",
                        "thoughts": [{"title": "Audit Log Rollback Execution", "detail": f"Reverted action '{data.get('action_type')}' (ID: {log_id}). Restored '{data.get('restored_path')}'."}],
                        "operations": [{"type": "command", "title": "Undo Action", "command_or_tool": "undo_action", "args": {"log_id": log_id}, "summary": f"Restored {data.get('restored_path')}", "status": "COMPLETED"}],
                        "subagents": [{"role": "Undo & Rollback Subagent", "name": "Audit Reversal Engine", "task": f"Reverted action '{data.get('action_type')}'", "status": "COMPLETED"}],
                    }
                    self._send_json(200, resp_data)
                else:
                    self._send_json(400, {"error": undo_res.error})

            elif path == "/api/operations/history":
                limit = int(self._read_json().get("limit", 40) if self.headers.get("Content-Length") else 40)
                logs = app.store.list_audit_logs(limit=limit)
                self._send_json(200, {"operations": logs, "count": len(logs)})

            elif path == "/api/operations/revert-last":
                payload = self._read_json() if self.headers.get("Content-Length") else {}
                session_id = payload.get("session_id", "").strip()
                undo_res = app.tools.revert_last_operation()
                if undo_res.success:
                    data = undo_res.data or {}
                    answer_text = f"↩️ **Action Undone**: Restored `{data.get('restored_path')}` (Action: `{data.get('action_type')}`)."
                    if session_id:
                        msg_id = uuid.uuid4().hex[:12]
                        app.store.add_chat_message(
                            msg_id,
                            session_id,
                            "assistant",
                            answer_text,
                            thoughts=[{"title": "Audit Log Rollback Execution", "detail": f"Reverted action '{data.get('action_type')}' (ID: {data.get('undone_log_id')}). Restored '{data.get('restored_path')}'."}],
                            tool_calls=[{"type": "command", "title": "Undo Action", "command_or_tool": "undo_action", "args": {"log_id": data.get("undone_log_id")}, "summary": f"Restored {data.get('restored_path')}", "status": "COMPLETED"}],
                            subagents=[{"role": "Undo & Rollback Subagent", "name": "Audit Reversal Engine", "task": f"Reverted action '{data.get('action_type')}'", "status": "COMPLETED"}],
                            status="COMPLETED",
                        )
                    resp_data = {
                        **data,
                        "answer": answer_text,
                        "content": answer_text,
                        "role": "assistant",
                        "status": "COMPLETED",
                        "thoughts": [{"title": "Audit Log Rollback Execution", "detail": f"Reverted action '{data.get('action_type')}' (ID: {data.get('undone_log_id')}). Restored '{data.get('restored_path')}'."}],
                        "operations": [{"type": "command", "title": "Undo Action", "command_or_tool": "undo_action", "args": {"log_id": data.get("undone_log_id")}, "summary": f"Restored {data.get('restored_path')}", "status": "COMPLETED"}],
                        "subagents": [{"role": "Undo & Rollback Subagent", "name": "Audit Reversal Engine", "task": f"Reverted action '{data.get('action_type')}'", "status": "COMPLETED"}],
                    }
                    self._send_json(200, resp_data)
                else:
                    self._send_json(400, {"error": undo_res.error})

            elif path == "/api/workspaces/recent/remove":
                payload = self._read_json()
                target_p = payload.get("path", "").strip()
                if target_p:
                    app.global_store.remove_recent_workspace(target_p)
                self._send_json(200, {"status": "removed", "path": target_p})

            elif path == "/api/config/budget":
                payload = self._read_json()
                max_mb = float(payload.get("max_rss_mb", 2048.0))
                app.resources.set_budget(max_mb)
                self._send_json(200, {
                    "max_rss_mb": max_mb,
                    "budget_gb": round(max_mb / 1024.0, 2),
                    "status": "updated"
                })

            elif path == "/api/chat/sessions":
                payload = self._read_json()
                sid = payload.get("session_id") or uuid.uuid4().hex[:12]
                title = payload.get("title") or "New Chat"
                m_name = payload.get("model_name") or (app.model_manager.active_model_path if hasattr(app, "model_manager") else None)
                session = app.store.create_chat_session(sid, title, m_name)
                self._send_json(200, session)

            elif path == "/api/chat/sessions/delete":
                payload = self._read_json()
                sid = payload.get("session_id", "").strip()
                if sid:
                    app.store.delete_chat_session(sid)
                self._send_json(200, {"status": "deleted", "session_id": sid})

            elif path == "/api/chat/message":
                payload = self._read_json()
                session_id = payload.get("session_id", "").strip()
                query = payload.get("query", "").strip()
                if not query:
                    self._send_json(400, {"error": "Missing query parameter"})
                    return

                if not session_id:
                    session_id = uuid.uuid4().hex[:12]
                
                existing_sess = app.store.get_chat_session(session_id)
                if not existing_sess:
                    first_title = (query[:30] + "...") if len(query) > 30 else query
                    m_name = app.model_manager.active_model_path if hasattr(app, "model_manager") else None
                    app.store.create_chat_session(session_id, first_title, m_name)

                user_msg_id = uuid.uuid4().hex[:12]
                app.store.add_chat_message(user_msg_id, session_id, "user", query)

                history = app.store.get_session_messages(session_id)
                resp = app.agent.ask(query, history=history)

                asst_msg_id = uuid.uuid4().hex[:12]
                plan_dict = resp.plan.model_dump() if resp.plan else None
                thoughts_list = [t.model_dump() for t in resp.thoughts]
                ops_list = [o.model_dump() for o in resp.operations]
                subagents_list = [s.model_dump() for s in resp.subagents]

                app.store.add_chat_message(
                    asst_msg_id,
                    session_id,
                    "assistant",
                    resp.answer,
                    thoughts=thoughts_list,
                    tool_calls=ops_list,
                    explored_files=resp.explored_files,
                    subagents=subagents_list,
                    plan_id=resp.plan.plan_id if resp.plan else None,
                    plan_data=plan_dict,
                    status=resp.status,
                    can_undo=resp.can_undo,
                    undo_log_ids=resp.undo_log_ids,
                )

                if len(history) <= 2:
                    app.store.update_chat_session(session_id, title=(query[:30] + "...") if len(query) > 30 else query)

                resp_dict = resp.model_dump()
                resp_dict["session_id"] = session_id
                resp_dict["message_id"] = asst_msg_id
                resp_dict["plan_data"] = plan_dict
                resp_dict["plan_id"] = resp.plan.plan_id if resp.plan else None
                self._send_json(200, resp_dict)

            elif path == "/api/ask":
                payload = self._read_json()
                query = payload.get("query", "").strip()
                if not query:
                    self._send_json(400, {"error": "Missing query parameter"})
                    return

                resp = app.agent.ask(query)
                self._send_json(200, resp.model_dump())

            else:
                self._send_json(404, {"error": f"Endpoint '{path}' not found"})

    return MindFSHTTPHandler


def run_ui_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    config_path: Optional[str] = None,
    workspace_root: Optional[str] = None,
) -> None:
    """Starts the MindFS web application HTTP server."""
    app = MindFSServer(config_path=config_path, workspace_root=workspace_root)
    handler_class = create_request_handler(app)
    server = ThreadingHTTPServer((host, port), handler_class)
    print(f"MindFS UI Server active at: http://{host}:{port}")
    print(f"Active Workspace: {app.config.resolved_workspace_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMindFS UI Server stopped.")
        server.server_close()
