"""Lightweight, offline HTTP server and REST API for the MindFS Web UI."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import urllib.parse
import threading

from mindfs.agent.llm import LLMEngine
from mindfs.agent.loop import MindFSAgent
from mindfs.agent.tools import FilesystemTools
from mindfs.config.settings import MindFSConfig, load_config
from mindfs.filesystem.sandbox import FilesystemSandbox
from mindfs.indexing.embeddings import EmbeddingPipeline
from mindfs.indexing.indexer import Indexer
from mindfs.indexing.vector_store import VectorStore
from mindfs.processors import create_default_registry
from mindfs.resources.manager import ResourceManager
from mindfs.retrieval.search import SearchEngine
from mindfs.storage.sqlite_store import SQLiteStore


class MindFSServer:
    """Manages the MindFS application instance for the Web UI."""

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
        self.sandbox = FilesystemSandbox(self.config.resolved_workspace_root)
        self.resources = ResourceManager(self.config)
        self.store = SQLiteStore(self.config.resolved_db_path)
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
        self.agent = MindFSAgent(config=self.config, tools=self.tools, llm_engine=self.llm_engine)

    def set_workspace(self, new_root: str) -> Dict[str, Any]:
        """Dynamically switches active workspace root."""
        p = Path(new_root).expanduser().resolve()
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        self._init_system(workspace_root=str(p))
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
            body = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len > 0:
                raw = self.rfile.read(content_len)
                return json.loads(raw.decode("utf-8"))
            return {}

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = urllib.parse.parse_qs(parsed.query)

            if path == "/" or path == "/index.html":
                html_path = Path(__file__).parent / "static" / "index.html"
                if html_path.exists():
                    body = html_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                else:
                    self._send_json(404, {"error": "UI index.html not found"})
                    return

            elif path == "/api/status":
                status = app.tools.get_index_status()
                res = status.data if status.success else {"error": status.error}
                res["workspace_root"] = str(app.config.resolved_workspace_root)
                res["peak_rss_mb"] = app.resources.get_peak_rss_mb()
                res["current_rss_mb"] = app.resources.get_current_rss_mb()
                self._send_json(200, res)

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
                            files_list.append({"name": entry.name, "size": sz})
                except (PermissionError, Exception):
                    pass

                parent = str(target_dir.parent) if target_dir != target_dir.parent else None

                home = Path.home()
                shortcuts = [
                    {"label": "Current Workspace", "path": str(app.config.resolved_workspace_root)},
                    {"label": "Home (~)", "path": str(home)},
                ]
                if (home / "Desktop").exists():
                    shortcuts.append({"label": "Desktop", "path": str(home / "Desktop")})
                if (home / "Documents").exists():
                    shortcuts.append({"label": "Documents", "path": str(home / "Documents")})
                if (home / "Downloads").exists():
                    shortcuts.append({"label": "Downloads", "path": str(home / "Downloads")})
                if (home / "code").exists():
                    shortcuts.append({"label": "Code", "path": str(home / "code")})
                if (Path("/") / "Users").exists():
                    shortcuts.append({"label": "/Users", "path": "/Users"})

                self._send_json(200, {
                    "current_path": str(target_dir),
                    "parent_path": parent,
                    "directories": subdirs,
                    "files": files_list,
                    "shortcuts": shortcuts,
                })

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

            elif path == "/api/index":
                payload = self._read_json()
                target_p = payload.get("path", "").strip()
                recursive = payload.get("recursive", True)

                # If target_p is an absolute path
                if target_p and Path(target_p).is_absolute():
                    ws = Path(app.config.resolved_workspace_root).resolve()
                    tp = Path(target_p).resolve()
                    if tp == ws or ws in tp.parents:
                        try:
                            target_p = str(tp.relative_to(ws))
                        except Exception:
                            target_p = ""
                    else:
                        # Switch workspace to new target directory
                        app.set_workspace(str(tp))
                        target_p = ""

                with app.indexing_lock:
                    app.indexing_progress.update({
                        "is_indexing": True,
                        "target_path": target_p,
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

            elif path == "/api/ask":
                payload = self._read_json()
                query = payload.get("query", "").strip()
                if not query:
                    self._send_json(400, {"error": "Missing query parameter"})
                    return

                resp = app.agent.ask(query)
                actions_data = [
                    {
                        "step": a.step,
                        "thought": a.thought,
                        "tool": a.tool_call.name if a.tool_call else None,
                        "success": a.tool_result.success if a.tool_result else False,
                    }
                    for a in resp.actions_taken
                ]
                self._send_json(200, {
                    "query": query,
                    "answer": resp.answer,
                    "total_steps": resp.total_steps,
                    "actions": actions_data,
                })

            else:
                self._send_json(404, {"error": f"Endpoint '{path}' not found"})

    return MindFSHTTPHandler


def run_ui_server(
    host: str = "127.0.0.1",
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
