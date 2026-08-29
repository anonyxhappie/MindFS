"""Deterministic filesystem and indexing tools for the MindFS agent."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
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
            
            # If not yet indexed, derive live identification
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

        return ToolResult(tool_name=name, success=False, data=None, error=f"Unknown tool '{name}'")

