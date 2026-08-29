"""Text and source code document processor."""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mindfs.artifacts.models import SemanticArtifact
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory, FileInfo
from mindfs.processors.base import FileProcessor


class TextProcessor(FileProcessor):
    """Processes plain text, markdown, code, logs, and other plaintext documents."""

    name: str = "text"
    version: str = "1.0.0"
    supported_categories: List[FileCategory] = [FileCategory.DOCUMENT]
    supported_mimes: List[str] = [
        "text/plain",
        "text/markdown",
        "text/x-python",
        "text/javascript",
        "text/typescript",
        "text/x-java-source",
        "text/x-c",
        "text/x-c++",
        "text/x-c-header",
        "text/x-c++-header",
        "text/x-rust",
        "text/x-go",
        "text/x-shellscript",
        "text/html",
        "text/rtf",
    ]
    supported_extensions: List[str] = [
        ".txt", ".md", ".log", ".py", ".js", ".ts", ".java",
        ".cpp", ".c", ".h", ".hpp", ".rs", ".go", ".sh",
        ".html", ".htm", ".rtf"
    ]

    def _read_bounded_text(self, file_path: Path, max_bytes: int) -> Tuple[str, bool]:
        """Reads file up to max_bytes safely with UTF-8 decoding fallback to latin-1."""
        truncated = False
        with open(file_path, "rb") as f:
            raw = f.read(max_bytes + 1)
            if len(raw) > max_bytes:
                truncated = True
                raw = raw[:max_bytes]
        
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
        
        return text, truncated

    def inspect(self, file_info: FileInfo) -> Dict[str, Any]:
        path = Path(file_info.canonical_path)
        max_preview_bytes = 2048
        text, truncated = self._read_bounded_text(path, max_preview_bytes)
        lines = text.splitlines()
        
        return {
            "processor": self.name,
            "line_count_sample": len(lines),
            "preview": text[:500],
            "is_truncated_preview": truncated or file_info.size_bytes > max_preview_bytes,
            "format": file_info.extension or "text",
        }

    def extract(self, file_info: FileInfo) -> List[SemanticArtifact]:
        path = Path(file_info.canonical_path)
        max_bytes = int(self.config.index.max_file_size_mb * 1024 * 1024)
        
        text, truncated = self._read_bounded_text(path, max_bytes)
        
        lines = text.splitlines()
        first_few = " ".join([l.strip() for l in lines[:5] if l.strip()])
        summary = first_few[:200] if first_few else f"{file_info.filename} text document"

        metadata = {
            "total_lines": len(lines),
            "size_bytes": file_info.size_bytes,
            "truncated": truncated,
            "extension": file_info.extension,
        }

        artifact = SemanticArtifact(
            file_id=file_info.file_id,
            artifact_type="document_text",
            source_path=file_info.canonical_path,
            source_offset=None,
            text=text,
            summary=summary,
            metadata=metadata,
            entities=[],
            processor=self.name,
            processor_version=self.version,
        )

        return [artifact]
