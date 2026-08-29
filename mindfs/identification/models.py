"""Data models for file identification in MindFS."""

from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class FileCategory(str, Enum):
    DOCUMENT = "DOCUMENT"
    STRUCTURED = "STRUCTURED"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    ARCHIVE = "ARCHIVE"
    BINARY = "BINARY"
    UNKNOWN = "UNKNOWN"
    DIRECTORY = "DIRECTORY"


class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNSUPPORTED = "UNSUPPORTED"


class FileInfo(BaseModel):
    file_id: str
    canonical_path: str
    relative_path: str
    filename: str
    extension: str
    mime_type: str = "application/octet-stream"
    category: FileCategory = FileCategory.UNKNOWN
    size_bytes: int = 0
    mtime: float = 0.0
    mtime_ns: int = 0
    ctime: Optional[float] = None
    sha256: Optional[str] = None
    processor: Optional[str] = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    status_reason: Optional[str] = None

