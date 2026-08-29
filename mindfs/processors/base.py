"""Base FileProcessor interface definition."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from mindfs.artifacts.models import SemanticArtifact
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory, FileInfo


class FileProcessor(ABC):
    """Abstract base class for all MindFS file processors."""

    name: str = "base"
    version: str = "1.0.0"
    supported_categories: List[FileCategory] = []
    supported_mimes: List[str] = []
    supported_extensions: List[str] = []

    def __init__(self, config: MindFSConfig):
        self.config = config

    def can_handle(self, file_info: FileInfo) -> bool:
        """Determines whether this processor is suitable for the given file."""
        if file_info.category in self.supported_categories:
            return True
        if file_info.mime_type in self.supported_mimes:
            return True
        if file_info.extension.lower() in self.supported_extensions:
            return True
        return False

    @abstractmethod
    def inspect(self, file_info: FileInfo) -> Dict[str, Any]:
        """
        Performs fast, lightweight inspection of the file without heavy extraction.
        Returns technical facts, structure, dimensions, member counts, or summary info.
        """
        pass

    @abstractmethod
    def extract(self, file_info: FileInfo) -> List[SemanticArtifact]:
        """
        Performs semanticization and extracts one or more SemanticArtifact objects.
        Must be bounded and stream large files where possible.
        """
        pass

    def estimate_resources(self, file_info: FileInfo) -> Dict[str, Any]:
        """
        Estimates memory (MB) and time (sec) needed to process the given file.
        """
        size_mb = file_info.size_bytes / (1024 * 1024)
        return {
            "estimated_rss_mb": round(min(size_mb * 1.5, 50.0), 2),
            "estimated_duration_sec": round(max(0.01, size_mb * 0.1), 2),
        }

