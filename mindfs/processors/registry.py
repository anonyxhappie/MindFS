"""Processor Registry for selecting and managing file processors."""

from typing import Any, Dict, List, Optional, Type

from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory, FileInfo
from mindfs.processors.base import FileProcessor


class ProcessorRegistry:
    """Manages available processors and routes files to the most specific processor."""

    def __init__(self, config: MindFSConfig):
        self.config = config
        self._processors: List[FileProcessor] = []
        self._fallback_processor: Optional[FileProcessor] = None

    def register(self, processor: FileProcessor) -> None:
        """Registers an initialized processor instance."""
        if processor.name == "fallback":
            self._fallback_processor = processor
        else:
            self._processors.append(processor)

    def get_processor(self, file_info: FileInfo) -> FileProcessor:
        """Finds the most specific processor capable of handling the file."""
        for proc in self._processors:
            if proc.can_handle(file_info):
                return proc
        if self._fallback_processor:
            return self._fallback_processor
        raise RuntimeError("No processor or fallback registered to handle file.")

    def list_processors(self) -> List[Dict[str, Any]]:
        """Lists metadata for all registered processors."""
        result = []
        for p in self._processors:
            result.append({
                "name": p.name,
                "version": p.version,
                "categories": [c.value for c in p.supported_categories],
                "extensions": p.supported_extensions[:10],
            })
        if self._fallback_processor:
            result.append({
                "name": self._fallback_processor.name,
                "version": self._fallback_processor.version,
                "categories": ["ALL"],
                "extensions": ["*"],
            })
        return result
