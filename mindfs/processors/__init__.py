"""Processors package."""

from mindfs.config.settings import MindFSConfig
from mindfs.processors.base import FileProcessor
from mindfs.processors.registry import ProcessorRegistry
from mindfs.processors.text import TextProcessor
from mindfs.processors.pdf import PDFProcessor
from mindfs.processors.structured import StructuredDataProcessor
from mindfs.processors.image import ImageProcessor
from mindfs.processors.audio import AudioProcessor
from mindfs.processors.video import VideoProcessor
from mindfs.processors.archive import ArchiveProcessor
from mindfs.processors.binary import BinaryProcessor
from mindfs.processors.fallback import FallbackProcessor


def create_default_registry(config: MindFSConfig) -> ProcessorRegistry:
    """Instantiates and registers all standard MindFS file processors."""
    registry = ProcessorRegistry(config)
    
    # Register specific processors in priority order
    registry.register(PDFProcessor(config))
    registry.register(StructuredDataProcessor(config))
    registry.register(TextProcessor(config))
    registry.register(ImageProcessor(config))
    registry.register(AudioProcessor(config))
    registry.register(VideoProcessor(config))
    registry.register(ArchiveProcessor(config))
    registry.register(BinaryProcessor(config))
    registry.register(FallbackProcessor(config))
    
    return registry


__all__ = [
    "FileProcessor",
    "ProcessorRegistry",
    "TextProcessor",
    "PDFProcessor",
    "StructuredDataProcessor",
    "ImageProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "ArchiveProcessor",
    "BinaryProcessor",
    "FallbackProcessor",
    "create_default_registry",
]

