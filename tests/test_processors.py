"""Tests for all 9 modular file processors."""

from pathlib import Path
import pytest

from mindfs.config.settings import MindFSConfig
from mindfs.filesystem.sandbox import FilesystemSandbox
from mindfs.identification.detector import FileDetector
from mindfs.processors import (
    ArchiveProcessor,
    AudioProcessor,
    BinaryProcessor,
    FallbackProcessor,
    ImageProcessor,
    PDFProcessor,
    StructuredDataProcessor,
    TextProcessor,
    VideoProcessor,
)


def test_text_processor(temp_workspace):
    ws_root, fixtures = temp_workspace
    config = MindFSConfig(workspace_root=str(ws_root))
    sandbox = FilesystemSandbox(ws_root)
    detector = FileDetector(sandbox)
    proc = TextProcessor(config)

    info = detector.identify("markdown.md")
    assert proc.can_handle(info)

    inspection = proc.inspect(info)
    assert "line_count_sample" in inspection

    artifacts = proc.extract(info)
    assert len(artifacts) == 1
    assert "Project Apollo" in artifacts[0].text
    assert artifacts[0].source_path == info.canonical_path


def test_pdf_processor(temp_workspace):
    ws_root, fixtures = temp_workspace
    config = MindFSConfig(workspace_root=str(ws_root))
    sandbox = FilesystemSandbox(ws_root)
    detector = FileDetector(sandbox)
    proc = PDFProcessor(config)

    info = detector.identify("sample.pdf")
    assert proc.can_handle(info)

    inspection = proc.inspect(info)
    assert "pages" in inspection

    artifacts = proc.extract(info)
    assert len(artifacts) >= 1
    assert artifacts[0].source_offset is not None
    assert "page" in artifacts[0].source_offset


def test_structured_processor(temp_workspace):
    ws_root, fixtures = temp_workspace
    config = MindFSConfig(workspace_root=str(ws_root))
    sandbox = FilesystemSandbox(ws_root)
    detector = FileDetector(sandbox)
    proc = StructuredDataProcessor(config)

    # CSV
    info_csv = detector.identify("data.csv")
    assert proc.can_handle(info_csv)
    arts_csv = proc.extract(info_csv)
    assert len(arts_csv) == 1
    assert "Columns" in arts_csv[0].text
    assert "customer_id" in arts_csv[0].text

    # JSON
    info_json = detector.identify("data.json")
    arts_json = proc.extract(info_json)
    assert len(arts_json) == 1
    assert "service" in arts_json[0].text


def test_image_processor(temp_workspace):
    ws_root, fixtures = temp_workspace
    config = MindFSConfig(workspace_root=str(ws_root))
    sandbox = FilesystemSandbox(ws_root)
    detector = FileDetector(sandbox)
    proc = ImageProcessor(config)

    info = detector.identify("image.jpg")
    assert proc.can_handle(info)

    inspection = proc.inspect(info)
    assert inspection["width"] == 200
    assert inspection["height"] == 200

    artifacts = proc.extract(info)
    assert len(artifacts) == 1
    assert "Dimensions: 200 x 200" in artifacts[0].text


def test_audio_processor(temp_workspace):
    ws_root, fixtures = temp_workspace
    config = MindFSConfig(workspace_root=str(ws_root))
    sandbox = FilesystemSandbox(ws_root)
    detector = FileDetector(sandbox)
    proc = AudioProcessor(config)

    info = detector.identify("audio.wav")
    assert proc.can_handle(info)

    inspection = proc.inspect(info)
    assert "duration_sec" in inspection
    assert inspection["sample_rate_hz"] == 16000

    artifacts = proc.extract(info)
    assert len(artifacts) == 1
    assert "Audio Track: audio.wav" in artifacts[0].text


def test_video_processor(temp_workspace):
    ws_root, fixtures = temp_workspace
    config = MindFSConfig(workspace_root=str(ws_root))
    sandbox = FilesystemSandbox(ws_root)
    detector = FileDetector(sandbox)
    proc = VideoProcessor(config)

    info = detector.identify("video.mp4")
    assert proc.can_handle(info)

    artifacts = proc.extract(info)
    assert len(artifacts) == 1
    assert "Video File: video.mp4" in artifacts[0].text


def test_archive_processor(temp_workspace):
    ws_root, fixtures = temp_workspace
    config = MindFSConfig(workspace_root=str(ws_root))
    sandbox = FilesystemSandbox(ws_root)
    detector = FileDetector(sandbox)
    proc = ArchiveProcessor(config)

    info = detector.identify("archive.zip")
    assert proc.can_handle(info)

    inspection = proc.inspect(info)
    assert inspection["member_count"] == 2

    artifacts = proc.extract(info)
    assert len(artifacts) == 1
    assert "Archive Manifest: archive.zip" in artifacts[0].text
    assert "internal_report.txt" in artifacts[0].text


def test_binary_processor(temp_workspace):
    ws_root, fixtures = temp_workspace
    config = MindFSConfig(workspace_root=str(ws_root))
    sandbox = FilesystemSandbox(ws_root)
    detector = FileDetector(sandbox)
    proc = BinaryProcessor(config)

    info = detector.identify("sample_elf")
    assert proc.can_handle(info)

    inspection = proc.inspect(info)
    assert inspection.get("binary_format") == "ELF"
    assert inspection.get("architecture") == "x86_64 / AMD64"

    artifacts = proc.extract(info)
    assert len(artifacts) == 1
    assert "Architecture: x86_64 / AMD64" in artifacts[0].text
    assert "/usr/lib/libc.so.6" in artifacts[0].text


def test_fallback_processor(temp_workspace):
    ws_root, fixtures = temp_workspace
    config = MindFSConfig(workspace_root=str(ws_root))
    sandbox = FilesystemSandbox(ws_root)
    detector = FileDetector(sandbox)
    proc = FallbackProcessor(config)

    info = detector.identify("unknown.dat")
    assert proc.can_handle(info)

    inspection = proc.inspect(info)
    assert "entropy" in inspection

    artifacts = proc.extract(info)
    assert len(artifacts) == 1
    assert "Unknown/Unsupported File: unknown.dat" in artifacts[0].text

