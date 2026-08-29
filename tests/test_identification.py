"""Tests for multi-tier file identification and hashing."""

from pathlib import Path
import pytest

from mindfs.filesystem.sandbox import FilesystemSandbox
from mindfs.identification.detector import FileDetector
from mindfs.identification.models import FileCategory


def test_file_categories(temp_workspace):
    ws_root, fixtures = temp_workspace
    sandbox = FilesystemSandbox(ws_root)
    detector = FileDetector(sandbox)

    # Documents
    info = detector.identify("plain.txt")
    assert info.category == FileCategory.DOCUMENT
    assert info.mime_type == "text/plain"

    # PDF
    info_pdf = detector.identify("sample.pdf")
    assert info_pdf.category == FileCategory.DOCUMENT
    assert info_pdf.mime_type == "application/pdf"

    # Structured
    info_json = detector.identify("data.json")
    assert info_json.category == FileCategory.STRUCTURED
    assert info_json.mime_type == "application/json"

    info_csv = detector.identify("data.csv")
    assert info_csv.category == FileCategory.STRUCTURED
    assert info_csv.mime_type == "text/csv"

    # Images
    info_jpg = detector.identify("image.jpg")
    assert info_jpg.category == FileCategory.IMAGE

    info_png = detector.identify("image_with_text.png")
    assert info_png.category == FileCategory.IMAGE

    # Audio
    info_wav = detector.identify("audio.wav")
    assert info_wav.category == FileCategory.AUDIO

    # Video
    info_mp4 = detector.identify("video.mp4")
    assert info_mp4.category == FileCategory.VIDEO

    # Archive
    info_zip = detector.identify("archive.zip")
    assert info_zip.category == FileCategory.ARCHIVE

    # Binary
    info_elf = detector.identify("sample_elf")
    assert info_elf.category == FileCategory.BINARY


def test_streaming_sha256(temp_workspace):
    ws_root, fixtures = temp_workspace
    sandbox = FilesystemSandbox(ws_root)
    detector = FileDetector(sandbox)

    info = detector.identify("plain.txt", compute_hash=True)
    assert info.sha256 is not None
    assert len(info.sha256) == 64

