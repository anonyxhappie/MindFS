"""Image intelligence processor for extracting EXIF, dimensions, OCR, and visual metadata."""

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional
from PIL import Image, ExifTags

from mindfs.artifacts.models import SemanticArtifact
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory, FileInfo
from mindfs.processors.base import FileProcessor


class ImageProcessor(FileProcessor):
    """Processes images to extract dimensions, color profiles, EXIF metadata, and optional OCR text."""

    name: str = "image"
    version: str = "1.0.0"
    supported_categories: List[FileCategory] = [FileCategory.IMAGE]
    supported_mimes: List[str] = [
        "image/jpeg", "image/png", "image/gif", "image/webp",
        "image/bmp", "image/tiff", "image/svg+xml"
    ]
    supported_extensions: List[str] = [
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg"
    ]

    def _read_exif(self, img: Image.Image) -> Dict[str, Any]:
        exif_data = {}
        try:
            raw_exif = img.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Skip binary blobs or GPS if not allowed
                    if tag_name == "GPSInfo" and not self.config.media.allow_gps:
                        continue
                    if isinstance(value, (str, int, float)):
                        exif_data[tag_name] = value
                    elif isinstance(value, bytes) and len(value) < 64:
                        exif_data[tag_name] = value.decode("utf-8", errors="ignore")
        except Exception:
            pass
        return exif_data

    def _run_ocr(self, image_path: Path) -> Optional[str]:
        """Runs tesseract OCR if available on system without pulling heavy libraries."""
        if not self.config.media.enable_ocr:
            return None
        tesseract_bin = shutil.which("tesseract")
        if not tesseract_bin:
            return None

        try:
            res = subprocess.run(
                [tesseract_bin, str(image_path), "stdout", "--oem", "1", "-l", "eng"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        return None

    def inspect(self, file_info: FileInfo) -> Dict[str, Any]:
        path = Path(file_info.canonical_path)
        try:
            with Image.open(path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "has_exif": bool(img.getexif()),
                }
        except Exception as exc:
            return {"error": f"Cannot inspect image: {str(exc)}"}

    def extract(self, file_info: FileInfo) -> List[SemanticArtifact]:
        path = Path(file_info.canonical_path)
        artifacts: List[SemanticArtifact] = []

        width, height, img_format, mode = 0, 0, "UNKNOWN", "UNKNOWN"
        exif_info: Dict[str, Any] = {}

        try:
            with Image.open(path) as img:
                width = img.width
                height = img.height
                img_format = img.format or file_info.extension.lstrip(".").upper()
                mode = img.mode
                exif_info = self._read_exif(img)
        except Exception as exc:
            # Fallback if image opening fails
            img_format = file_info.extension.lstrip(".").upper()

        lines = [
            f"Image File: {file_info.filename}",
            f"Format: {img_format}",
            f"Dimensions: {width} x {height} pixels (Aspect: {round(width/height, 2) if height else 'N/A'})",
            f"Color Mode: {mode}",
        ]

        if exif_info:
            lines.append("EXIF Metadata:")
            for k, v in list(exif_info.items())[:10]:
                lines.append(f"  - {k}: {v}")

        ocr_text = self._run_ocr(path)
        if ocr_text:
            lines.append("")
            lines.append("Extracted Text (OCR):")
            lines.append(ocr_text[:1500])

        text_content = "\n".join(lines)
        summary = f"Image '{file_info.filename}' ({width}x{height} {img_format})"
        if ocr_text:
            summary += f" containing text: {ocr_text[:80]}..."

        metadata = {
            "width": width,
            "height": height,
            "format": img_format,
            "mode": mode,
            "has_ocr": bool(ocr_text),
            "exif": exif_info,
        }

        art = SemanticArtifact(
            file_id=file_info.file_id,
            artifact_type="image_metadata",
            source_path=file_info.canonical_path,
            source_offset=None,
            text=text_content,
            summary=summary,
            metadata=metadata,
            entities=list(exif_info.keys())[:10],
            processor=self.name,
            processor_version=self.version,
        )
        artifacts.append(art)
        return artifacts

