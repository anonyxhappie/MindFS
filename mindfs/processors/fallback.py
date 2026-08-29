"""Fallback processor for unsupported and unknown files."""

import math
from pathlib import Path
from typing import Any, Dict, List

from mindfs.artifacts.models import SemanticArtifact
from mindfs.config.settings import MindFSConfig
from mindfs.identification.detector import FileDetector
from mindfs.identification.models import FileCategory, FileInfo
from mindfs.processors.base import FileProcessor


class FallbackProcessor(FileProcessor):
    """Generates technical metadata and heuristics for unknown or unsupported files."""

    name: str = "fallback"
    version: str = "1.0.0"
    supported_categories: List[FileCategory] = [FileCategory.UNKNOWN]
    supported_mimes: List[str] = ["*"]
    supported_extensions: List[str] = ["*"]

    def can_handle(self, file_info: FileInfo) -> bool:
        return True

    def _analyze_bytes(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, "rb") as f:
                sample = f.read(4096)
            if not sample:
                return {
                    "is_empty": True,
                    "printable_ratio": 0.0,
                    "entropy": 0.0,
                    "magic_hex": "",
                    "suspicion": "empty_file",
                }

            magic_hex = sample[:16].hex(" ")
            printable_bytes = sum(1 for b in sample if 32 <= b <= 126 or b in (9, 10, 13))
            ratio = printable_bytes / len(sample)

            # Shannon Entropy calculation
            byte_counts = [0] * 256
            for b in sample:
                byte_counts[b] += 1
            entropy = 0.0
            for count in byte_counts:
                if count > 0:
                    p = count / len(sample)
                    entropy -= p * math.log2(p)

            suspicion = "plain_text" if ratio > 0.9 else ("encrypted_or_compressed" if entropy > 7.2 else "binary_data")

            return {
                "is_empty": False,
                "printable_ratio": round(ratio, 3),
                "entropy": round(entropy, 3),
                "magic_hex": magic_hex,
                "suspicion": suspicion,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def inspect(self, file_info: FileInfo) -> Dict[str, Any]:
        path = Path(file_info.canonical_path)
        stats = self._analyze_bytes(path)
        stats["size_bytes"] = file_info.size_bytes
        stats["mime_type"] = file_info.mime_type
        return stats

    def extract(self, file_info: FileInfo) -> List[SemanticArtifact]:
        path = Path(file_info.canonical_path)
        analysis = self._analyze_bytes(path)

        magic_hex = analysis.get("magic_hex", "")
        entropy = analysis.get("entropy", 0.0)
        printable_ratio = analysis.get("printable_ratio", 0.0)
        suspicion = analysis.get("suspicion", "unknown")

        lines = [
            f"Unknown/Unsupported File: {file_info.filename}",
            f"Extension: {file_info.extension or 'none'}",
            f"Guessed MIME: {file_info.mime_type}",
            f"Size: {file_info.size_bytes} bytes",
            f"Magic Bytes (Hex): {magic_hex}",
            f"Entropy: {entropy} / 8.0 (Heuristic: {suspicion})",
            f"Printable Character Ratio: {printable_ratio}",
        ]

        text_content = "\n".join(lines)
        summary = f"Unknown file '{file_info.filename}' ({file_info.size_bytes} bytes, entropy: {entropy})"

        metadata = {
            "mime_type": file_info.mime_type,
            "analysis": analysis,
            "status": "unsupported",
        }

        art = SemanticArtifact(
            file_id=file_info.file_id,
            artifact_type="fallback_metadata",
            source_path=file_info.canonical_path,
            source_offset=None,
            text=text_content,
            summary=summary,
            metadata=metadata,
            entities=[file_info.extension] if file_info.extension else [],
            processor=self.name,
            processor_version=self.version,
        )

        return [art]

