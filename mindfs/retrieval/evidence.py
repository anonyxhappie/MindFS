"""Evidence data models and assembly helpers for MindFS retrieval."""

import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Represents a single verified chunk/artifact retrieved from the index."""
    chunk_id: str
    file_id: str
    source_path: str
    relative_path: str
    artifact_type: str
    source_location: Optional[Union[str, int, Dict[str, Any]]] = None
    similarity_score: float = 0.0
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def formatted_citation(self) -> str:
        """Generates a human-readable citation header."""
        prov = self.format_source_provenance()
        return f"[{prov} | Score: {round(self.similarity_score, 3)}]"

    def format_source_provenance(self) -> str:
        """
        Formats concise source provenance for user presentation.
        e.g.:
        - project/architecture.md — section "Database"
        - meeting.mp4 — 00:14:22
        - sample.pdf — page 2
        - mindfs/agent/loop.py — relevant code
        """
        loc = ""
        # 1. Check source_location dict
        if isinstance(self.source_location, dict):
            if "page" in self.source_location:
                loc = f"page {self.source_location['page']}"
            elif "timestamp" in self.source_location:
                ts = str(self.source_location["timestamp"])
                if " - " in ts:
                    ts = ts.split(" - ")[0]
                loc = ts
            elif "timestamps" in self.source_location and self.source_location["timestamps"]:
                loc = str(self.source_location["timestamps"][0])
            elif "section" in self.source_location:
                loc = f'section "{self.source_location["section"]}"'
            elif "lines" in self.source_location:
                loc = f"lines {self.source_location['lines']}"
            elif "duration_sec" in self.source_location:
                sec = float(self.source_location["duration_sec"])
                mins = int(sec // 60)
                secs = int(sec % 60)
                loc = f"{mins:02d}:{secs:02d}"
        elif isinstance(self.source_location, (str, int)):
            loc_str = str(self.source_location).strip()
            if loc_str.isdigit():
                loc = f"page {loc_str}"
            elif ":" in loc_str and any(c.isdigit() for c in loc_str):
                loc = loc_str
            elif loc_str:
                loc = loc_str

        # 2. If no location found from source_location, inspect text & filename
        if not loc:
            fn_lower = self.relative_path.lower()
            if fn_lower.endswith(".pdf") or self.artifact_type == "pdf_page":
                page_match = re.search(r"page\s*[:\s]?\s*(\d+)", self.text, re.IGNORECASE)
                if page_match:
                    loc = f"page {page_match.group(1)}"
                else:
                    loc = "document page"
            elif any(fn_lower.endswith(ext) for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm")):
                time_match = re.search(r"\b(\d{2}:\d{2}(?::\d{2})?)\b", self.text)
                if time_match:
                    loc = time_match.group(1)
                else:
                    loc = "video keyframe"
            elif any(fn_lower.endswith(ext) for ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a")):
                time_match = re.search(r"\b(\d{2}:\d{2}(?::\d{2})?)\b", self.text)
                if time_match:
                    loc = time_match.group(1)
                else:
                    loc = "audio track"
            elif any(fn_lower.endswith(ext) for ext in (".json", ".csv", ".yaml", ".yml", ".xml")):
                loc = "structured schema & data"
            elif any(fn_lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                loc = "visual metadata"
            elif any(fn_lower.endswith(ext) for ext in (".zip", ".tar", ".gz", ".tar.gz")):
                loc = "archive manifest"
            elif any(fn_lower.endswith(ext) for ext in (".py", ".ts", ".js", ".go", ".rs", ".cpp", ".c", ".java")):
                func_match = re.search(r"(?:def|class|function|fn|pub fn)\s+([a-zA-Z0-9_]+)", self.text)
                if func_match:
                    loc = f"definition `{func_match.group(1)}`"
                else:
                    loc = "relevant code"
            elif any(fn_lower.endswith(ext) for ext in (".md", ".txt", ".rst")):
                head_match = re.search(r"^#+\s+(.+)$", self.text, re.MULTILINE)
                if head_match:
                    loc = f'section "{head_match.group(1).strip()}"'
                else:
                    loc = "relevant section"
            else:
                loc = "relevant section"

        return f"{self.relative_path} — {loc}"


class RetrievalResult(BaseModel):
    """Enriched result of a semantic search query with structured evidence."""
    query: str
    total_candidates: int
    evidence: List[EvidenceItem]
    has_sufficient_evidence: bool = True
    diagnostic_info: Dict[str, Any] = Field(default_factory=dict)
