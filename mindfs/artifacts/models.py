"""Data models for Semantic Artifacts in MindFS."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import uuid
from pydantic import BaseModel, Field


class SemanticArtifact(BaseModel):
    """
    Standardized semantic representation derived from a file or portion of a file.
    """
    artifact_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    file_id: str
    artifact_type: str = "text"
    source_path: str
    source_offset: Optional[Union[str, int, Dict[str, Any]]] = None
    text: str = ""
    summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    entities: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processor: str = "base"
    processor_version: str = "1.0.0"

    def to_searchable_representation(self) -> str:
        """Returns the complete text representation suitable for chunking and embedding."""
        parts = []
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.text:
            parts.append(self.text)
        if self.entities:
            parts.append(f"Entities: {', '.join(self.entities)}")
        return "\n\n".join(parts) if parts else self.text


class ChunkItem(BaseModel):
    """A bounded text chunk derived from a SemanticArtifact for vector embedding."""
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    artifact_id: str
    file_id: str
    source_path: str
    source_offset: Optional[Union[str, int, Dict[str, Any]]] = None
    chunk_index: int = 0
    text: str
    char_start: int = 0
    char_end: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

