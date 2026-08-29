"""Format-aware text chunker with source offset and location tracking."""

import re
from typing import Any, Dict, List, Optional
import uuid

from mindfs.artifacts.models import ChunkItem, SemanticArtifact
from mindfs.config.settings import MindFSConfig


class Chunker:
    """Chunks text from SemanticArtifacts using structure-aware boundaries (paragraphs, lines, sentences)."""

    def __init__(self, config: MindFSConfig):
        self.config = config
        self.target_size = config.index.chunk_target_size
        self.overlap_size = int(self.target_size * config.index.chunk_overlap_pct)
        self.max_size = config.index.chunk_max_size

    def chunk_artifact(self, artifact: SemanticArtifact) -> List[ChunkItem]:
        """Splits an artifact's searchable text into bounded chunk items."""
        full_text = artifact.to_searchable_representation().strip()
        if not full_text:
            return []

        # If full text is already within chunk size, return single chunk
        if len(full_text) <= self.max_size:
            return [
                ChunkItem(
                    chunk_id=uuid.uuid4().hex[:16],
                    artifact_id=artifact.artifact_id,
                    file_id=artifact.file_id,
                    source_path=artifact.source_path,
                    source_offset=artifact.source_offset,
                    chunk_index=0,
                    text=full_text,
                    char_start=0,
                    char_end=len(full_text),
                    metadata={"artifact_type": artifact.artifact_type},
                )
            ]

        # Break by double newline (paragraphs), single newline, or sentences
        paragraphs = re.split(r"(\n\n+)", full_text)
        chunks: List[ChunkItem] = []
        
        current_chunk_text = ""
        current_start = 0
        chunk_idx = 0

        for segment in paragraphs:
            if not segment:
                continue

            # If adding this segment exceeds max_size and we already have content, emit chunk
            if len(current_chunk_text) + len(segment) > self.max_size and len(current_chunk_text) >= self.target_size:
                c_text = current_chunk_text.strip()
                chunks.append(
                    ChunkItem(
                        chunk_id=uuid.uuid4().hex[:16],
                        artifact_id=artifact.artifact_id,
                        file_id=artifact.file_id,
                        source_path=artifact.source_path,
                        source_offset=artifact.source_offset,
                        chunk_index=chunk_idx,
                        text=c_text,
                        char_start=current_start,
                        char_end=current_start + len(c_text),
                        metadata={"artifact_type": artifact.artifact_type},
                    )
                )
                chunk_idx += 1
                
                # Keep overlap tail
                overlap_text = current_chunk_text[-self.overlap_size:] if self.overlap_size > 0 else ""
                current_start = current_start + len(current_chunk_text) - len(overlap_text)
                current_chunk_text = overlap_text + segment
            else:
                current_chunk_text += segment

        if current_chunk_text.strip():
            c_text = current_chunk_text.strip()
            chunks.append(
                ChunkItem(
                    chunk_id=uuid.uuid4().hex[:16],
                    artifact_id=artifact.artifact_id,
                    file_id=artifact.file_id,
                    source_path=artifact.source_path,
                    source_offset=artifact.source_offset,
                    chunk_index=chunk_idx,
                    text=c_text,
                    char_start=current_start,
                    char_end=current_start + len(c_text),
                    metadata={"artifact_type": artifact.artifact_type},
                )
            )

        return chunks

