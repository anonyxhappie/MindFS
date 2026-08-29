"""Configuration settings for MindFS."""

from pathlib import Path
from typing import Optional
import os
import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model_path: Optional[str] = None
    context_tokens: int = 2048
    max_output_tokens: int = 256
    temperature: float = 0.1
    backend: str = "auto"  # "llama_cpp", "mock", "auto"


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    embedding_dim: int = 384


class IndexConfig(BaseModel):
    db_path: str = ".mindfs/metadata.db"
    faiss_path: str = ".mindfs/index.faiss"
    max_file_size_mb: float = 5.0
    chunk_target_size: int = 500
    chunk_overlap_pct: float = 0.10
    chunk_max_size: int = 1000
    retrieval_candidates: int = 12
    final_evidence_count: int = 5
    recursive_default: bool = True


class AgentConfig(BaseModel):
    max_steps: int = 10


class ResourceConfig(BaseModel):
    max_rss_mb: float = 1740.0  # 1.7 GB hard ceiling


class MediaConfig(BaseModel):
    video_frame_interval_seconds: float = 30.0
    audio_segment_seconds: float = 30.0
    enable_ocr: bool = True
    enable_vision: bool = False
    enable_asr: bool = False
    allow_gps: bool = False


class ArchiveConfig(BaseModel):
    max_members: int = 1000
    max_expanded_size_mb: float = 100.0
    max_nesting_depth: int = 3
    max_member_size_mb: float = 25.0


class MindFSConfig(BaseModel):
    workspace_root: str = "."
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    archive: ArchiveConfig = Field(default_factory=ArchiveConfig)

    @property
    def resolved_workspace_root(self) -> Path:
        return Path(self.workspace_root).expanduser().resolve()

    @property
    def resolved_db_path(self) -> Path:
        p = Path(self.index.db_path)
        if p.is_absolute():
            return p
        return (self.resolved_workspace_root / p).resolve()

    @property
    def resolved_faiss_path(self) -> Path:
        p = Path(self.index.faiss_path)
        if p.is_absolute():
            return p
        return (self.resolved_workspace_root / p).resolve()


def load_config(config_path: Optional[str | Path] = None, workspace_root: Optional[str | Path] = None) -> MindFSConfig:
    """Load configuration from a YAML file, environment variables, or defaults."""
    cfg_data = {}
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
    elif os.path.exists("config.yaml"):
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

    config = MindFSConfig(**cfg_data)
    
    if workspace_root:
        config.workspace_root = str(workspace_root)
    elif "WORKSPACE_ROOT" in os.environ:
        config.workspace_root = os.environ["WORKSPACE_ROOT"]

    return config

