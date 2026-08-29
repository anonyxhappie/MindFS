"""Pytest fixtures for MindFS test suites."""

import os
from pathlib import Path
import shutil
import tempfile
import pytest

from mindfs.config.settings import MindFSConfig
from mindfs.filesystem.sandbox import FilesystemSandbox
from mindfs.indexing.embeddings import EmbeddingPipeline
from mindfs.indexing.indexer import Indexer
from mindfs.indexing.vector_store import VectorStore
from mindfs.processors import create_default_registry
from mindfs.resources.manager import ResourceManager
from mindfs.retrieval.search import SearchEngine
from mindfs.storage.sqlite_store import SQLiteStore
from tests.fixtures_generator import generate_test_corpus


@pytest.fixture
def temp_workspace():
    """Creates a temporary workspace folder with synthetic test files."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="mindfs_test_ws_")).resolve()
    fixtures = generate_test_corpus(tmp_dir)
    yield tmp_dir, fixtures
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def mindfs_env(temp_workspace):
    """Provides a fully wired MindFS environment pointing to the temp workspace."""
    ws_root, fixtures = temp_workspace
    config = MindFSConfig(
        workspace_root=str(ws_root),
        index={
            "db_path": str(ws_root / ".mindfs" / "metadata.db"),
            "faiss_path": str(ws_root / ".mindfs" / "index.faiss"),
            "max_file_size_mb": 5.0,
            "retrieval_candidates": 10,
            "final_evidence_count": 5,
        }
    )
    sandbox = FilesystemSandbox(ws_root)
    resources = ResourceManager(config)
    store = SQLiteStore(config.resolved_db_path)
    registry = create_default_registry(config)
    vector_store = VectorStore(
        embedding_dim=config.embedding.embedding_dim,
        index_path=config.resolved_faiss_path,
    )
    embedding_pipeline = EmbeddingPipeline(config)
    indexer = Indexer(
        config=config,
        sandbox=sandbox,
        store=store,
        registry=registry,
        vector_store=vector_store,
        embedding_pipeline=embedding_pipeline,
        resource_manager=resources,
    )
    search_engine = SearchEngine(
        config=config,
        sandbox=sandbox,
        store=store,
        vector_store=vector_store,
        embedding_pipeline=embedding_pipeline,
        resource_manager=resources,
    )

    return {
        "workspace_root": ws_root,
        "fixtures": fixtures,
        "config": config,
        "sandbox": sandbox,
        "resources": resources,
        "store": store,
        "registry": registry,
        "vector_store": vector_store,
        "embedding_pipeline": embedding_pipeline,
        "indexer": indexer,
        "search_engine": search_engine,
    }
