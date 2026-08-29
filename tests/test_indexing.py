"""Tests for bounded chunking, vector store, and incremental indexing."""

import time
from pathlib import Path
import pytest

from mindfs.artifacts.models import SemanticArtifact
from mindfs.indexing.chunker import Chunker


def test_chunker_bounded_size(mindfs_env):
    config = mindfs_env["config"]
    chunker = Chunker(config)

    # Large text artifact
    art = SemanticArtifact(
        artifact_id="art_large",
        file_id="f_large",
        source_path="/fake/path.txt",
        text="Sentence number one about database migration.\n\n" * 50,
    )
    chunks = chunker.chunk_artifact(art)
    assert len(chunks) > 1
    for chk in chunks:
        assert len(chk.text) <= config.index.chunk_max_size
        assert chk.file_id == "f_large"


def test_incremental_skip_unchanged(mindfs_env):
    indexer = mindfs_env["indexer"]
    store = mindfs_env["store"]
    ws_root = mindfs_env["workspace_root"]

    # Initial indexing run
    res1 = indexer.index_path("", recursive=True)
    assert res1["files_indexed"] > 0

    status_1 = store.get_status_summary()
    total_arts_1 = status_1["artifacts_count"]

    # Second indexing run without file changes
    res2 = indexer.index_path("", recursive=True)
    # All unchanged files should be skipped
    assert res2["files_indexed"] == 0
    assert res2["files_skipped"] > 0

    status_2 = store.get_status_summary()
    assert status_2["artifacts_count"] == total_arts_1


def test_incremental_reprocess_modified(mindfs_env):
    indexer = mindfs_env["indexer"]
    store = mindfs_env["store"]
    ws_root = mindfs_env["workspace_root"]

    # Initial index
    indexer.index_path("", recursive=True)

    # Modify markdown.md
    md_file = ws_root / "markdown.md"
    time.sleep(0.02)
    md_file.write_text("# Project Apollo Updated\nNew section on GraphQL.\n", encoding="utf-8")

    # Second index run
    res2 = indexer.index_path("", recursive=True)
    assert res2["files_indexed"] >= 1


def test_incremental_prune_deleted(mindfs_env):
    indexer = mindfs_env["indexer"]
    store = mindfs_env["store"]
    ws_root = mindfs_env["workspace_root"]

    # Create temporary extra file
    extra = ws_root / "temp_extra.txt"
    extra.write_text("temporary file for deletion test")

    indexer.index_path("", recursive=True)
    assert store.get_file_by_path(str(extra)) is not None

    # Delete file from disk
    extra.unlink()

    # Re-scan index
    indexer.index_path("", recursive=True)
    assert store.get_file_by_path(str(extra)) is None

