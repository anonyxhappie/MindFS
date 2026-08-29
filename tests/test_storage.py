"""Tests for SQLite authoritative metadata store."""

from pathlib import Path
import pytest

from mindfs.artifacts.models import ChunkItem, SemanticArtifact
from mindfs.identification.models import FileCategory, FileInfo, ProcessingStatus
from mindfs.resources.manager import OperationDiagnostic
from mindfs.storage.sqlite_store import SQLiteStore


def test_sqlite_crud(tmp_path):
    db_path = tmp_path / "metadata.db"
    store = SQLiteStore(db_path)

    finfo = FileInfo(
        file_id="f123",
        canonical_path=str(tmp_path / "test.txt"),
        relative_path="test.txt",
        filename="test.txt",
        extension=".txt",
        mime_type="text/plain",
        category=FileCategory.DOCUMENT,
        size_bytes=100,
        mtime=12345.67,
        mtime_ns=12345670000,
        processing_status=ProcessingStatus.COMPLETED
    )

    store.upsert_file(finfo)
    retrieved = store.get_file_by_path(str(tmp_path / "test.txt"))
    assert retrieved is not None
    assert retrieved.file_id == "f123"
    assert retrieved.category == FileCategory.DOCUMENT

    # Artifacts & Chunks
    art = SemanticArtifact(
        artifact_id="art1",
        file_id="f123",
        source_path=str(tmp_path / "test.txt"),
        text="Hello world from MindFS artifact",
        summary="Hello world",
    )
    store.save_artifacts([art])
    arts = store.get_artifacts_by_file("f123")
    assert len(arts) == 1
    assert arts[0].artifact_id == "art1"

    chunk = ChunkItem(
        chunk_id="chk1",
        artifact_id="art1",
        file_id="f123",
        source_path=str(tmp_path / "test.txt"),
        text="Hello world",
    )
    store.save_chunks([(chunk, 0)])

    chk_lookup = store.get_chunk_by_vector_id(0)
    assert chk_lookup is not None
    assert chk_lookup["text"] == "Hello world"
    assert chk_lookup["category"] == "DOCUMENT"

    # Cascade delete
    vids = store.delete_file(str(tmp_path / "test.txt"))
    assert vids == [0]
    assert store.get_file_by_path(str(tmp_path / "test.txt")) is None
    assert len(store.get_artifacts_by_file("f123")) == 0
    assert store.get_chunk_by_vector_id(0) is None


def test_sqlite_diagnostics_and_runs(tmp_path):
    db_path = tmp_path / "metadata.db"
    store = SQLiteStore(db_path)

    diag = OperationDiagnostic(
        operation="test_op",
        peak_rss_mb=45.2,
        current_rss_mb=40.1,
        duration_seconds=0.15,
        files_processed=5,
    )
    store.save_diagnostic(diag)

    summary = store.get_status_summary()
    assert summary["files_total"] == 0


def test_sqlite_recent_workspaces(tmp_path):
    db_path = tmp_path / "metadata.db"
    store = SQLiteStore(db_path)

    # 1. Record workspaces
    ws1 = str(tmp_path / "WorkspaceA")
    ws2 = str(tmp_path / "WorkspaceB")
    
    store.record_recent_workspace(ws1, name="WorkspaceA")
    store.record_recent_workspace(ws2, name="WorkspaceB")

    # 2. List workspaces
    recent = store.list_recent_workspaces(limit=10)
    assert len(recent) == 2
    assert recent[0]["name"] == "WorkspaceB"
    assert recent[1]["name"] == "WorkspaceA"

    # 3. Update existing workspace (moves to top)
    store.record_recent_workspace(ws1, name="WorkspaceA Updated")
    recent2 = store.list_recent_workspaces(limit=10)
    assert len(recent2) == 2
    assert recent2[0]["path"] == str(Path(ws1).resolve())
    assert recent2[0]["name"] == "WorkspaceA Updated"

    # 4. Remove workspace
    store.remove_recent_workspace(ws2)
    recent3 = store.list_recent_workspaces(limit=10)
    assert len(recent3) == 1
    assert recent3[0]["path"] == str(Path(ws1).resolve())

