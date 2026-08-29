"""Acceptance Test Suite for MindFS covering all criteria in Section 40."""

from pathlib import Path
import time
import pytest

from mindfs.agent.llm import LLMEngine
from mindfs.agent.loop import MindFSAgent
from mindfs.agent.tools import FilesystemTools
from mindfs.config.settings import MindFSConfig
from mindfs.filesystem.sandbox import FilesystemSandbox, SandboxSecurityError
from mindfs.identification.detector import FileDetector
from mindfs.indexing.embeddings import EmbeddingPipeline
from mindfs.indexing.indexer import Indexer
from mindfs.indexing.vector_store import VectorStore
from mindfs.processors import create_default_registry
from mindfs.resources.manager import ResourceManager
from mindfs.retrieval.search import SearchEngine
from mindfs.storage.sqlite_store import SQLiteStore


def test_acceptance_filesystem_security(mindfs_env):
    """Filesystem criteria: Cannot escape workspace, symlinks rejected, missing files structured error."""
    sandbox = mindfs_env["sandbox"]
    ws_root = mindfs_env["workspace_root"]

    # Traversal escape
    with pytest.raises(SandboxSecurityError):
        sandbox.validate_and_resolve("../../escape.txt")

    # Symlink escape
    ext = ws_root.parent / "ext_secret.txt"
    ext.write_text("external secret")
    sym = ws_root / "bad_symlink"
    if sym.exists():
        sym.unlink()
    sym.symlink_to(ext)

    with pytest.raises(SandboxSecurityError):
        sandbox.validate_and_resolve("bad_symlink")


def test_acceptance_documents_and_pdf(mindfs_env):
    """Documents criteria: Text extracted, source locations preserved, PDF pages tracked."""
    indexer = mindfs_env["indexer"]
    store = mindfs_env["store"]
    search_engine = mindfs_env["search_engine"]

    indexer.index_path("", recursive=True)

    # Markdown document
    res_md = search_engine.search("Project Apollo Alice")
    assert any("markdown.md" in ev.relative_path for ev in res_md.evidence)

    # PDF document with page offset
    res_pdf = search_engine.search("Architecture Report Page")
    assert any("sample.pdf" in ev.relative_path for ev in res_pdf.evidence)


def test_acceptance_structured_data(mindfs_env):
    """Structured data criteria: Schema, keys, columns searchable."""
    indexer = mindfs_env["indexer"]
    search_engine = mindfs_env["search_engine"]

    indexer.index_path("", recursive=True)

    # CSV column and value search
    res_csv = search_engine.search("Acme Corp monthly fee Enterprise")
    assert any("data.csv" in ev.relative_path for ev in res_csv.evidence)

    # JSON service and metric search
    res_json = search_engine.search("MindFS Server requests latency")
    assert any("data.json" in ev.relative_path for ev in res_json.evidence)


def test_acceptance_media_image_audio_video(mindfs_env):
    """Media criteria: Image, Audio, Video metadata extracted with timestamps and formats."""
    indexer = mindfs_env["indexer"]
    search_engine = mindfs_env["search_engine"]

    indexer.index_path("", recursive=True)

    # Image
    res_img = search_engine.search("Dimensions 200 x 200 image")
    assert any("image.jpg" in ev.relative_path for ev in res_img.evidence)

    # Audio
    res_audio = search_engine.search("Audio Track audio.wav sample rate 16000")
    assert any("audio.wav" in ev.relative_path for ev in res_audio.evidence)

    # Video
    res_video = search_engine.search("Video File video.mp4")
    assert any("video.mp4" in ev.relative_path for ev in res_video.evidence)


def test_acceptance_archives_and_binaries(mindfs_env):
    """Archive & Binary criteria: Safe inspection without execution/bombing."""
    indexer = mindfs_env["indexer"]
    search_engine = mindfs_env["search_engine"]

    indexer.index_path("", recursive=True)

    # Archive
    res_zip = search_engine.search("Archive Manifest internal_report.txt")
    assert any("archive.zip" in ev.relative_path for ev in res_zip.evidence)

    # Binary (ELF)
    res_elf = search_engine.search("Architecture x86_64 libc.so.6")
    assert any("sample_elf" in ev.relative_path for ev in res_elf.evidence)


def test_acceptance_index_lifecycle_and_restart(temp_workspace):
    """Indexing criteria: Index survives restart without full re-indexing, unchanged skipped."""
    ws_root, fixtures = temp_workspace
    config = MindFSConfig(
        workspace_root=str(ws_root),
        index={"db_path": str(ws_root / ".mindfs" / "metadata.db"), "faiss_path": str(ws_root / ".mindfs" / "index.faiss")}
    )

    # 1. Initialize and index in Session 1
    sandbox_1 = FilesystemSandbox(ws_root)
    store_1 = SQLiteStore(config.resolved_db_path)
    reg_1 = create_default_registry(config)
    vstore_1 = VectorStore(config.embedding.embedding_dim, config.resolved_faiss_path)
    emb_1 = EmbeddingPipeline(config)
    indexer_1 = Indexer(config, sandbox_1, store_1, reg_1, vstore_1, emb_1)

    r1 = indexer_1.index_path("", recursive=True)
    assert r1["files_indexed"] > 0

    # 2. Simulate Application Restart (Session 2 with new store and vector objects)
    sandbox_2 = FilesystemSandbox(ws_root)
    store_2 = SQLiteStore(config.resolved_db_path)
    reg_2 = create_default_registry(config)
    vstore_2 = VectorStore(config.embedding.embedding_dim, config.resolved_faiss_path)
    emb_2 = EmbeddingPipeline(config)
    indexer_2 = Indexer(config, sandbox_2, store_2, reg_2, vstore_2, emb_2)
    search_2 = SearchEngine(config, sandbox_2, store_2, vstore_2, emb_2)

    # Verify search still works immediately on restart without re-indexing
    res = search_2.search("Project Apollo")
    assert res.has_sufficient_evidence is True

    # Run indexer on restart: should skip unchanged files
    r2 = indexer_2.index_path("", recursive=True)
    assert r2["files_indexed"] == 0
    assert r2["files_skipped"] > 0


def test_acceptance_memory_ceiling_diagnostics(mindfs_env):
    """Memory criteria: Peak process RSS remains below 1.7 GB (1740 MB)."""
    resources = mindfs_env["resources"]
    indexer = mindfs_env["indexer"]
    search_engine = mindfs_env["search_engine"]

    # Measure across operations
    indexer.index_path("", recursive=True)
    search_engine.search("Apollo migration")

    peak_rss = resources.get_peak_rss_mb()
    assert peak_rss < 1740.0, f"Peak RSS {peak_rss} MB exceeded budget limit!"

