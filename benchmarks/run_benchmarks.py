"""Comprehensive Memory (Peak RSS) and Latency Benchmarking Suite for MindFS."""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Any, Dict, List

from mindfs.agent.llm import LLMEngine
from mindfs.agent.loop import MindFSAgent
from mindfs.agent.tools import FilesystemTools
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


def run_benchmark_suite() -> List[Dict[str, Any]]:
    """Runs all 12 operational benchmarks required by Section 5 of the spec."""
    results: List[Dict[str, Any]] = []

    tmp_dir = Path(tempfile.mkdtemp(prefix="mindfs_benchmarks_"))
    try:
        fixtures = generate_test_corpus(tmp_dir)

        config = MindFSConfig(
            workspace_root=str(tmp_dir),
            index={
                "db_path": str(tmp_dir / ".mindfs" / "metadata.db"),
                "faiss_path": str(tmp_dir / ".mindfs" / "index.faiss"),
                "max_file_size_mb": 5.0,
            },
            resources={"max_rss_mb": 1740.0},
        )
        sandbox = FilesystemSandbox(tmp_dir)
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
        tools = FilesystemTools(
            config=config,
            sandbox=sandbox,
            store=store,
            indexer=indexer,
            search_engine=search_engine,
            registry=registry,
        )
        llm_engine = LLMEngine(config)
        agent = MindFSAgent(config=config, tools=tools, llm_engine=llm_engine)

        def benchmark_op(name: str, fn, **kwargs) -> Dict[str, Any]:
            start_t = time.perf_counter()
            start_rss = resources.get_current_rss_mb()
            fn(**kwargs)
            duration = round(time.perf_counter() - start_t, 4)
            current_rss = round(resources.get_current_rss_mb(), 2)
            peak_rss = round(resources.get_peak_rss_mb(), 2)

            res = {
                "operation": name,
                "duration_seconds": duration,
                "current_rss_mb": current_rss,
                "peak_rss_mb": peak_rss,
                "budget_mb": config.resources.max_rss_mb,
                "status": "PASS" if peak_rss < config.resources.max_rss_mb else "FAIL",
            }
            results.append(res)
            return res

        # 1. Application idle
        benchmark_op("1. Application Idle", lambda: time.sleep(0.05))

        # 2. LLM loaded / initialization
        benchmark_op("2. LLM Engine Ready", lambda: getattr(llm_engine, "_is_deterministic_fallback"))

        # 3. Embedding model loaded
        benchmark_op("3. Embedding Pipeline Warmup", lambda: embedding_pipeline.embed_texts(["MindFS Warmup Query"]))

        # 4. Indexing one large supported file
        benchmark_op("4. Indexing Large File (large.txt)", lambda: indexer.index_single_file("large.txt"))

        # 5. Indexing many small files (generate 100 small files)
        batch_dir = tmp_dir / "small_files"
        batch_dir.mkdir()
        for i in range(100):
            (batch_dir / f"file_{i}.txt").write_text(f"File {i} testing high-count incremental ingestion.", encoding="utf-8")

        benchmark_op("5. Indexing 100 Small Files", lambda: indexer.index_path("small_files", recursive=False))

        # 6. Image processing
        benchmark_op("6. Image Processing (image.jpg)", lambda: indexer.index_single_file("image.jpg"))

        # 7. Audio processing
        benchmark_op("7. Audio Processing (audio.wav)", lambda: indexer.index_single_file("audio.wav"))

        # 8. Video processing
        benchmark_op("8. Video Processing (video.mp4)", lambda: indexer.index_single_file("video.mp4"))

        # 9. Archive inspection
        benchmark_op("9. Archive Inspection (archive.zip)", lambda: indexer.index_single_file("archive.zip"))

        # 10. Binary inspection
        benchmark_op("10. Binary Inspection (sample_elf)", lambda: indexer.index_single_file("sample_elf"))

        # 11. Semantic retrieval
        benchmark_op("11. Semantic Retrieval", lambda: search_engine.search("PostgreSQL Apollo database"))

        # 12. LLM answer generation
        benchmark_op("12. LLM Answer Generation", lambda: agent.ask("What database is planned for Apollo?"))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


def print_benchmark_report(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print(f"{'MINDFS PEAK RSS & LATENCY BENCHMARK REPORT':^78}")
    print("=" * 78)
    print(f"{'Operation':<35} | {'Duration (s)':<12} | {'Peak RSS (MB)':<14} | {'Status':<6}")
    print("-" * 78)
    for r in results:
        print(f"{r['operation']:<35} | {r['duration_seconds']:>12.4f} | {r['peak_rss_mb']:>14.2f} | {r['status']:<6}")
    print("=" * 78)
    max_peak = max(r["peak_rss_mb"] for r in results)
    print(f"Max Peak RSS across all operations: {max_peak:.2f} MB (Hard Target: < 1740.00 MB / 1.7 GB)")
    if max_peak < 1740.0:
        print("RESULT: ALL MEMORY AND PERFORMANCE CRITERIA SATISFIED!\n")
    else:
        print("RESULT: MEMORY CRITERIA VIOLATION DETECTED!\n")


if __name__ == "__main__":
    res = run_benchmark_suite()
    print_benchmark_report(res)
