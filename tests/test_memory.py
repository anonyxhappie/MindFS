"""Tests for MindFS Working Memory, Entity Extraction, and Multi-Turn Coreference Resolution."""

import pytest
from mindfs.agent.memory import MemoryManager, WorkingMemory


def test_working_memory_extraction():
    history = [
        {"role": "user", "content": "Move disk images to folder dmg"},
        {"role": "assistant", "content": "Proposed plan to move docs/Docker.dmg and docs/Brave.dmg to dmg/", "plan_id": "p123"},
        {"role": "user", "content": "What about report.pdf?"},
        {"role": "assistant", "content": "Inspected report.pdf in documents/", "explored_files": ["docs/report.pdf"]},
    ]
    
    mem = MemoryManager.extract_working_memory(history)
    assert mem.turn_count == 4
    assert mem.last_plan_id == "p123"
    assert "dmg" in mem.active_folders
    assert any("Docker.dmg" in f for f in mem.active_files)
    assert any("report.pdf" in f for f in mem.active_files)
    assert ".dmg" in mem.active_extensions
    assert ".pdf" in mem.active_extensions


def test_contextualize_query_retry():
    history = [
        {"role": "user", "content": "Move disk images to folder dmg"},
        {"role": "assistant", "content": "Plan proposed"},
        {"role": "user", "content": "Try again"},
    ]
    ctx = MemoryManager.contextualize_query("Try again", history)
    assert ctx == "Move disk images to folder dmg"


def test_contextualize_query_coreference_fallback():
    history = [
        {"role": "user", "content": "Explain project/architecture.md"},
        {"role": "assistant", "content": "architecture.md contains system design."},
        {"role": "user", "content": "What database does it use?"},
    ]
    ctx = MemoryManager.contextualize_query("What database does it use?", history)
    assert "architecture.md" in ctx


def test_format_history_for_llm():
    history = [
        {"role": "user", "content": "Hello MindFS"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Summarize workspace"},
        {"role": "assistant", "content": "Workspace contains 5 files."},
        {"role": "user", "content": "What about that file?"},
    ]
    formatted = MemoryManager.format_history_for_llm(history, max_turns=4)
    assert "<|im_start|>user\nHello MindFS<|im_end|>" in formatted
    assert "<|im_start|>assistant\nWorkspace contains 5 files.<|im_end|>" in formatted
    # The current incoming query (history[-1]) should not be duplicated in history
    assert "What about that file?" not in formatted


def test_token_counting_and_compression():
    from mindfs.agent.memory import count_tokens, ContextMetrics

    # Test token estimation
    assert count_tokens("Hello world") >= 2
    assert count_tokens("") == 0

    history = [
        {"role": "user", "content": "Tell me about the architecture of MindFS. " * 30},
        {"role": "assistant", "content": "MindFS is a local-first intelligent filesystem assistant. " * 40},
        {"role": "user", "content": "What about the vector database and search pipeline? " * 30},
        {"role": "assistant", "content": "MindFS uses FAISS with BAAI embeddings and SQLite metadata store. " * 40},
        {"role": "user", "content": "How are files processed?"},
    ]

    evidence_blocks = [
        "Source: docs/architecture.md\n" + "Architecture specification details for offline semantic indexing. " * 30,
        "Source: docs/retrieval.md\n" + "Retrieval pipeline ranking and cosine similarity algorithms. " * 30,
    ]

    system_prompt = "<|im_start|>system\nYou are MindFS assistant.<|im_end|>"

    # 1. Under threshold (large limit = 20,000 tokens) -> No compression
    prompt_uncompressed, metrics_uncompressed, comp_event = MemoryManager.build_compressed_context(
        history=history,
        evidence_blocks=evidence_blocks,
        query="How are files processed?",
        system_prompt=system_prompt,
        total_context_limit=20000,
        threshold_pct=0.70,
    )
    assert not metrics_uncompressed.is_compressed
    assert comp_event is None
    assert metrics_uncompressed.used_tokens > 500

    # 2. Exceeds threshold (small limit = 1,000 tokens, threshold = 0.50 -> 500 tokens) -> Auto-compress!
    prompt_compressed, metrics_compressed, comp_event = MemoryManager.build_compressed_context(
        history=history,
        evidence_blocks=evidence_blocks,
        query="How are files processed?",
        system_prompt=system_prompt,
        total_context_limit=1000,
        threshold_pct=0.50,
    )
    assert metrics_compressed.is_compressed
    assert comp_event is not None
    assert metrics_compressed.tokens_saved > 0
    assert metrics_compressed.compression_ratio_pct > 30.0
    assert metrics_compressed.used_tokens < metrics_uncompressed.used_tokens
    assert "[Compressed Memory" in prompt_compressed
