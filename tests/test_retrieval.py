"""Tests for semantic search, candidate filtering, deduplication, and evidence retrieval."""

from pathlib import Path
import pytest


def test_semantic_search_with_evidence(mindfs_env):
    indexer = mindfs_env["indexer"]
    search_engine = mindfs_env["search_engine"]

    # Index workspace
    indexer.index_path("", recursive=True)

    # Search for database migration
    res = search_engine.search("database migration PostgreSQL")
    assert len(res.evidence) > 0
    assert res.has_sufficient_evidence is True

    # Check citation provenance
    top_ev = res.evidence[0]
    assert top_ev.relative_path is not None
    assert top_ev.similarity_score > 0
    assert top_ev.text != ""


def test_search_path_and_type_filters(mindfs_env):
    indexer = mindfs_env["indexer"]
    search_engine = mindfs_env["search_engine"]

    indexer.index_path("", recursive=True)

    # Path filter
    res_path = search_engine.search("MindFS", path_filter="plain.txt")
    for ev in res_path.evidence:
        assert "plain.txt" in ev.relative_path

    # Category/Type filter
    res_csv = search_engine.search("Enterprise monthly fee", file_type_filter="csv")
    for ev in res_csv.evidence:
        assert "csv" in ev.relative_path or ev.artifact_type == "structured_csv"


def test_insufficient_evidence_empty_query(mindfs_env):
    search_engine = mindfs_env["search_engine"]
    res = search_engine.search("")
    assert res.has_sufficient_evidence is False
    assert len(res.evidence) == 0

