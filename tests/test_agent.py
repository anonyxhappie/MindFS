"""Tests for deterministic tools, agent loop, and evidence-grounded synthesis."""

from pathlib import Path
import pytest

from mindfs.agent.llm import LLMEngine
from mindfs.agent.loop import MindFSAgent
from mindfs.agent.tools import FilesystemTools, ToolCall


def test_agent_tools_execution(mindfs_env):
    tools = FilesystemTools(
        config=mindfs_env["config"],
        sandbox=mindfs_env["sandbox"],
        store=mindfs_env["store"],
        indexer=mindfs_env["indexer"],
        search_engine=mindfs_env["search_engine"],
        registry=mindfs_env["registry"],
    )

    # 1. list_directory
    res_list = tools.execute_tool(ToolCall(name="list_directory", arguments={}))
    assert res_list.success is True
    assert any(item["name"] == "plain.txt" for item in res_list.data)

    # 2. inspect_file
    res_insp = tools.execute_tool(ToolCall(name="inspect_file", arguments={"file_path": "data.json"}))
    assert res_insp.success is True
    assert res_insp.data["category"] == "STRUCTURED"

    # 3. index_path
    res_idx = tools.execute_tool(ToolCall(name="index_path", arguments={"path": "", "recursive": True}))
    assert res_idx.success is True

    # 4. rag_search
    res_search = tools.execute_tool(ToolCall(name="rag_search", arguments={"query": "Project Apollo"}))
    assert res_search.success is True
    assert res_search.data["has_sufficient_evidence"] is True

    # 5. get_index_status
    res_status = tools.execute_tool(ToolCall(name="get_index_status", arguments={}))
    assert res_status.success is True
    assert res_status.data["files_indexed"] > 0


def test_agent_bounded_loop_grounded_answer(mindfs_env):
    tools = FilesystemTools(
        config=mindfs_env["config"],
        sandbox=mindfs_env["sandbox"],
        store=mindfs_env["store"],
        indexer=mindfs_env["indexer"],
        search_engine=mindfs_env["search_engine"],
        registry=mindfs_env["registry"],
    )
    llm_engine = LLMEngine(mindfs_env["config"])
    agent = MindFSAgent(config=mindfs_env["config"], tools=tools, llm_engine=llm_engine)

    # Index files
    tools.index_path("", recursive=True)

    # Query with evidence
    resp = agent.ask("What database is planned for Project Apollo?")
    assert resp.total_steps <= 10
    assert "markdown.md" in resp.answer or "Project Apollo" in resp.answer

    # Query with insufficient evidence
    resp_empty = agent.ask("What is the capital of Mars in 2099?")
    assert resp_empty.total_steps <= 10

