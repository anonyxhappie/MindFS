"""Tests for deterministic tools, agent loop, safe human-in-the-loop plans, and undo rollback."""

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


def test_consequential_tools_and_undo(mindfs_env):
    tools = FilesystemTools(
        config=mindfs_env["config"],
        sandbox=mindfs_env["sandbox"],
        store=mindfs_env["store"],
        indexer=mindfs_env["indexer"],
        search_engine=mindfs_env["search_engine"],
        registry=mindfs_env["registry"],
    )

    # A. Create File
    res_create = tools.execute_tool(ToolCall(name="create_file", arguments={"path": "new_note.txt", "content": "Hello MindFS"}))
    assert res_create.success is True
    assert (mindfs_env["workspace_root"] / "new_note.txt").exists()
    assert res_create.audit_id is not None

    # Undo Create File
    res_undo_create = tools.execute_tool(ToolCall(name="undo_action", arguments={"log_id": res_create.audit_id}))
    assert res_undo_create.success is True
    assert not (mindfs_env["workspace_root"] / "new_note.txt").exists()

    # B. Move / Rename Path
    res_create2 = tools.execute_tool(ToolCall(name="create_file", arguments={"path": "report.txt", "content": "Financial report"}))
    assert res_create2.success is True
    
    res_move = tools.execute_tool(ToolCall(name="move_path", arguments={"source_path": "report.txt", "destination_path": "docs/report.txt"}))
    assert res_move.success is True
    assert (mindfs_env["workspace_root"] / "docs" / "report.txt").exists()
    assert not (mindfs_env["workspace_root"] / "report.txt").exists()

    # Undo Move Path
    res_undo_move = tools.execute_tool(ToolCall(name="undo_action", arguments={"log_id": res_move.audit_id}))
    assert res_undo_move.success is True
    assert (mindfs_env["workspace_root"] / "report.txt").exists()
    assert not (mindfs_env["workspace_root"] / "docs" / "report.txt").exists()

    # C. Delete File with Safe Backup to Trash
    res_del = tools.execute_tool(ToolCall(name="delete_file", arguments={"path": "report.txt"}))
    assert res_del.success is True
    assert not (mindfs_env["workspace_root"] / "report.txt").exists()

    # Undo Delete File (restores from trash)
    res_undo_del = tools.execute_tool(ToolCall(name="undo_action", arguments={"log_id": res_del.audit_id}))
    assert res_undo_del.success is True
    assert (mindfs_env["workspace_root"] / "report.txt").exists()

    # D. Test list_operation_history and revert_last_operation tools
    res_hist = tools.execute_tool(ToolCall(name="list_operation_history", arguments={"limit": 10}))
    assert res_hist.success is True
    assert "operations" in res_hist.data
    assert res_hist.data["total_count"] >= 3

    # Create a new file, then call revert_last_operation tool
    res_create3 = tools.execute_tool(ToolCall(name="create_file", arguments={"path": "temp_note.txt", "content": "To revert"}))
    assert res_create3.success is True
    assert (mindfs_env["workspace_root"] / "temp_note.txt").exists()

    res_rev_last = tools.execute_tool(ToolCall(name="revert_last_operation", arguments={}))
    assert res_rev_last.success is True
    assert not (mindfs_env["workspace_root"] / "temp_note.txt").exists()


def test_agent_approval_and_rejection_workflow(mindfs_env):
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

    # 1. Ask agent to create file -> Should generate PENDING_APPROVAL plan
    resp_plan = agent.ask("create file test_task.txt with content Task completed")
    assert resp_plan.status == "PENDING_APPROVAL"
    assert resp_plan.plan is not None
    assert len(resp_plan.plan.proposed_actions) == 1
    plan_id = resp_plan.plan.plan_id

    # File should NOT exist yet before approval
    assert not (mindfs_env["workspace_root"] / "test_task.txt").exists()

    # 2. Reject the plan
    resp_reject = agent.execute_plan(plan_id, approved=False)
    assert resp_reject.status == "REJECTED"
    assert not (mindfs_env["workspace_root"] / "test_task.txt").exists()

    # 3. Create plan again and approve
    resp_plan2 = agent.ask("create file test_task2.txt with content Important task")
    assert resp_plan2.status == "PENDING_APPROVAL"
    plan_id2 = resp_plan2.plan.plan_id

    resp_approve = agent.execute_plan(plan_id2, approved=True)
    assert resp_approve.status == "COMPLETED"
    assert (mindfs_env["workspace_root"] / "test_task2.txt").exists()
    assert resp_approve.can_undo is True


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
