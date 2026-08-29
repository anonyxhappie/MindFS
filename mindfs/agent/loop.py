"""Stateful, bounded filesystem intelligence and action agent for MindFS with multi-turn memory and CoT."""

from datetime import datetime, timezone
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from mindfs.agent.llm import LLMEngine
from mindfs.agent.tools import FilesystemTools, ToolCall, ToolResult
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory
from mindfs.retrieval.evidence import RetrievalResult


class ProposedAction(BaseModel):
    """Detailed preview of an individual filesystem action before user confirmation."""
    action_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    description: str
    is_destructive: bool = False
    impact_summary: str
    diff_preview: Optional[Dict[str, Any]] = None


class ActionPlan(BaseModel):
    """Structured, human-readable execution plan requiring explicit user approval."""
    plan_id: str
    intent: str
    proposed_actions: List[ProposedAction] = Field(default_factory=list)
    requires_approval: bool = True
    status: str = "PENDING_APPROVAL"  # PENDING_APPROVAL, APPROVED, REJECTED, EXECUTED
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentAction(BaseModel):
    step: int
    thought: str
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None


class AgentThought(BaseModel):
    """Structured Chain-of-Thought (CoT) step."""
    title: str
    detail: str
    duration_seconds: float = 0.0


class AgentOperation(BaseModel):
    """Execution step detail for collapsible operations UI."""
    type: str = "tool"  # tool, command, file_read, subagent
    title: str
    command_or_tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    summary: str
    duration_seconds: float = 0.0
    status: str = "COMPLETED"


class SubagentTask(BaseModel):
    """Subagent delegation step."""
    role: str
    name: str
    task: str
    status: str = "COMPLETED"


class AgentResponse(BaseModel):
    query: str
    intent: str
    answer: str
    status: str = "COMPLETED"  # COMPLETED, PENDING_APPROVAL, REJECTED, ERROR
    plan: Optional[ActionPlan] = None
    actions_taken: List[AgentAction] = Field(default_factory=list)
    retrieval_evidence: Optional[RetrievalResult] = None
    total_steps: int = 0
    can_undo: bool = False
    undo_log_ids: List[str] = Field(default_factory=list)
    thoughts: List[AgentThought] = Field(default_factory=list)
    operations: List[AgentOperation] = Field(default_factory=list)
    explored_files: List[str] = Field(default_factory=list)
    subagents: List[SubagentTask] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
    model_name: Optional[str] = None
    context_usage: Optional[Dict[str, Any]] = None


class MindFSAgent:
    """Orchestrates deterministic tools, multi-turn memory, LangGraph workflows, and LLM synthesis within a bounded step budget."""

    def __init__(self, config: MindFSConfig, tools: FilesystemTools, llm_engine: LLMEngine):
        self.config = config
        self.tools = tools
        self.llm_engine = llm_engine
        self.max_steps = config.agent.max_steps
        self.pending_plans: Dict[str, ActionPlan] = {}
        from mindfs.agent.graph import MindFSLangGraphAgent
        self.graph_agent = MindFSLangGraphAgent(config=config, tools=tools, llm_engine=llm_engine)

    def _is_file_listing_query(self, query_lower: str) -> bool:
        """Determines if query asks for files/folders listing or counting."""
        patterns = [
            r"\b(how\s+many|count|number\s+of|total)\b.*\b(files?|documents?|pdfs?|images?|programs?|scripts?|code|py|python|js|ts|json|markdown|items?|folders?|dirs?)\b",
            r"\blist\b.*\b(files?|documents?|pdfs?|images?|programs?|scripts?|code|all|contents|directory|folder|workspace|py|python|js|ts|json|markdown)\b",
            r"\bshow\b.*\b(files?|documents?|pdfs?|images?|programs?|scripts?|code|all|contents|directory|folder|workspace|py|python|js|ts|json|markdown)\b",
            r"\bdisplay\b.*\b(files?|documents?|pdfs?|images?|programs?|scripts?|code|all|contents|py|python|js|ts|json|markdown)\b",
            r"\bwhat\b.*\b(files?|documents?|pdfs?|contents?)\b",
            r"\bwhich\b.*\b(files?|documents?|pdfs?)\b",
            r"\bfiles?\b.*\b(in|here|under|available|indexed|workspace|folder|directory)\b",
            r"\bfile\s+inventory\b",
            r"\bdirectory\s+contents\b",
            r"^list$",
            r"^ls$",
            r"^dir$",
            r"^count$",
        ]
        return any(re.search(p, query_lower) for p in patterns)

    def _is_status_query(self, query_lower: str) -> bool:
        """Determines if query asks for index or resource status."""
        patterns = [
            r"status",
            r"how many files indexed",
            r"index stats",
            r"database size",
            r"memory usage",
            r"peak rss",
            r"diagnostics",
        ]
        return any(re.search(p, query_lower) for p in patterns)

    def _find_referenced_file(self, query: str) -> Optional[str]:
        """Checks if the query specifically references a known file in workspace."""
        all_files = self.tools.store.list_all_files()
        q_lower = query.lower()
        for f in all_files:
            fn = f.filename.lower()
            if fn in q_lower or f.relative_path.lower() in q_lower:
                return f.relative_path
        return None

    def ask(self, user_query: str, history: Optional[List[Dict[str, Any]]] = None) -> AgentResponse:
        """
        Executes query through the compiled LangGraph workflow with specialized subagents.
        """
        state = self.graph_agent.ask(user_query, history=history)

        plan_obj = None
        if state.get("plan"):
            p_dict = state["plan"]
            plan_obj = ActionPlan(
                plan_id=p_dict["plan_id"],
                intent=p_dict["intent"],
                proposed_actions=[ProposedAction(**a) for a in p_dict.get("proposed_actions", [])],
                requires_approval=p_dict.get("requires_approval", True),
                status=p_dict.get("status", "PENDING_APPROVAL"),
                created_at=p_dict.get("created_at", datetime.now(timezone.utc).isoformat()),
            )
            self.pending_plans[plan_obj.plan_id] = plan_obj

        thoughts = [AgentThought(**t) if isinstance(t, dict) else t for t in state.get("thoughts", [])]
        operations = [AgentOperation(**op) if isinstance(op, dict) else op for op in state.get("operations", [])]
        subagents = [SubagentTask(**sa) if isinstance(sa, dict) else sa for sa in state.get("subagents", [])]

        return AgentResponse(
            query=user_query,
            intent=state.get("intent", "Filesystem Intelligence"),
            answer=state.get("answer", ""),
            status=state.get("status", "COMPLETED"),
            plan=plan_obj,
            actions_taken=[],
            thoughts=thoughts,
            operations=operations,
            explored_files=state.get("explored_files", []),
            subagents=subagents,
            total_steps=len(operations),
            can_undo=state.get("can_undo", False),
            undo_log_ids=state.get("undo_log_ids", []),
            model_name=state.get("model_name") or self.llm_engine.active_model_name or "Local FastEmbed Engine",
            total_duration_seconds=state.get("total_duration_seconds", 0.0),
            context_usage=state.get("context_usage"),
        )

    def execute_plan(self, plan_id: str, approved: bool = True) -> AgentResponse:
        """Executes or rejects a held ActionPlan."""
        res = self.graph_agent.execute_plan(plan_id, approved=approved)
        plan_obj = None
        if res.get("plan"):
            p_dict = res["plan"]
            plan_obj = ActionPlan(
                plan_id=p_dict["plan_id"],
                intent=p_dict["intent"],
                proposed_actions=[ProposedAction(**a) for a in p_dict.get("proposed_actions", [])],
                requires_approval=p_dict.get("requires_approval", True),
                status=p_dict.get("status", "EXECUTED"),
                created_at=p_dict.get("created_at", datetime.now(timezone.utc).isoformat()),
            )
        thoughts = [AgentThought(**t) if isinstance(t, dict) else t for t in res.get("thoughts", [])]
        operations = [AgentOperation(**op) if isinstance(op, dict) else op for op in res.get("operations", [])]
        subagents = [SubagentTask(**sa) if isinstance(sa, dict) else sa for sa in res.get("subagents", [])]

        return AgentResponse(
            query=res.get("query", f"approve:{plan_id}"),
            intent=res.get("intent", "Execute Action Plan"),
            answer=res.get("answer", ""),
            status=res.get("status", "COMPLETED"),
            plan=plan_obj,
            actions_taken=[],
            thoughts=thoughts,
            operations=operations,
            explored_files=res.get("explored_files", []),
            subagents=subagents,
            total_steps=len(operations),
            can_undo=res.get("can_undo", False),
            undo_log_ids=res.get("undo_log_ids", []),
            model_name=self.llm_engine.active_model_name or "Local FastEmbed Engine",
            total_duration_seconds=res.get("total_duration_seconds", 0.0),
            context_usage=res.get("context_usage"),
        )
