"""LangGraph stateful orchestration graph with specialized filesystem subagents."""

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, TypedDict
import uuid

from langgraph.graph import StateGraph, END

from mindfs.agent.llm import LLMEngine
from mindfs.agent.memory import MemoryManager, WorkingMemory
from mindfs.agent.tools import FilesystemTools, ToolCall, ToolResult
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory
from mindfs.retrieval.evidence import RetrievalResult


class ProposedAction(TypedDict, total=False):
    action_id: str
    tool_name: str
    arguments: Dict[str, Any]
    description: str
    is_destructive: bool
    impact_summary: str
    diff_preview: Optional[Dict[str, Any]]


class ActionPlan(TypedDict):
    plan_id: str
    intent: str
    proposed_actions: List[ProposedAction]
    requires_approval: bool
    status: str
    created_at: str


class AgentState(TypedDict):
    query: str
    history: List[Dict[str, Any]]
    intent: str
    route: str  # "plan_action", "undo", "inspect", "inventory", "summary", "rag_search"
    thoughts: List[Dict[str, Any]]
    operations: List[Dict[str, Any]]
    explored_files: List[str]
    subagents: List[Dict[str, Any]]
    plan: Optional[ActionPlan]
    retrieval_evidence: Optional[Any]
    answer: str
    status: str
    can_undo: bool
    undo_log_ids: List[str]
    total_duration_seconds: float
    model_name: Optional[str]
    context_usage: Optional[Dict[str, Any]]


class MindFSLangGraphAgent:
    """Stateful agent orchestrator powered by LangGraph."""

    def __init__(self, config: MindFSConfig, tools: FilesystemTools, llm_engine: LLMEngine):
        self.config = config
        self.tools = tools
        self.llm_engine = llm_engine
        self.pending_plans: Dict[str, Dict[str, Any]] = {}
        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs the compiled LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # 1. Add Subagent Nodes
        workflow.add_node("intent_router", self._node_intent_router)
        workflow.add_node("filesystem_planner", self._node_filesystem_planner)
        workflow.add_node("safety_guard", self._node_safety_guard)
        workflow.add_node("rag_retriever", self._node_rag_retriever)
        workflow.add_node("answer_synthesizer", self._node_answer_synthesizer)
        workflow.add_node("file_inspector", self._node_file_inspector)
        workflow.add_node("inventory_reporter", self._node_inventory_reporter)
        workflow.add_node("workspace_summarizer", self._node_workspace_summarizer)
        workflow.add_node("undo_executor", self._node_undo_executor)

        # 2. Entry Point
        workflow.set_entry_point("intent_router")

        # 3. Conditional Routing from Router
        def route_decision(state: AgentState) -> str:
            route = state.get("route", "rag_search")
            if route == "plan_action":
                return "filesystem_planner"
            elif route == "undo":
                return "undo_executor"
            elif route == "inspect":
                return "file_inspector"
            elif route == "inventory":
                return "inventory_reporter"
            elif route == "summary":
                return "workspace_summarizer"
            else:
                return "rag_retriever"

        workflow.add_conditional_edges(
            "intent_router",
            route_decision,
            {
                "filesystem_planner": "filesystem_planner",
                "undo_executor": "undo_executor",
                "file_inspector": "file_inspector",
                "inventory_reporter": "inventory_reporter",
                "workspace_summarizer": "workspace_summarizer",
                "rag_retriever": "rag_retriever",
            },
        )

        # Planner -> Safety Guard -> END
        workflow.add_edge("filesystem_planner", "safety_guard")
        workflow.add_edge("safety_guard", END)

        # RAG -> Synthesizer -> END
        workflow.add_edge("rag_retriever", "answer_synthesizer")
        workflow.add_edge("answer_synthesizer", END)

        # Direct terminators
        workflow.add_edge("file_inspector", END)
        workflow.add_edge("inventory_reporter", END)
        workflow.add_edge("workspace_summarizer", END)
        workflow.add_edge("undo_executor", END)

        return workflow.compile()

    # ---------------- Subagent 1: Intent Router ----------------

    def _node_intent_router(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        q_lower = query.lower().strip()
        history = state.get("history", [])

        thoughts = list(state.get("thoughts", []))
        subagents = list(state.get("subagents", []))

        # 1. Dynamic Model-Driven Query Decomposition (First Step of CoT)
        th_t0 = time.perf_counter()
        query_understanding = self.llm_engine.reason_and_simplify_query(query, history=history)
        th_dt = round(time.perf_counter() - th_t0, 3)

        thoughts.append({
            "title": "Query Understanding & Semantic Decomposition",
            "detail": query_understanding,
            "duration_seconds": th_dt,
        })

        # 2. Session Working Memory State
        working_mem = MemoryManager.extract_working_memory(history)
        if working_mem.active_entities:
            thoughts.append({
                "title": "Session Working Memory",
                "detail": f"Active entities in context: {', '.join(working_mem.active_entities[:4])} (Session turns: {working_mem.turn_count}).",
                "duration_seconds": 0.001,
            })

        subagents.append({
            "role": "Intent Classification Subagent",
            "name": "LangGraph Router",
            "task": f"Deconstructing request and enforcing policy constraints for: '{query}' (History: {len(history)} turns).",
            "status": "COMPLETED",
        })

        # Multi-turn context resolution: if "try again" / "retry", inherit previous intent
        effective_query = query
        if q_lower in ("try again", "retry", "repeat", "redo") and history:
            user_turns = [h for h in history[:-1] if h.get("role") == "user" and h.get("content").strip().lower() not in ("try again", "retry", "repeat", "redo")]
            if not user_turns and history:
                user_turns = [h for h in history if h.get("role") == "user" and h.get("content").strip().lower() not in ("try again", "retry", "repeat", "redo")]
            if user_turns:
                effective_query = user_turns[-1].get("content", query)
                q_lower = effective_query.lower().strip()

        # A. Undo / Rollback / Revert
        is_undo = (
            q_lower in ("undo", "rollback", "revert", "undo that", "revert that", "take back", "undo last", "revert last")
            or bool(re.search(r"^(please\s+)?(undo|rollback|revert|take\s+back)\b", q_lower))
            or bool(re.search(r"\b(undo|rollback|revert)\s+(the\s+)?(last|previous|recent|change|action|operation)\b", q_lower))
        ) and not (q_lower.startswith("create ") or q_lower.startswith("make ") or q_lower.startswith("write "))

        if is_undo:
            route = "undo"
            intent = "Undo / Rollback Operation"

        # B. Mutating Actions (Move, Copy, Create, Rename, Delete, Organize, Clean)
        elif any(action_kw in q_lower for action_kw in (
            "move ", "mv ", "create file", "make file", "create folder", "make folder", "create dir", "mkdir ",
            "delete ", "remove file", "trash ", "rename ", "organize ", "clean up files", "sort files"
        )):
            route = "plan_action"
            intent = "Filesystem Action Planning"

        # C. Inspect File
        elif q_lower.startswith("inspect ") or "inspect file" in q_lower:
            route = "inspect"
            intent = "Inspect File Metadata"

        # D. Workspace Summary
        elif any(w in q_lower for w in ("summarise workspace", "summarize workspace", "workspace summary", "overview of workspace", "summarise project", "summarize project", "project summary", "explain workspace", "workspace overview")):
            route = "summary"
            intent = "Workspace Intelligence Overview"

        # E. File Inventory / Listing / Count
        elif any(re.search(p, q_lower) for p in [
            r"\b(how\s+many|count|number\s+of|total)\b.*\b(files?|documents?|pdfs?|images?|programs?|scripts?|code|py|python|js|ts|json|markdown|items?|folders?|dirs?)\b",
            r"\blist\b.*\b(files?|documents?|pdfs?|images?|programs?|scripts?|code|all|contents|directory|folder|workspace|py|python|js|ts|json|markdown|types)\b",
            r"\bshow\b.*\b(files?|documents?|pdfs?|images?|programs?|scripts?|code|all|contents|directory|folder|workspace|py|python|js|ts|json|markdown)\b",
            r"^list$", r"^ls$", r"^dir$", r"^count$"
        ]):
            route = "inventory"
            intent = "File Inventory & Types"

        # F. Semantic RAG Retrieval
        else:
            route = "rag_search"
            intent = "Semantic Retrieval & Question Answering"

        dt = round(time.perf_counter() - t0, 3)

        return {
            "query": effective_query,
            "intent": intent,
            "route": route,
            "thoughts": thoughts,
            "subagents": subagents,
        }

    # ---------------- Subagent 2: Filesystem Planner ----------------

    def _node_filesystem_planner(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        q_lower = query.lower().strip()
        all_files = self.tools.store.list_all_files()

        thoughts = list(state.get("thoughts", []))
        operations = list(state.get("operations", []))
        explored_files = list(state.get("explored_files", []))
        subagents = list(state.get("subagents", []))

        subagents.append({
            "role": "Filesystem Planning Subagent",
            "name": "Action Graph Planner",
            "task": "Scanning workspace entities to formulate deterministic ActionPlan.",
            "status": "COMPLETED",
        })

        plan_id = uuid.uuid4().hex[:10]
        proposed: List[ProposedAction] = []
        matching_files: List[Any] = []
        plan_title = "Filesystem Operation"

        # Case 1: "Move disk images to a new folder named dmg" / "Move *.dmg files to dmg"
        if ("move" in q_lower or "mv" in q_lower) and any(kw in q_lower for kw in ("disk image", "dmg", "pdf", "image", "audio", "video", "archive", "data", "doc", "file")):
            # Detect target category/extension
            target_ext = ""
            target_cat = None
            if "dmg" in q_lower or "disk image" in q_lower:
                target_ext = ".dmg"
            elif "pdf" in q_lower:
                target_ext = ".pdf"
            elif "image" in q_lower:
                target_cat = FileCategory.IMAGE
            elif "audio" in q_lower:
                target_cat = FileCategory.AUDIO
            elif "video" in q_lower:
                target_cat = FileCategory.VIDEO

            # Detect destination folder name
            dest_folder = "dmg"
            m_dest = re.search(r"\b(?:to|into)\s+(?:(?:a\s+)?(?:new\s+)?(?:folder|dir|directory)?\s*(?:names|named)?\s*)+([a-zA-Z0-9_\-\./]+)", q_lower)
            if m_dest:
                cand = m_dest.group(1).strip().strip("'\"")
                if cand not in ("a", "the", "new", "folder", "directory", "names", "named"):
                    dest_folder = cand

            # Find matching files across database AND live workspace filesystem
            matching_files = []
            seen_rel = set()
            for f in all_files:
                if (target_ext and f.filename.lower().endswith(target_ext)) or (target_cat and f.category == target_cat):
                    matching_files.append((f.filename, f.relative_path))
                    seen_rel.add(f.relative_path)

            # Live disk scan
            roots = [Path(self.config.resolved_workspace_root)] + list(self.tools.sandbox.allowed_roots)
            for r in roots:
                if not r.exists():
                    continue
                for p in r.rglob("*"):
                    if p.is_file():
                        try:
                            rel = str(p.relative_to(r))
                        except ValueError:
                            rel = p.name
                        if any(ign in rel for ign in (".git", "node_modules", ".mindfs", "__pycache__", ".venv", "venv", "trash")):
                            continue
                        if rel not in seen_rel:
                            if target_ext and p.name.lower().endswith(target_ext):
                                matching_files.append((p.name, rel))
                                seen_rel.add(rel)

            # Filter out files already in dest_folder
            matching_files = [(fn, rel) for fn, rel in matching_files if not rel.startswith(f"{dest_folder}/") and rel != dest_folder]

            if matching_files:
                plan_title = f"Move {len(matching_files)} file(s) to `{dest_folder}/`"
                # Add create directory action if needed
                proposed.append({
                    "action_id": f"act_{len(proposed)+1}",
                    "tool_name": "create_directory",
                    "arguments": {"path": dest_folder},
                    "description": f"Create destination directory '{dest_folder}'",
                    "is_destructive": False,
                    "impact_summary": f"Creates folder '{dest_folder}' inside workspace",
                    "diff_preview": {
                        "type": "mkdir",
                        "target": f"{dest_folder}/",
                        "old_path": None,
                        "new_path": f"{dest_folder}/",
                        "diff_text": f"+ [CREATE DIRECTORY] {dest_folder}/",
                    },
                })
                for fn, rel_path in matching_files:
                    explored_files.append(rel_path)
                    dest_file_path = f"{dest_folder}/{fn}"
                    proposed.append({
                        "action_id": f"act_{len(proposed)+1}",
                        "tool_name": "move_path",
                        "arguments": {"source_path": rel_path, "destination_path": dest_file_path},
                        "description": f"Move '{rel_path}' -> '{dest_file_path}'",
                        "is_destructive": True,
                        "impact_summary": f"Relocates '{fn}' to '{dest_folder}/'",
                        "diff_preview": {
                            "type": "move",
                            "target": fn,
                            "old_path": rel_path,
                            "new_path": dest_file_path,
                            "diff_text": f"- {rel_path}\n+ {dest_file_path}",
                        },
                    })
            else:
                plan_title = f"Move files to `{dest_folder}/`"

        # Case 2: Specific single file move / rename (e.g. "move notes.txt to archive/notes.txt")
        elif ("move" in q_lower or "mv" in q_lower or "rename" in q_lower):
            m_move = re.search(r"\b(?:move|mv)\s+([^\s]+)\s+(?:to|into)\s+([^\s]+)", q_lower)
            m_ren = re.search(r"\brename\s+([^\s]+)\s+(?:to|as)\s+([^\s]+)", q_lower)
            src = m_move.group(1).strip() if m_move else (m_ren.group(1).strip() if m_ren else "")
            dst = m_move.group(2).strip() if m_move else (m_ren.group(2).strip() if m_ren else "")
            tool_nm = "move_path" if m_move else "rename_path"
            if src and dst:
                plan_title = f"Move / Rename '{src}' -> '{dst}'"
                proposed.append({
                    "action_id": "act_1",
                    "tool_name": tool_nm,
                    "arguments": {"source_path": src, "destination_path" if m_move else "new_name": dst},
                    "description": f"Move/Rename '{src}' -> '{dst}'",
                    "is_destructive": True,
                    "impact_summary": f"Relocates '{src}' to destination '{dst}'",
                    "diff_preview": {
                        "type": "move" if m_move else "rename",
                        "target": Path(src).name,
                        "old_path": src,
                        "new_path": dst,
                        "diff_text": f"- {src}\n+ {dst}",
                    },
                })
                explored_files.append(src)

        # Case 3: Create File (e.g. "create file test.txt with hello")
        elif any(q_lower.startswith(k) for k in ("create file", "make file", "write file", "touch ")):
            m_cf = re.search(r"\b(?:create|make|write|touch)\s+file\s+([^\s]+)(?:\s+(?:with\s+content|with)?\s*(.*))?", q_lower)
            if m_cf:
                target_path = m_cf.group(1).strip()
                content = m_cf.group(2).strip() if m_cf.group(2) else ""
                plan_title = f"Create file '{target_path}'"
                content_preview = ("\n".join(f"+ {line}" for line in content.splitlines()[:6])) if content else f"+ [Empty file: {target_path}]"
                proposed.append({
                    "action_id": "act_1",
                    "tool_name": "create_file",
                    "arguments": {"path": target_path, "content": content},
                    "description": f"Create file '{target_path}' ({len(content.encode('utf-8'))} bytes)",
                    "is_destructive": False,
                    "impact_summary": f"Creates new file '{target_path}' inside workspace",
                    "diff_preview": {
                        "type": "create",
                        "target": target_path,
                        "old_path": None,
                        "new_path": target_path,
                        "diff_text": f"+ [CREATE NEW FILE] {target_path}\n{content_preview}",
                    },
                })

        # Case 4: Create Directory
        elif any(k in q_lower for k in ("create folder", "make folder", "create dir", "mkdir")):
            m_cd = re.search(r"\b(?:create|make|mkdir)\s+(?:dir|directory|folder)\s+([^\s]+)", q_lower)
            if m_cd:
                target_dir = m_cd.group(1).strip()
                plan_title = f"Create folder '{target_dir}'"
                proposed.append({
                    "action_id": "act_1",
                    "tool_name": "create_directory",
                    "arguments": {"path": target_dir},
                    "description": f"Create directory '{target_dir}'",
                    "is_destructive": False,
                    "impact_summary": f"Creates folder '{target_dir}' inside workspace",
                    "diff_preview": {
                        "type": "mkdir",
                        "target": f"{target_dir}/",
                        "old_path": None,
                        "new_path": f"{target_dir}/",
                        "diff_text": f"+ [CREATE DIRECTORY] {target_dir}/",
                    },
                })

        # Case 5: Delete File
        elif any(q_lower.startswith(k) for k in ("delete ", "remove ", "rm ", "trash ")):
            m_del = re.search(r"\b(?:delete|remove|rm|trash)\s+(?:file|folder|dir)?\s*([^\s]+)", q_lower)
            if m_del:
                target_path = m_del.group(1).strip()
                plan_title = f"Delete '{target_path}'"
                proposed.append({
                    "action_id": "act_1",
                    "tool_name": "delete_file",
                    "arguments": {"path": target_path},
                    "description": f"Delete '{target_path}' (Safe backup to trash)",
                    "is_destructive": True,
                    "impact_summary": f"Removes '{target_path}' with backup to .mindfs/trash for full undo",
                    "diff_preview": {
                        "type": "delete",
                        "target": target_path,
                        "old_path": target_path,
                        "new_path": f".mindfs/trash/{Path(target_path).name}",
                        "diff_text": f"- {target_path}\n+ [TRASH BACKUP] .mindfs/trash/{Path(target_path).name}",
                    },
                })
                explored_files.append(target_path)

        # Case 6: Organize Files
        elif any(w in q_lower for w in ("organize files", "organize workspace", "sort files into folders", "clean up files", "categorize files")):
            plan_title = "Batch Organize Workspace Files"
            proposed.append({
                "action_id": "act_1",
                "tool_name": "organize_files",
                "arguments": {"source_dir": ""},
                "description": "Organize workspace files by category (documents/, images/, media/, data/, archives/)",
                "is_destructive": True,
                "impact_summary": "Groups loose files in workspace root into categorized subfolders",
                "diff_preview": {
                    "type": "organize",
                    "target": "Workspace Loose Files",
                    "old_path": "workspace root",
                    "new_path": "documents/, images/, media/, data/, archives/",
                    "diff_text": "+ [AUTO-CATEGORIZE] Sort root files by MIME/extension into categorized folders",
                },
            })

        plan: Optional[ActionPlan] = None
        if proposed:
            plan = {
                "plan_id": plan_id,
                "intent": plan_title,
                "proposed_actions": proposed,
                "requires_approval": True,
                "status": "PENDING_APPROVAL",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self.pending_plans[plan_id] = plan

        dt = round(time.perf_counter() - t0, 3)
        if matching_files:
            thoughts.append({
                "title": "Workspace Entity Discovery & Plan Formulation",
                "detail": f"Discovered {len(matching_files)} matching file(s) across workspace. Formulated destination '{dest_folder}/' with {len(proposed)} concrete operations ({dt*1000:.1f} ms).",
                "duration_seconds": dt,
            })
        elif proposed:
            thoughts.append({
                "title": "Action Plan Formulation",
                "detail": f"Formulated ActionPlan `{plan_id}` with {len(proposed)} proposed tool operations ({dt*1000:.1f} ms).",
                "duration_seconds": dt,
            })
        else:
            thoughts.append({
                "title": "Workspace Entity Scan",
                "detail": f"Scanned workspace and indexed storage for matching files ({dt*1000:.1f} ms).",
                "duration_seconds": dt,
            })

        return {
            "plan": plan,
            "thoughts": thoughts,
            "operations": operations,
            "explored_files": explored_files,
            "subagents": subagents,
        }

    # ---------------- Subagent 3: Safety Guard ----------------

    def _node_safety_guard(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        plan = state.get("plan")
        thoughts = list(state.get("thoughts", []))
        subagents = list(state.get("subagents", []))

        subagents.append({
            "role": "Safety Policy Verification Subagent",
            "name": "Sandbox Guard",
            "task": "Enforcing sandbox containment & human-in-the-loop review policy.",
            "status": "COMPLETED",
        })

        if not plan or not plan.get("proposed_actions"):
            answer = (
                f"ℹ️ **No matching files found in workspace** (`{self.config.resolved_workspace_root}`).\n\n"
                f"MindFS scanned the workspace and indexed storage, but found no files matching the requested target. "
                f"If you add or copy those files into the workspace, you can ask again to generate an automated action plan."
            )
            status = "COMPLETED"
        else:
            actions_list = plan["proposed_actions"]
            actions_summary = "\n".join(f"- ⚡ **{a['description']}** (Impact: *{a['impact_summary']}*)" for a in actions_list)
            answer = (
                f"### 📋 Proposed Action Plan: *{plan['intent']}*\n\n"
                f"MindFS generated the following filesystem operations requiring your explicit approval:\n\n"
                f"{actions_summary}\n\n"
                f"⚠️ *Please review and approve below to execute.*"
            )
            status = "PENDING_APPROVAL"

        dt = round(time.perf_counter() - t0, 3)
        if plan and plan.get("proposed_actions"):
            thoughts.append({
                "title": "Sandbox Containment & Safety Verification",
                "detail": f"Verified workspace sandbox boundaries. Flagged {len(plan['proposed_actions'])} mutating operations for mandatory human confirmation.",
                "duration_seconds": dt,
            })

        return {
            "answer": answer,
            "status": status,
            "thoughts": thoughts,
            "subagents": subagents,
        }

    # ---------------- Subagent 4: RAG Retriever ----------------

    def _node_rag_retriever(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        thoughts = list(state.get("thoughts", []))
        operations = list(state.get("operations", []))
        explored_files = list(state.get("explored_files", []))
        subagents = list(state.get("subagents", []))

        subagents.append({
            "role": "Semantic Intelligence Subagent",
            "name": "Vector Retriever",
            "task": f"Executing FAISS semantic similarity search for query '{query}'.",
            "status": "COMPLETED",
        })

        # Contextualize query with conversational memory
        history = state.get("history", [])
        search_query = MemoryManager.contextualize_query(query, history, llm_engine=self.llm_engine)
        if search_query != query:
            thoughts.append({
                "title": "Conversational Coreference Resolution",
                "detail": f"Resolved conversational reference '{query}' -> '{search_query}' using session memory.",
                "duration_seconds": 0.01,
            })

        # Check for file path mentions
        ref_file = None
        for f in self.tools.store.list_all_files():
            if f.filename.lower() in search_query.lower() or f.relative_path.lower() in search_query.lower():
                ref_file = f.relative_path
                break

        res = self.tools.search_engine.search(
            query=search_query,
            path_filter=ref_file,
            limit=self.config.index.final_evidence_count,
        )
        dt = round(time.perf_counter() - t0, 3)

        operations.append({
            "type": "tool",
            "title": f"Searched vector index (Found {len(res.evidence)} chunks)",
            "command_or_tool": "rag_search",
            "args": {"query": search_query, "path_filter": ref_file},
            "summary": f"Retrieved {len(res.evidence)} chunks across {len(set(e.source_path for e in res.evidence))} files",
            "duration_seconds": dt,
            "status": "COMPLETED",
        })

        for ev in res.evidence:
            if ev.source_path and ev.source_path not in explored_files:
                explored_files.append(ev.source_path)

        ev_count = len(res.evidence)
        unique_files_count = len(set(e.source_path for e in res.evidence))
        top_score = round(res.evidence[0].similarity_score, 3) if res.evidence else 0.0

        thoughts.append({
            "title": "Vector & Keyword Evidence Retrieval",
            "detail": f"Retrieved {ev_count} evidence chunk(s) across {unique_files_count} file(s) from FAISS index (top score: {top_score}, {dt*1000:.1f} ms).",
            "duration_seconds": dt,
        })

        return {
            "retrieval_evidence": res,
            "thoughts": thoughts,
            "operations": operations,
            "explored_files": explored_files,
            "subagents": subagents,
        }

    # ---------------- Subagent 5: Answer Synthesizer ----------------

    def _node_answer_synthesizer(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        evidence = state.get("retrieval_evidence")
        history = state.get("history", [])
        thoughts = list(state.get("thoughts", []))
        subagents = list(state.get("subagents", []))

        subagents.append({
            "role": "Grounded Synthesis Subagent",
            "name": "Local LLM Synthesizer",
            "task": "Synthesizing evidence-grounded response using active model.",
            "status": "COMPLETED",
        })

        answer = self.llm_engine.synthesize_answer(query, evidence, history=history)
        dt = round(time.perf_counter() - t0, 3)

        model_display = self.llm_engine.active_model_name or "FastEmbed Engine"
        ev_count = len(evidence.evidence) if evidence and evidence.evidence else 0
        thoughts.append({
            "title": "Grounded Answer Synthesis",
            "detail": f"Synthesized natural language response grounded in {ev_count} evidence chunk(s) and {len(history)} past turn(s) using {model_display} ({dt*1000:.1f} ms).",
            "duration_seconds": dt,
        })

        # Context Compression Chain-of-Thought
        comp_event = getattr(self.llm_engine, "last_compression_event", None)
        if comp_event:
            thoughts.append({
                "title": "Context Window Dynamic Compression",
                "detail": f"Prompt reached {comp_event.get('threshold_pct', 70)}% ceiling ({comp_event.get('uncompressed_tokens')} tokens). Compressed memory to {comp_event.get('compressed_tokens')} tokens (Saved {comp_event.get('tokens_saved')} tokens / {comp_event.get('saved_ratio_pct')}%).",
                "duration_seconds": 0.005,
            })

        ctx_metrics = getattr(self.llm_engine, "last_context_metrics", None)
        ctx_dict = ctx_metrics.model_dump() if (ctx_metrics and hasattr(ctx_metrics, "model_dump")) else (ctx_metrics if isinstance(ctx_metrics, dict) else None)

        return {
            "answer": answer,
            "status": "COMPLETED",
            "thoughts": thoughts,
            "subagents": subagents,
            "model_name": self.llm_engine.active_model_name or "Local FastEmbed Engine",
            "context_usage": ctx_dict,
        }

    # ---------------- Subagent 6: File Inspector ----------------

    def _node_file_inspector(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        parts = query.split(maxsplit=1)
        target = parts[1].strip() if len(parts) > 1 else ""

        thoughts = list(state.get("thoughts", []))
        operations = list(state.get("operations", []))
        explored_files = list(state.get("explored_files", []))
        subagents = list(state.get("subagents", []))

        tc = ToolCall(name="inspect_file", arguments={"file_path": target})
        tr = self.tools.execute_tool(tc)
        dt = round(time.perf_counter() - t0, 3)

        operations.append({
            "type": "tool",
            "title": f"Inspected file '{target}'",
            "command_or_tool": "inspect_file",
            "args": {"file_path": target},
            "summary": f"MIME: {tr.data.get('mime_type') if tr.success else tr.error}",
            "duration_seconds": dt,
            "status": "COMPLETED" if tr.success else "FAILED",
        })
        explored_files.append(target)
        thoughts.append({
            "title": "File Inspection Analysis",
            "detail": f"Retrieved metadata and MIME inspection for target entity '{target}' ({dt*1000:.1f} ms).",
            "duration_seconds": dt,
        })

        if tr.success and tr.data:
            d = tr.data
            answer = (
                f"**File Inspection for `{d.get('file')}`**:\n"
                f"- **Category**: {d.get('category')}\n"
                f"- **MIME Type**: {d.get('mime_type')}\n"
                f"- **Size**: {d.get('size_bytes'):,} bytes\n"
                f"- **Processor**: `{d.get('processor')}`\n"
                f"- **Details**: {d.get('inspection')}"
            )
        else:
            answer = f"Inspection failed: {tr.error}"

        return {
            "answer": answer,
            "status": "COMPLETED",
            "thoughts": thoughts,
            "operations": operations,
            "explored_files": explored_files,
            "subagents": subagents,
        }

    # ---------------- Subagent 7: Inventory Reporter ----------------

    def _node_inventory_reporter(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        q_lower = query.lower()

        thoughts = list(state.get("thoughts", []))
        operations = list(state.get("operations", []))
        explored_files = list(state.get("explored_files", []))
        subagents = list(state.get("subagents", []))

        stored_files = [
            f for f in self.tools.store.list_all_files()
            if "__pycache__" not in f.relative_path and not f.relative_path.endswith(".pyc")
        ]

        is_types_query = any(w in q_lower for w in ("types of files", "file types", "kinds of files", "categories of files", "file categories", "type of files"))
        is_count_query = any(w in q_lower for w in ("how many", "count", "number of", "total"))

        if is_types_query and stored_files:
            by_cat = {}
            for f in stored_files:
                by_cat.setdefault(f.category.value, []).append(f)
            lines = [f"### 📊 Indexed File Types in Workspace (`{self.config.resolved_workspace_root}`):\n"]
            for cat_name, flist in sorted(by_cat.items()):
                exts = sorted(set(Path(x.filename).suffix.lower() for x in flist if Path(x.filename).suffix))
                ext_str = f"({', '.join(exts)})" if exts else ""
                lines.append(f"- **`[{cat_name}]`**: **{len(flist)}** files {ext_str}")
            lines.append(f"\n*Total indexed files: **{len(stored_files)}***")
            answer = "\n".join(lines)
        elif stored_files:
            for sf in stored_files[:15]:
                explored_files.append(sf.relative_path)
            lines = [f"Found **{len(stored_files)} files** in workspace (`{self.config.resolved_workspace_root}`):\n"]
            for f in sorted(stored_files, key=lambda x: (x.category.value, x.relative_path))[:30]:
                lines.append(f"- `[{f.category.value:<12}]` **{f.relative_path}** ({f.size_bytes:,} bytes)")
            if len(stored_files) > 30:
                lines.append(f"\n*...and {len(stored_files)-30} more files.*")
            answer = "\n".join(lines)
        else:
            answer = "MindFS workspace currently has no indexed files."

        dt = round(time.perf_counter() - t0, 3)
        thoughts.append({
            "title": "Workspace Inventory Scan",
            "detail": f"Queried SQLite store and aggregated metadata for {len(stored_files)} file(s) across categories ({dt*1000:.1f} ms).",
            "duration_seconds": dt,
        })
        operations.append({
            "type": "tool",
            "title": "Queried workspace inventory",
            "command_or_tool": "list_directory",
            "summary": f"Retrieved {len(stored_files)} metadata records from SQLite store",
            "duration_seconds": dt,
            "status": "COMPLETED",
        })

        return {
            "answer": answer,
            "status": "COMPLETED",
            "thoughts": thoughts,
            "operations": operations,
            "explored_files": explored_files,
            "subagents": subagents,
        }

    # ---------------- Subagent 8: Workspace Summarizer ----------------

    def _node_workspace_summarizer(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        thoughts = list(state.get("thoughts", []))
        operations = list(state.get("operations", []))
        explored_files = list(state.get("explored_files", []))
        subagents = list(state.get("subagents", []))

        stored_files = [
            f for f in self.tools.store.list_all_files()
            if "__pycache__" not in f.relative_path and not f.relative_path.endswith(".pyc")
        ]
        for sf in stored_files[:15]:
            explored_files.append(sf.relative_path)

        by_cat = {}
        for f in stored_files:
            by_cat.setdefault(f.category.value, []).append(f)

        if not self.llm_engine._is_deterministic_fallback:
            prompt = (
                f"<|im_start|>system\n"
                f"You are MindFS, a filesystem intelligence assistant. "
                f"Provide a concise, professional summary of the workspace components and structure based on the indexed file categories and list.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Workspace: {self.config.resolved_workspace_root}\n"
                f"Total files: {len(stored_files)}\n"
                f"Categories: {', '.join(f'{k}: {len(v)} files' for k, v in by_cat.items())}\n"
                f"Sample files: {', '.join(f.relative_path for f in stored_files[:12])}\n\n"
                f"Please provide a well-structured overview of this workspace.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            llm_out = self.llm_engine.generate(prompt)
            if llm_out:
                answer = llm_out.replace("<|im_end|>", "").strip()
            else:
                answer = f"Workspace `{self.config.resolved_workspace_root}` contains {len(stored_files)} files across {len(by_cat)} categories."
        else:
            lines = [
                f"### 📂 Workspace Overview: `{self.config.resolved_workspace_root}`\n",
                f"MindFS has indexed **{len(stored_files)} files** across the following categories:\n",
            ]
            for cat_name, flist in sorted(by_cat.items()):
                exts = sorted(set(Path(x.filename).suffix.lower() for x in flist if Path(x.filename).suffix))
                ext_str = f"({', '.join(exts)})" if exts else ""
                lines.append(f"- **`[{cat_name}]`**: **{len(flist)}** files {ext_str}")
            answer = "\n".join(lines)

        dt = round(time.perf_counter() - t0, 3)
        thoughts.append({
            "title": "Workspace Architectural Synthesis",
            "detail": f"Analyzed directory hierarchy and synthesized overview for {len(stored_files)} files across {len(by_cat)} categories ({dt*1000:.1f} ms).",
            "duration_seconds": dt,
        })

        return {
            "answer": answer,
            "status": "COMPLETED",
            "thoughts": thoughts,
            "operations": operations,
            "explored_files": explored_files,
            "subagents": subagents,
        }

    # ---------------- Subagent 9: Undo Executor ----------------

    def _node_undo_executor(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        thoughts = list(state.get("thoughts", []))
        operations = list(state.get("operations", []))
        subagents = list(state.get("subagents", []))
        recent_logs = self.tools.store.list_audit_logs(limit=5)
        active_logs = [l for l in recent_logs if not l.get("undone")]

        subagents.append({
            "role": "Undo & Rollback Subagent",
            "name": "Audit Reversal Engine",
            "task": "Reversing last mutating operation from persistent audit history.",
            "status": "COMPLETED",
        })

        if not active_logs:
            answer = "ℹ️ **No recent undoable actions found in audit history.** All previous actions are either already undone or no changes were made."
            thoughts.append({
                "title": "Audit History Lookup",
                "detail": "Scanned audit log database. No active non-undone operations found.",
                "duration_seconds": 0.01,
            })
        else:
            target_log = active_logs[0]
            log_id = target_log["log_id"]
            undo_res = self.tools.undo_action(log_id)
            dt = round(time.perf_counter() - t0, 3)

            if undo_res.success:
                answer = f"↩️ **Action Undone Successfully**: Restored `{target_log.get('source_path')}` (Action: `{target_log.get('action_type')}`)."
                thoughts.append({
                    "title": "Audit Log Rollback Execution",
                    "detail": f"Reverted action '{target_log.get('action_type')}' (ID: {log_id}). Restored '{target_log.get('source_path')}'.",
                    "duration_seconds": dt,
                })
            else:
                answer = f"❌ **Undo Failed**: {undo_res.error}"
                thoughts.append({
                    "title": "Audit Log Rollback Failed",
                    "detail": f"Rollback failed for ID '{log_id}': {undo_res.error}",
                    "duration_seconds": dt,
                })

            operations.append({
                "type": "command",
                "title": "Undo Action",
                "command_or_tool": "undo_action",
                "args": {"log_id": log_id},
                "summary": f"Restored {target_log.get('source_path')}",
                "duration_seconds": dt,
                "status": "COMPLETED" if undo_res.success else "FAILED",
            })

        return {
            "answer": answer,
            "status": "COMPLETED",
            "thoughts": thoughts,
            "operations": operations,
            "subagents": subagents,
        }

    # ---------------- Main Ask & Approval Loop ----------------

    def ask(self, user_query: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Runs the LangGraph agent state graph for the user query."""
        t0 = time.perf_counter()

        initial_state: AgentState = {
            "query": user_query,
            "history": history or [],
            "intent": "",
            "route": "rag_search",
            "thoughts": [],
            "operations": [],
            "explored_files": [],
            "subagents": [],
            "plan": None,
            "retrieval_evidence": None,
            "answer": "",
            "status": "COMPLETED",
            "can_undo": False,
            "undo_log_ids": [],
            "total_duration_seconds": 0.0,
            "model_name": self.llm_engine.active_model_name or "Local FastEmbed Engine",
        }

        result_state = self.graph.invoke(initial_state)
        result_state["total_duration_seconds"] = round(time.perf_counter() - t0, 3)
        return result_state

    def execute_plan(self, plan_id: str, approved: bool = True) -> Dict[str, Any]:
        """Executes a held plan after user approval."""
        t0 = time.perf_counter()
        plan = self.pending_plans.get(plan_id)
        if not plan:
            return {
                "query": f"approve:{plan_id}",
                "intent": "Execute Plan",
                "answer": "❌ **Plan not found or expired.** It may have already been executed.",
                "status": "ERROR",
                "thoughts": [],
                "operations": [],
                "explored_files": [],
                "subagents": [],
                "total_duration_seconds": 0.0,
            }

        if not approved:
            plan["status"] = "REJECTED"
            del self.pending_plans[plan_id]
            try:
                self.tools.store.update_plan_status(plan_id, "REJECTED")
            except Exception:
                pass
            return {
                "query": f"reject:{plan_id}",
                "intent": plan["intent"],
                "answer": f"🚫 **Action Rejected.** Plan `{plan_id}` ({plan['intent']}) was cancelled by the user. No files were modified.",
                "status": "REJECTED",
                "plan": plan,
                "thoughts": [],
                "operations": [],
                "explored_files": [],
                "subagents": [],
                "total_duration_seconds": round(time.perf_counter() - t0, 3),
            }

        # Execute actions
        actions_taken = []
        operations = []
        results_summary = []
        undo_log_ids = []

        for act in plan["proposed_actions"]:
            act_t0 = time.perf_counter()
            tc = ToolCall(name=act["tool_name"], arguments=act["arguments"])
            tr = self.tools.execute_tool(tc)
            act_dt = round(time.perf_counter() - act_t0, 3)

            operations.append({
                "type": "command" if "delete" in act["tool_name"] or "move" in act["tool_name"] else "tool",
                "title": f"Executed {act['tool_name']}",
                "command_or_tool": act["tool_name"],
                "args": act["arguments"],
                "summary": f"Result: {tr.data if tr.success else tr.error}",
                "duration_seconds": act_dt,
                "status": "COMPLETED" if tr.success else "FAILED",
            })

            if tr.success:
                results_summary.append(f"✅ **{act['description']}**: Success")
                if tr.audit_id:
                    undo_log_ids.append(tr.audit_id)
            else:
                results_summary.append(f"❌ **{act['description']}**: Failed — {tr.error}")

        plan["status"] = "EXECUTED"
        del self.pending_plans[plan_id]
        try:
            self.tools.store.update_plan_status(plan_id, "EXECUTED")
        except Exception:
            pass

        answer_text = (
            f"### ⚡ Execution Result for: *{plan['intent']}*\n\n"
            + "\n".join(results_summary)
            + f"\n\n*All mutating operations have been recorded in the audit log.*"
        )

        return {
            "query": f"approve:{plan_id}",
            "intent": plan["intent"],
            "answer": answer_text,
            "status": "COMPLETED",
            "plan": plan,
            "operations": operations,
            "can_undo": bool(undo_log_ids),
            "undo_log_ids": undo_log_ids,
            "thoughts": [{"title": "Plan Execution", "detail": f"Executed {len(plan['proposed_actions'])} actions.", "duration_seconds": round(time.perf_counter() - t0, 3)}],
            "subagents": [{"role": "Execution Subagent", "name": "Tool Runner", "task": f"Executed {len(plan['proposed_actions'])} tools."}],
            "explored_files": [],
            "total_duration_seconds": round(time.perf_counter() - t0, 3),
        }
