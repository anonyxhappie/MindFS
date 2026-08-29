import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mindfs.agent.llm import LLMEngine
from mindfs.agent.tools import FilesystemTools, ToolCall, ToolResult
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory
from mindfs.retrieval.evidence import RetrievalResult


class AgentAction(BaseModel):
    step: int
    thought: str
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None


class AgentResponse(BaseModel):
    query: str
    answer: str
    actions_taken: List[AgentAction] = Field(default_factory=list)
    retrieval_evidence: Optional[RetrievalResult] = None
    total_steps: int = 0


class MindFSAgent:
    """Orchestrates deterministic tools and LLM synthesis within a bounded step budget."""

    def __init__(self, config: MindFSConfig, tools: FilesystemTools, llm_engine: LLMEngine):
        self.config = config
        self.tools = tools
        self.llm_engine = llm_engine
        self.max_steps = config.agent.max_steps

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
        q_lower = query.lower().strip()

        # Sort files by relative_path length descending so more specific names match first
        all_files_sorted = sorted(all_files, key=lambda f: len(f.relative_path), reverse=True)
        for f in all_files_sorted:
            fname_lower = f.filename.lower()
            rel_lower = f.relative_path.lower()

            # Direct presence of filename or relative path with word boundaries / delimiters
            if fname_lower in q_lower or rel_lower in q_lower:
                return f.relative_path

            # Match exact stem word boundary if not a common generic stem
            stem_lower = Path(f.filename).stem.lower()
            if len(stem_lower) >= 4 and stem_lower not in ("data", "test", "main", "base", "sample", "file", "text", "info"):
                if re.search(r"\b" + re.escape(stem_lower) + r"\b", q_lower):
                    return f.relative_path

        return None

    def ask(self, user_query: str) -> AgentResponse:
        """
        Executes a bounded reasoning and tool execution loop to answer the user query.
        """
        actions: List[AgentAction] = []
        q_lower = user_query.lower().strip()
        step = 1

        # 1. Intent: Explicit file inspection (e.g. "inspect <file>")
        if q_lower.startswith("inspect ") or "inspect file" in q_lower:
            parts = user_query.split(maxsplit=1)
            target = parts[1].strip() if len(parts) > 1 else ""
            tc = ToolCall(name="inspect_file", arguments={"file_path": target})
            tr = self.tools.execute_tool(tc)
            actions.append(AgentAction(step=step, thought=f"Inspecting file '{target}'", tool_call=tc, tool_result=tr))
            step += 1
            if tr.success and tr.data:
                data = tr.data
                answer = (
                    f"**File Inspection for `{data.get('file')}`**:\n"
                    f"- **Category**: {data.get('category')}\n"
                    f"- **MIME Type**: {data.get('mime_type')}\n"
                    f"- **Size**: {data.get('size_bytes'):,} bytes\n"
                    f"- **Processor**: `{data.get('processor')}`\n"
                    f"- **Technical Details**: {data.get('inspection')}"
                )
            else:
                answer = f"Inspection failed: {tr.error}"
            return AgentResponse(query=user_query, answer=answer, actions_taken=actions, total_steps=step - 1)

        # 2. Intent: File listing / directory inventory query (e.g. "how many py files", "list program files")
        if self._is_file_listing_query(q_lower):
            tc = ToolCall(name="list_directory", arguments={})
            tr = self.tools.execute_tool(tc)
            actions.append(AgentAction(step=step, thought="Retrieving file inventory for workspace", tool_call=tc, tool_result=tr))
            step += 1
            
            stored_files = [
                f for f in self.tools.store.list_all_files()
                if "__pycache__" not in f.relative_path and not f.relative_path.endswith(".pyc")
            ]

            # Filter by requested category/extension if specified
            filter_label = "files"
            code_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".java", ".sh", ".rb", ".php", ".html", ".css", ".json", ".toml", ".yaml", ".yml", ".sql"}
            
            if re.search(r"\b(py|python)\b", q_lower):
                filter_label = "Python (.py) files"
                stored_files = [f for f in stored_files if f.filename.lower().endswith(".py")]
            elif re.search(r"\b(js|javascript)\b", q_lower):
                filter_label = "JavaScript (.js) files"
                stored_files = [f for f in stored_files if f.filename.lower().endswith((".js", ".jsx"))]
            elif re.search(r"\b(ts|typescript)\b", q_lower):
                filter_label = "TypeScript (.ts) files"
                stored_files = [f for f in stored_files if f.filename.lower().endswith((".ts", ".tsx"))]
            elif re.search(r"\bjson\b", q_lower):
                filter_label = "JSON (.json) files"
                stored_files = [f for f in stored_files if f.filename.lower().endswith(".json")]
            elif re.search(r"\b(md|markdown)\b", q_lower):
                filter_label = "Markdown (.md) files"
                stored_files = [f for f in stored_files if f.filename.lower().endswith(".md")]
            elif any(w in q_lower for w in ("pdf", "statement")):
                filter_label = "PDF documents"
                stored_files = [f for f in stored_files if f.filename.lower().endswith(".pdf")]
            elif any(w in q_lower for w in ("program", "code", "script", "source")):
                filter_label = "program/code files"
                stored_files = [f for f in stored_files if Path(f.filename).suffix.lower() in code_exts or f.category in (FileCategory.DOCUMENT, FileCategory.STRUCTURED)]
            elif any(w in q_lower for w in ("image", "photo", "screenshot", "picture")):
                filter_label = "images"
                stored_files = [f for f in stored_files if f.category == FileCategory.IMAGE]
            elif any(w in q_lower for w in ("audio", "sound", "recording")):
                filter_label = "audio files"
                stored_files = [f for f in stored_files if f.category == FileCategory.AUDIO]
            elif any(w in q_lower for w in ("video", "movie", "clip")):
                filter_label = "video files"
                stored_files = [f for f in stored_files if f.category == FileCategory.VIDEO]
            elif any(w in q_lower for w in ("archive", "zip", "tar", "dmg")):
                filter_label = "archives"
                stored_files = [f for f in stored_files if f.category == FileCategory.ARCHIVE]
            elif any(w in q_lower for w in ("binary", "exe", "executable")):
                filter_label = "binary files"
                stored_files = [f for f in stored_files if f.category == FileCategory.BINARY]

            is_count_query = any(w in q_lower for w in ("how many", "count", "number of", "total"))

            if stored_files:
                if is_count_query:
                    header = f"There are **{len(stored_files)} {filter_label}** in workspace (`{self.config.resolved_workspace_root}`):\n"
                else:
                    header = f"Found **{len(stored_files)} {filter_label}** in workspace (`{self.config.resolved_workspace_root}`):\n"
                
                lines = [header]
                for f in sorted(stored_files, key=lambda x: (x.category.value, x.relative_path)):
                    cat_badge = f"[{f.category.value}]"
                    status_badge = f"({f.processing_status.value})"
                    size_str = f"{f.size_bytes:,} bytes"
                    lines.append(f"- `{cat_badge:<13}` **{f.relative_path}** ({size_str}) — {status_badge}")
                answer = "\n".join(lines)
            elif filter_label != "files":
                answer = f"MindFS found 0 {filter_label} in workspace `{self.config.resolved_workspace_root}`."
            elif tr.success and isinstance(tr.data, list) and tr.data:
                clean_items = [it for it in tr.data if "__pycache__" not in it["name"] and not it["name"].endswith(".pyc")]
                lines = [f"Found **{len(clean_items)} entries** in workspace (`{self.config.resolved_workspace_root}`):\n"]
                for it in clean_items:
                    kind = it["type"].upper()
                    lines.append(f"- `[{kind}]` **{it['name']}** ({it['size_bytes']:,} bytes)")
                answer = "\n".join(lines)
            else:
                answer = (
                    f"MindFS found 0 {filter_label} in workspace `{self.config.resolved_workspace_root}`.\n"
                    "Please verify that files are present and click 'Index Selected Folder'."
                )
            
            return AgentResponse(query=user_query, answer=answer, actions_taken=actions, total_steps=step - 1)

        # 3. Intent: Status / Diagnostics query
        if self._is_status_query(q_lower):
            tc = ToolCall(name="get_index_status", arguments={})
            tr = self.tools.execute_tool(tc)
            actions.append(AgentAction(step=step, thought="Retrieving MindFS index and resource status", tool_call=tc, tool_result=tr))
            step += 1
            if tr.success and isinstance(tr.data, dict):
                st = tr.data
                answer = (
                    f"**MindFS Index & Storage Status**:\n"
                    f"- **Workspace**: `{self.config.resolved_workspace_root}`\n"
                    f"- **Files Indexed**: {st.get('files_indexed', 0)} / {st.get('files_total', 0)}\n"
                    f"- **Completed**: {st.get('files_completed', 0)}\n"
                    f"- **Skipped (Oversized/Unchanged)**: {st.get('files_skipped', 0)}\n"
                    f"- **Semantic Artifacts**: {st.get('artifacts_count', 0)}\n"
                    f"- **Chunks**: {st.get('chunks_count', 0)}\n"
                    f"- **FAISS Vectors**: {st.get('vectors_count', 0)}\n"
                    f"- **SQLite Database Size**: {st.get('db_size_mb', 0)} MB"
                )
            else:
                answer = f"Failed to get index status: {tr.error}"
            return AgentResponse(query=user_query, answer=answer, actions_taken=actions, total_steps=step - 1)

        # 4. Intent: Specific referenced file query (e.g. "summarise Credit Card Statement.pdf")
        referenced_file = self._find_referenced_file(user_query)
        if referenced_file:
            tc = ToolCall(name="inspect_file", arguments={"file_path": referenced_file})
            tr = self.tools.execute_tool(tc)
            actions.append(AgentAction(step=step, thought=f"Inspecting and summarizing file '{referenced_file}'", tool_call=tc, tool_result=tr))
            step += 1

            file_rec = self.tools.store.get_file_by_path(str(self.tools.sandbox.validate_and_resolve(referenced_file)))
            if file_rec:
                arts = self.tools.store.get_artifacts_by_file(file_rec.file_id)
                if arts:
                    meaningful_sections = []
                    sources_list = []
                    evidence_snippets = []

                    for idx, art in enumerate(arts, 1):
                        loc_str = art.artifact_type
                        if art.source_offset:
                            if isinstance(art.source_offset, dict):
                                if "page" in art.source_offset:
                                    loc_str = f"page {art.source_offset['page']}"
                                elif "timestamp" in art.source_offset:
                                    loc_str = f"{art.source_offset['timestamp']}"
                                elif "section" in art.source_offset:
                                    loc_str = f"section \"{art.source_offset['section']}\""
                            elif isinstance(art.source_offset, str):
                                loc_str = art.source_offset

                        prov_loc = f"{referenced_file} — {loc_str}"
                        sources_list.append(f"• {prov_loc}")

                        raw_t = art.text.strip()
                        cleaned_lines = []
                        for l in raw_t.splitlines():
                            l_str = l.strip()
                            if not l_str:
                                continue
                            if l_str.startswith("Summary: Page ") or l_str.startswith("Summary: "):
                                if ":" in l_str:
                                    after = l_str.split(":", 1)[1].strip()
                                    if after and not after.startswith("[Image-only") and not after.startswith("[empty page"):
                                        cleaned_lines.append(after)
                                continue
                            if "[Image-only" in l_str or "[empty page" in l_str:
                                continue
                            cleaned_lines.append(l_str)

                        if cleaned_lines:
                            section_snippet = " ".join(cleaned_lines[:4])
                            if len(section_snippet) > 280:
                                section_snippet = section_snippet[:280] + "..."
                            meaningful_sections.append((loc_str, section_snippet))

                        evidence_snippets.append((idx, prov_loc, raw_t))

                    # Format rich synthesized answer
                    if meaningful_sections:
                        if len(meaningful_sections) == 1:
                            synthesis_text = f"`{referenced_file}`: {meaningful_sections[0][1]}"
                        else:
                            section_summaries = []
                            for loc, snip in meaningful_sections:
                                loc_label = f" ({loc})" if loc and loc != "document" else ""
                                section_summaries.append(f"• **{referenced_file}{loc_label}**: {snip}")
                            synthesis_text = (
                                f"`{referenced_file}` is a {file_rec.category.value} document ({len(arts)} pages/sections, {file_rec.size_bytes:,} bytes).\n\n"
                                + "\n\n".join(section_summaries)
                            )
                    else:
                        clean_doc = self.llm_engine._clean_doc_text(arts[0].summary or "")
                        if clean_doc:
                            synthesis_text = f"`{referenced_file}`: {clean_doc}"
                        else:
                            synthesis_text = f"`{referenced_file}` is a {file_rec.category.value} file ({file_rec.size_bytes:,} bytes)."

                    unique_sources = list(dict.fromkeys(sources_list))
                    sources_formatted = "\n".join(unique_sources[:8])

                    evidence_blocks = []
                    for idx, prov, raw_t in evidence_snippets:
                        evidence_blocks.append(f"#### [{idx}] {prov}\n```text\n{raw_t[:600]}\n```")

                    evidence_formatted = "\n\n".join(evidence_blocks)

                    answer = (
                        f"🧠 **MindFS found 1 relevant file**\n\n"
                        f"{synthesis_text}\n\n"
                        f"**Sources**\n"
                        f"{sources_formatted}\n\n"
                        f"<details>\n"
                        f"<summary><strong>View evidence ▾</strong></summary>\n\n"
                        f"{evidence_formatted}\n"
                        f"</details>"
                    )
                    return AgentResponse(query=user_query, answer=answer, actions_taken=actions, total_steps=step - 1)

        # 5. Semantic Vector Retrieval
        tc = ToolCall(name="rag_search", arguments={"query": user_query})
        res = self.tools.search_engine.search(query=user_query)
        tr = ToolResult(
            tool_name="rag_search",
            success=True,
            data={"evidence_count": len(res.evidence), "has_evidence": res.has_sufficient_evidence},
        )
        actions.append(AgentAction(step=step, thought="Executing semantic vector search across indexed evidence", tool_call=tc, tool_result=tr))
        step += 1

        # Grounded LLM / Fallback Synthesis
        answer = self.llm_engine.synthesize_answer(user_query, res)

        return AgentResponse(
            query=user_query,
            answer=answer,
            actions_taken=actions,
            retrieval_evidence=res,
            total_steps=step - 1,
        )
