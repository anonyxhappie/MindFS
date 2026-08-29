"""Conversational and Working Memory Manager for MindFS Agent."""

import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class WorkingMemory(BaseModel):
    """Structured working memory extracted from multi-turn conversation sessions."""
    active_entities: List[str] = Field(default_factory=list)
    active_files: List[str] = Field(default_factory=list)
    active_folders: List[str] = Field(default_factory=list)
    active_extensions: List[str] = Field(default_factory=list)
    active_topic: Optional[str] = None
    last_plan_id: Optional[str] = None
    last_action_type: Optional[str] = None
    last_user_query: Optional[str] = None
    last_assistant_answer: Optional[str] = None
    turn_count: int = 0


class ContextMetrics(BaseModel):
    """Realtime token usage and compression metrics for the active context window."""
    used_tokens: int = 0
    total_tokens: int = 2048
    usage_pct: float = 0.0
    is_compressed: bool = False
    tokens_saved: int = 0
    compression_ratio_pct: float = 0.0


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """Estimates or calculates token length with tokenizer or fast heuristic."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "encode"):
                return len(tokenizer.encode(text))
            elif callable(tokenizer):
                return len(tokenizer(text))
        except Exception:
            pass
    # Standard heuristic: ~3.8 chars/token for code/markdown/English text
    return max(1, int(len(text) / 3.8))


class MemoryManager:
    """Manages multi-turn conversation memory, entity extraction, and query contextualization."""

    @staticmethod
    def extract_working_memory(history: List[Dict[str, Any]]) -> WorkingMemory:
        """Extracts structured entities, referenced files, and goals from conversation turns."""
        mem = WorkingMemory()
        if not history:
            return mem

        mem.turn_count = len(history)
        user_msgs = [h for h in history if h.get("role") == "user"]
        asst_msgs = [h for h in history if h.get("role") == "assistant"]

        if user_msgs:
            mem.last_user_query = user_msgs[-1].get("content", "")
        if asst_msgs:
            mem.last_assistant_answer = asst_msgs[-1].get("content", "")
            for am in asst_msgs:
                if am.get("plan_id"):
                    mem.last_plan_id = am.get("plan_id")
                if am.get("plan_data"):
                    pd = am.get("plan_data", {})
                    mem.last_action_type = pd.get("intent")

        # Scan all user & assistant messages for file paths, folder names, and extensions
        seen_files = set()
        seen_folders = set()
        seen_exts = set()

        for msg in history:
            content = msg.get("content", "")
            if not content:
                continue

            # 1. File paths (e.g. docs/Docker.dmg, README.md, src/main.py)
            file_matches = re.findall(r"\b([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]{1,6})\b", content)
            for fm in file_matches:
                if not fm.startswith("http") and not fm.endswith((".com", ".org", ".net", ".io")):
                    seen_files.add(fm)
                    ext = "." + fm.split(".")[-1].lower()
                    seen_exts.add(ext)

            # 2. Folder mentions (e.g. folder named 'dmg', 'docs/', 'src/')
            folder_matches = re.findall(r"\b(?:folder|directory|dir)\s+(?:named|names)?\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?", content, re.I)
            for fld in folder_matches:
                if fld.lower() not in ("a", "the", "new", "this", "that"):
                    seen_folders.add(fld)

            # 3. Explored files from message metadata
            for ef in msg.get("explored_files", []):
                seen_files.add(ef)

        mem.active_files = sorted(list(seen_files))
        mem.active_folders = sorted(list(seen_folders))
        mem.active_extensions = sorted(list(seen_exts))
        mem.active_entities = mem.active_files + mem.active_folders

        if user_msgs:
            mem.active_topic = user_msgs[0].get("content", "")[:50]

        return mem

    @staticmethod
    def contextualize_query(
        query: str,
        history: List[Dict[str, Any]],
        llm_engine: Optional[Any] = None,
    ) -> str:
        """
        Rewrites/expands ambiguous follow-up queries using conversation memory.
        e.g. 'what else is in it?' -> 'what else is in project/architecture.md?'
        """
        q_lower = query.lower().strip()
        if not history or len(history) <= 1:
            return query

        # Direct retry requests
        if q_lower in ("try again", "retry", "repeat", "redo"):
            user_turns = [h for h in history[:-1] if h.get("role") == "user" and h.get("content").strip().lower() not in ("try again", "retry", "repeat", "redo")]
            if user_turns:
                return user_turns[-1].get("content", query)
            return query

        # Check for coreference pronouns: "it", "that", "this file", "them", "those", "the folder"
        has_coref = bool(re.search(r"\b(it|that|this file|the file|them|those|the folder|this folder|its)\b", q_lower))
        if not has_coref:
            return query

        mem = MemoryManager.extract_working_memory(history)
        if not mem.active_files and not mem.active_folders and not mem.last_user_query:
            return query

        # If LLM engine is active, use fast query reformulation
        if llm_engine and not getattr(llm_engine, "_is_deterministic_fallback", True):
            prompt = (
                f"<|im_start|>system\n"
                f"You are the memory coreference resolution module of MindFS. "
                f"Given the conversation context, reformulate the ambiguous follow-up query into a standalone query. "
                f"Replace pronouns like 'it', 'that', 'them' with specific file or entity names. Output ONLY the reformulated query.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Previous turn: \"{mem.last_user_query}\"\n"
                f"Referenced entities: {', '.join(mem.active_entities[:5])}\n"
                f"Follow-up query: \"{query}\"\n"
                f"Standalone query:<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            out = llm_engine.generate(prompt, max_tokens=40)
            if out:
                clean_out = out.replace("<|im_end|>", "").strip().strip('\"\'')
                if len(clean_out) > 3 and not clean_out.startswith(("<|", "{")):
                    return clean_out

        # Rule-based fallback resolution
        target_entity = mem.active_files[-1] if mem.active_files else (mem.active_folders[-1] if mem.active_folders else "")
        if target_entity:
            expanded = re.sub(r"\b(it|that|this file|the file)\b", f"'{target_entity}'", query, flags=re.I)
            return expanded

        return query

    @staticmethod
    def format_history_for_llm(
        history: List[Dict[str, Any]],
        max_turns: int = 6,
        max_chars_per_msg: int = 350,
    ) -> str:
        """
        Formats rolling conversation history for LLM chat prompt injection within token limits.
        """
        if not history:
            return ""

        past_msgs = history[:-1] if len(history) > 1 else []
        if not past_msgs:
            return ""

        recent_msgs = past_msgs[-max_turns:]
        formatted_turns: List[str] = []

        for msg in recent_msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if not content:
                continue

            # Strip bulky approval banners or details
            if content.startswith("### 📋 Proposed Action Plan:"):
                first_line = content.splitlines()[0].replace("### 📋 Proposed Action Plan:", "Action plan:").strip()
                content = first_line
            elif "<details>" in content:
                content = content.split("<details>")[0].strip()

            if len(content) > max_chars_per_msg:
                content = content[:max_chars_per_msg] + "..."

            formatted_turns.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        return "\n".join(formatted_turns)

    @staticmethod
    def build_compressed_context(
        history: List[Dict[str, Any]],
        evidence_blocks: List[str],
        query: str,
        system_prompt: str,
        total_context_limit: int = 2048,
        threshold_pct: float = 0.70,
        tokenizer: Optional[Any] = None,
        llm_engine: Optional[Any] = None,
    ) -> Tuple[str, ContextMetrics, Optional[Dict[str, Any]]]:
        """
        Calculates context window usage. If context exceeds threshold_pct, dynamically
        compresses older conversation turns and lengthy evidence chunks to stay within budget.
        """
        # 1. Base uncompressed assembly
        uncompressed_history_text = MemoryManager.format_history_for_llm(history, max_turns=8, max_chars_per_msg=500)
        uncompressed_evidence_text = "\n---\n".join(evidence_blocks) if evidence_blocks else "No files indexed."
        hist_clause = f"{uncompressed_history_text}\n" if uncompressed_history_text else ""

        raw_prompt = (
            f"{system_prompt}\n"
            f"{hist_clause}"
            f"<|im_start|>user\n"
            f"Workspace Evidence:\n{uncompressed_evidence_text}\n\nQuestion: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        raw_tokens = count_tokens(raw_prompt, tokenizer)
        threshold_tokens = int(total_context_limit * threshold_pct)

        # Under threshold: no compression needed
        if raw_tokens <= threshold_tokens:
            usage_pct = round((raw_tokens / max(1, total_context_limit)) * 100.0, 1)
            metrics = ContextMetrics(
                used_tokens=raw_tokens,
                total_tokens=total_context_limit,
                usage_pct=usage_pct,
                is_compressed=False,
                tokens_saved=0,
                compression_ratio_pct=0.0,
            )
            return raw_prompt, metrics, None

        # 2. Threshold exceeded: execute multi-stage cognitive compression!
        past_msgs = history[:-1] if len(history) > 1 else []
        summary_block = ""
        recent_history_text = ""

        # A. Condense older turns
        if len(past_msgs) > 2:
            older_msgs = past_msgs[:-2]
            latest_msgs = past_msgs[-2:]

            older_user = [m.get("content", "")[:80] for m in older_msgs if m.get("role") == "user"]
            summary_block = f"<|im_start|>system\n[Compressed Memory of Past {len(older_msgs)} Turns: User asked about {'; '.join(older_user[-2:])}]<|im_end|>\n"
            recent_history_text = MemoryManager.format_history_for_llm(latest_msgs + [history[-1]], max_turns=2, max_chars_per_msg=200)
        elif past_msgs:
            recent_history_text = MemoryManager.format_history_for_llm(past_msgs + [history[-1]], max_turns=2, max_chars_per_msg=200)

        # B. Condense evidence blocks (keep top 3 items, trim to 240 chars each)
        compressed_eb = [eb[:240] + "..." if len(eb) > 240 else eb for eb in evidence_blocks[:3]]
        compressed_evidence_text = "\n---\n".join(compressed_eb) if compressed_eb else "No files indexed."
        rec_hist_clause = f"{recent_history_text}\n" if recent_history_text else ""

        compressed_prompt = (
            f"{system_prompt}\n"
            f"{summary_block}"
            f"{rec_hist_clause}"
            f"<|im_start|>user\n"
            f"Workspace Evidence:\n{compressed_evidence_text}\n\nQuestion: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        compressed_tokens = count_tokens(compressed_prompt, tokenizer)
        saved_tokens = max(0, raw_tokens - compressed_tokens)
        saved_ratio = round((saved_tokens / max(1, raw_tokens)) * 100.0, 1) if raw_tokens > 0 else 0.0
        comp_usage_pct = round((compressed_tokens / max(1, total_context_limit)) * 100.0, 1)

        metrics = ContextMetrics(
            used_tokens=compressed_tokens,
            total_tokens=total_context_limit,
            usage_pct=comp_usage_pct,
            is_compressed=True,
            tokens_saved=saved_tokens,
            compression_ratio_pct=saved_ratio,
        )

        compression_event = {
            "uncompressed_tokens": raw_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": saved_tokens,
            "saved_ratio_pct": saved_ratio,
            "threshold_pct": round(threshold_pct * 100.0, 1),
        }

        return compressed_prompt, metrics, compression_event
