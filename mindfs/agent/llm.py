"""Local LLM inference runner supporting Apple Silicon MLX models, GGUF, and robust grounded synthesis."""

import gc
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from mindfs.config.settings import MindFSConfig
from mindfs.retrieval.evidence import EvidenceItem, RetrievalResult


class LLMEngine:
    """Local LLM Engine supporting Apple Silicon MLX, GGUF models, and evidence synthesis."""

    def __init__(self, config: MindFSConfig):
        self.config = config
        self.model_path = config.llm.model_path
        self.context_tokens = config.llm.context_tokens
        self.max_output_tokens = config.llm.max_output_tokens
        self.temperature = config.llm.temperature
        self._llm = None
        self._mlx_model = None
        self._is_deterministic_fallback = True
        self.active_model_name: Optional[str] = None
        self.last_context_metrics: Optional[Any] = None
        self.last_compression_event: Optional[Dict[str, Any]] = None

        self._init_backend()

    @staticmethod
    def _clean_model_name(p_str: str) -> str:
        if "models--" in p_str:
            return p_str.split("models--")[-1].split("/snapshots")[0].replace("--", "/")
        p = Path(p_str)
        return p.stem if p.is_file() else p.name

    def _init_backend(self) -> None:
        """Initializes MLX or GGUF backend for the configured model_path."""
        self._llm = None
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._is_deterministic_fallback = True

        if not self.model_path:
            return

        p = Path(self.model_path).expanduser().resolve()
        if not p.exists():
            return

        # 1. Try MLX / HuggingFace Backend (for directories or safetensors models)
        if p.is_dir() or "safetensors" in p.name.lower():
            try:
                import mlx_lm
                self._mlx_model, self._mlx_tokenizer = mlx_lm.load(str(p))
                self._is_deterministic_fallback = False
                self.active_model_name = self._clean_model_name(str(p))
                return
            except Exception:
                pass

        # 2. Try Llama.cpp for GGUF files
        if p.is_file() and p.suffix.lower() in (".gguf", ".bin"):
            try:
                from llama_cpp import Llama
                self._llm = Llama(
                    model_path=str(p),
                    n_ctx=self.context_tokens,
                    n_threads=os.cpu_count() or 4,
                    verbose=False,
                )
                self._is_deterministic_fallback = False
                self.active_model_name = self._clean_model_name(str(p))
                return
            except Exception:
                pass

            try:
                import mlx_lm
                self._mlx_model, self._mlx_tokenizer = mlx_lm.load(str(p))
                self._is_deterministic_fallback = False
                self.active_model_name = self._clean_model_name(str(p))
                return
            except Exception:
                pass

        self._is_deterministic_fallback = True

    def reason_and_simplify_query(self, user_query: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Dynamically analyzes, deconstructs, and simplifies the user's intent and context.
        Uses active LLM if loaded, with intelligent fallback.
        """
        q_lower = user_query.lower().strip()
        last_turn_text = ""
        if history and len(history) > 0:
            user_turns = [h for h in history[:-1] if h.get("role") == "user"]
            if not user_turns and history:
                user_turns = [h for h in history if h.get("role") == "user" and h.get("content").strip().lower() != q_lower]
            if user_turns:
                last_turn_text = user_turns[-1].get("content", "")

        # If LLM is active, run prompt to deconstruct request
        if not self._is_deterministic_fallback:
            ctx_clause = f"Previous user context: \"{last_turn_text}\"\n" if last_turn_text else ""
            prompt = (
                f"<|im_start|>system\n"
                f"You are the cognitive reasoning module of MindFS AI assistant. "
                f"In 1 concise, clear sentence, analyze and simplify what the user is requesting. "
                f"Resolve references like 'it', 'try again', or implied file operations.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{ctx_clause}"
                f"User query: \"{user_query}\"\n"
                f"Deconstruct request:<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            out = self.generate(prompt, max_tokens=70)
            if out:
                clean = out.replace("<|im_end|>", "").strip()
                if len(clean) > 5 and not clean.startswith(("<|", "{")):
                    return clean

        # Context-aware deterministic breakdown fallback
        if q_lower in ("try again", "retry", "repeat", "redo"):
            if last_turn_text:
                return f"Re-evaluating previous intent: '{last_turn_text}'. Re-running workspace discovery and execution plan."
            return "Re-evaluating workspace state and re-running recent operation."

        if any(w in q_lower for w in ("move", "mv", "relocate")):
            return f"User requested relocating files in the workspace matching '{user_query}'. Identifying source files and target directory."

        if any(w in q_lower for w in ("delete", "remove", "trash", "rm")):
            return f"User requested deleting file entities from workspace. Formulating safety snapshot to trash before execution."

        if any(w in q_lower for w in ("summarise", "summarize", "overview", "what is in this")):
            return f"User requested a high-level architectural overview of workspace components and file inventory."

        if any(w in q_lower for w in ("list", "count", "how many")):
            return f"User requested inventory metadata and counts for workspace files."

        return f"Analyzing query '{user_query}' to locate relevant files, semantic vector embeddings, and filesystem entities."

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generates text from the active local model if loaded."""
        tokens_limit = max_tokens or self.max_output_tokens
        
        # 1. MLX Backend
        if not self._is_deterministic_fallback and self._mlx_model is not None:
            try:
                import mlx_lm
                output = mlx_lm.generate(
                    self._mlx_model,
                    self._mlx_tokenizer,
                    prompt=prompt,
                    max_tokens=tokens_limit,
                    verbose=False,
                )
                return output.strip()
            except Exception:
                pass

        # 2. Llama.cpp Backend
        if not self._is_deterministic_fallback and self._llm is not None:
            try:
                out = self._llm(
                    prompt,
                    max_tokens=tokens_limit,
                    temperature=self.temperature,
                )
                return out["choices"][0]["text"].strip()
            except Exception:
                pass

        return ""

    def _clean_doc_text(self, text: str) -> str:
        """Removes comment markers, docstrings, and artifacts."""
        cleaned = text.strip()
        cleaned = re.sub(r'^["\']{3}|["\']{3}$', '', cleaned).strip()
        cleaned = re.sub(r'^(#+|//|/\*|\*)\s*', '', cleaned, flags=re.MULTILINE).strip()
        return cleaned

    def _extract_clean_summary_from_evidence(self, item: EvidenceItem) -> Optional[str]:
        """Extracts a grammatically complete, meaningful summary statement from evidence."""
        fn = item.relative_path
        text = item.text.strip()
        if not text or text in (";", ":", "...", "{", "}"):
            return None

        lines = [l.strip() for l in text.splitlines() if l.strip() and l.strip() not in (";", ":", "...", "{", "}")]

        # 1. Structured data (JSON, YAML, CSV)
        if any(fn.endswith(ext) for ext in (".json", ".csv", ".yaml", ".yml", ".toml")):
            key_lines = [l for l in lines if (":" in l or "=" in l) and not l.startswith(("{", "}", "[", "]"))]
            if key_lines:
                preview = ", ".join(key_lines[:3])
                return f"`{fn}`: contains configuration & structured records ({preview})"
            return f"`{fn}`: structured configuration schema data"

        # 2. Images & Screenshots
        if any(fn.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".svg", ".webp")):
            for l in lines:
                if "containing text:" in l:
                    ocr_part = l.split("containing text:", 1)[1].strip()
                    if ocr_part and len(ocr_part) > 3:
                        return f"`{fn}`: screenshot containing visual text *\"{ocr_part[:120]}\"*"
                if l.startswith("Image '") and "pixels" in l:
                    return f"`{fn}`: UI graphic asset ({l})"
            return f"`{fn}`: UI screenshot & image asset"

        # 3. Audio & Video
        if any(fn.endswith(ext) for ext in (".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav")):
            props = [l for l in lines if any(k in l.lower() for k in ("duration", "resolution", "codec", "audio track"))]
            if props:
                return f"`{fn}`: media recording ({', '.join(props[:2])})"
            return f"`{fn}`: media recording"

        # 4. Source Code (Python, TypeScript, JS, etc.)
        if any(fn.endswith(ext) for ext in (".py", ".ts", ".js", ".go", ".rs", ".cpp", ".c", ".java")):
            # Look for docstring
            clean_raw = text
            if clean_raw.startswith(('"""', "'''")):
                doc_end = clean_raw.find('"""', 3) if clean_raw.startswith('"""') else clean_raw.find("'''", 3)
                if doc_end != -1:
                    doc = self._clean_doc_text(clean_raw[3:doc_end])
                    first_sentence = doc.split("\n")[0].strip()
                    if first_sentence and len(first_sentence) > 5:
                        return f"`{fn}`: {first_sentence}"
            # Check function/class defs
            defs = [l for l in lines if l.startswith(("class ", "def ", "export function ", "function ", "pub fn "))]
            if defs:
                def_name = defs[0].split("(")[0].split(":")[0].strip()
                return f"`{fn}`: implements `{def_name}`"
            return f"`{fn}`: implementation module"

        # 5. Documents & Specifications (Markdown, PDF, Text)
        valid_sentences = []
        for l in lines:
            if l.startswith("Summary:"):
                l = l[8:].strip()
            clean_l = re.sub(r'^[#\-\*\s>]+', '', l).strip()
            # Ignore broken partial words or short noise
            if len(clean_l) >= 15 and not clean_l.endswith((":", ";", ",")) and not clean_l.startswith(("n ", "g ", ";")):
                valid_sentences.append(clean_l)

        if valid_sentences:
            chosen = valid_sentences[0]
            if len(chosen) > 160:
                chosen = chosen[:160] + "..."
            return f"`{fn}`: {chosen}"

        return f"`{fn}`: document containing workspace specifications"

    def synthesize_answer(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Synthesizes a direct, comprehensive natural language response with multi-turn memory."""
        from mindfs.agent.memory import MemoryManager

        if not retrieval_result.has_sufficient_evidence or not retrieval_result.evidence:
            return (
                f"MindFS could not find enough indexed evidence in your workspace for: **\"{query}\"**.\n\n"
                f"**Suggestions:**\n"
                f"• Index additional folders or files from the Files tab.\n"
                f"• Check that relevant files are supported and within size limits."
            )

        unique_files = list(dict.fromkeys(item.relative_path for item in retrieval_result.evidence))
        num_files = len(unique_files)
        model_tag = f" ({self.active_model_name})" if (not self._is_deterministic_fallback and self.active_model_name) else ""
        header = f"🧠 **MindFS found {num_files} relevant file{'s' if num_files != 1 else ''}**{model_tag}\n"

        direct_answer = ""

        # 1. Real Local Model Generation with Multi-Turn Conversational Memory & Dynamic Compression
        context_blocks = []
        if retrieval_result and retrieval_result.evidence:
            for item in retrieval_result.evidence[:5]:
                prov = item.format_source_provenance()
                context_blocks.append(f"Source: {prov}\n{item.text[:600]}")

        system_prompt = (
            "<|im_start|>system\n"
            "You are MindFS, an intelligent filesystem assistant with persistent memory across conversation turns. "
            "Answer the user's question directly, clearly, and concisely in natural language using the provided evidence and conversation context. "
            "Synthesize across files. Provide meaningful complete sentences. Do not invent facts.<|im_end|>"
        )

        prompt, metrics, comp_event = MemoryManager.build_compressed_context(
            history=history or [],
            evidence_blocks=context_blocks,
            query=query,
            system_prompt=system_prompt,
            total_context_limit=self.context_tokens or 2048,
            threshold_pct=getattr(self.config.llm, "compression_threshold_pct", 0.70),
            tokenizer=self._mlx_tokenizer,
            llm_engine=self,
        )
        self.last_context_metrics = metrics
        self.last_compression_event = comp_event

        if not self._is_deterministic_fallback:
            gen = self.generate(prompt, max_tokens=self.max_output_tokens)
            if gen:
                clean_gen = gen.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
                if clean_gen and len(clean_gen) > 10:
                    direct_answer = clean_gen

        # 2. Coherent Grounded Synthesis Fallback
        if not direct_answer:
            answer_points = []
            seen_files = set()
            for item in retrieval_result.evidence:
                if item.relative_path in seen_files:
                    continue
                summary_stmt = self._extract_clean_summary_from_evidence(item)
                if summary_stmt:
                    seen_files.add(item.relative_path)
                    answer_points.append(summary_stmt)

            if answer_points:
                direct_answer = "\n\n".join(f"• {pt}" for pt in answer_points)
            else:
                direct_answer = f"Found relevant files in workspace for '{query}'."

        # 3. Sources List
        sources_lines = ["\n**Sources**"]
        seen_provs = set()
        for item in retrieval_result.evidence:
            prov = item.format_source_provenance()
            if prov not in seen_provs:
                seen_provs.add(prov)
                sources_lines.append(f"• {prov}")

        sources_section = "\n".join(sources_lines)

        # 4. Collapsible View Evidence Block
        evidence_lines = [
            "\n<details>",
            "<summary><strong>View evidence ▾</strong></summary>\n",
        ]
        for idx, item in enumerate(retrieval_result.evidence, 1):
            prov = item.format_source_provenance()
            evidence_lines.append(f"#### [{idx}] {prov} (Score: {round(item.similarity_score, 3)})")
            clean_snippet = item.text.strip()
            if len(clean_snippet) > 600:
                clean_snippet = clean_snippet[:600] + "\n..."
            evidence_lines.append(f"```text\n{clean_snippet}\n```\n")
        evidence_lines.append("</details>")
        evidence_section = "\n".join(evidence_lines)

        return f"{header}\n{direct_answer}\n{sources_section}\n{evidence_section}"
