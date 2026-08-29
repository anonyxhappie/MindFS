"""Local LLM inference runner and evidence-grounded answer synthesizer."""

import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from mindfs.config.settings import MindFSConfig
from mindfs.retrieval.evidence import EvidenceItem, RetrievalResult


class LLMEngine:
    """Local LLM Engine supporting GGUF model execution and deterministic grounded synthesis."""

    def __init__(self, config: MindFSConfig):
        self.config = config
        self.model_path = config.llm.model_path
        self.context_tokens = config.llm.context_tokens
        self.max_output_tokens = config.llm.max_output_tokens
        self.temperature = config.llm.temperature
        self._llm = None
        self._is_deterministic_fallback = True

        self._init_backend()

    def _init_backend(self) -> None:
        if self.model_path and Path(self.model_path).exists():
            try:
                from llama_cpp import Llama
                self._llm = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.context_tokens,
                    n_threads=os.cpu_count() or 4,
                    verbose=False,
                )
                self._is_deterministic_fallback = False
            except Exception:
                self._is_deterministic_fallback = True
        else:
            self._is_deterministic_fallback = True

    def _clean_doc_text(self, text: str) -> str:
        """Removes comment markers and docstring wrappers."""
        cleaned = text.strip()
        cleaned = re.sub(r'^["\']{3}|["\']{3}$', '', cleaned).strip()
        cleaned = re.sub(r'^(#|//|/\*|\*)\s*', '', cleaned, flags=re.MULTILINE).strip()
        return cleaned

    def _extract_file_semantic_summary(self, item: EvidenceItem, query: str) -> str:
        """Extracts a clear, natural language explanation from an evidence item."""
        fn = item.relative_path
        text = item.text.strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # 1. Structured data (JSON, CSV, YAML, XML)
        if any(fn.endswith(ext) for ext in (".json", ".csv", ".yaml", ".yml", ".xml")):
            key_lines = [l for l in lines if ":" in l or "=" in l]
            if key_lines:
                preview = ", ".join(key_lines[:3])
                return f"In `{fn}`, the structured data specifies: {preview}."
            return f"In `{fn}`, structured schema records are present for this topic."

        # 2. Audio & Video
        if any(fn.endswith(ext) for ext in (".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            props = []
            for l in lines:
                if any(k in l.lower() for k in ("duration", "codec", "resolution", "channels", "sample rate", "audio track")):
                    props.append(l)
            if props:
                return f"`{fn}` ({'; '.join(props[:3])})."
            return f"`{fn}` contains media recordings relevant to the query."

        clean_raw = text
        if clean_raw.startswith("Summary: "):
            clean_raw = clean_raw[9:].strip()

        # 3. Code (Python, TypeScript, Go, Rust, C++, Java, etc.)
        if any(fn.endswith(ext) for ext in (".py", ".ts", ".js", ".go", ".rs", ".cpp", ".c", ".java")):
            # Look for leading module/class docstring
            if clean_raw.startswith(('"""', "'''")):
                doc_end = clean_raw.find('"""', 3) if clean_raw.startswith('"""') else clean_raw.find("'''", 3)
                if doc_end != -1:
                    doc = self._clean_doc_text(clean_raw[3:doc_end])
                    if doc:
                        first_sentence = doc.split("\n")[0].strip()
                        return f"`{fn}`: {first_sentence}"

            # Check class or function definitions
            defs = [l for l in lines if l.startswith(("class ", "def ", "function ", "pub fn ", "struct "))]
            if defs:
                def_name = defs[0].split("(")[0].split(":")[0].strip()
                return f"`{fn}` implements `{def_name}`."

            return f"`{fn}` contains implementation logic for this component."

        # 4. Documents & PDFs (Markdown, Text, PDF)
        filtered_lines = []
        for l in lines:
            if l.startswith("Summary: Page ") or l.startswith("Summary: "):
                if ":" in l:
                    after_colon = l.split(":", 1)[1].strip()
                    if after_colon and not after_colon.startswith("[Image-only") and not after_colon.startswith("[empty page"):
                        filtered_lines.append(after_colon)
                continue
            if "[Image-only" in l or "[empty page" in l or "[Binary or unparseable" in l:
                continue
            if len(l) > 10 and not l.startswith("```") and not l.startswith("<!--") and not l.startswith("#"):
                filtered_lines.append(l)

        if filtered_lines:
            summary_snippet = " ".join(filtered_lines[:3])
            summary_snippet = self._clean_doc_text(summary_snippet)
            if len(summary_snippet) > 280:
                summary_snippet = summary_snippet[:280] + "..."
            return f"`{fn}`: {summary_snippet}"

        return f"`{fn}` provides documented evidence on this subject."

    def synthesize_answer(self, query: str, retrieval_result: RetrievalResult) -> str:
        """
        Synthesizes a direct natural language answer with concise sources and collapsible evidence.
        """
        if not retrieval_result.has_sufficient_evidence or not retrieval_result.evidence:
            return (
                f"MindFS could not find enough indexed evidence in your workspace for: **\"{query}\"**.\n\n"
                f"**Possible reasons:**\n"
                f"• The relevant files have not been indexed yet (try running `⚡ Index Selected Folder` or `Index Workspace`).\n"
                f"• The files may be unsupported or exceeded size limits.\n"
                f"• No matching content exists in the indexed files."
            )

        unique_files = list(dict.fromkeys(item.relative_path for item in retrieval_result.evidence))
        num_files = len(unique_files)
        header = f"🧠 **MindFS found {num_files} relevant file{'s' if num_files != 1 else ''}**\n"

        # 1. LLM Model synthesis if local GGUF loaded
        direct_answer = ""
        if not self._is_deterministic_fallback and self._llm is not None:
            context_blocks = []
            for item in retrieval_result.evidence:
                prov = item.format_source_provenance()
                context_blocks.append(f"Source: {prov}\nContent:\n{item.text}")

            prompt = (
                f"System: You are MindFS, a filesystem intelligence engine. "
                f"Answer the user's question directly and concisely in natural language using ONLY the provided evidence. "
                f"Synthesize across sources if multiple files contribute. Do not invent facts.\n\n"
                f"Evidence:\n" + "\n---\n".join(context_blocks) + "\n\n"
                f"Question: {query}\n"
                f"Direct Answer:"
            )

            try:
                output = self._llm(
                    prompt,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    stop=["\nSystem:", "\nQuestion:", "\nSources:"],
                )
                direct_answer = output["choices"][0]["text"].strip()
            except Exception:
                direct_answer = ""

        # 2. Deterministic grounded synthesis fallback
        if not direct_answer:
            answer_points = []
            for item in retrieval_result.evidence:
                summary_stmt = self._extract_file_semantic_summary(item, query)
                if summary_stmt and summary_stmt not in answer_points:
                    answer_points.append(summary_stmt)

            if len(answer_points) == 1:
                direct_answer = answer_points[0]
            else:
                direct_answer = "\n\n".join(f"• {pt}" for pt in answer_points)

        # 3. Build Sources List
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
