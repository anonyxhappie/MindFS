"""Semantic retrieval engine implementing the 7-step evidence retrieval pipeline."""

import re
from typing import Any, Dict, List, Optional

from mindfs.config.settings import MindFSConfig
from mindfs.filesystem.sandbox import FilesystemSandbox
from mindfs.indexing.embeddings import EmbeddingPipeline
from mindfs.indexing.vector_store import VectorStore
from mindfs.resources.manager import ResourceManager
from mindfs.retrieval.evidence import EvidenceItem, RetrievalResult
from mindfs.storage.sqlite_store import SQLiteStore

STOP_WORDS = {
    "what", "is", "the", "are", "for", "and", "in", "on", "at", "to",
    "with", "about", "how", "why", "does", "do", "of", "a", "an", "this",
    "that", "these", "those", "can", "tell", "me", "find", "show", "give",
    "using", "use", "uses", "used", "from", "into", "over", "after", "before",
    "between", "under", "above", "such", "some", "any", "all", "each", "every",
    "more", "most", "other", "many", "much", "very", "also", "just", "then",
    "where", "when", "which", "who", "whom", "whose", "will", "would", "should",
    "could", "have", "has", "had", "been", "being", "were", "was",
}


class SearchEngine:
    """Orchestrates candidate retrieval, metadata filtering, deduplication, and evidence assembly."""

    def __init__(
        self,
        config: MindFSConfig,
        sandbox: FilesystemSandbox,
        store: SQLiteStore,
        vector_store: VectorStore,
        embedding_pipeline: EmbeddingPipeline,
        resource_manager: Optional[ResourceManager] = None,
    ):
        self.config = config
        self.sandbox = sandbox
        self.store = store
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline
        self.resources = resource_manager or ResourceManager(config)

    def search(
        self,
        query: str,
        path_filter: Optional[str] = None,
        file_type_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Executes the 7-step semantic retrieval pipeline:
        1. Normalization & safety check
        2. Query embedding
        3. FAISS candidate retrieval (8-12)
        4. SQLite metadata filtering
        5. Deduplication
        6. Source diversity enforcement
        7. Evidence assembly & relevance verification
        """
        norm_query = query.strip()
        if not norm_query:
            return RetrievalResult(
                query="",
                total_candidates=0,
                evidence=[],
                has_sufficient_evidence=False,
            )

        cand_limit = self.config.index.retrieval_candidates
        final_limit = limit or self.config.index.final_evidence_count

        with self.resources.track_operation(f"search:{norm_query[:30]}") as diag:
            # 1. Embed query
            query_vec = self.embedding_pipeline.embed_query(norm_query)

            # 2. Retrieve candidates from FAISS
            faiss_results = self.vector_store.search(query_vec, top_k=cand_limit)
            diag.details["raw_candidates_found"] = len(faiss_results)

            if not faiss_results:
                return RetrievalResult(
                    query=query,
                    total_candidates=0,
                    evidence=[],
                    has_sufficient_evidence=False,
                    diagnostic_info={"duration_sec": diag.duration_seconds},
                )

            # Extract content words from query for lexical relevance check
            q_words = [w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", norm_query.lower()) if w not in STOP_WORDS]

            # 3. Lookup details in SQLite & apply filters
            candidates: List[EvidenceItem] = []
            for vid, score in faiss_results:
                chunk_data = self.store.get_chunk_by_vector_id(vid)
                if not chunk_data:
                    continue

                source_path = chunk_data["source_path"]
                rel_path = self.sandbox.relative_path(source_path)

                # Path filter
                if path_filter and path_filter.lower() not in rel_path.lower():
                    continue

                # File type / category filter
                cat = chunk_data.get("category", "")
                mime = chunk_data.get("mime_type", "")
                art_type = chunk_data.get("artifact_type", "")
                if file_type_filter:
                    ft = file_type_filter.lower()
                    if ft not in cat.lower() and ft not in mime.lower() and ft not in art_type.lower() and ft not in rel_path.lower():
                        continue

                # Relevance check: if query has specific content words, ensure at least one full keyword match or high embedding similarity
                text_lower = (chunk_data["text"] + " " + rel_path).lower()
                has_kw_match = any(re.search(r"\b" + re.escape(qw) + r"\b", text_lower) for qw in q_words) if q_words else True
                
                # If content words exist but none match in chunk, require high similarity score (>= 0.60)
                if q_words and not has_kw_match and score < 0.60:
                    continue
                if score < 0.28 and not has_kw_match:
                    continue

                item = EvidenceItem(
                    chunk_id=chunk_data["chunk_id"],
                    file_id=chunk_data["file_id"],
                    source_path=source_path,
                    relative_path=rel_path,
                    artifact_type=art_type,
                    source_location=chunk_data["source_offset"],
                    similarity_score=score,
                    text=chunk_data["text"],
                    metadata=chunk_data["metadata"],
                )
                candidates.append(item)

            if not candidates:
                return RetrievalResult(
                    query=query,
                    total_candidates=0,
                    evidence=[],
                    has_sufficient_evidence=False,
                    diagnostic_info={"duration_sec": diag.duration_seconds},
                )

            # 4. Deduplicate near-identical text
            deduped: List[EvidenceItem] = []
            seen_texts = set()
            for cand in candidates:
                # Use first 80 chars as signature
                sig = cand.text.strip()[:80].lower()
                if sig not in seen_texts:
                    seen_texts.add(sig)
                    deduped.append(cand)

            # 5. Enforce source diversity (max 2 chunks from same file in final set)
            final_evidence: List[EvidenceItem] = []
            file_counts: Dict[str, int] = {}
            for item in deduped:
                cnt = file_counts.get(item.file_id, 0)
                if cnt < 2:
                    final_evidence.append(item)
                    file_counts[item.file_id] = cnt + 1
                if len(final_evidence) >= final_limit:
                    break

            # Fallback if diversity was too restrictive
            if len(final_evidence) < final_limit and len(deduped) > len(final_evidence):
                for item in deduped:
                    if item not in final_evidence:
                        final_evidence.append(item)
                    if len(final_evidence) >= final_limit:
                        break

            has_evidence = len(final_evidence) > 0

            return RetrievalResult(
                query=query,
                total_candidates=len(candidates),
                evidence=final_evidence,
                has_sufficient_evidence=has_evidence,
                diagnostic_info={
                    "duration_sec": diag.duration_seconds,
                    "peak_rss_mb": diag.peak_rss_mb,
                },
            )
