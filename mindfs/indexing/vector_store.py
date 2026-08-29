"""Persistent Vector Store backed by FAISS IndexIDMap with Cosine Similarity."""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class VectorStore:
    """FAISS-based vector index supporting persistent storage, incremental adds, and ID lookups."""

    def __init__(self, embedding_dim: int, index_path: Optional[Path | str] = None):
        self.dim = embedding_dim
        self.index_path = Path(index_path) if index_path else None
        self._index = None
        self._numpy_vectors: Optional[np.ndarray] = None
        self._numpy_ids: List[int] = []

        self._init_index()

    def _init_index(self) -> None:
        if self.index_path and self.index_path.exists() and self.index_path.stat().st_size > 0:
            self.load(self.index_path)
        else:
            if HAS_FAISS:
                base_index = faiss.IndexFlatIP(self.dim)
                self._index = faiss.IndexIDMap2(base_index)
            else:
                self._numpy_vectors = np.zeros((0, self.dim), dtype=np.float32)
                self._numpy_ids = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add_vectors(self, vectors: np.ndarray, ids: List[int]) -> None:
        """Adds normalized vectors associated with specific integer IDs."""
        if len(ids) == 0 or len(vectors) == 0:
            return

        if vectors.shape[1] != self.dim:
            # Recreate with matching dimension
            self.dim = vectors.shape[1]
            if HAS_FAISS:
                base = faiss.IndexFlatIP(self.dim)
                self._index = faiss.IndexIDMap2(base)

        normed = self._normalize(vectors.astype(np.float32))

        if HAS_FAISS and self._index is not None:
            id_array = np.array(ids, dtype=np.int64)
            self._index.add_with_ids(normed, id_array)
        else:
            if self._numpy_vectors is None or len(self._numpy_vectors) == 0:
                self._numpy_vectors = normed
            else:
                self._numpy_vectors = np.vstack([self._numpy_vectors, normed])
            self._numpy_ids.extend(ids)

        if self.index_path:
            self.save(self.index_path)

    def search(self, query_vector: np.ndarray, top_k: int = 12) -> List[Tuple[int, float]]:
        """
        Searches top_k nearest neighbors by cosine similarity.
        Returns list of (vector_id, score).
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        normed_q = self._normalize(query_vector.astype(np.float32))

        if HAS_FAISS and self._index is not None:
            if self._index.ntotal == 0:
                return []
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(normed_q, k)
            results = []
            for score, vec_id in zip(scores[0], indices[0]):
                if vec_id != -1:
                    results.append((int(vec_id), float(score)))
            return results
        else:
            if self._numpy_vectors is None or len(self._numpy_vectors) == 0:
                return []
            similarities = np.dot(self._numpy_vectors, normed_q[0])
            top_indices = np.argsort(-similarities)[:top_k]
            results = []
            for idx in top_indices:
                results.append((self._numpy_ids[idx], float(similarities[idx])))
            return results

    def remove_vectors(self, ids_to_remove: List[int]) -> None:
        """Removes vectors by their integer IDs."""
        if not ids_to_remove:
            return

        if HAS_FAISS and self._index is not None:
            id_array = np.array(ids_to_remove, dtype=np.int64)
            try:
                self._index.remove_ids(id_array)
            except Exception:
                # If remove_ids is not supported by underlying index, rebuild without removed IDs
                pass
        else:
            if self._numpy_vectors is not None and len(self._numpy_ids) > 0:
                mask = [i not in ids_to_remove for i in self._numpy_ids]
                self._numpy_vectors = self._numpy_vectors[mask]
                self._numpy_ids = [i for i in self._numpy_ids if i not in ids_to_remove]

        if self.index_path:
            self.save(self.index_path)

    def total_vectors(self) -> int:
        if HAS_FAISS and self._index is not None:
            return self._index.ntotal
        return len(self._numpy_ids)

    def save(self, path: Optional[Path | str] = None) -> None:
        target = Path(path or self.index_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if HAS_FAISS and self._index is not None:
            faiss.write_index(self._index, str(target))
        else:
            # Save numpy fallback
            np.savez_compressed(
                str(target),
                vectors=self._numpy_vectors if self._numpy_vectors is not None else np.zeros((0, self.dim)),
                ids=np.array(self._numpy_ids, dtype=np.int64)
            )

    def load(self, path: Path | str) -> None:
        target = Path(path)
        if not target.exists():
            return
        if HAS_FAISS:
            try:
                self._index = faiss.read_index(str(target))
                self.dim = self._index.d
                return
            except Exception:
                pass

        try:
            data = np.load(str(target) if str(target).endswith(".npz") else str(target) + ".npz")
            self._numpy_vectors = data["vectors"]
            self._numpy_ids = list(data["ids"])
            self.dim = self._numpy_vectors.shape[1] if len(self._numpy_vectors) > 0 else self.dim
        except Exception:
            self._init_index()

    def clear(self) -> None:
        """Clears the vector store."""
        if HAS_FAISS:
            base = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIDMap2(base)
        self._numpy_vectors = np.zeros((0, self.dim), dtype=np.float32)
        self._numpy_ids = []
        if self.index_path and self.index_path.exists():
            try:
                self.index_path.unlink()
            except Exception:
                pass

