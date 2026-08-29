"""Embedding pipeline using FastEmbed (ONNX Runtime CPU) with deterministic offline fallback."""

from typing import List, Optional
import numpy as np

from mindfs.config.settings import MindFSConfig


class EmbeddingPipeline:
    """Computes text embeddings in bounded batches using CPU ONNX models."""

    def __init__(self, config: MindFSConfig):
        self.config = config
        self.model_name = config.embedding.model_name
        self.embedding_dim = config.embedding.embedding_dim
        self.batch_size = config.embedding.batch_size
        self._model = None
        self._is_fallback = False

    def _get_model(self):
        if self._model is not None:
            return self._model
        
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
            self._is_fallback = False
        except Exception:
            # Deterministic offline lightweight bag-of-words / hash embedding fallback
            self._is_fallback = True
            self._model = "fallback"

        return self._model

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Generates deterministic pseudo-semantic embeddings when offline without models."""
        import hashlib
        import re

        def _dhash(s: str) -> int:
            return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")

        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.findall(r"[A-Za-z0-9_]+", text.lower())
            if not words:
                continue
            for word in words:
                # Word-level feature hashing
                h = _dhash(word) % self.embedding_dim
                embeddings[i, h] += 1.0
                # Sub-token n-grams
                if len(word) > 3:
                    for j in range(len(word) - 2):
                        sub_h = _dhash(word[j:j+3]) % self.embedding_dim
                        embeddings[i, sub_h] += 0.25
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        return embeddings

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts in bounded batches."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        model = self._get_model()
        if self._is_fallback:
            return self._fallback_embed(texts)

        try:
            # Use FastEmbed generator in bounded batches
            all_vectors = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                vec_gen = model.embed(batch)
                for vec in vec_gen:
                    all_vectors.append(vec)
            res = np.array(all_vectors, dtype=np.float32)
            if res.shape[1] != self.embedding_dim:
                self.embedding_dim = res.shape[1]
            return res
        except Exception:
            self._is_fallback = True
            return self._fallback_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string."""
        vectors = self.embed_texts([query])
        return vectors[0]
