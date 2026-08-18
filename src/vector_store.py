"""
vector_store.py
----------------
Thin wrapper around a FAISS index that stores chunk embeddings alongside
their metadata (document name, chunk index, raw text) so retrieval results
can be traced back to a source for citation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import faiss

from src.document_loader import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class FaissVectorStore:
    """
    Stores L2-normalized embeddings in a FAISS IndexFlatIP index (inner
    product on normalized vectors == cosine similarity). Keeps a parallel
    Python list of Chunk objects so results can be mapped back to their
    document + chunk index for source citation.
    """

    def __init__(self, dim: int):
        if not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise ValueError(f"Embedding dimension must be a positive integer, got {dim!r}")
        self.dim = int(dim)
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings were provided to add() -- embedding generation may have failed.")
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) must match number of chunks ({len(chunks)})"
            )
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} does not match index dim {self.dim}. "
                f"This usually means the embedder changed between ingestions -- try re-uploading "
                f"all documents."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or infinite values -- cannot index them.")

        try:
            self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"FAISS failed to index the new embeddings: {exc}") from exc
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[SearchResult]:
        if self.index.ntotal == 0:
            return []
        if top_k <= 0:
            return []

        query_embedding = np.ascontiguousarray(query_embedding, dtype="float32").reshape(1, -1)
        if query_embedding.shape[1] != self.dim:
            raise ValueError(
                f"Query embedding dim {query_embedding.shape[1]} does not match index dim {self.dim}."
            )

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, top_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results

    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []
