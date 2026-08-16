"""
vector_store.py
----------------
Thin wrapper around a FAISS index that stores chunk embeddings alongside
their metadata (document name, chunk index, raw text) so retrieval results
can be traced back to a source for citation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import faiss

from document_loader import Chunk


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
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} does not match index dim {self.dim}"
            )
        self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[SearchResult]:
        if self.index.ntotal == 0:
            return []
        query_embedding = np.ascontiguousarray(query_embedding, dtype="float32").reshape(1, -1)
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
