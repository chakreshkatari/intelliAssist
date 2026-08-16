"""
tfidf_search.py
----------------
Classical NLP baseline: a pure TF-IDF + cosine-similarity search over the
same document chunks used by the BERT/embedding-based retriever. Included
so the Evaluation Report can directly compare classical keyword-based
retrieval against modern semantic (embedding) retrieval on the same corpus
and the same queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.document_loader import Chunk


@dataclass
class TfidfResult:
    chunk: Chunk
    score: float


class TfidfSearchIndex:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.matrix = None
        self.chunks: List[Chunk] = []
        self._fitted = False

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        if not texts:
            self._fitted = False
            return
        self.matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True

    def search(self, query: str, top_k: int = 4) -> List[TfidfResult]:
        if not self._fitted or not self.chunks:
            return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix)[0]
        top_k = min(top_k, len(self.chunks))
        top_indices = similarities.argsort()[::-1][:top_k]
        results = [
            TfidfResult(chunk=self.chunks[i], score=float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0
        ]
        return results

    def is_empty(self) -> bool:
        return not self._fitted or not self.chunks
