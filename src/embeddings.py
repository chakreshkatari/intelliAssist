"""
embeddings.py
-------------
Provides a single EmbeddingModel interface used across the app.

Default / documented path: Hugging Face sentence-transformers model
('all-MiniLM-L6-v2'), downloaded on first use from the HF Hub. This is the
model the project is designed around and is what will run on a normal
machine with internet access.

Automatic offline fallback: if the sentence-transformers model cannot be
downloaded (no internet access to huggingface.co, e.g. in a locked-down
sandbox / CI environment), the model falls back to a local, dependency-free
"LSA" embedder (TF-IDF + Truncated SVD) that produces fixed-size dense
vectors with the same .encode() interface. This keeps the whole pipeline
runnable end-to-end even without external network access, while keeping the
BERT-based path as the default for real deployments.

Both paths are genuinely exercised by this project: the sentence-transformer
path is the documented default, and the LSA fallback is what was actually
used to test this project end-to-end inside a network-restricted sandbox
(see README / Evaluation Report for details).
"""

from __future__ import annotations

import os
import pickle
from typing import List, Optional

import numpy as np


class BaseEmbedder:
    name: str = "base"
    dim: int = 0

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wraps a Hugging Face sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily

        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers/{model_name}"
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype="float32")


class LocalLSAEmbedder(BaseEmbedder):
    """
    Offline fallback embedder: TF-IDF vectors compressed with Truncated SVD
    (Latent Semantic Analysis) into a fixed-size dense vector. This is a
    classical, well-established way to get dense semantic-ish embeddings
    without any deep learning model or network access.

    NOTE: This is intentionally kept separate from the TF-IDF baseline in
    tfidf_search.py. TF-IDF search uses raw sparse TF-IDF vectors, while this
    embedder compresses them into dense vectors via SVD purely so the RAG /
    FAISS pipeline has a working embedding source with no external
    dependencies.
    """

    def __init__(self, n_components: int = 128, random_state: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self._target_components = n_components  # fixed target; never shrunk permanently
        self._random_state = random_state
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._fitted = False
        self.name = "local-tfidf-svd"
        self.dim = n_components

    def fit(self, corpus: List[str]) -> None:
        from sklearn.decomposition import TruncatedSVD

        # Always compute the component count from the fixed target and the
        # CURRENT corpus size -- not from whatever the component count
        # happened to shrink to on a previous, smaller corpus. This lets the
        # effective embedding dimensionality grow back up as more documents
        # are ingested, rather than being permanently capped by the first
        # (smallest) corpus it was ever fit on.
        n_components = min(self._target_components, max(2, len(corpus) - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=self._random_state)
        self.dim = n_components
        tfidf_matrix = self._vectorizer.fit_transform(corpus)
        self._svd.fit(tfidf_matrix)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            # Fit on-the-fly if not fitted yet (e.g. single ad-hoc query encode)
            self.fit(texts)
        tfidf_matrix = self._vectorizer.transform(texts)
        vectors = self._svd.transform(tfidf_matrix)
        # L2 normalize so cosine similarity == dot product, matching FAISS IP index usage
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        return vectors.astype("float32")


def get_embedder(model_name: str = "all-MiniLM-L6-v2", force_offline: bool = False) -> BaseEmbedder:
    """
    Factory that returns a working embedder. Tries the real Hugging Face
    sentence-transformers model first (this is the intended default for a
    normal machine with internet access); if that fails for any reason
    (no internet, model not cached, offline sandbox, etc.) it transparently
    falls back to the local LSA embedder so the app never hard-crashes.
    """
    if not force_offline:
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a graceful fallback
            import sys
            print(
                f"[embeddings] Could not load sentence-transformers model "
                f"'{model_name}' ({exc.__class__.__name__}: {exc}). "
                f"Falling back to local offline TF-IDF+SVD embedder.",
                file=sys.stderr,
            )
    return LocalLSAEmbedder()
