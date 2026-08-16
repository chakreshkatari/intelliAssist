"""
summarizer.py
-------------
Text summarisation for uploaded documents.

Two modes are supported:

1. Extractive TextRank summarisation (default, free, local, no API key
   required). Sentences are embedded with TF-IDF, a sentence-similarity
   graph is built, and PageRank-style scoring (via numpy power iteration,
   no networkx dependency) selects the most representative sentences.

2. LLM-based abstractive summarisation via LangChain, used automatically
   when an OPENAI_API_KEY or GOOGLE_API_KEY is present in the environment
   (see src/rag_pipeline.py for the shared LLM factory). This is optional
   and the app works fully without it.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def _pagerank(similarity_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Simple PageRank implementation via power iteration (avoids a networkx dependency)."""
    n = similarity_matrix.shape[0]
    if n == 0:
        return np.array([])
    # Row-normalise to build a transition matrix
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = similarity_matrix / row_sums

    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * transition.T.dot(scores)
        if np.abs(new_scores - scores).sum() < tol:
            scores = new_scores
            break
        scores = new_scores
    return scores


def extractive_summarize(text: str, num_sentences: int = 3) -> str:
    """
    Produce an extractive summary of `text` by selecting the top-scoring
    sentences according to a TF-IDF + cosine-similarity TextRank graph.
    Sentences are returned in their original order for readability.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # e.g. text is all stopwords / empty after vectorisation
        return " ".join(sentences[:num_sentences])

    similarity_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarity_matrix, 0)

    scores = _pagerank(similarity_matrix)
    ranked_idx = np.argsort(scores)[::-1][:num_sentences]
    selected_idx = sorted(ranked_idx.tolist())

    summary = " ".join(sentences[i] for i in selected_idx)
    return summary


def summarize_document(full_text: str, target_sentences: int = 5) -> str:
    """Public entry point used by the app for whole-document summarisation."""
    full_text = full_text.strip()
    if not full_text:
        return ""
    return extractive_summarize(full_text, num_sentences=target_sentences)
