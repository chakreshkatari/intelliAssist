"""
summarizer.py
-------------
Text summarisation for uploaded documents.
Supports both LLM-based abstractive summarisation (Gemini) and local extractive summarisation fallback.
"""

from __future__ import annotations
import os
import re
from typing import List
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _get_gemini_llm():
    """Returns initialized ChatGoogleGenerativeAI if API key is present."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    if not api_key and "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]

    if api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.3
            )
        except Exception:
            return None
    return None


def _split_sentences(text: str) -> List[str]:
    """Cleans extra whitespace/slide artifacts and splits text into clean sentences."""
    clean_text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def _pagerank(similarity_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Simple PageRank implementation via power iteration."""
    n = similarity_matrix.shape[0]
    if n == 0:
        return np.array([])
    
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M = similarity_matrix / row_sums
    
    v = np.ones(n) / n
    for _ in range(max_iter):
        v_next = (1 - damping) / n + damping * M.T.dot(v)
        if np.linalg.norm(v_next - v, 1) < tol:
            break
        v = v_next
    return v


def summarize_document(text: str, target_sentences: int = 5) -> str:
    """Summarizes document using Gemini LLM if available, otherwise falls back to TextRank."""
    if not text or not text.strip():
        return "No text available to summarize."

    # 1. Try Gemini Abstractive Summarization
    llm = _get_gemini_llm()
    if llm:
        try:
            prompt = (
                f"You are an expert AI summarizer. Summarize the following document clearly "
                f"in approximately {target_sentences} key bullet points. Clean up any slide deck artifacts, "
                f"broken sentences, or awkward formatting.\n\n"
                f"Document Text:\n{text[:12000]}\n\nSummary:"
            )
            response = llm.invoke(prompt)
            return response.content
        except Exception:
            pass  # Fallback to local extractive if API call fails

    # 2. Local Extractive Fallback (TextRank)
    sentences = _split_sentences(text)
    if not sentences:
        return text[:500]
    if len(sentences) <= target_sentences:
        return "\n\n".join([f"• {s}" for s in sentences])

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)

    scores = _pagerank(sim_matrix)
    top_indices = np.argsort(scores)[::-1][:target_sentences]
    top_indices = sorted(top_indices)  # Retain original document flow

    summary_bullets = [f"• {sentences[i]}" for i in top_indices]
    return "\n\n".join(summary_bullets)
