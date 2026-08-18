"""
rag_pipeline.py
----------------
Core Retrieval-Augmented Generation orchestration for IntelliAssist AI.

- Retrieval: FAISS vector store over document-chunk embeddings (src/vector_store.py,
  src/embeddings.py), plus a parallel TF-IDF baseline (src/tfidf_search.py) for comparison.
- Prompt construction: LangChain PromptTemplate.
- Generation: pluggable.
    * If OPENAI_API_KEY or GOOGLE_API_KEY is set in the environment, a real LLM
      (via langchain-openai's ChatOpenAI or langchain-google-genai's
      ChatGoogleGenerativeAI) is used through a LangChain LLMChain for
      natural-language answer generation grounded in the retrieved context.
    * Otherwise, a local, free, extractive generator composes an answer directly
      from the retrieved chunks (no external API calls, no cost). This is the
      default so the whole project runs without any paid API key.
- Conversational intents (greetings, thanks) bypass retrieval entirely and get a
  clean canned response instead of an irrelevant document dump.
- Low-relevance retrieval results are filtered out via a similarity threshold, so
  a query that doesn't match anything in the corpus gets an honest "not found"
  answer instead of the system inventing an answer from unrelated chunks.
- Every answer carries structured source citations (document name + chunk index)
  so the UI can show exactly which chunk(s) supported the answer.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain.prompts import PromptTemplate

from src.document_loader import Chunk, load_and_chunk, normalize_whitespace
from src.embeddings import get_embedder, BaseEmbedder
from src.vector_store import FaissVectorStore, SearchResult
from src.tfidf_search import TfidfSearchIndex
from src.sentiment_intent import analyze_query


RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are IntelliAssist, a helpful document assistant. Answer the "
        "question using ONLY the context below. If the answer is not "
        "contained in the context, say you don't have enough information "
        "in the uploaded documents to answer.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer, citing which part of the context you used where relevant:"
    ),
)

# Intents that should never trigger a document search -- these are purely
# conversational and answering them with retrieved chunks would just dump
# irrelevant document text on the user (the "greeting bug").
CONVERSATIONAL_INTENTS = {
    "greeting": (
        "Hello! 👋 I'm IntelliAssist, your document AI assistant. Upload a "
        "PDF, TXT, or DOCX file using the sidebar and ask me anything about "
        "it -- I can answer questions with source citations, summarise "
        "documents, and compare classical vs. semantic search."
    ),
    "gratitude": (
        "You're very welcome! Let me know if you have any other questions "
        "about your documents."
    ),
}

# Minimum cosine-similarity score a retrieved chunk must clear to be treated
# as genuinely relevant. Below this, results are discarded and the user gets
# an honest "nothing relevant found" answer rather than a guess built from
# unrelated chunks. Tuned against this project's embedders (both the
# sentence-transformers model and the offline TF-IDF+SVD fallback return
# L2-normalised vectors, so cosine similarity is in [-1, 1] and unrelated
# chunks typically score well under 0.15).
DEFAULT_MIN_RELEVANCE_SCORE = 0.15


@dataclass
class Citation:
    doc_name: str
    chunk_index: int
    score: float
    text_preview: str


@dataclass
class RagAnswer:
    answer: str
    citations: List[Citation]
    generation_mode: str  # "llm" | "extractive" | "conversational" | "no_match" | "none"
    query_analysis: dict = field(default_factory=dict)


def _get_llm():
    """
    Returns a LangChain chat model if a usable API key is configured, else None.
    Supports OpenAI (OPENAI_API_KEY) and Google Gemini (GOOGLE_API_KEY).
    Requires the optional langchain-openai / langchain-google-genai packages
    to be installed (see requirements.txt comments) -- these are NOT required
    for the app to run in its default, free, local mode.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            print(f"[rag_pipeline] OPENAI_API_KEY set but ChatOpenAI unavailable: {exc}")

    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"), temperature=0.2
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[rag_pipeline] GOOGLE_API_KEY set but ChatGoogleGenerativeAI unavailable: {exc}")

    return None


def _extractive_answer(question: str, retrieved: List[SearchResult]) -> str:
    """
    Free, local, no-API-key answer composer. Selects the sentences within the
    retrieved chunks that are most relevant to the question (TF-IDF cosine
    similarity) and stitches them into a short, direct answer. This keeps the
    default RAG pipeline fully functional without any paid LLM.

    Every sentence is passed through normalize_whitespace() before being
    joined, so stray newlines embedded in the source chunk (a common
    byproduct of PDF/DOCX line-wrapping) can never surface as broken,
    vertical-looking text in the rendered answer.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not retrieved:
        return (
            "I don't have enough information in the uploaded documents to "
            "answer that yet. Try uploading a relevant document first."
        )

    all_sentences = []
    for result in retrieved:
        clean_chunk_text = normalize_whitespace(result.chunk.text)
        sentences = re.split(r"(?<=[.!?])\s+", clean_chunk_text)
        for s in sentences:
            s = normalize_whitespace(s)
            if len(s) > 15:
                all_sentences.append((s, result.chunk.doc_name, result.chunk.chunk_index))

    if not all_sentences:
        return "I found related content, but could not extract a clear answer from it."

    texts = [s[0] for s in all_sentences]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts + [question])
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    except ValueError:
        sims = [0] * len(texts)

    ranked = sorted(zip(texts, sims), key=lambda x: x[1], reverse=True)
    top_sentences = [t for t, score in ranked[:3] if score > 0]

    if not top_sentences:
        # Fall back to the single most relevant chunk's opening sentence(s)
        top_sentences = [all_sentences[0][0]]

    # De-duplicate while preserving order (overlapping chunks can surface the
    # same sentence twice) and join into one clean paragraph.
    seen = set()
    deduped = []
    for s in top_sentences:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    answer = normalize_whitespace(" ".join(deduped))
    return answer


class DocumentAssistant:
    """
    Top-level orchestration object used by the Streamlit app. Owns the
    embedder, vector store, TF-IDF index, and raw per-document text, and
    exposes ingest / summarize / ask methods.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    ):
        self.embedder: BaseEmbedder = get_embedder(embedding_model_name)
        self.vector_store: Optional[FaissVectorStore] = None
        self.tfidf_index = TfidfSearchIndex()
        self.chunks: List[Chunk] = []
        self.doc_texts: dict = {}   # doc_name -> full concatenated text
        self.llm = _get_llm()
        self.min_relevance_score = min_relevance_score

    # ---------------------------------------------------------------- ingest
    def ingest_file(self, file_path: str, chunk_size: int = 800, chunk_overlap: int = 120) -> int:
        """
        Load, chunk, embed, and index a single document. Returns #chunks added.

        Raises ValueError (with a human-readable message) on anything that
        goes wrong during extraction, chunking, or embedding, so the caller
        (the Streamlit app) can surface a clear st.error() instead of an
        unhandled traceback.
        """
        new_chunks, full_text = load_and_chunk(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not new_chunks:
            raise ValueError(
                "No usable text could be extracted from this file. It may be "
                "empty, corrupted, or an image-only/scanned document."
            )

        doc_name = new_chunks[0].doc_name
        # Store the original de-duplicated full text (not the overlapping chunks)
        # so whole-document summarisation doesn't repeat sentences from chunk overlap.
        self.doc_texts[doc_name] = full_text

        self.chunks.extend(new_chunks)

        try:
            # (Re)build embeddings for all chunks. For the LSA fallback embedder
            # this ensures the TF-IDF/SVD vocabulary reflects the full corpus;
            # for the sentence-transformer embedder this is just a plain re-encode.
            if hasattr(self.embedder, "fit"):
                self.embedder.fit([c.text for c in self.chunks])  # type: ignore[attr-defined]

            all_embeddings = self.embedder.encode([c.text for c in self.chunks])
            new_store = FaissVectorStore(dim=all_embeddings.shape[1])
            new_store.add(all_embeddings, self.chunks)
            self.tfidf_index.build(self.chunks)
        except Exception as exc:
            # Roll back so a failed embedding pass doesn't leave the assistant
            # in an inconsistent state (chunks/doc_texts referencing a document
            # that never made it into the searchable index).
            self.chunks = self.chunks[: -len(new_chunks)]
            self.doc_texts.pop(doc_name, None)
            raise ValueError(f"Failed to generate embeddings for '{doc_name}': {exc}") from exc

        self.vector_store = new_store
        return len(new_chunks)

    # ------------------------------------------------------------- retrieval
    def semantic_search(self, query: str, top_k: int = 4) -> List[SearchResult]:
        if self.vector_store is None or self.vector_store.is_empty():
            return []
        query_embedding = self.embedder.encode([query])[0]
        return self.vector_store.search(query_embedding, top_k=top_k)

    def tfidf_search(self, query: str, top_k: int = 4):
        return self.tfidf_index.search(query, top_k=top_k)

    # ------------------------------------------------------------- generation
    def ask(
        self,
        question: str,
        top_k: int = 4,
        min_relevance_score: Optional[float] = None,
    ) -> RagAnswer:
        """
        Answer a user's question.

        top_k and min_relevance_score are read fresh on every call (not
        cached from __init__), so a Streamlit sidebar slider bound to these
        parameters takes effect immediately on the very next question --
        there is no stale/static value anywhere in this path.
        """
        query_analysis = analyze_query(question)
        intent = query_analysis["intent"]
        threshold = self.min_relevance_score if min_relevance_score is None else min_relevance_score

        # --- Bypass retrieval entirely for purely conversational turns ---
        # Fixes the "greeting bug": previously every message, including
        # "hi" / "hello" / "thanks", still ran a full vector search and
        # returned unrelated document chunks as if they were a real answer.
        if intent in CONVERSATIONAL_INTENTS:
            return RagAnswer(
                answer=CONVERSATIONAL_INTENTS[intent],
                citations=[],
                generation_mode="conversational",
                query_analysis=query_analysis,
            )

        if self.vector_store is None or self.vector_store.is_empty():
            return RagAnswer(
                answer=(
                    "No documents have been indexed yet. Please upload a PDF, "
                    "TXT, or DOCX file in the sidebar first."
                ),
                citations=[],
                generation_mode="none",
                query_analysis=query_analysis,
            )

        all_retrieved = self.semantic_search(question, top_k=top_k)

        # --- Relevance thresholding ---
        # Discard chunks that don't clear the minimum similarity score.
        # Without this, a question unrelated to the uploaded documents would
        # still return the "least bad" chunks and the extractive/LLM
        # generator would try to construct an answer from them -- producing
        # a confident-sounding but meaningless response instead of an honest
        # "not found".
        retrieved = [r for r in all_retrieved if r.score >= threshold]

        if not retrieved:
            return RagAnswer(
                answer=(
                    "No relevant information found in the uploaded documents "
                    "for that question. Try rephrasing, or upload a document "
                    "that covers this topic."
                ),
                citations=[],
                generation_mode="no_match",
                query_analysis=query_analysis,
            )

        citations = [
            Citation(
                doc_name=r.chunk.doc_name,
                chunk_index=r.chunk.chunk_index,
                score=r.score,
                text_preview=normalize_whitespace(
                    (r.chunk.text[:220] + "...") if len(r.chunk.text) > 220 else r.chunk.text
                ),
            )
            for r in retrieved
        ]

        context = "\n\n".join(
            f"[{r.chunk.doc_name} | chunk {r.chunk.chunk_index}] {normalize_whitespace(r.chunk.text)}"
            for r in retrieved
        )

        if self.llm is not None:
            try:
                chain = RAG_PROMPT | self.llm
                response = chain.invoke({"context": context, "question": question})
                answer_text = normalize_whitespace(getattr(response, "content", str(response)))
                return RagAnswer(
                    answer=answer_text,
                    citations=citations,
                    generation_mode="llm",
                    query_analysis=query_analysis,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[rag_pipeline] LLM generation failed, falling back to extractive: {exc}")

        answer_text = _extractive_answer(question, retrieved)
        return RagAnswer(
            answer=answer_text,
            citations=citations,
            generation_mode="extractive",
            query_analysis=query_analysis,
        )

    # ------------------------------------------------------------- summaries
    def summarize_document(self, doc_name: str, num_sentences: int = 5) -> str:
        from src.summarizer import summarize_document as _summarize

        text = self.doc_texts.get(doc_name, "")
        if not text:
            raise ValueError(f"'{doc_name}' was not found or contains no extractable text.")
        try:
            return normalize_whitespace(_summarize(text, target_sentences=num_sentences))
        except Exception as exc:
            raise ValueError(f"Failed to summarise '{doc_name}': {exc}") from exc

    def list_documents(self) -> List[str]:
        return list(self.doc_texts.keys())
