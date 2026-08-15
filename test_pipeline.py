"""
tests/test_pipeline.py
-----------------------
Automated tests proving the core pipeline works:
- document loading & chunking for PDF, TXT, DOCX
- embedding + FAISS retrieval returns relevant, correctly-cited chunks
- TF-IDF baseline retrieval works
- summarisation produces a shorter, non-empty summary
- sentiment/intent analysis on sample queries

Run with:  pytest tests/ -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.document_loader import load_and_chunk, SUPPORTED_EXTENSIONS
from src.rag_pipeline import DocumentAssistant
from src.summarizer import summarize_document
from src.sentiment_intent import analyze_sentiment, analyze_intent

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")


@pytest.fixture(scope="module")
def assistant():
    a = DocumentAssistant()
    for fname in sorted(os.listdir(SAMPLE_DIR)):
        fpath = os.path.join(SAMPLE_DIR, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            a.ingest_file(fpath)
    return a


# --------------------------------------------------------------------------
# Document loading
# --------------------------------------------------------------------------

def test_txt_loading_produces_chunks():
    chunks, full_text = load_and_chunk(os.path.join(SAMPLE_DIR, "ai_overview.txt"))
    assert len(chunks) > 0
    assert len(full_text) > 0
    assert all(c.doc_name == "ai_overview.txt" for c in chunks)


def test_docx_loading_produces_chunks():
    chunks, full_text = load_and_chunk(os.path.join(SAMPLE_DIR, "remote_work_policy.docx"))
    assert len(chunks) > 0
    assert "remote" in full_text.lower()


def test_pdf_loading_produces_chunks():
    chunks, full_text = load_and_chunk(os.path.join(SAMPLE_DIR, "climate_change.pdf"))
    assert len(chunks) > 0
    assert "climate" in full_text.lower() or "temperature" in full_text.lower()


# --------------------------------------------------------------------------
# Ingestion / indexing
# --------------------------------------------------------------------------

def test_assistant_ingests_all_sample_docs(assistant):
    docs = assistant.list_documents()
    assert set(docs) == {"ai_overview.txt", "remote_work_policy.docx", "climate_change.pdf"}
    assert len(assistant.chunks) > 0
    assert not assistant.vector_store.is_empty()
    assert not assistant.tfidf_index.is_empty()


# --------------------------------------------------------------------------
# Retrieval (semantic + TF-IDF)
# --------------------------------------------------------------------------

def test_semantic_search_retrieves_relevant_chunk(assistant):
    results = assistant.semantic_search("remote work probation period eligibility", top_k=3)
    assert len(results) > 0
    top_doc_names = [r.chunk.doc_name for r in results]
    assert "remote_work_policy.docx" in top_doc_names


def test_semantic_search_retrieves_climate_chunk(assistant):
    results = assistant.semantic_search("sea level rise since 1880", top_k=3)
    assert len(results) > 0
    assert results[0].chunk.doc_name == "climate_change.pdf"


def test_tfidf_search_retrieves_relevant_chunk(assistant):
    results = assistant.tfidf_search("VPN encryption security devices", top_k=3)
    assert len(results) > 0
    assert results[0].chunk.doc_name == "remote_work_policy.docx"


# --------------------------------------------------------------------------
# RAG answer + citations
# --------------------------------------------------------------------------

def test_rag_answer_has_citations_from_correct_document(assistant):
    result = assistant.ask("How much is the home office stipend?", top_k=3)
    assert result.answer
    assert len(result.citations) > 0
    assert any(c.doc_name == "remote_work_policy.docx" for c in result.citations)
    assert result.generation_mode in {"llm", "extractive"}


def test_rag_answer_on_empty_index_is_graceful():
    empty_assistant = DocumentAssistant()
    result = empty_assistant.ask("What is BERT?")
    assert result.citations == []
    assert "upload" in result.answer.lower() or "not" in result.answer.lower()


# --------------------------------------------------------------------------
# Summarisation
# --------------------------------------------------------------------------

def test_summary_is_shorter_than_original(assistant):
    original = assistant.doc_texts["ai_overview.txt"]
    summary = assistant.summarize_document("ai_overview.txt", num_sentences=3)
    assert len(summary) > 0
    assert len(summary) < len(original)


def test_standalone_summarizer_handles_short_text():
    text = "This is one short sentence."
    summary = summarize_document(text, target_sentences=3)
    assert summary.strip() != ""


# --------------------------------------------------------------------------
# Sentiment & intent
# --------------------------------------------------------------------------

def test_sentiment_positive():
    result = analyze_sentiment("This tool is amazing and super helpful, thank you!")
    assert result.label == "positive"


def test_sentiment_negative():
    result = analyze_sentiment("This is broken and completely useless, I hate it.")
    assert result.label == "negative"


def test_intent_summary_request():
    result = analyze_intent("Can you summarize this document for me?")
    assert result.label == "summary_request"


def test_intent_question():
    result = analyze_intent("What is the refund policy?")
    assert result.label == "question"
