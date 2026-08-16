import os
import sys
import streamlit as st

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="IntelliAssist AI",
    page_icon="🤖",
    layout="wide"
)

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline import DocumentAssistant
from src.document_loader import SUPPORTED_EXTENSIONS
from src.sentiment_intent import analyze_query
"""
IntelliAssist AI - Smart Document AI Assistant
Streamlit front-end tying together: document upload, RAG chatbot, semantic
search, summarisation, sentiment/intent analysis, source citations, and
persisted conversation history (persisted within the browser session).
"""

import os
import sys
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))



load_dotenv()

st.set_page_config(
    page_title="IntelliAssist AI",
    page_icon="🧠",
    layout="wide",
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "assistant" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.assistant = DocumentAssistant(
            embedding_model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: role, content, meta

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

assistant: DocumentAssistant = st.session_state.assistant

# --------------------------------------------------------------------------
# Sidebar: upload + settings
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("🧠 IntelliAssist AI")
    st.caption("Smart Document AI Assistant")

    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name in st.session_state.ingested_files:
                continue
            # Preserve the original filename (document_loader uses the basename
            # as the doc_name for citations, so it must match what the user
            # uploaded rather than a random temp-file name).
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, uf.name)
            with open(tmp_path, "wb") as tmp:
                tmp.write(uf.getbuffer())
            try:
                with st.spinner(f"Processing {uf.name}..."):
                    n_chunks = assistant.ingest_file(tmp_path)
                st.session_state.ingested_files.add(uf.name)
                st.success(f"✅ {uf.name}: {n_chunks} chunks indexed")
            except Exception as e:
                st.error(f"Failed to process {uf.name}: {e}")
            finally:
                os.unlink(tmp_path)
                os.rmdir(tmp_dir)

    st.markdown("### 📄 Indexed Documents")
    docs = assistant.list_documents()
    if docs:
        for d in docs:
            st.write(f"- {d}")
    else:
        st.info("No documents indexed yet. Upload files above, or use the bundled sample_docs/ folder.")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=8, value=4)
    generation_note = "LLM (API key detected)" if assistant.llm is not None else "Local extractive (free, no API key)"
    st.caption(f"**Embedding model:** {assistant.embedder.name}")
    st.caption(f"**Generation mode:** {generation_note}")

    if st.button("🗑️ Clear conversation history"):
        st.session_state.history = []
        st.rerun()

# --------------------------------------------------------------------------
# Main layout: Chat | Summary & Search tools
# --------------------------------------------------------------------------

tab_chat, tab_summary, tab_compare, tab_history = st.tabs(
    ["💬 Chat", "📝 Summarize", "🔍 TF-IDF vs Semantic Search", "🕒 History"]
)

# ---------------------------------------------------------------- Chat tab
with tab_chat:
    st.subheader("Ask a question about your documents")

    if not docs:
        st.warning("Upload at least one document in the sidebar to start chatting.")

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn["role"] == "assistant" and turn.get("citations"):
                with st.expander("📌 Sources"):
                    for c in turn["citations"]:
                        st.markdown(
                            f"**{c['doc_name']}** — chunk {c['chunk_index']} "
                            f"(relevance score: {c['score']:.3f})"
                        )
                        st.caption(c["text_preview"])
            if turn["role"] == "user" and turn.get("meta"):
                meta = turn["meta"]
                st.caption(
                    f"Sentiment: **{meta['sentiment']}** ({meta['sentiment_compound']:.2f}) "
                    f"| Intent: **{meta['intent']}**"
                )

    user_query = st.chat_input("Ask a question about your uploaded documents...")

    if user_query:
        query_meta = analyze_query(user_query)
        st.session_state.history.append(
            {"role": "user", "content": user_query, "meta": query_meta}
        )
        with st.chat_message("user"):
            st.write(user_query)
            st.caption(
                f"Sentiment: **{query_meta['sentiment']}** ({query_meta['sentiment_compound']:.2f}) "
                f"| Intent: **{query_meta['intent']}**"
            )

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating an answer..."):
                result = assistant.ask(user_query, top_k=top_k)
            st.write(result.answer)
            st.caption(f"Generation mode: {result.generation_mode}")
            citations = [
                {
                    "doc_name": c.doc_name,
                    "chunk_index": c.chunk_index,
                    "score": c.score,
                    "text_preview": c.text_preview,
                }
                for c in result.citations
            ]
            if citations:
                with st.expander("📌 Sources"):
                    for c in citations:
                        st.markdown(
                            f"**{c['doc_name']}** — chunk {c['chunk_index']} "
                            f"(relevance score: {c['score']:.3f})"
                        )
                        st.caption(c["text_preview"])

        st.session_state.history.append(
            {
                "role": "assistant",
                "content": result.answer,
                "citations": citations,
                "generation_mode": result.generation_mode,
            }
        )

# ----------------------------------------------------------- Summary tab
with tab_summary:
    st.subheader("Document summarisation")
    if not docs:
        st.info("Upload a document first to summarise it.")
    else:
        selected_doc = st.selectbox("Choose a document to summarise", docs)
        num_sentences = st.slider("Summary length (sentences)", 2, 10, 5)
        if st.button("Generate summary"):
            with st.spinner("Summarising..."):
                summary = assistant.summarize_document(selected_doc, num_sentences=num_sentences)
            st.markdown("#### Summary")
            st.write(summary)
            original_len = len(assistant.doc_texts.get(selected_doc, ""))
            st.caption(
                f"Original length: {original_len} characters → "
                f"Summary length: {len(summary)} characters "
                f"({100 - int(100 * len(summary) / max(original_len, 1))}% shorter)"
            )

# --------------------------------------------------------- Comparison tab
with tab_compare:
    st.subheader("Classical (TF-IDF) vs Semantic (embedding) search")
    st.caption(
        "Runs the same query through both retrieval methods over the same document "
        "chunks, so you can compare keyword-based vs meaning-based retrieval."
    )
    if not docs:
        st.info("Upload a document first to compare retrieval methods.")
    else:
        compare_query = st.text_input("Enter a search query to compare")
        compare_k = st.slider("Results per method", 1, 8, 4, key="compare_k")
        if st.button("Compare") and compare_query.strip():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔤 TF-IDF (classical)")
                tfidf_results = assistant.tfidf_search(compare_query, top_k=compare_k)
                if not tfidf_results:
                    st.write("No matches found.")
                for r in tfidf_results:
                    st.markdown(f"**{r.chunk.doc_name}** [chunk {r.chunk.chunk_index}] — score {r.score:.3f}")
                    st.caption(r.chunk.text[:220] + ("..." if len(r.chunk.text) > 220 else ""))
            with col2:
                st.markdown(f"#### 🧠 Semantic ({assistant.embedder.name})")
                semantic_results = assistant.semantic_search(compare_query, top_k=compare_k)
                if not semantic_results:
                    st.write("No matches found.")
                for r in semantic_results:
                    st.markdown(f"**{r.chunk.doc_name}** [chunk {r.chunk.chunk_index}] — score {r.score:.3f}")
                    st.caption(r.chunk.text[:220] + ("..." if len(r.chunk.text) > 220 else ""))

# -------------------------------------------------------------- History tab
with tab_history:
    st.subheader("Full conversation history (this session)")
    if not st.session_state.history:
        st.info("No conversation yet. Ask something in the Chat tab.")
    else:
        for i, turn in enumerate(st.session_state.history):
            role_label = "🧑 You" if turn["role"] == "user" else "🤖 IntelliAssist"
            st.markdown(f"**{role_label}:** {turn['content']}")
        if st.button("Export history as text"):
            text_dump = "\n\n".join(
                f"{'You' if t['role']=='user' else 'IntelliAssist'}: {t['content']}"
                for t in st.session_state.history
            )
            st.download_button("Download", text_dump, file_name="conversation_history.txt")

st.markdown("---")
st.caption(
    "IntelliAssist AI — Smart Document AI Assistant | RAG pipeline: LangChain + FAISS + "
    "sentence-transformers (with local offline fallback) | Built for the LaunchED AI Internship Capstone Project"
)
