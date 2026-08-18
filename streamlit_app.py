"""
IntelliAssist AI - Smart Document AI Assistant
Streamlit front-end tying together: document upload, RAG chatbot, semantic
search, summarisation, sentiment/intent analysis, source citations, and
persisted conversation history (persisted within the browser session).
"""

import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import DocumentAssistant, DEFAULT_MIN_RELEVANCE_SCORE
from src.document_loader import normalize_whitespace
from src.sentiment_intent import analyze_query

load_dotenv()

st.set_page_config(
    page_title="IntelliAssist AI",
    page_icon="🧠",
    layout="wide",
)

GENERATION_MODE_LABELS = {
    "llm": ":blue[LLM-generated]",
    "extractive": ":green[Local extractive (free)]",
    "conversational": ":violet[Conversational]",
    "no_match": ":orange[No relevant match]",
    "none": ":gray[No documents indexed]",
}

SENTIMENT_COLOR = {"positive": "green", "negative": "red", "neutral": "gray"}


# ==========================================================================
# Session state initialisation
# ==========================================================================
# Every piece of state the app depends on across reruns is initialised here,
# once, guarded by "not in st.session_state" checks. Streamlit reruns this
# entire script top-to-bottom on every interaction, so anything NOT stored in
# st.session_state (e.g. a plain local variable) would silently reset on the
# next click/keystroke -- this is what keeps chat history, indexed documents,
# and sidebar parameters stable across reruns.

if "assistant" not in st.session_state:
    try:
        with st.spinner("Loading embedding model..."):
            st.session_state.assistant = DocumentAssistant(
                embedding_model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            )
    except Exception as exc:
        st.error(
            "IntelliAssist AI failed to start up. This usually means a required "
            "package is missing or misconfigured.\n\n"
            f"**Details:** {exc}"
        )
        st.stop()

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: role, content, meta/citations

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

if "top_k" not in st.session_state:
    st.session_state.top_k = 4

if "min_relevance" not in st.session_state:
    st.session_state.min_relevance = DEFAULT_MIN_RELEVANCE_SCORE

assistant: DocumentAssistant = st.session_state.assistant


def render_citations(citations: list) -> None:
    """Render a list of citation dicts in a consistent, clean format."""
    with st.expander(f"📌 Sources ({len(citations)})", expanded=False):
        for c in citations:
            st.markdown(f"**{c['doc_name']}** — chunk {c['chunk_index']}  ·  relevance `{c['score']:.3f}`")
            st.caption(normalize_whitespace(c["text_preview"]))
            st.markdown("&nbsp;", unsafe_allow_html=False)


def render_query_meta(meta: dict) -> None:
    color = SENTIMENT_COLOR.get(meta.get("sentiment", "neutral"), "gray")
    st.caption(
        f"Sentiment: :{color}[**{meta['sentiment']}**] ({meta['sentiment_compound']:.2f})"
        f"  |  Intent: **{meta['intent']}**"
    )


# ==========================================================================
# Sidebar: upload + settings
# ==========================================================================

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

            # Preserve the original filename (document_loader uses the
            # basename as the doc_name for citations, so it must match what
            # the user uploaded rather than a random temp-file name).
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, uf.name)
            try:
                with open(tmp_path, "wb") as tmp:
                    tmp.write(uf.getbuffer())

                with st.spinner(f"Processing {uf.name}..."):
                    n_chunks = assistant.ingest_file(tmp_path)

                st.session_state.ingested_files.add(uf.name)
                st.toast(f"✅ {uf.name} indexed ({n_chunks} chunks)", icon="✅")
                st.success(f"✅ {uf.name}: {n_chunks} chunks indexed")

            except ValueError as exc:
                # Expected, "clean" failures raised deliberately by the
                # pipeline (bad file, no extractable text, embedding failure)
                # -- show the message as-is, no traceback.
                st.error(f"❌ Couldn't process **{uf.name}**: {exc}")

            except Exception as exc:  # noqa: BLE001
                # Anything unexpected: still fail loudly but keep the app alive
                # so one bad upload doesn't take down the whole session.
                st.error(f"❌ Unexpected error while processing **{uf.name}**: {exc}")

            finally:
                # Clean up the temp file/dir regardless of success or failure.
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                if os.path.isdir(tmp_dir):
                    os.rmdir(tmp_dir)

    st.markdown("### 📄 Indexed Documents")
    docs = assistant.list_documents()
    if docs:
        for d in docs:
            st.write(f"- {d}")
    else:
        st.info("No documents indexed yet. Upload files above, or use the bundled sample_docs/ folder.")

    st.divider()
    st.markdown("### ⚙️ Settings")

    # Bound directly to st.session_state via `key`, and re-read at the top of
    # every rerun -- this guarantees the retriever always uses the slider's
    # CURRENT value on the very next query, with no stale/cached top_k
    # anywhere in the call path (the earlier "slider doesn't update chunk
    # count" bug).
    st.slider(
        "Chunks to retrieve (top_k)",
        min_value=1, max_value=8, value=st.session_state.top_k, key="top_k",
        help="How many document chunks are retrieved and considered per question.",
    )

    with st.expander("🔧 Advanced"):
        st.slider(
            "Minimum relevance score",
            min_value=0.0, max_value=0.6, value=st.session_state.min_relevance, step=0.01,
            key="min_relevance",
            help=(
                "Retrieved chunks below this similarity score are treated as "
                "irrelevant and discarded, so an unrelated question gets an "
                "honest 'not found' answer instead of a guess built from the "
                "least-bad match."
            ),
        )

    generation_note = (
        "LLM (API key detected)" if assistant.llm is not None else "Local extractive (free, no API key)"
    )
    st.caption(f"**Embedding model:** {assistant.embedder.name}")
    st.caption(f"**Generation mode:** {generation_note}")
    st.caption(f"**Retrieving:** top {st.session_state.top_k} chunk(s)  ·  **min score:** {st.session_state.min_relevance:.2f}")

    st.divider()
    if st.button("🗑️ Clear conversation history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ==========================================================================
# Main layout: Chat | Summary & Search tools
# ==========================================================================

tab_chat, tab_summary, tab_compare, tab_history = st.tabs(
    ["💬 Chat", "📝 Summarize", "🔍 TF-IDF vs Semantic Search", "🕒 History"]
)

# ---------------------------------------------------------------- Chat tab
with tab_chat:
    st.subheader("Ask a question about your documents")

    if not docs:
        st.warning(
            "No documents uploaded yet. You can still say hello, but for "
            "document Q&A, upload a file in the sidebar first."
        )

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn["role"] == "assistant":
                if turn.get("generation_mode"):
                    st.caption(f"Mode: {GENERATION_MODE_LABELS.get(turn['generation_mode'], turn['generation_mode'])}")
                if turn.get("citations"):
                    render_citations(turn["citations"])
            if turn["role"] == "user" and turn.get("meta"):
                render_query_meta(turn["meta"])

    user_query = st.chat_input("Ask a question about your uploaded documents...")

    if user_query:
        query_meta = None
        with st.chat_message("user"):
            st.write(user_query)
            try:
                query_meta = analyze_query(user_query)
                render_query_meta(query_meta)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not analyse sentiment/intent for this message: {exc}")

        st.session_state.history.append({"role": "user", "content": user_query, "meta": query_meta})

        with st.chat_message("assistant"):
            result = None
            citations = []
            try:
                with st.spinner("Retrieving relevant context and generating an answer..."):
                    # top_k and min_relevance are read fresh from session_state
                    # on every single query -- always the CURRENT slider value.
                    result = assistant.ask(
                        user_query,
                        top_k=st.session_state.top_k,
                        min_relevance_score=st.session_state.min_relevance,
                    )
                st.write(result.answer)
                st.caption(f"Mode: {GENERATION_MODE_LABELS.get(result.generation_mode, result.generation_mode)}")

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
                    render_citations(citations)

            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong while answering: {exc}")

        if result is not None:
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "citations": citations,
                    "generation_mode": result.generation_mode,
                }
            )
        else:
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": "Sorry, I ran into an error trying to answer that -- please try again.",
                    "citations": [],
                    "generation_mode": "none",
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
        if st.button("Generate summary", type="primary"):
            try:
                with st.spinner("Summarising..."):
                    summary = assistant.summarize_document(selected_doc, num_sentences=num_sentences)
                st.markdown("#### Summary")
                st.write(summary)
                original_len = len(assistant.doc_texts.get(selected_doc, ""))
                summary_len = len(summary)
                compression = 100 - int(100 * summary_len / max(original_len, 1))
                st.caption(
                    f"Original length: {original_len} characters → "
                    f"Summary length: {summary_len} characters "
                    f"({compression}% shorter)"
                )
            except ValueError as exc:
                st.error(f"❌ Couldn't summarise this document: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Unexpected error while summarising: {exc}")

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
        if st.button("Compare", type="primary") and compare_query.strip():
            try:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🔤 TF-IDF (classical)")
                    tfidf_results = assistant.tfidf_search(compare_query, top_k=compare_k)
                    if not tfidf_results:
                        st.info("No matches found.")
                    for r in tfidf_results:
                        st.markdown(f"**{r.chunk.doc_name}** [chunk {r.chunk.chunk_index}] — score `{r.score:.3f}`")
                        preview = normalize_whitespace(r.chunk.text)
                        st.caption(preview[:220] + ("..." if len(preview) > 220 else ""))
                with col2:
                    st.markdown(f"#### 🧠 Semantic ({assistant.embedder.name})")
                    semantic_results = assistant.semantic_search(compare_query, top_k=compare_k)
                    if not semantic_results:
                        st.info("No matches found.")
                    for r in semantic_results:
                        st.markdown(f"**{r.chunk.doc_name}** [chunk {r.chunk.chunk_index}] — score `{r.score:.3f}`")
                        preview = normalize_whitespace(r.chunk.text)
                        st.caption(preview[:220] + ("..." if len(preview) > 220 else ""))
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Search comparison failed: {exc}")

# -------------------------------------------------------------- History tab
with tab_history:
    st.subheader("Full conversation history (this session)")
    if not st.session_state.history:
        st.info("No conversation yet. Ask something in the Chat tab.")
    else:
        for turn in st.session_state.history:
            role_label = "🧑 You" if turn["role"] == "user" else "🤖 IntelliAssist"
            st.markdown(f"**{role_label}:** {turn['content']}")
        st.divider()
        text_dump = "\n\n".join(
            f"{'You' if t['role'] == 'user' else 'IntelliAssist'}: {t['content']}"
            for t in st.session_state.history
        )
        st.download_button("⬇️ Export history as text", text_dump, file_name="conversation_history.txt")

st.divider()
st.caption(
    "IntelliAssist AI — Smart Document AI Assistant | RAG pipeline: LangChain + FAISS + "
    "sentence-transformers (with local offline fallback) | Built for the LaunchED AI Internship Capstone Project"
)
