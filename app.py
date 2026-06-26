"""
app.py — Streamlit web UI for the RAG paper Q&A system

Run:
    streamlit run app.py

Requires:
    GROQ_API_KEY set in a local .env file:
        GROQ_API_KEY="gsk_..."
"""

import os
import time

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv  # <-- Added to load environment variables

from query import retrieve, build_prompt, generate, EMBED_MODEL, CHROMA_DIR, COLLECTION

# ── Load local environment variables ──────────────────────────────────────────
load_dotenv()  # This reads your local .env file automatically

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Research Paper Q&A",
    page_icon  = "📄",
    layout     = "centered",
)

st.title("📄 Research Paper Q&A")
st.caption("Powered by sentence-transformers + ChromaDB + Groq (free)")

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    # 🛑 REMOVED: groq_key text input widget is gone.

    model_choice = st.selectbox(
        "LLM model",
        options = [
            "llama-3.3-70b-versatile",
            "gemma-7b-it",
        ],
        help = "All are free on Groq's free tier",
    )

    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=5)

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Your question is embedded locally\n"
        "2. Most similar paper chunks retrieved\n"
        "3. LLM answers using only that context\n"
        "4. Sources shown for every answer"
    )

    st.divider()
    if st.button("🔄 Reload index"):
        st.cache_resource.clear()
        st.success("Index reloaded!")


# ── Cached resource loading ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def load_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Connecting to ChromaDB…")
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        return None


model      = load_model()
collection = load_collection()

# ── Index status ──────────────────────────────────────────────────────────────

if collection is None:
    st.error(
        "No indexed documents found. Run the ingestion pipeline first:\n\n"
        "```bash\npython ingest.py --docs_dir ./papers\n```"
    )
    st.stop()

chunk_count = collection.count()
st.info(f"Index loaded: **{chunk_count} chunks** indexed", icon="✅")

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources retrieved"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**{s['source']}** — chunk {s['chunk_idx']} "
                        f"(similarity: `{s['similarity']}`)\n\n"
                        f"> {s['text'][:300]}…"
                    )

# ── Input ─────────────────────────────────────────────────────────────────────

question = st.chat_input("Ask a question about your papers…")

if question:
    # 🔄 UPDATED: Directly checking the environment for the key now
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing! Please add it to your local `.env` file.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Retrieve
    with st.spinner("Searching papers…"):
        t0     = time.perf_counter()
        chunks = retrieve(question, model, collection, top_k)
        t_ret  = time.perf_counter() - t0

    # Generate
    with st.chat_message("assistant"):
        prompt     = build_prompt(question, chunks)
        answer_box = st.empty()
        full_ans   = ""

        # Stream token-by-token into the UI
        from groq import Groq
        from query import SYSTEM_PROMPT

        # 🔄 UPDATED: Uses the key automatically sourced from os.getenv
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        t1     = time.perf_counter()
        completion = client.chat.completions.create(
            model    = model_choice,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            stream = True,
        )
        for token in completion:
            delta   = token.choices[0].delta.content or ""
            full_ans += delta
            answer_box.markdown(full_ans + "▌")   # cursor effect

        answer_box.markdown(full_ans)
        t_gen = time.perf_counter() - t1

        # Timing footer
        st.caption(
            f"Retrieval: {t_ret*1000:.0f}ms · Generation: {t_gen:.1f}s · "
            f"Model: {model_choice} · top-k: {top_k}"
        )

        # Sources expander
        with st.expander(f"📚 {len(chunks)} sources retrieved"):
            for i, c in enumerate(chunks, 1):
                st.markdown(
                    f"**{i}. {c['source']}** (chunk {c['chunk_idx']}, similarity `{c['similarity']}`)"
                )
                st.markdown(f"> {c['text'][:400]}…")
                st.divider()

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_ans,
        "sources": chunks,
    })