"""
query.py — Embed a question → retrieve relevant chunks → generate answer via Groq

Usage:
    python query.py "What is retrieval-augmented generation?"
    python query.py "Summarise the methodology in the attention paper" --top_k 6
    python query.py "..." --model llama3-70b-8192

Environment:
    GROQ_API_KEY  — free key from https://console.groq.com (required)
"""

import argparse
import os
import sys
import time

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR     = "./chroma_db"
COLLECTION     = "papers"
EMBED_MODEL    = "all-MiniLM-L6-v2"
DEFAULT_MODEL  = "mixtral-8x7b-32768"   # fast, free on Groq
TOP_K          = 5                       # chunks to retrieve

SYSTEM_PROMPT = """You are a research assistant specialising in academic literature.
Answer the question using ONLY the context excerpts provided below.
If the context does not contain enough information, say so honestly.
Always cite the source document name(s) at the end of your answer like: [Source: filename]."""


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str, model: SentenceTransformer, collection, top_k: int) -> list[dict]:
    """Embed the question and return the top-k most similar chunks with metadata."""
    query_vec = model.encode(question).tolist()
    results   = collection.query(
        query_embeddings = [query_vec],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":       doc,
            "source":     meta.get("source", "unknown"),
            "chunk_idx":  meta.get("chunk_index", "?"),
            "similarity": round(1 - dist, 4),   # cosine distance → similarity
        })
    return chunks


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict]) -> str:
    """Assemble the context + question into an LLM prompt."""
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[Excerpt {i} | Source: {c['source']} | similarity: {c['similarity']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return f"Context excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"


# ── Generation ────────────────────────────────────────────────────────────────

def generate(prompt: str, groq_model: str, stream: bool = True) -> str:
    """Call Groq API and stream (or return) the response."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit(
            "Error: GROQ_API_KEY not set.\n"
            "  1. Sign up free at https://console.groq.com\n"
            "  2. export GROQ_API_KEY=your_key_here"
        )

    client   = Groq(api_key=api_key)
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": prompt},
    ]

    if stream:
        response_text = ""
        completion = client.chat.completions.create(
            model    = groq_model,
            messages = messages,
            stream   = True,
        )
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            response_text += delta
        print()  # newline after stream
        return response_text
    else:
        completion = client.chat.completions.create(
            model    = groq_model,
            messages = messages,
        )
        return completion.choices[0].message.content


# ── Main ──────────────────────────────────────────────────────────────────────

def ask(question: str, top_k: int = TOP_K, groq_model: str = DEFAULT_MODEL) -> dict:
    """
    Full RAG pipeline: retrieve → prompt → generate.
    Returns a dict with answer, chunks, and timing info.
    """
    # Load model + DB
    model      = SentenceTransformer(EMBED_MODEL)
    client     = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        sys.exit(
            f"Collection '{COLLECTION}' not found. Run ingest.py first:\n"
            "  python ingest.py --docs_dir ./papers"
        )

    # Retrieve
    t0     = time.perf_counter()
    chunks = retrieve(question, model, collection, top_k)
    t_ret  = time.perf_counter() - t0

    # Print retrieved sources (useful for debugging)
    print(f"\n{'─'*60}")
    print(f"Retrieved {len(chunks)} chunks in {t_ret*1000:.0f}ms")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c['source']}  (similarity {c['similarity']})")
    print(f"{'─'*60}\n")

    # Generate
    prompt = build_prompt(question, chunks)
    t1     = time.perf_counter()
    answer = generate(prompt, groq_model)
    t_gen  = time.perf_counter() - t1

    print(f"\n{'─'*60}")
    print(f"Generation: {t_gen:.1f}s  |  Model: {groq_model}")
    print(f"{'─'*60}\n")

    return {
        "answer":     answer,
        "chunks":     chunks,
        "timing":     {"retrieval_ms": round(t_ret*1000), "generation_s": round(t_gen, 2)},
        "model":      groq_model,
        "top_k":      top_k,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question about your research papers")
    parser.add_argument("question",            help="Your question in quotes")
    parser.add_argument("--top_k", type=int,   default=TOP_K,          help=f"Chunks to retrieve (default {TOP_K})")
    parser.add_argument("--model",             default=DEFAULT_MODEL,   help=f"Groq model (default {DEFAULT_MODEL})")
    args = parser.parse_args()

    ask(args.question, top_k=args.top_k, groq_model=args.model)
