# RAG Paper Q&A

A production-quality Retrieval-Augmented Generation (RAG) system for querying research papers.

**Stack**: sentence-transformers (local embeddings, no API key) · ChromaDB (vector store) · Groq API (free LLM inference)

---

## Setup 

### 1. Clone and install

```bash
git clone https://github.com/Aashi8810/RAG_System.git
cd rag-paper-qa
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free Groq API key

1. Sign up at https://console.groq.com (no credit card)
2. Create an API key
3. Export it:

```bash
export GROQ_API_KEY=gsk_your_key_here   # Mac/Linux
set GROQ_API_KEY=gsk_your_key_here      # Windows CMD
```

### 3. Add your papers

Put `.txt`, `.pdf`, or `.md` files in `./papers/`. For testing, you can use any research paper PDF.

### 4. Ingest (index) your papers

```bash
python ingest.py --docs_dir ./papers
```

Sample output:
```
Loading embedding model 'all-MiniLM-L6-v2' (downloads ~80 MB on first run)...
  Model loaded in 3.2s

Processing: attention_is_all_you_need.pdf
  → 12043 words → 43 chunks
  ✓ Added 43 new chunks in 1.8s

Done. ChromaDB now contains 43 total chunks (43 added this run).
```

### 5. Ask questions

**CLI:**
```bash
python query.py "What attention mechanism is used in transformers?"
python query.py "What dataset was used for training?" --top_k 6
```

**Web UI:**
```bash
streamlit run app.py
```

---

## Architecture

```
Indexing (run once):
  PDF/TXT → chunk (300 words, 50 overlap) → embed (MiniLM) → ChromaDB

Query (per question):
  question → embed → cosine search → top-5 chunks → prompt → Groq LLM → answer + sources
```

### Why these choices?

| Decision | Choice | Why |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | 80 MB, runs locally, excellent quality/speed tradeoff |
| Vector DB | ChromaDB | Zero infra, persists to disk, free forever |
| LLM | Groq (Mixtral) | 500 tok/s, free tier, no credit card |
| Chunking | Word-count sliding window | Simple, controllable overlap |

---

## Benchmarks

Run after ingesting multiple papers to compare embedding strategies:

```bash
# Compare chunk sizes
python ingest.py --chunk_size 200 --chunk_overlap 40   # smaller = more precise
python ingest.py --chunk_size 600 --chunk_overlap 100  # larger = more context
```

Metrics to track:
- Retrieval latency (logged per query)
- Similarity scores of top-k results (logged per query)
- Answer quality (manual evaluation)

---

## Project structure

```
rag-paper-qa/
├── ingest.py        # Indexing pipeline: load → chunk → embed → store
├── query.py         # Query pipeline: embed → retrieve → generate
├── app.py           # Streamlit web UI
├── requirements.txt
├── README.md
└── papers/          # Put your PDFs/TXTs here
```

---

## Free API limits (Groq)

| Model | Free RPM | Free TPM |
|---|---|---|
| Mixtral 8x7b | 30 | 5,000 |
| Llama3 70b | 30 | 6,000 |
| Llama3 8b | 30 | 30,000 |

More than enough for a portfolio demo. Limits reset every minute.

---

## Next steps (Week 5+)

- Benchmark `all-MiniLM-L6-v2` vs `multi-qa-mpnet-base-dot-v1` vs `bge-small-en`
- Add re-ranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Experiment with chunk sizes: 100 vs 300 vs 600 words
- Add hybrid search (BM25 + vector)
- Deploy to Streamlit Cloud (free)
