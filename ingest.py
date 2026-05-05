"""
ingest.py — Load research papers → chunk → embed → store in ChromaDB

Usage:
    python ingest.py --docs_dir ./papers
    python ingest.py --docs_dir ./papers --chunk_size 400 --chunk_overlap 80

Supported file types: .txt, .pdf (via pypdf), .md
"""

import argparse
import os
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── Optional PDF support ─────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "papers"
EMBED_MODEL  = "all-MiniLM-L6-v2"   # 80 MB, fast, good quality
BATCH_SIZE   = 64                    # embeddings per batch


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text(path: Path) -> str:
    """Return plain text from .txt, .md, or .pdf files."""
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        if not PDF_SUPPORT:
            print(f"  [skip] {path.name} — install pypdf: pip install pypdf")
            return ""
        reader = PdfReader(str(path))
        pages  = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)

    print(f"  [skip] {path.name} — unsupported file type")
    return ""


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple word-count based sliding window chunker.

    Why not character-based? Word boundaries are more semantically meaningful,
    and sentence-transformers work on word sequences anyway.
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap   # slide forward with overlap
    return chunks


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest(docs_dir: str, chunk_size: int, chunk_overlap: int) -> None:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Directory not found: {docs_dir}")

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"Loading embedding model '{EMBED_MODEL}' (downloads ~80 MB on first run)...")
    t0    = time.perf_counter()
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  Model loaded in {time.perf_counter()-t0:.1f}s\n")

    # ── Connect to ChromaDB ───────────────────────────────────────────────────
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name     = COLLECTION,
        metadata = {"hnsw:space": "cosine"},   # cosine similarity for retrieval
    )

    existing_ids = set(collection.get()["ids"])
    print(f"ChromaDB: {len(existing_ids)} existing chunks in '{COLLECTION}'\n")

    # ── Process files ─────────────────────────────────────────────────────────
    files     = sorted(docs_path.rglob("*"))
    files     = [f for f in files if f.suffix.lower() in (".txt", ".md", ".pdf")]
    new_total = 0

    for file in files:
        print(f"Processing: {file.name}")
        text = extract_text(file)
        if not text.strip():
            print("  [empty] skipping\n")
            continue

        chunks = chunk_text(text, chunk_size, chunk_overlap)
        print(f"  → {len(text.split())} words → {len(chunks)} chunks")

        # Build IDs, skip already-ingested chunks (idempotent re-runs)
        ids       = [f"{file.stem}__chunk{i}" for i in range(len(chunks))]
        new_mask  = [i for i, cid in enumerate(ids) if cid not in existing_ids]
        new_chunks = [chunks[i] for i in new_mask]
        new_ids    = [ids[i]    for i in new_mask]

        if not new_chunks:
            print("  [cached] all chunks already indexed\n")
            continue

        metadatas = [{"source": file.name, "chunk_index": i} for i in new_mask]

        # Embed in batches
        t1   = time.perf_counter()
        vecs = []
        for b in range(0, len(new_chunks), BATCH_SIZE):
            batch = new_chunks[b : b + BATCH_SIZE]
            vecs.extend(model.encode(batch, show_progress_bar=False).tolist())
        elapsed = time.perf_counter() - t1

        collection.add(
            ids        = new_ids,
            documents  = new_chunks,
            embeddings = vecs,
            metadatas  = metadatas,
        )

        new_total += len(new_ids)
        print(f"  ✓ Added {len(new_ids)} new chunks in {elapsed:.1f}s\n")

    total = collection.count()
    print(f"Done. ChromaDB now contains {total} total chunks ({new_total} added this run).")
    print(f"DB saved to: {os.path.abspath(CHROMA_DIR)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest research papers into ChromaDB")
    parser.add_argument("--docs_dir",      default="./papers", help="Folder with your documents")
    parser.add_argument("--chunk_size",    type=int, default=300, help="Words per chunk (default 300)")
    parser.add_argument("--chunk_overlap", type=int, default=50,  help="Overlap between chunks (default 50)")
    args = parser.parse_args()

    ingest(
        docs_dir     = args.docs_dir,
        chunk_size   = args.chunk_size,
        chunk_overlap= args.chunk_overlap,
    )
