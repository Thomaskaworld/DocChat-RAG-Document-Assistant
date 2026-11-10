"""
DocChat · ingest.py
Load PDFs / text files -> chunk -> embed locally -> persist to ChromaDB.

Run once per document set:
    python ingest.py docs/
    python ingest.py docs/ --reset --chunk-size 500 --overlap 80

No API key is needed: embeddings run locally with all-MiniLM-L6-v2 (~80 MB, CPU is fine).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DB_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION = "docchat"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
SUPPORTED = {".pdf", ".txt", ".md"}


# --------------------------------------------------------------------------- loading
def read_pdf(path: Path) -> list[dict]:
    """One record per page, so a citation can point at a page number."""
    pages = []
    for i, page in enumerate(PdfReader(str(path)).pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "source": path.name, "page": i + 1})
    if not pages:
        print(f"  !! {path.name}: no extractable text (scanned PDF?). "
              f"Run OCR first, e.g. with docling or pytesseract.")
    return pages


def read_text(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [{"text": text, "source": path.name, "page": 1}] if text else []


def load_documents(target: Path) -> list[dict]:
    """Accepts a single file or a directory (searched recursively)."""
    files = [target] if target.is_file() else sorted(
        p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED
    )
    if not files:
        sys.exit(f"No {'/'.join(sorted(SUPPORTED))} files found under {target}")

    records: list[dict] = []
    for f in files:
        if f.suffix.lower() not in SUPPORTED:
            continue
        got = read_pdf(f) if f.suffix.lower() == ".pdf" else read_text(f)
        records.extend(got)
        print(f"  loaded {f.name}: {len(got)} page(s)")
    return records


# --------------------------------------------------------------------------- chunking
def chunk(records: list[dict], size: int, overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks: list[dict] = []
    for rec in records:
        for piece in splitter.split_text(rec["text"]):
            piece = piece.strip()
            if len(piece) < 30:            # drop headers/footers and stray page numbers
                continue
            # Deterministic id: re-ingesting the same file overwrites instead of duplicating.
            uid = hashlib.sha1(f"{rec['source']}|{rec['page']}|{piece[:120]}".encode()).hexdigest()[:16]
            chunks.append({"id": uid, "text": piece, "source": rec["source"], "page": rec["page"]})
    return chunks


# --------------------------------------------------------------------------- store
def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))
    if reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"Deleted existing collection '{COLLECTION}'.")
        except Exception:
            pass
    # cosine distance; Chroma returns distance = 1 - cosine_similarity
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest documents into ChromaDB for DocChat.")
    ap.add_argument("path", nargs="?", default="docs", help="file or folder to ingest (default: docs/)")
    ap.add_argument("--chunk-size", type=int, default=800, help="characters per chunk (default 800)")
    ap.add_argument("--overlap", type=int, default=120, help="characters repeated between chunks")
    ap.add_argument("--reset", action="store_true", help="wipe the collection before ingesting")
    args = ap.parse_args()

    if args.overlap >= args.chunk_size:
        sys.exit("overlap must be smaller than chunk-size")

    target = Path(args.path)
    if not target.exists():
        sys.exit(f"{target} does not exist. Put your PDFs in ./docs/ first.")

    print(f"Reading {target} ...")
    records = load_documents(target)
    chunks = chunk(records, args.chunk_size, args.overlap)
    if not chunks:
        sys.exit("Nothing to index — no text survived chunking.")
    print(f"{len(records)} page(s) -> {len(chunks)} chunks "
          f"(size={args.chunk_size}, overlap={args.overlap})")

    print(f"Embedding with {EMBED_MODEL} (first run downloads ~80 MB) ...")
    embedder = SentenceTransformer(EMBED_MODEL)
    vectors = embedder.encode([c["text"] for c in chunks],
                              normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    col = get_collection(reset=args.reset)
    B = 256                                            # batch the writes; Chroma dislikes huge single adds
    for i in range(0, len(chunks), B):
        part = chunks[i:i + B]
        col.upsert(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in part],
            embeddings=[v.tolist() for v in vectors[i:i + B]],
        )

    print(f"\nDone. Collection '{COLLECTION}' now holds {col.count()} chunks in ./{DB_DIR}/")
    print("Next:  python rag.py \"your question\"      or      streamlit run app.py")


if __name__ == "__main__":
    main()
