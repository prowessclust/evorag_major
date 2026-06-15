"""
embedder.py — EvoRAG Phase 2: Embedding + Vector Store

Public API
----------
    build_index(chunks)          → None   (embeds chunks, saves index to disk)
    load_index()                 → (faiss.Index, List[dict])
    query(text, top_k)           → List[dict]   (chunk dicts + 'score' field)
    build_or_load(chunks)        → (faiss.Index, List[dict])

Usage (CLI — build index from latest chunk file):
    python embedder.py --build
    python embedder.py --query "semiconductor supply chain" --top-k 5

Usage (module — import from Phase 3):
    from embedder import query
    results = query("AI regulation policy", top_k=5)
"""

import argparse
import gc
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# EvoRAG uses PyTorch sentence-transformers; avoid accidental TensorFlow/Keras imports.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    EMBED_MODEL,
    INDEX_DIR,
    FAISS_INDEX_FILE,
    META_FILE,
    EMBED_BATCH_SIZE,
    TOP_K_DEFAULT,
    DATA_DIR,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    RETRIEVAL_MIN_CANDIDATES,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Lazy model singleton ───────────────────────────────────────────────────────
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it for the process lifetime."""
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
        log.info("Model loaded.")
    return _model


def release_model() -> None:
    """Unload the cached embedding model to free RAM before LLM generation."""
    global _model
    if _model is not None:
        log.info("Releasing embedding model to free memory for Ollama.")
        del _model
        _model = None
        gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD — embed all chunks and persist index + metadata
# ══════════════════════════════════════════════════════════════════════════════

def build_index(chunks: List[Dict]) -> None:
    """
    Embed all chunk texts with MiniLM-L6-v2, build a FAISS IndexFlatL2,
    and persist both the index and a metadata sidecar to INDEX_DIR.

    Args:
        chunks: List of chunk dicts from fetcher.fetch() — must have 'text' key.

    Side effects:
        Writes INDEX_DIR/evorag.faiss and INDEX_DIR/evorag_meta.json to disk.
        Creates INDEX_DIR if it does not exist.
    """
    if not chunks:
        raise ValueError("build_index() received an empty chunk list.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    model = _get_model()
    texts = [c["text"] for c in chunks]

    log.info(f"Embedding {len(texts)} chunks in batches of {EMBED_BATCH_SIZE}…")
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,   # L2 distance; no need to normalize
    )

    # embeddings shape: (N, D)  — D=384 for MiniLM-L6-v2
    embeddings = embeddings.astype(np.float32)
    dimension = embeddings.shape[1]

    log.info(f"Building FAISS IndexFlatL2 — {len(texts)} vectors, dim={dimension}")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # ── Persist index ──────────────────────────────────────────────────────────
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    log.info(f"Saved FAISS index → {FAISS_INDEX_FILE}")

    # ── Persist metadata sidecar ───────────────────────────────────────────────
    # Store only the fields Phase 3 needs; excludes raw_text (already cleaned → 'text')
    meta_keys = (
        "chunk_id", "source", "url", "url_hash",
        "title", "published_at", "chunk_index", "total_chunks",
        "word_count", "text",
    )
    metadata = [{k: c.get(k, "") for k in meta_keys} for c in chunks]
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log.info(f"Saved metadata sidecar → {META_FILE}  ({len(metadata)} entries)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD — restore persisted index + metadata from disk
# ══════════════════════════════════════════════════════════════════════════════

def load_index() -> Tuple[faiss.Index, List[Dict]]:
    """
    Load the persisted FAISS index and metadata sidecar from INDEX_DIR.

    Returns:
        (faiss.Index, List[dict]) — the index and the metadata list.

    Raises:
        FileNotFoundError if either file is missing (run build_index first).
    """
    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_FILE}. "
            "Run embedder.build_index(chunks) or 'python embedder.py --build' first."
        )
    if not META_FILE.exists():
        raise FileNotFoundError(
            f"Metadata sidecar not found at {META_FILE}. "
            "Run embedder.build_index(chunks) or 'python embedder.py --build' first."
        )

    log.info(f"Loading FAISS index from {FAISS_INDEX_FILE}…")
    index = faiss.read_index(str(FAISS_INDEX_FILE))

    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    log.info(f"Index loaded: {index.ntotal} vectors, dim={index.d} | "
             f"{len(metadata)} metadata entries")
    return index, metadata


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUERY — semantic similarity search
# ══════════════════════════════════════════════════════════════════════════════

def _document_keys(chunk: Dict) -> List[str]:
    """Return all stable article-level identities for deduplicating chunks."""
    keys = []

    title = " ".join(str(chunk.get("title") or "").lower().split())
    if title:
        keys.append(f"title:{title}")

    url_hash = str(chunk.get("url_hash") or "").strip().lower()
    if url_hash:
        keys.append(f"url_hash:{url_hash}")

    url = str(chunk.get("url") or "").strip().lower().rstrip("/")
    if url:
        keys.append(f"url:{url}")

    if not keys:
        keys.append(f"chunk:{chunk.get('chunk_id', id(chunk))}")
    return keys


def deduplicate_results(results: List[Dict], limit: int) -> List[Dict]:
    """Keep the highest-ranked chunk for each unique article/document."""
    unique: List[Dict] = []
    seen = set()

    for chunk in results:
        keys = _document_keys(chunk)
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        unique.append(chunk)
        if len(unique) >= limit:
            break

    return unique


def query(
    text: str,
    top_k: int = TOP_K_DEFAULT,
    index: Optional[faiss.Index] = None,
    metadata: Optional[List[Dict]] = None,
    deduplicate: bool = True,
) -> List[Dict]:
    """
    Embed `text`, search the FAISS index, and return the top-k most similar
    chunk dicts, each augmented with a 'score' field (L2 distance — lower = better).

    Args:
        text:     The query string.
        top_k:    Number of results to return (default from config).
        index:    Pre-loaded faiss.Index (optional — avoids repeated disk I/O).
        metadata: Pre-loaded metadata list (optional — must pair with `index`).
        deduplicate: When true, return top-k unique source documents instead of
            allowing multiple chunks from the same article.

    Returns:
        List of chunk dicts (up to top_k), sorted by ascending L2 distance.
        Each dict has all original chunk fields plus a float 'score' key.
    """
    if not text or not text.strip():
        raise ValueError("query() received an empty text string.")

    # Auto-load from disk if caller didn't provide pre-loaded objects
    if index is None or metadata is None:
        index, metadata = load_index()

    search_k = top_k
    if deduplicate:
        search_k = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, RETRIEVAL_MIN_CANDIDATES)

    actual_k = min(search_k, index.ntotal)
    if actual_k == 0:
        log.warning("Index is empty — no results to return.")
        return []

    model = _get_model()
    query_vec = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    distances, indices = index.search(query_vec, actual_k)  # shape: (1, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:          # FAISS returns -1 for unfilled slots
            continue
        chunk = dict(metadata[idx])
        chunk["score"] = float(dist)   # L2 distance (lower = more similar)
        results.append(chunk)

    if deduplicate:
        unique = deduplicate_results(results, limit=top_k)
        log.info(
            "Query returned %s unique documents from %s candidate chunks.",
            len(unique),
            len(results),
        )
        return unique

    return results[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONVENIENCE — build or load
# ══════════════════════════════════════════════════════════════════════════════

def build_or_load(chunks: Optional[List[Dict]] = None) -> Tuple[faiss.Index, List[Dict]]:
    """
    Load the index from disk if it exists, otherwise build it from `chunks`.

    This is the recommended entry point for Phase 3's FastAPI startup:

        index, metadata = build_or_load()

    Args:
        chunks: Required only when no persisted index exists yet.

    Returns:
        (faiss.Index, List[dict])
    """
    if FAISS_INDEX_FILE.exists() and META_FILE.exists():
        return load_index()
    if not chunks:
        raise ValueError(
            "No persisted index found and no chunks provided to build one. "
            "Pass chunks=fetch('your query') or run 'python embedder.py --build'."
        )
    build_index(chunks)
    return load_index()


# ══════════════════════════════════════════════════════════════════════════════
# 5. HELPERS — load chunks from latest data file
# ══════════════════════════════════════════════════════════════════════════════

def _load_latest_chunks() -> List[Dict]:
    """Load chunks from the most recently modified chunks_*.json in DATA_DIR."""
    files = sorted(DATA_DIR.glob("chunks_*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(
            f"No chunk files found in {DATA_DIR}. "
            "Run 'python fetcher.py --query <topic>' first."
        )
    latest = files[-1]
    log.info(f"Loading chunks from {latest.name}…")
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="EvoRAG Phase 2 — Build FAISS index or run a semantic query."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--build", "-b",
        action="store_true",
        help="Embed all chunks from the latest data/chunks_*.json and build the index.",
    )
    group.add_argument(
        "--query", "-q",
        type=str,
        metavar="TEXT",
        help="Run a semantic similarity query against the persisted index.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=TOP_K_DEFAULT,
        help=f"Number of results to return (default: {TOP_K_DEFAULT}).",
    )
    args = parser.parse_args()

    if args.build:
        chunks = _load_latest_chunks()
        log.info(f"Loaded {len(chunks)} chunks.")
        build_index(chunks)
        print(f"\n[OK] Index built and saved.")
        print(f"     Vectors : {faiss.read_index(str(FAISS_INDEX_FILE)).ntotal}")
        print(f"     Index   : {FAISS_INDEX_FILE}")
        print(f"     Metadata: {META_FILE}")

    elif args.query:
        results = query(args.query, top_k=args.top_k)
        if not results:
            print("[!] No results found.")
            sys.exit(1)

        print(f"\nQuery : '{args.query}'")
        print(f"Top-{len(results)} results (L2 distance — lower = more similar):\n")
        print(f"  {'#':<3}  {'Score':>8}  {'Source':<20}  {'Title'}")
        print("  " + "-" * 80)
        for i, r in enumerate(results):
            title = (r["title"] or "(no title)")[:55]
            source = r["source"][:18]
            print(f"  {i:<3}  {r['score']:>8.4f}  {source:<20}  {title}")
            print(f"       {r['text'][:120].strip()}…")
            print()


if __name__ == "__main__":
    main()
