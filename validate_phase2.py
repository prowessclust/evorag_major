"""
validate_phase2.py - EvoRAG Phase 2 sanity check.

Run this after installing dependencies and building the index:
    python validate_phase2.py

Exit code 0 = PASSED, 1 = FAILED.
"""

import json
import sys
from pathlib import Path


def fail(msg: str):
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str):
    print(f"[OK]   {msg}")


def doc_key(result):
    """Return article-level identity for duplicate retrieval checks."""
    title = " ".join((result.get("title") or "").lower().split())
    return title or result.get("url_hash") or result.get("url")


print("\n" + "=" * 60)
print("  EvoRAG Phase 2 - Validation")
print("=" * 60)

# -- Step 1: Load latest chunks from disk (no network call) -------------------
from config import DATA_DIR, FAISS_INDEX_FILE, META_FILE

chunk_files = sorted(DATA_DIR.glob("chunks_*.json"), key=lambda f: f.stat().st_mtime)
if not chunk_files:
    fail(f"No chunk files in {DATA_DIR} - run fetcher.py --query first.")

latest = chunk_files[-1]
with open(latest, "r", encoding="utf-8") as f:
    chunks = json.load(f)

if not chunks:
    fail(f"{latest.name} exists but is empty.")

ok(f"Loaded {len(chunks)} chunks from {latest.name}")

# -- Step 2: Build index -------------------------------------------------------
print("\n[...] Building FAISS index (first run downloads the model ~80 MB)...")
from embedder import build_index, load_index, query as embed_query

try:
    build_index(chunks)
except Exception as e:
    fail(f"build_index() raised: {e}")

ok("build_index() completed without errors.")

# -- Step 3: Confirm files on disk --------------------------------------------
if not FAISS_INDEX_FILE.exists():
    fail(f"FAISS index not found at {FAISS_INDEX_FILE}")
ok(f"File present: {FAISS_INDEX_FILE.name}  ({FAISS_INDEX_FILE.stat().st_size // 1024} KB)")

if not META_FILE.exists():
    fail(f"Metadata sidecar not found at {META_FILE}")
ok(f"File present: {META_FILE.name}  ({META_FILE.stat().st_size // 1024} KB)")

# -- Step 4: Load index and verify vector count --------------------------------
try:
    index, metadata = load_index()
except Exception as e:
    fail(f"load_index() raised: {e}")

if index.ntotal != len(chunks):
    fail(
        f"Vector count mismatch - index has {index.ntotal} vectors "
        f"but we embedded {len(chunks)} chunks."
    )
ok(f"Index vectors = {index.ntotal}, embedding dim = {index.d}")
ok(f"Metadata entries = {len(metadata)}")

# -- Step 5: Semantic query (primary topic) ------------------------------------
# Derive the query from the chunk file name (e.g. "artificial_intelligence")
topic_raw = latest.stem  # e.g. "chunks_artificial_intelligence_20260323_163813"
parts = topic_raw.split("_")
# Strip leading "chunks" and trailing timestamp (date + time = last 2 parts)
topic_words = parts[1:-2]  # everything between "chunks_" and "_YYYYMMDD_HHMMSS"
topic_q1 = " ".join(topic_words) if topic_words else "news analysis"
topic_q2 = "global economy policy"   # a generic second query to test generalization

print(f"\n[...] Query 1: '{topic_q1}'")
try:
    results1 = embed_query(topic_q1, top_k=5, index=index, metadata=metadata)
except Exception as e:
    fail(f"query() raised on query 1: {e}")

if not results1:
    fail("Query 1 returned zero results.")
ok(f"Query 1 -> {len(results1)} results")
for i, r in enumerate(results1):
    title = (r.get("title") or "(no title)")[:60]
    print(f"       [{i}] score={r['score']:.4f} | {r['source'][:20]} | {title}")

if len({doc_key(r) for r in results1}) != len(results1):
    fail("Query 1 returned duplicate articles. Retrieval deduplication is not working.")
ok("Query 1 returned unique articles.")

print(f"\n[...] Query 2: '{topic_q2}'")
try:
    results2 = embed_query(topic_q2, top_k=5, index=index, metadata=metadata)
except Exception as e:
    fail(f"query() raised on query 2: {e}")

if not results2:
    fail("Query 2 returned zero results.")
ok(f"Query 2 -> {len(results2)} results")
for i, r in enumerate(results2):
    title = (r.get("title") or "(no title)")[:60]
    print(f"       [{i}] score={r['score']:.4f} | {r['source'][:20]} | {title}")

if len({doc_key(r) for r in results2}) != len(results2):
    fail("Query 2 returned duplicate articles. Retrieval deduplication is not working.")
ok("Query 2 returned unique articles.")

# -- Step 6: Validate result schema -------------------------------------------
required_keys = {
    "chunk_id", "source", "url", "url_hash", "title",
    "published_at", "chunk_index", "total_chunks", "word_count", "text", "score",
}
missing = [k for k in required_keys if k not in results1[0]]
if missing:
    fail(f"Result dict missing keys: {missing}")
ok(f"All required result keys present: {sorted(required_keys)}")

# -- Step 7: Score sanity check ------------------------------------------------
scores = [r["score"] for r in results1]
if any(s < 0 for s in scores):
    fail(f"Negative L2 distances found - FAISS issue: {scores}")
ok(f"Score range for query 1: {min(scores):.4f} - {max(scores):.4f} (L2, lower = better)")

print("\n" + "=" * 60)
print("  === Phase 2 PASSED ===")
print("=" * 60 + "\n")

