"""Build a broader AI-focused chunk corpus for EvoRAG."""

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List

from config import AI_COVERAGE_QUERIES, DATA_DIR, MAX_ARTICLES_PER_SOURCE
from fetcher import fetch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks across the multi-query corpus."""
    seen = set()
    unique = []

    for chunk in chunks:
        key = (
            chunk.get("url_hash")
            or chunk.get("url")
            or f"{chunk.get('title', '')}:{chunk.get('chunk_index', '')}"
        )
        chunk_text = chunk.get("text", "")
        text_key = hash(chunk_text[:500])
        combined_key = (key, text_key)
        if combined_key in seen:
            continue
        seen.add(combined_key)
        unique.append(chunk)

    for chunk_id, chunk in enumerate(unique):
        chunk["chunk_id"] = chunk_id
    return unique


def build_ai_corpus(max_per_source: int = MAX_ARTICLES_PER_SOURCE) -> List[Dict]:
    """Fetch all configured AI coverage queries and write one combined corpus file."""
    all_chunks: List[Dict] = []

    for query in AI_COVERAGE_QUERIES:
        log.info("Fetching coverage query: %s", query)
        chunks = fetch(query, max_per_source=max_per_source)
        for chunk in chunks:
            chunk["coverage_query"] = query
        all_chunks.extend(chunks)

    unique_chunks = _dedupe_chunks(all_chunks)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"chunks_ai_corpus_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(unique_chunks, handle, ensure_ascii=False, indent=2)

    log.info("Saved combined AI corpus: %s chunks -> %s", len(unique_chunks), out_path)
    return unique_chunks


def main() -> None:
    """Run the AI corpus builder from the command line."""
    parser = argparse.ArgumentParser(description="Build a broad AI corpus for EvoRAG.")
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=MAX_ARTICLES_PER_SOURCE,
        help=f"Max articles per source for each coverage query (default: {MAX_ARTICLES_PER_SOURCE}).",
    )
    args = parser.parse_args()
    chunks = build_ai_corpus(max_per_source=args.max_per_source)
    print(f"\n[OK] Combined AI corpus chunks: {len(chunks)}")
    print("[OK] Next run: python embedder.py --build")


if __name__ == "__main__":
    main()
