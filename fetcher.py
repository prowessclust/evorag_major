"""
fetcher.py — EvoRAG Phase 1: Data Ingestion

Usage (CLI):
    python fetcher.py --query "artificial intelligence"
    python fetcher.py --query "climate change" --max-per-source 5

Usage (module):
    from fetcher import fetch
    chunks = fetch("artificial intelligence")   # returns List[dict]
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from config import (
    DATA_DIR,
    MAX_ARTICLES_PER_SOURCE,
    MAX_CHUNK_FILES,
    CHUNK_MIN_WORDS,
    CHUNK_MAX_WORDS,
    CHUNK_OVERLAP_WORDS,
    CHUNK_TARGET_WORDS,
    RSS_FEEDS,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Load .env once at import time
load_dotenv()

# Browser-like headers so sites don't block us
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ══════════════════════════════════════════════════════════════════════════════
# 0. FULL-TEXT FETCHER — get the real article body from the source URL
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_full_text(url: str, fallback: str = "") -> str:
    """
    GET the article URL and extract body text via BeautifulSoup.
    Returns fallback string on any error (paywall, timeout, JS-only page, etc.).
    """
    if not url:
        return fallback
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=12, allow_redirects=True)
        resp.raise_for_status()
        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception:
            soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script, style, nav, header, footer, ads
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        # Prefer <article> or <main> blocks — they hold the real content
        body = soup.find("article") or soup.find("main") or soup.find("body")
        if body:
            text = body.get_text(separator=" ")
        else:
            text = soup.get_text(separator=" ")

        # Quick sanity check — if we got less than 100 words, fallback
        if len(text.split()) < 100:
            return fallback
        return text
    except Exception as e:
        log.debug(f"Full-text fetch failed for {url}: {e}")
        return fallback


# ══════════════════════════════════════════════════════════════════════════════
# 1. FETCH — retrieve raw articles from each source
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_newsapi(query: str, max_results: int) -> List[Dict]:
    """Fetch articles from NewsAPI /everything endpoint."""
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        log.warning("NEWSAPI_KEY not set — skipping NewsAPI.")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        log.info(f"NewsAPI: fetched {len(articles)} articles — fetching full text...")
        results = []
        for a in articles:
            snippet = (a.get("content") or "") + " " + (a.get("description") or "")
            article_url = a.get("url", "")
            full_text = _fetch_full_text(article_url, fallback=snippet)
            results.append({
                "source": "newsapi",
                "url": article_url,
                "title": a.get("title", ""),
                "published_at": a.get("publishedAt", ""),
                "raw_text": full_text,
            })
        return results
    except Exception as e:
        log.error(f"NewsAPI error: {e}")
        return []


def _fetch_gnews(query: str, max_results: int) -> List[Dict]:
    """Fetch articles from GNews /search endpoint."""
    api_key = os.getenv("GNEWS_KEY", "")
    if not api_key:
        log.warning("GNEWS_KEY not set — skipping GNews.")
        return []

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "max": max_results,
        "apikey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        log.info(f"GNews: fetched {len(articles)} articles — fetching full text...")
        results = []
        for a in articles:
            snippet = (a.get("content") or "") + " " + (a.get("description") or "")
            article_url = a.get("url", "")
            full_text = _fetch_full_text(article_url, fallback=snippet)
            results.append({
                "source": "gnews",
                "url": article_url,
                "title": a.get("title", ""),
                "published_at": a.get("publishedAt", ""),
                "raw_text": full_text,
            })
        return results
    except Exception as e:
        log.error(f"GNews error: {e}")
        return []


def _fetch_rss(query: str, max_results: int) -> List[Dict]:
    """
    Parse RSS feeds and return articles whose title/summary mention the query.
    Falls back to returning the most-recent articles if no keyword match.
    """
    query_terms = set(query.lower().split())
    all_results = []

    for feed_info in RSS_FEEDS:
        name = feed_info["name"]
        feed_url = feed_info["url"]
        try:
            feed = feedparser.parse(feed_url)
            entries = feed.entries[:max_results * 3]  # fetch more, then filter
            matched = []
            for entry in entries:
                combined = (
                    (entry.get("title") or "") + " " +
                    (entry.get("summary") or "") + " " +
                    (entry.get("description") or "")
                ).lower()
                if any(term in combined for term in query_terms):
                    matched.append(entry)

            # If no keyword match, just take the most recent ones
            chosen = matched[:max_results] if matched else entries[:max_results]
            log.info(f"RSS {name}: {len(chosen)} articles "
                     f"({'filtered' if matched else 'unfiltered'}).")

            for entry in chosen:
                snippet = (
                    (entry.get("summary") or "") + " " +
                    (entry.get("description") or "")
                )
                article_url = entry.get("link", "")
                full_text = _fetch_full_text(article_url, fallback=snippet)
                published = ""
                if hasattr(entry, "published"):
                    published = entry.published
                all_results.append({
                    "source": f"rss:{name}",
                    "url": article_url,
                    "title": entry.get("title", ""),
                    "published_at": published,
                    "raw_text": full_text,
                })
        except Exception as e:
            log.error(f"RSS {name} error: {e}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN — strip HTML, normalize whitespace
# ══════════════════════════════════════════════════════════════════════════════

def _clean_text(raw: str) -> str:
    """Remove HTML tags, collapse whitespace, strip boilerplate patterns."""
    if not raw:
        return ""

    # Remove HTML — prefer lxml for speed, fall back to stdlib html.parser
    try:
        soup = BeautifulSoup(raw, "lxml")
    except Exception:
        soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator=" ")


    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove common boilerplate phrases
    boilerplate_patterns = [
        r"\[?\+?\d+ chars?\]?",           # "[+1234 chars]" NewsAPI truncation marker
        r"Sign up for.*?newsletter",
        r"Subscribe to.*?updates",
        r"Read more at.*",
        r"©.*?\d{4}",
        r"All rights reserved.*",
        r"Cookie Policy.*",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# 3. DEDUPLICATE — by URL SHA-256 hash
# ══════════════════════════════════════════════════════════════════════════════

def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _deduplicate(articles: List[Dict]) -> List[Dict]:
    """Remove articles with duplicate URLs (keeps first occurrence)."""
    seen_hashes = set()
    unique = []
    for article in articles:
        h = _url_hash(article.get("url", ""))
        if h not in seen_hashes and article.get("url"):
            seen_hashes.add(h)
            article["url_hash"] = h
            unique.append(article)
    log.info(f"Deduplication: {len(articles)} → {len(unique)} unique articles.")
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# 4. CHUNK — sliding window (300–400 words, 50-word overlap)
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str) -> List[str]:
    """
    Split text into overlapping chunks of ~CHUNK_TARGET_WORDS words.
    Returns a list of chunk strings; skips chunks under CHUNK_MIN_WORDS.
    """
    words = text.split()
    if not words:
        return []

    step = CHUNK_TARGET_WORDS - CHUNK_OVERLAP_WORDS  # advance by this many words
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_TARGET_WORDS, len(words))
        chunk_words = words[start:end]

        # If the final chunk is too small and we already have chunks, extend last one
        if len(chunk_words) < CHUNK_MIN_WORDS and chunks:
            # Merge into last chunk instead of creating a tiny leftover
            last_words = chunks[-1].split()
            merged = " ".join(last_words + chunk_words[-CHUNK_OVERLAP_WORDS:])
            chunks[-1] = merged
            break

        if len(chunk_words) >= CHUNK_MIN_WORDS:
            chunks.append(" ".join(chunk_words))

        if end == len(words):
            break
        start += step

    return chunks


# Maximum chunks taken from any single article (identified by url_hash)
MAX_CHUNKS_PER_ARTICLE = 4


def _build_chunks(articles: List[Dict]) -> List[Dict]:
    """Turn cleaned articles into chunk records with full metadata.

    Applies a per-article chunk cap (MAX_CHUNKS_PER_ARTICLE) so that no single
    article can dominate the retrieval corpus.
    """
    all_chunks = []
    chunk_id_counter = 0
    articles_capped = 0  # how many articles hit the cap

    for article in articles:
        text = article.get("clean_text", "")
        if not text:
            continue

        raw_chunks = _chunk_text(text)
        total = len(raw_chunks)

        # Apply per-article cap
        if total > MAX_CHUNKS_PER_ARTICLE:
            articles_capped += 1
            log.debug(
                f"Cap applied to '{article['title'][:60]}': "
                f"{total} chunks → {MAX_CHUNKS_PER_ARTICLE}"
            )
        capped_chunks = raw_chunks[:MAX_CHUNKS_PER_ARTICLE]

        for idx, chunk_text in enumerate(capped_chunks):
            word_count = len(chunk_text.split())
            all_chunks.append({
                "chunk_id":     chunk_id_counter,
                "source":       article["source"],
                "url":          article["url"],
                "url_hash":     article["url_hash"],
                "title":        article["title"],
                "published_at": article["published_at"],
                "chunk_index":  idx,
                "total_chunks": min(total, MAX_CHUNKS_PER_ARTICLE),
                "word_count":   word_count,
                "text":         chunk_text,
            })
            chunk_id_counter += 1

    log.info(
        f"Chunking: produced {len(all_chunks)} chunks from {len(articles)} articles "
        f"(cap={MAX_CHUNKS_PER_ARTICLE}, articles capped: {articles_capped})."
    )
    return all_chunks, articles_capped


# ══════════════════════════════════════════════════════════════════════════════
# 5. ROTATE — keep only MAX_CHUNK_FILES files in data/
# ══════════════════════════════════════════════════════════════════════════════

def _rotate_data_files():
    """Delete oldest chunk JSON files beyond MAX_CHUNK_FILES."""
    files = sorted(DATA_DIR.glob("chunks_*.json"), key=lambda f: f.stat().st_mtime)
    excess = len(files) - MAX_CHUNK_FILES + 1  # +1 because we're about to add one
    if excess > 0:
        for old_file in files[:excess]:
            old_file.unlink()
            log.info(f"Rotated out old file: {old_file.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def fetch(query: str, max_per_source: int = MAX_ARTICLES_PER_SOURCE) -> List[Dict]:
    """
    Main entry point — importable by Phase 2 and beyond.

    Steps:
        1. Fetch from NewsAPI, GNews, RSS in parallel (sequential for simplicity here)
        2. Clean text
        3. Deduplicate by URL hash
        4. Chunk with sliding window (with per-article cap)
        5. Write to disk (rotating to keep ≤ MAX_CHUNK_FILES)
        6. Return chunk list in memory

    Returns:
        List of chunk dicts, each with full metadata.
    """
    log.info(f"Starting fetch for query: '{query}' (max {max_per_source}/source)")

    # --- Step 1: Fetch ---
    raw_articles: List[Dict] = []
    raw_articles.extend(_fetch_newsapi(query, max_per_source))
    raw_articles.extend(_fetch_gnews(query, max_per_source))
    raw_articles.extend(_fetch_rss(query, max_per_source))

    if not raw_articles:
        log.error("No articles fetched from any source. Check your API keys and network.")
        return []

    # --- Step 2: Clean ---
    for article in raw_articles:
        article["clean_text"] = _clean_text(article.get("raw_text", ""))

    # Filter articles with too little text to chunk
    min_text_words = CHUNK_MIN_WORDS
    viable = [a for a in raw_articles if len(a["clean_text"].split()) >= min_text_words]
    log.info(f"Articles with enough text to chunk: {len(viable)}/{len(raw_articles)}")

    # --- Step 3: Deduplicate ---
    unique_articles = _deduplicate(viable)

    # --- Step 4: Chunk (with per-article cap) ---
    chunks, articles_capped = _build_chunks(unique_articles)

    if not chunks:
        log.warning("Produced zero chunks. Articles may be too short after cleaning.")
        return []

    # --- Print cap summary ---
    sources: Dict[str, int] = {}
    for c in chunks:
        sources[c["source"]] = sources.get(c["source"], 0) + 1
    print(f"[OK] Chunk cap applied: max {MAX_CHUNKS_PER_ARTICLE} chunks per article")
    print(f"[OK] Articles affected by cap: {articles_capped}")
    print(f"[OK] Total chunks after cap: {len(chunks)}")
    by_source = ", ".join(f"{src}={cnt}" for src, cnt in sorted(sources.items()))
    print(f"[OK] By source: {by_source}")

    # --- Step 5: Write to disk ---
    _rotate_data_files()
    safe_query = re.sub(r"[^\w]", "_", query)[:40]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"chunks_{safe_query}_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(chunks)} chunks → {out_path}")

    # --- Step 6: Return in memory ---
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(chunks: List[Dict]):
    """Print a human-readable summary of the fetch results."""
    if not chunks:
        print("\n[!] No chunks produced.")
        return

    sources: Dict[str, int] = {}
    for c in chunks:
        sources[c["source"]] = sources.get(c["source"], 0) + 1

    print("\n" + "=" * 60)
    print(f"  EvoRAG Phase 1 — Fetch complete")
    print("=" * 60)
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Word range   : {min(c['word_count'] for c in chunks)}–"
          f"{max(c['word_count'] for c in chunks)} words")
    print(f"  Sources:")
    for src, count in sorted(sources.items()):
        print(f"    {src:30s} {count:4d} chunks")
    print("=" * 60)
    print(f"\nSample chunk [0]:")
    print(f"  Title : {chunks[0]['title'][:80]}")
    print(f"  Source: {chunks[0]['source']}")
    print(f"  Words : {chunks[0]['word_count']}")
    print(f"  Text  : {chunks[0]['text'][:200]}...")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="EvoRAG Phase 1 — Fetch, clean, and chunk news articles."
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Search query, e.g. 'artificial intelligence'"
    )
    parser.add_argument(
        "--max-per-source", "-m",
        type=int,
        default=MAX_ARTICLES_PER_SOURCE,
        help=f"Max articles per source (default: {MAX_ARTICLES_PER_SOURCE})"
    )
    args = parser.parse_args()

    chunks = fetch(args.query, max_per_source=args.max_per_source)
    _print_summary(chunks)

    # Exit non-zero if nothing was produced
    sys.exit(0 if chunks else 1)


if __name__ == "__main__":
    main()
