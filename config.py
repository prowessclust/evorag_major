"""
config.py — EvoRAG central configuration

All tunable constants live here so every phase imports from one place.
"""
# ── Phase 2 — Embedding + Vector Store (added below Phase 1 constants)

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ── Data directory ────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# How many chunk JSON files to keep in data/ before rotating (deleting oldest)
MAX_CHUNK_FILES = 20

# ── Fetcher settings ──────────────────────────────────────────────────────────
MAX_ARTICLES_PER_SOURCE = 15   # articles to request from each source per query

# Suggested topic sweep for building a broader AI-focused corpus.
AI_COVERAGE_QUERIES = [
    "OpenAI recent developments",
    "Anthropic Claude AI news",
    "Google DeepMind Gemini AI research",
    "Meta AI Llama artificial intelligence",
    "Microsoft AI Copilot OpenAI",
    "Nvidia AI chips data centers",
    "AI regulation policy safety",
    "generative AI enterprise adoption",
]

# ── Chunking settings ─────────────────────────────────────────────────────────
CHUNK_TARGET_WORDS   = 350    # aim for this word count per chunk
CHUNK_MIN_WORDS      = 300    # discard chunks shorter than this
CHUNK_MAX_WORDS      = 400    # hard ceiling (split further if exceeded)
CHUNK_OVERLAP_WORDS  = 50     # overlap between consecutive chunks

# ── RSS feed list ─────────────────────────────────────────────────────────────
RSS_FEEDS = [
    {"name": "BBC World",    "url": "http://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "Reuters",      "url": "https://feeds.reuters.com/reuters/topNews"},
    {"name": "Al Jazeera",   "url": "https://www.aljazeera.com/xml/rss/all.xml"},
    {"name": "MIT Technology Review AI", "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed"},
    {"name": "The Verge AI",  "url": "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml"},
    {"name": "VentureBeat AI", "url": "https://venturebeat.com/category/ai/feed/"},
    {"name": "Google AI Blog", "url": "https://blog.google/technology/ai/rss/"},
    {"name": "OpenAI News",   "url": "https://openai.com/news/rss.xml"},
]

# ── Phase 2 — Embedding + Vector Store ───────────────────────────────────────
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR        = ROOT_DIR / "index"          # persisted FAISS files live here
FAISS_INDEX_FILE = INDEX_DIR / "evorag.faiss"
META_FILE        = INDEX_DIR / "evorag_meta.json"
EMBED_BATCH_SIZE = 64                          # chunks embedded per batch
TOP_K_DEFAULT    = 5                           # default retrieval count

# ── Phase 3 — Retrieval Pipeline + FastAPI ────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"   # default Ollama address
OLLAMA_MODEL     = "phi3:mini"               # was: phi3:latest — 3-5x faster on CPU
# Keeps the model resident in Ollama for the whole consensus run (17 sequential
# calls). Without this, Ollama's default 5-min idle unload can evict the model
# between calls, forcing an expensive reload or leaving the server unresponsive
# mid-pipeline — observed as every scoring call failing right after generation.
OLLAMA_KEEP_ALIVE = "30m"
API_HOST         = "0.0.0.0"
API_PORT         = 8000
RETRIEVAL_TOP_K  = 5                          # chunks fed into prompt
RETRIEVAL_CANDIDATE_MULTIPLIER = 6            # over-fetch before deduplication
RETRIEVAL_MIN_CANDIDATES = 20                 # minimum candidate pool size

# ── Phase 4 — Multi-Persona Engine ───────────────────────────────────────────
PERSONA_TIMEOUT_SECONDS = 300  # was: 180 — raised because complex queries (e.g. judicial AI) cause
                               # personas to generate 2,500-3,000 char responses taking 180-270s on CPU.
                               # 300s matches SCORING_TIMEOUT_SECONDS and prevents mid-pipeline timeouts.
SCORING_TIMEOUT_SECONDS = 300   # Scoring prompts are 3-4× longer; needs extra headroom
OLLAMA_MAX_CONCURRENT = 1   # serialize Ollama calls to avoid CPU overload empties
CONSENSUS_TOP_K = 3
SCORE_STORE_FILE = ROOT_DIR / "score_store.json"

PERSONAS = [
    {
        "id": "analytical-critical",
        "name": "Analytical-Critical",
        "system_prompt": (
            "You are a rigorous analytical thinker. Break down the information in the "
            "context into its core components. Identify logical structure, highlight "
            "assumptions, and evaluate the strength of any claims made. Be precise and "
            "structured. Use only the provided context."
        ),
    },
    {
        "id": "empirical-evidential",
        "name": "Empirical-Evidential",
        "system_prompt": (
            "You are an empirical analyst who focuses strictly on evidence. Cite specific "
            "facts, statistics, and concrete events from the context. Avoid speculation. "
            "Distinguish clearly between what is stated and what is inferred. "
            "Use only the provided context."
        ),
    },
    {
        "id": "adversarial-skeptical",
        "name": "Adversarial-Skeptical",
        "system_prompt": (
            "You are a devil's advocate. Challenge the main claims in the context. "
            "Identify weaknesses, missing evidence, alternative explanations, and "
            "potential biases. Ask hard questions. Do not simply accept what is stated. "
            "Use only the provided context."
        ),
    },
    {
        "id": "synthesis-integrative",
        "name": "Synthesis-Integrative",
        "system_prompt": (
            "You are a synthesis expert. Connect the dots across the different passages "
            "in the context. Identify common themes, tensions, and patterns. Produce a "
            "coherent, integrated summary that is more than the sum of its parts. "
            "Use only the provided context."
        ),
    },
    {
        "id": "domain-expert",
        "name": "Domain-Expert",
        "system_prompt": (
            "You are a domain expert with deep technical knowledge. Explain the topic "
            "with precision using appropriate terminology. Provide depth and nuance that "
            "a specialist audience would appreciate. Correct any oversimplifications. "
            "Use only the provided context."
        ),
    },
    {
        "id": "temporal-contextual",
        "name": "Temporal-Contextual",
        "system_prompt": (
            "You are a temporal analyst. Focus on the timeline and sequence of events "
            "in the context. Identify what is recent vs. historical, what is developing, "
            "and what trajectory or trend is implied. Frame your answer in terms of "
            "time and change. Use only the provided context."
        ),
    },
    {
        "id": "risk-evaluative",
        "name": "Risk-Evaluative",
        "system_prompt": (
            "You are a risk analyst. Identify the key risks, threats, and uncertainties "
            "described or implied in the context. Assess their likelihood and potential "
            "impact. Highlight what could go wrong and for whom. "
            "Use only the provided context."
        ),
    },
    {
        "id": "creative-lateral",
        "name": "Creative-Lateral",
        "system_prompt": (
            "You are a lateral thinker. Go beyond the obvious interpretation. Draw "
            "unexpected connections, propose novel framings, and surface non-obvious "
            "implications from the context. Be creative but grounded in what is stated. "
            "Use only the provided context."
        ),
    },
]

# Fast-mode persona IDs used by validate_phase4.py (avoids 16-min CPU wait)
PERSONAS_FAST_MODE = ["analytical-critical", "empirical-evidential"]
