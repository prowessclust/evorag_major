# EvoRAG

A local-first Retrieval-Augmented Generation (RAG) system that routes every query through
eight analytical personas running on a local Ollama LLM. Personas peer-score each other,
the top-scoring ones are synthesized into a final consensus answer, and an evolution engine
tracks persona performance over time and can automatically swap out weak personas.

See [project_state_report.md](project_state_report.md) for a full breakdown of what's
implemented and what's remaining.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- A NewsAPI key and a GNews key (only needed for building/refreshing the corpus — not needed to just run queries against the existing index)

## Setup

```bash
git clone https://github.com/prowessclust/evorag_major.git
cd evorag_major

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# then fill in NEWSAPI_KEY / GNEWS_KEY in .env

ollama pull phi3:mini
```

## Running

Three terminals, in order:

```bash
# Terminal 1
ollama serve

# Terminal 2
python api.py

# Terminal 3 (optional) — sanity-check each phase
python validate_phase1.py
python validate_phase2.py
python validate_phase3.py
python validate_phase4.py
python validate_phase5.py
```

`validate_phase5.py` runs a full 8-persona consensus query (17 sequential Ollama calls) and
can take 15-30+ minutes on CPU-only hardware — this is expected, not a hang. Progress is
logged in the `python api.py` terminal.

Frontend: open [frontend/index.html](frontend/index.html) directly in a browser once the API
is running on `http://localhost:8000`.

## Rebuilding the corpus / index

```bash
python build_ai_corpus.py   # fetches fresh articles into data/
python embedder.py          # rebuilds the FAISS index from data/
```

Restart `python api.py` afterward — it loads the index into memory once at startup and won't
pick up a rebuilt index otherwise.

## Project layout

| Path | What it is |
|---|---|
| `fetcher.py`, `build_ai_corpus.py` | News ingestion (NewsAPI/GNews/RSS) |
| `embedder.py` | Embedding + FAISS index build/load |
| `retriever.py` | Single-RAG retrieval + Ollama call helpers |
| `personas.py` | 8-persona generation engine |
| `voting.py` | Peer scoring between personas |
| `synthesis.py` | Top-K synthesis into final answer |
| `pipeline.py` | Full consensus orchestration (`/query/consensus`) |
| `score_store.py` | Persists per-query persona scores |
| `persona_evolution/` | Tracks persona performance, detects/replaces weak personas |
| `api.py` | FastAPI app — all HTTP routes |
| `frontend/` | Static HTML/JS/CSS demo UI |
| `validate_phase*.py` | Per-phase smoke tests |
