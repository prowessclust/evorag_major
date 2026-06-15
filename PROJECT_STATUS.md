# EvoRAG Project Status

## Project Description

EvoRAG is an Evolutionary Multi-Personality Retrieval-Augmented Generation system for real-time information analysis. A normal RAG system retrieves relevant chunks and sends them to one LLM for one answer. EvoRAG extends that flow by sending the same retrieved context to eight analytical personas, collecting multiple perspectives, using peer voting to identify stronger responses, and synthesizing a consensus answer for the user.

The project synopsis defines four major objectives:

1. Build a multi-personality RAG architecture using eight analytical personas on a single lightweight language model.
2. Add consensus-based peer voting for internal quality assurance.
3. Add performance-driven persona replacement so weak personas can be replaced over time.
4. Add user adaptation through implicit feedback so the system can align with each user's reasoning style.

## What Is Completed

### Phase 1: Data Ingestion

Status: complete.

Implemented in `fetcher.py`.

- Fetches articles from NewsAPI, GNews, and RSS sources.
- Extracts full article text where possible.
- Cleans HTML and boilerplate.
- Deduplicates articles by URL hash.
- Splits text into overlapping chunks with metadata.
- Writes chunk files into `data/`.
- Has validation through `validate_phase1.py`.

### Phase 2: Embedding and Vector Store

Status: complete.

Implemented in `embedder.py`.

- Uses `sentence-transformers/all-MiniLM-L6-v2`.
- Builds a FAISS vector index.
- Saves `index/evorag.faiss` and `index/evorag_meta.json`.
- Supports semantic top-k retrieval.
- Has validation through `validate_phase2.py`.

### Phase 3: Single-Answer RAG API

Status: complete.

Implemented in `retriever.py` and `api.py`.

- Retrieves top-k chunks from FAISS.
- Builds grounded RAG prompts with source passage references.
- Calls Ollama directly through HTTP.
- Exposes `POST /query`.
- Exposes `GET /health`.
- Has validation through `validate_phase3.py`.

### Phase 4: Eight-Persona Generation

Status: mostly complete.

Implemented in `config.py`, `personas.py`, and `api.py`.

- Defines eight personas:
  - Analytical-Critical
  - Empirical-Evidential
  - Adversarial-Skeptical
  - Synthesis-Integrative
  - Domain-Expert
  - Temporal-Contextual
  - Risk-Evaluative
  - Creative-Lateral
- Retrieves context once and shares it across all personas.
- Runs personas concurrently with `asyncio.gather()`.
- Retries empty persona responses.
- Exposes `POST /query/personas`.
- Has validation through `validate_phase4.py`.

Note: persona prompts differ slightly from the attached implementation prompt names, but they satisfy the same goal of orthogonal analytical perspectives.

### Phase 5: Consensus Voting Layer

Status: initial implementation complete.

Implemented in `pipeline.py`, `voting.py`, `synthesis.py`, `score_store.py`, `utils.py`, and `api.py`.

- Runs the eight-persona generation round.
- Runs peer scoring across persona responses.
- Builds a score matrix where each persona scores peers and leaves self-score empty.
- Computes final persona scores from factual grounding, reasoning quality, and completeness.
- Selects the top 3 personas.
- Calls Ollama once more to synthesize the selected responses.
- Appends query-level scoring data to `score_store.json`.
- Exposes `POST /query/consensus`.

## What Remains

### 1. Stronger Consensus Validation

The consensus code exists, but it needs a dedicated validation script similar to earlier phases.

Recommended next file:

- `validate_phase5.py`

It should check:

- `POST /query/consensus` returns `final_answer`.
- Exactly 8 raw persona responses are present.
- `score_matrix` has peer-score rows.
- Top personas are selected.
- `score_store.json` receives a new entry.

### 2. Improve LLM JSON Robustness

Voting depends on personas returning valid JSON. The current parser is safe and will not crash, but local LLMs may still produce malformed score JSON.

Remaining work:

- Add stricter retry for malformed scoring JSON.
- Add fallback scoring when too many score rows fail.
- Consider separating scoring model settings from answer-generation settings.

### 3. Persona Replacement Strategy

This is part of the synopsis but not yet implemented.

Needed pieces:

- Read historical scores from `score_store.json`.
- Track moving average performance per persona.
- Detect consistently weak personas.
- Generate or load candidate replacement personas.
- Replace underperforming personas only after enough evidence.
- Store persona versions so experiments are reproducible.

### 4. User Adaptation

This is part of the synopsis but not yet implemented.

Needed pieces:

- Capture user feedback signals.
- Store per-user preference history.
- Learn which personas or answer styles a user prefers.
- Adjust synthesis weighting or top-K selection per user.
- Add privacy-aware user/session identifiers.

### 5. Frontend / Demonstration Interface

The current project is backend/API focused.

Needed pieces:

- Simple UI for entering a query.
- Show final consensus answer.
- Optionally show persona responses, scores, and sources.
- Show score evolution over time for demonstration.

### 6. Evaluation and Report Evidence

The report directory exists under `evorag_phase2/`, but project claims need measurable evidence.

Needed pieces:

- Accuracy/reliability comparison between normal RAG and EvoRAG.
- Latency measurements for single RAG, persona generation, and consensus.
- User study or small manual evaluation.
- Screenshots/API outputs for final report.

## Current Mental Model

The implemented path is now:

1. User asks a question.
2. EvoRAG retrieves relevant chunks from FAISS.
3. Eight personas read the same chunks through different analytical lenses.
4. Personas score each other's responses.
5. The top 3 responses are selected.
6. A final consensus answer is synthesized.
7. Scores are stored for future evolution and adaptation.

## Suggested Next Milestones

1. Run and stabilize `/query/consensus` with a local Ollama model.
2. Add `validate_phase5.py`.
3. Add score-history analysis for persona performance.
4. Add persona replacement logic.
5. Add user feedback and personalization.
6. Build a small frontend demo.
