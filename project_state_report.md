# EvoRAG — Current Project State Report
*Generated: 2026-09-16 | Based on full codebase inspection. No changes made.*

---

## 1. Project Overview

### What this project currently does
EvoRAG is a **local-first Retrieval-Augmented Generation (RAG) system** that routes every query through **eight distinct analytical personas** running on a local Ollama LLM. After each persona generates its answer, they **peer-score each other**, the top 3 are selected, and a final consensus answer is synthesized. Scores are stored for performance tracking. A background evolution engine can detect underperforming personas and swap them out automatically.

### Intended final project
The full system adds:
- **Corpus management**: a richer, refreshable AI-news knowledge base
- **Persona evolution**: automatic replacement of consistently weak personas (already built)
- **User adaptation**: implicit feedback signals that personalize which personas/styles the user sees
- **Evaluation and reporting**: measurable accuracy comparison between single-RAG and EvoRAG
- **Full frontend UI**: a polished demonstration interface for all features

### Main technologies
| Layer | Technology |
|---|---|
| LLM inference | Ollama (local), `phi3:mini` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (CPU) |
| Backend API | FastAPI + uvicorn |
| HTTP client | httpx (async) |
| Frontend | Vanilla HTML/CSS/JS (single file) |
| Charting | Chart.js (CDN) |
| News data | NewsAPI + GNews + RSS |
| Persistence | JSON files (`score_store.json`, `persona_versions.json`) |
| Config | `.env` for API keys, `config.py` for all constants |

---

## 2. What Is Actually Implemented

| Component | Status | Evidence in code | Notes |
|---|---|---|---|
| Phase 1 — Data ingestion | Complete | `fetcher.py` (20971 bytes), `build_ai_corpus.py` | Fetches from NewsAPI/GNews/RSS, dedupes, chunks, writes to `data/` |
| Phase 2 — Embedding + FAISS | Complete | `embedder.py` (16121 bytes), `index/evorag.faiss` exists | Uses MiniLM, builds/loads FAISS, index on disk |
| Phase 3 — Single RAG endpoint | Complete | `retriever.py`, `api.py` `/query` route | Retrieves chunks, builds prompt, calls Ollama, returns answer + sources |
| Phase 4 — 8-persona engine | Complete | `personas.py`, `config.py`, `/query/personas` route | Semaphore-controlled, sequential with retry, 8 personas configured |
| Phase 5 — Consensus voting | Complete | `voting.py`, `synthesis.py`, `pipeline.py`, `score_store.py`, `/query/consensus` | Peer scoring matrix, top-K selection, synthesis, score persistence |
| Phase 5 validator | Complete | `validate_phase5.py` | 30-min timeout, checks all 5 required fields, score store growth |
| Score store | Complete | `score_store.py`, `score_store.json` (40KB — has real data) | File-locked JSON append, data exists from previous runs |
| Persona evolution — tracker | Complete | `persona_evolution/tracker.py` | Rolling averages over last N queries, MIN_QUERIES_BEFORE_EVAL gate |
| Persona evolution — detector | Complete | `persona_evolution/detector.py` | 4-gate threshold system, hysteresis, returns flagged list |
| Persona evolution — mutator | Complete | `persona_evolution/mutator.py` | Hybrid strategy: predefined pool (5 candidates) + LLM fallback |
| Persona evolution — replacer | Complete | `persona_evolution/replacer.py` | In-memory swap, `persona_versions.json` persistence, file locking |
| Evolution integration in API | Complete | `api.py` `_run_evolution_check()`, BackgroundTasks | Runs after every `/query/consensus`, zero-latency, full logging |
| `/persona/status` endpoint | Complete | `api.py` lines 350-431 | Returns active persona health, rolling stats, replacement history |
| Frontend UI | Substantial | `frontend/index.html` (1136 lines, 40KB) | Query input, consensus display, persona cards, score matrix, source list, Chart.js evolution chart, stats table |
| Validation scripts (Phase 1-5) | All exist | `validate_phase1.py` through `validate_phase5.py` | Phase 5 script is fully complete with 30-min timeout |
| CORS middleware | Complete | `api.py` line 77 | Allow-all origins for local dev |
| `/health` endpoint | Complete | `api.py` | Returns vector count, dim, model name |
| Config centralization | Complete | `config.py` | All constants in one place, model now `phi3:mini` |
| User adaptation | Not implemented | No files | Planned only in PROJECT_STATUS.md |
| Evaluation / report evidence | Not implemented | `evorag_phase2/` has LaTeX report draft, no measurement scripts | Report directory exists but no accuracy comparison or latency scripts |
| `persona_versions.json` | Not yet created | `replacer.py` creates it on first replacement | Will auto-create; no replacement has fired yet |
| Full corpus rescrape | Partial | `build_ai_corpus.py` exists, `data/` has 2 files from March + July 2026 | Script is ready but last scrape was July 2026; stale data |

---

## 3. PROJECT_STATUS.md vs Actual Code

### Things PROJECT_STATUS.md says are done but have changed since it was written
| Claim | Actual state |
|---|---|
| "Phase 5: initial implementation complete" | Phase 5 is now fully complete including validator. The status file is stale. |
| "Recommended next file: validate_phase5.py" | `validate_phase5.py` already exists with 30-min timeout — written since the status file. |
| "LLM JSON robustness: add stricter retry" | `utils.py` `safe_json_loads()` handles markdown code fences; `voting.py` has dedicated `SCORING_TIMEOUT_SECONDS`. These fixes exist. |
| Model implicitly `phi3:latest` | `config.py` now shows `phi3:mini` with comment "was: phi3:latest — 3-5x faster on CPU" |

### Things implemented in code but missing from PROJECT_STATUS.md
- **Complete persona evolution system** (`persona_evolution/` package — all 4 modules) is fully implemented. PROJECT_STATUS.md listed it as "not yet implemented."
- **`/persona/status` API endpoint** exists in `api.py` — not mentioned in PROJECT_STATUS.md.
- **Frontend is substantially built** — PROJECT_STATUS.md describes it as "needed pieces." The actual frontend is 1136 lines with query, consensus, score matrix, persona cards, sources, Chart.js evolution chart, and stats table.
- **BackgroundTask evolution check** is wired into `/query/consensus` — not in PROJECT_STATUS.md.
- **`build_ai_corpus.py`** exists as a multi-query corpus builder — not mentioned.
- **`score_store.json` contains real data** (40KB) from actual previous consensus runs.

### Outdated information in PROJECT_STATUS.md
- "What Remains: Stronger Consensus Validation" — now done.
- "What Remains: Improve LLM JSON Robustness" — partially done.
- "What Remains: Persona Replacement Strategy" — now fully done.
- "What Remains: Frontend" — now substantially done.
- Milestone list at the bottom is mostly completed through step 4.

---

## 4. Current Architecture

```
User (browser)
    |
    v
frontend/index.html  (static, opened directly from filesystem)
    |  POST /query/consensus   GET /persona/status
    v
FastAPI (api.py — port 8000)
    |
    |-- GET  /health
    |-- POST /query              -> retriever.py -> Ollama
    |-- POST /query/personas     -> personas.py  -> Ollama x 8
    |-- POST /query/consensus    -> pipeline.py
    |       |
    |       |-- personas.py      -> Ollama x 8  (generation)
    |       |-- voting.py        -> Ollama x 8  (peer scoring)
    |       |-- synthesis.py     -> Ollama x 1  (synthesis)
    |       |-- score_store.py   -> score_store.json
    |       `-- BackgroundTask: _run_evolution_check()
    |               `-- persona_evolution/
    |                       tracker.py  <- score_store.json
    |                       detector.py
    |                       mutator.py  -> Ollama (if LLM strategy)
    |                       replacer.py -> persona_versions.json
    |
    `-- GET  /persona/status    -> tracker.py + replacer.py

embedder.py
    |-- sentence-transformers (MiniLM)
    |-- FAISS index (index/evorag.faiss)
    `-- Metadata (index/evorag_meta.json)

fetcher.py  -> NewsAPI, GNews, RSS -> data/*.json
```

**External services:**
- Ollama running locally at `http://localhost:11434` with `phi3:mini`
- NewsAPI key (in `.env`) — required for corpus fetching only
- GNews key (in `.env`) — required for corpus fetching only
- Chart.js loaded from jsDelivr CDN (frontend-only)

**Data flow:**
1. `fetcher.py` scrapes news → `data/chunks_*.json`
2. `embedder.py` embeds → `index/evorag.faiss` + `evorag_meta.json`
3. API loads index at startup into `_state`
4. `/query/consensus` → retrieve → 8 personas → 8 scorers → synthesis → persist scores → background evolution check

---

## 5. Existing Frontend

### Pages / structure
Single `frontend/index.html` file — no framework, no build step, opened directly in browser.

### Sections that exist
| Section | What it shows | Status |
|---|---|---|
| Header | EvoRAG branding, badge, tagline | Working |
| Query input | Textarea, "Analyse" button, 3 toggle checkboxes | Working |
| Progress bar | Shimmer animation + rotating stage messages | Working |
| Error/warning banners | Shown on API failure | Working |
| Consensus Answer | Synthesized answer text, Query ID, top-persona pills | Working |
| Persona Breakdown — Cards | 8 persona cards sorted by score, top-3 highlighted, read-more expand | Working |
| Persona Breakdown — Score Matrix | 8x8 peer-score table with color heatmap and hover tooltips | Working |
| Persona Breakdown — Sources | Retrieved source articles with title links and L2 distance | Working |
| Evolution Chart | Chart.js line chart — persona scores over sessions | Working (in-session only) |
| Stats Table | Per-persona avg/best/worst/top-K counts per session | Working (in-session only) |
| Toast | "Copied!" notification on answer click | Working |

### Connected APIs
- `POST /query/consensus` — main query (hardcoded to `http://localhost:8000`)
- **NOT connected**: `GET /persona/status` — endpoint exists in backend but frontend never calls it
- **NOT connected**: `GET /health` — frontend has no health indicator
- **NOT connected**: `POST /query` and `POST /query/personas` — not exposed in UI

### Missing functionality
- No `/persona/status` integration — the evolution panel is session-in-memory only
- No `/health` check on page load — user gets no warning if API is offline until they click "Analyse"
- Score evolution chart is session-only — refreshing the page loses all chart history
- No persona replacement events displayed — `persona_versions.json` data is not surfaced
- No "at_risk" or "replaced" persona indicators in the UI
- Score matrix rendering bug: API returns `dict[str, dict[str, score]]` but JS iterates it as array with `.forEach()` — **this is a bug**
- No configurable `top_k` in the UI

### UX problems
- No favicon
- No loading skeleton — page shows empty state until API responds (10-30 min)
- Answer text is plain text — no markdown rendering for structured answers
- Score matrix column headers only show first word of persona name

### Performance issues
- All 17 Ollama calls are blocking — browser `fetch()` blocks for up to 30 minutes; no streaming
- Evolution chart rebuilds all datasets on every query — degrades over many queries
- No request cancellation if user submits twice

---

## 6. Existing Performance (Backend)

| Area | Risk | Detail |
|---|---|---|
| LLM inference | Critical | 17 sequential Ollama calls. Each 60-120s on CPU. Total: up to 34 minutes. `phi3:mini` reduces to ~10-20 min. |
| Semaphore = 1 | Intentional | `OLLAMA_MAX_CONCURRENT=1` serializes all Ollama calls. Correct but unavoidably slow. |
| Scoring timeout | Acceptable | `SCORING_TIMEOUT_SECONDS=300` (5 min per scorer). Necessary for phi3 on CPU. |
| FAISS in-memory | Fine | Index loaded once at startup. Fast retrieval. |
| Score store reads | Low risk now | Reads entire `score_store.json` synchronously on each `/persona/status` call. Fine at 40KB. |
| Evolution check | Safe | Runs as BackgroundTask — zero latency added to user response. |
| No streaming | UX risk | Frontend blocks on `fetch()` for entire consensus duration. No partial results. |
| Bundle size | Fine | No npm, no bundler. Single HTML file ~40KB. Chart.js from CDN. |
| Embedding model | Moderate | MiniLM loaded per-process. `release_model()` called before Ollama to free RAM. Correct pattern. |
| Corpus staleness | Content risk | Last scrape was July 2026. Queries about current events will find stale or missing context. |

---

## 7. Remaining Work

### P0 — Blocking / Core

| Task | Why P0 |
|---|---|
| Verify Phase 5 passes `validate_phase5.py` | Score store has data but it's unclear if the full consensus pipeline completed without timeout. This is the system's core claim. |
| Fix score matrix rendering bug in frontend | `renderScoreMatrix()` calls `.forEach()` on a dict object. API returns `{scorer_id: {target_id: score}}` — JS needs `Object.values()` or similar fix. |
| Confirm evolution check imports are consistent | `_run_evolution_check` imports `MIN_QUERIES_BEFORE_REPLACE` from `detector.py` (correct), but verify naming matches throughout. |

### P1 — Important Functionality

| Task | Why P1 |
|---|---|
| Corpus refresh | `data/` has July 2026 chunks. Fresh scrape + index rebuild needed for current AI questions. |
| Wire `/persona/status` to frontend | Evolution section is session-in-memory only. Backend has all historical data. Endpoint exists but frontend never calls it. |
| Display persona replacement events in UI | Persona replacement is the key novelty claim. The UI doesn't show it at all. |
| User adaptation (Objective 4) | Core project objective not started: implicit feedback, per-user preference tracking, synthesis weighting. |
| Evaluation / measurement scripts | Project needs latency comparison (single RAG vs EvoRAG), accuracy comparison. Report has no actual numbers. |

### P2 — Improvements

| Task | Why P2 |
|---|---|
| Health check on frontend page load | User has no API status indicator before submitting a query. |
| Streaming or SSE for long-running queries | 10-30 min blocking `fetch()` is very poor UX. |
| Persist evolution chart history across page refreshes | Currently session-only. Should fetch from a `/scores/history` endpoint. |
| Markdown rendering in answer panel | Persona answers often contain bullet points and headers. `textContent` loses formatting. |
| "At Risk" and "Replaced" persona badges | Persona health status from `/persona/status` should be shown visually. |
| Configurable top_k in UI | Currently locked to server default in frontend. |

### P3 — Nice-to-Have

| Task |
|---|
| Favicon and meta tags |
| Dark/light mode toggle |
| Query history within session |
| Export query result as JSON/Markdown |
| Accessibility audit (ARIA, focus management) |
| Responsive mobile layout improvements |
| Replace CDN Chart.js with bundled version (offline capability) |

---

## 8. Project Objectives / Milestones

### Milestone 1 — Core Pipeline Verification — COMPLETE (needs one confirmed run)
**Objective:** Build and verify the multi-persona RAG pipeline (Phases 1-5).
**Status:** All 5 phases implemented. All validation scripts exist. `score_store.json` has real data.
**Caveat:** Phase 5 had timeout issues. Needs one successful end-to-end run with `phi3:mini` to confirm fully resolved.
**Definition of Done:** `validate_phase1.py` through `validate_phase5.py` all pass.

---

### Milestone 2 — Evolutionary Persona Replacement — Code complete, untested
**Objective:** Automatically detect and replace underperforming personas.
**What was built:** `persona_evolution/` package (tracker, detector, mutator, replacer), BackgroundTask integration, `/persona/status` endpoint.
**Needs verification:** Evolution check hasn't fired yet. Need to test by seeding `score_store.json` manually.
**Definition of Done:** One confirmed replacement event in `persona_versions.json`.

---

### Milestone 3 — Corpus Refresh
**Objective:** Replace stale July 2026 corpus with a fresh, larger AI-topic corpus.
**What needs to be built:**
- Run `python build_ai_corpus.py` with current API keys
- Cap at 500-800 chunks across 8 topic queries
- Add "topic" field to chunk metadata
- Rebuild FAISS index + restart API
- Re-run all 5 validation scripts
**Dependencies:** Valid NewsAPI/GNews keys in `.env`, Ollama running.
**Definition of Done:** Fresh `data/` files, rebuilt index, all validations pass.

---

### Milestone 4 — User Adaptation (Objective 4)
**Objective:** Learn user preferences from implicit feedback and personalize synthesis.
**What needs to be built:**
- Capture implicit feedback signals (follow-up questions, regenerate clicks, dwell time)
- Build per-user/session style profile
- Upweight preferred persona types in scoring
- Store user profiles separately from `score_store.json`
- New API endpoints: `POST /feedback`, `GET /user/profile`
**Dependencies:** Milestone 1 fully stable, `score_store.json` schema must not break.
**Definition of Done:** Repeated use by same user measurably shifts which personas appear in top-K.

---

### Milestone 5 — Frontend and UX
**Objective:** Turn the existing frontend into a complete, polished demonstration interface.
**What needs to be built:** See Section 9 below.
**Dependencies:** Milestones 1-2 stable. `/persona/status` endpoint working.
**Definition of Done:** Full demo interface that non-technical evaluators can use without reading docs.

---

### Milestone 6 — Evaluation and Report Evidence
**Objective:** Produce measurable, credible evidence for the academic report.
**What needs to be built:**
- Latency measurement script: single RAG vs EvoRAG timing
- Accuracy comparison: qualitative eval of 10-20 test queries
- Screenshots of replacement events from the UI
- API outputs for the appendix
**Dependencies:** Milestone 5 frontend complete, stable corpus (Milestone 3).
**Definition of Done:** LaTeX report in `evorag_phase2/` has filled-in results sections.

---

## 9. Proposed Milestone 5 — Frontend and UX

### A. Existing Frontend Capabilities
- Query input with toggles
- Consensus answer display with persona pills
- Persona response cards (expandable)
- 8x8 score matrix with color heatmap
- Source list with links
- Session-in-memory Chart.js evolution chart
- Session stats table

### B. Missing Features
- `/persona/status` integration (historical health data)
- Persona replacement event display
- Persistent score history (across page refreshes)
- Health check indicator on load
- "At Risk" / "Replaced" persona badges
- Markdown rendering in answer text
- Request streaming / SSE

### C. Required UI Pages/Components
- **Persona Health Panel** — shows per-persona rolling avg score, status badge (healthy/at-risk), fetched from `/persona/status`
- **Replacement History Timeline** — cards showing which persona was replaced, when, and why
- **Health Banner on load** — calls `/health`, shows model + vector count before first query
- **History Persistence** — on page load, call `/persona/status` to restore chart data

### D. Required API Integrations
| Frontend Feature | Endpoint |
|---|---|
| Health banner on load | `GET /health` |
| Persona health panel | `GET /persona/status` |
| Score history restore | New `GET /scores/history` or reuse `/persona/status` |
| User adaptation signals | New `POST /feedback` (Milestone 4) |

### E. Loading / Error / Empty States
- Loading skeleton for consensus answer (animated placeholder while waiting)
- Timeout warning after 2 minutes ("Still running, this can take up to 30 minutes on CPU")
- Empty state for score matrix when scoring failed (all zeros)
- Empty state for evolution chart before first query
- API offline state: health check on load, clear error banner

### F. Responsive Design Requirements
- 3-panel breakdown grid: collapses to 1 column on mobile (currently 3 columns)
- Persona card grid: 2 to 1 column at 640px (already partially done)
- Score matrix: horizontal scroll already implemented (correct)
- Header: already uses `clamp()` for font size (correct)
- Query actions: already collapses to column at 640px (correct)

### G. Accessibility Considerations
- Add `aria-live="polite"` to consensus answer div (new content appears after load)
- Score matrix cells: add `scope="col"` and `scope="row"` to `<th>` elements
- Persona pill buttons: ensure `aria-label` describes the persona name
- Test keyboard navigation through all sections

### H. User Interaction Improvements
- Keyboard shortcut: `Ctrl+Enter` already works; add tooltip hint in UI
- Persona card expand/collapse: animate height transition
- Score matrix: highlight entire row/column on cell hover
- "Clear" button to reset the page state
- Confirmation before submitting a new query while one is in progress

### I. Performance Improvements (Frontend-Specific)
- Abort controller: cancel in-flight `fetch()` if user navigates away
- Score history: cap `evolutionHistory` array to last 50 entries to avoid O(n) chart rebuilds
- Persona cards: use `DocumentFragment` for batch DOM insertion
- Avoid re-rendering the entire matrix on toggle — hide/show instead of rebuild

### J. Frontend Testing Requirements
- Unit test `renderScoreMatrix()` with known data shape (especially the dict-vs-array fix)
- Unit test `escHtml()` XSS safety
- Integration test: mock API response → verify all sections render
- E2E test: submit query → verify consensus section appears
- Responsive test: verify layout at 320px, 768px, 1024px

---

## 10. Recommended Execution Order

```
Milestone 1 (Phase 1-5 pipeline)
    - Run validate_phase5.py end-to-end and confirm PASSED
    |
    v
Milestone 2 (Persona evolution)
    - Seed score_store.json manually, trigger one replacement event, confirm persona_versions.json
    |
    v
Milestone 3 (Corpus refresh)
    - Run fresh scrape, rebuild index (~2h operational work)
    |
    v
Milestone 4 (User adaptation)
    - New backend: feedback signals + user profiles
    |
    v
Milestone 5 (Frontend + UX)
    - Wire /persona/status, fix matrix bug, add health check, history persistence, markdown
    |
    v
Milestone 6 (Evaluation + Report)
    - Measure latency, run eval queries, fill in LaTeX report
```

**Rationale:**
- The pipeline must be verified before adding features on top of it.
- Corpus refresh before user adaptation so the baseline corpus is stable.
- Frontend comes after all backend features are done, to avoid rewiring the UI multiple times.
- Report is last because it documents everything else.

---

## 11. Immediate Next Step

**Verify that Phase 5 actually works end-to-end with `phi3:mini`.**

1. Terminal 1: `ollama serve`
2. Terminal 2: `python api.py`
3. Terminal 3: `python validate_phase5.py`
4. Wait for it to complete (up to 30 minutes)
5. Confirm it exits with `=== Phase 5 PASSED ===`

This is the single highest-priority action because everything else — evolution checks, user adaptation, the frontend demo, the academic report — depends on the consensus pipeline being proven to work reliably. Until Phase 5 passes its own validator, you do not have a confirmed working system.

---

## 12. Project Recovery Summary

```
CURRENTLY WORKING:
  - Complete Phase 1-4 backend pipeline
  - Full consensus voting logic (Phase 5 code is complete)
  - All 5 validation scripts exist
  - Complete persona evolution package (tracker, detector, mutator, replacer)
  - /persona/status API endpoint
  - Background evolution check wired into /query/consensus
  - Score store has real data (40KB from previous runs)
  - Functional frontend with query, consensus, matrix, chart, stats
  - File-locked JSON persistence
  - CORS, health check, all API schemas

PARTIALLY WORKING:
  - Phase 5 consensus (code complete but validator timeout status unknown;
    last known state was FAILING; phi3:mini switch may have resolved it)
  - Frontend evolution section (session-only; /persona/status not connected)
  - Score matrix rendering (likely broken: JS iterates dict as array)
  - Persona evolution (code complete but no confirmed replacement event ever fired)
  - Corpus (exists but stale from July 2026)

NOT IMPLEMENTED:
  - User adaptation (Objective 4 — not started at all)
  - Evaluation / measurement scripts
  - Score history persistence across page refreshes
  - /persona/status -> frontend integration
  - Replacement event display in UI
  - Streaming / SSE for long queries

BROKEN / RISKY:
  - Score matrix JS rendering: matrix.forEach() called on a dict object — likely broken
  - Phase 5 timeout: unknown if phi3:mini resolved it; needs one successful run to confirm
  - Corpus staleness: July 2026 data causes poor RAG quality for current AI news questions
  - No request abort: browser tab must stay open for 30 min during consensus query

NEXT MILESTONE:
  - Milestone 1 verification: run validate_phase5.py end-to-end and confirm PASSED

FIRST TASK:
  - python validate_phase5.py
    (with ollama serve + python api.py running in separate terminals)
```
