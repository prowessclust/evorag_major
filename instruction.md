# EvoRAG — Timeout Fix + Model Swap + Persona Replacement
## Complete implementation prompt for AI coding assistants

---

> **How to use:** Copy everything below the horizontal line into your AI coding assistant. It is self-contained. It covers three things in order: (A) fix the two real issues from the latest validation run, (B) implement the full evolutionary persona replacement system, and (C) add the /persona/status API endpoint the frontend needs to display replacement events.

---

---

## CURRENT PROJECT STATE

All infrastructure is working:
- Phase 1–4: PASSING
- Phase 5: FAILING due to 15-minute timeout (not a logic bug — a latency + timeout config problem)
- Frontend: working
- Score store: writing correctly after each consensus query
- API: running at http://localhost:8000

Confirmed endpoints:
- `GET  /health`
- `POST /query`
- `POST /query/personas`
- `POST /query/consensus`  ← Phase 5 calls this, currently timing out

Model: `phi3:latest` running on CPU. Single call takes 60–120 seconds.
Consensus query requires 17 total Ollama calls (8 generation + 8 scoring + 1 synthesis).

---

## FIX A — Stale index after rebuild (operational fix, not a code fix)

This is not a code bug. Every time the FAISS index is rebuilt on disk, `api.py` must be restarted to load the new index into memory. The health check showing 108 vectors while the index has 150 is caused by the API process holding the old index in memory.

**Add this warning to the `embedder.py` output** so it is never forgotten:

```python
print("\n[!] IMPORTANT: Restart api.py now to load the new index into memory.")
print("    Health check will show old vector count until the API is restarted.\n")
```

Add the same warning at the end of any script that calls `build_index()`.

---

## FIX B — Phase 5 timeout: two parts

### Part 1 — Increase the validator timeout

In `validate_phase5.py`, find the timeout configuration for the `/query/consensus` POST request. It is currently set to 15 minutes (900 seconds). Increase it:

```python
CONSENSUS_TIMEOUT_SECONDS = 1800  # 30 minutes — covers worst-case 17 calls × 120s on CPU
```

Also add a progress indicator in the validator so it doesn't look frozen:

```python
print("      This can take up to 30 minutes on CPU (17 LLM calls).")
print("      Do not interrupt. Progress is logged in api.py terminal.")
```

### Part 2 — Switch to a faster model for all Ollama calls

`phi3:latest` is too slow for 17 sequential/parallel calls on CPU. Switch to `phi3:mini` which runs 3–5× faster with acceptable quality for short persona-style responses.

Locate wherever `OLLAMA_MODEL` is defined (likely `config.py`, `utils.py`, or top of `api.py`). Change it:

```python
OLLAMA_MODEL = "phi3:mini"  # was: phi3:latest — 3-5x faster on CPU
```

Then in the terminal, pull the model:
```bash
ollama pull phi3:mini
```

Do not change anything else — the model name is the only change needed. All prompt templates, API call logic, and parsing stay the same.

After making both changes:
1. Pull `phi3:mini` via ollama
2. Restart `api.py`
3. Re-run `validate_phase5.py`
4. Expected result: completes within 10–20 minutes instead of timing out

---

## PERSONA REPLACEMENT SYSTEM

### What needs to be built

A self-contained evolutionary persona replacement system that:
1. Reads historical scores from `score_store.json`
2. Tracks a rolling average performance score per persona over recent queries
3. Detects consistently underperforming personas using a threshold
4. Generates a candidate replacement persona (mutated variant of the weak one)
5. Replaces the weak persona with the candidate after sufficient evidence
6. Stores full version history so every experiment is reproducible
7. Exposes a `/persona/status` API endpoint for the frontend to display

---

### File structure to create

```
evorag/
├── persona_evolution/
│   ├── __init__.py
│   ├── tracker.py          # reads score_store.json, computes rolling averages
│   ├── detector.py         # detects underperformers using threshold logic
│   ├── mutator.py          # generates candidate replacement personas
│   ├── replacer.py         # executes the replacement and writes version history
│   └── persona_versions.json  # auto-created, stores all persona versions
```

---

### tracker.py — Rolling average calculator

```python
"""
Reads score_store.json and computes per-persona rolling average scores
over the last N queries (configurable window).
"""

ROLLING_WINDOW = 10  # number of recent queries to consider

def load_score_history(score_store_path: str = "score_store.json") -> list[dict]:
    """Load all entries from score_store.json. Returns empty list if file missing."""

def compute_rolling_averages(history: list[dict], window: int = ROLLING_WINDOW) -> dict[str, dict]:
    """
    For each persona, compute rolling average over the last `window` queries.
    
    Returns:
    {
        "analytical-critical": {
            "avg_score": 6.2,
            "query_count": 10,
            "scores": [7, 6, 5, 6, 7, 6, 5, 6, 7, 6],  # last N scores
            "times_in_top_k": 3,
            "fallback_count": 1   # times fallback default was used
        },
        ...
    }
    
    Rules:
    - Only include a persona if it has at least MIN_QUERIES_BEFORE_EVAL entries
    - If a score entry used a fallback default (detect this via a flag or score == 5),
      count it separately in fallback_count but still include in avg
    - times_in_top_k = count of entries where made_top_k == True
    """

MIN_QUERIES_BEFORE_EVAL = 5  # don't evaluate a persona until it has 5+ data points
```

---

### detector.py — Underperformer detection

```python
"""
Detects personas that are consistently underperforming and should be replaced.
Uses a threshold system with hysteresis to avoid premature replacement.
"""

REPLACEMENT_THRESHOLD = 5.0      # avg score below this triggers replacement candidate
REPLACEMENT_CERTAINTY = 0.7      # 70% of recent scores must be below threshold
TOP_K_MIN_RATE = 0.2             # persona must make top-K at least 20% of the time
MIN_QUERIES_BEFORE_REPLACEMENT = 8  # must have at least 8 queries before any replacement

def detect_underperformers(
    rolling_averages: dict[str, dict]
) -> list[dict]:
    """
    Returns a list of personas flagged for replacement, with reason.
    
    A persona is flagged if ALL of the following are true:
    1. query_count >= MIN_QUERIES_BEFORE_REPLACEMENT
    2. avg_score < REPLACEMENT_THRESHOLD
    3. More than REPLACEMENT_CERTAINTY fraction of recent scores are below threshold
    4. times_in_top_k / query_count < TOP_K_MIN_RATE
    
    Returns:
    [
        {
            "persona_id": "creative-lateral",
            "avg_score": 4.1,
            "top_k_rate": 0.1,
            "query_count": 10,
            "reason": "avg_score 4.1 < threshold 5.0, top_k_rate 10% < minimum 20%"
        }
    ]
    """

def should_trigger_replacement(underperformers: list[dict]) -> bool:
    """
    Only trigger replacement if there is at least one confirmed underperformer.
    Never replace more than one persona per trigger cycle.
    Always replace the WORST performer (lowest avg_score) if multiple qualify.
    """
```

---

### mutator.py — Candidate replacement persona generator

```python
"""
Generates a candidate replacement persona when a weak persona is detected.
Uses one of three strategies, selectable via config.
"""

MUTATION_STRATEGY = "llm_generated"  # options: "llm_generated", "predefined_pool", "hybrid"

# Predefined pool of replacement candidates — used if strategy is "predefined_pool" or "hybrid"
REPLACEMENT_POOL = {
    "historical-comparativist": """You are the Historical Comparativist. Your job is to find 
historical parallels and precedents for the current situation. When answering, ground 
every insight in a specific historical analogy or case study. Ask: has something like 
this happened before, and what was the outcome?""",

    "systems-thinker": """You are the Systems Thinker. Your job is to map the feedback loops, 
second-order effects, and interdependencies in any situation. Avoid linear cause-effect 
thinking. Always ask: what are the downstream consequences, and what feeds back into 
the system?""",

    "contrarian-optimist": """You are the Contrarian Optimist. Where others see risk, you find 
opportunity. Where the consensus is pessimistic, you find overlooked upsides. Your job 
is not to be blindly positive but to surface the strongest case that things will work 
out better than expected.""",

    "first-principles": """You are the First Principles Reasoner. You strip away assumptions 
and rebuild reasoning from the ground up. Reject analogies and conventional wisdom. 
Start from the most basic truths available in the retrieved context and build upward.""",

    "ethical-evaluator": """You are the Ethical Evaluator. Your job is to surface the moral 
dimensions of any situation — who benefits, who is harmed, whose interests are invisible 
in the mainstream analysis, and what values are implicitly embedded in the retrieved 
context.""",
}

def generate_replacement_persona(
    weak_persona_id: str,
    weak_persona_prompt: str,
    existing_persona_ids: list[str],
    strategy: str = MUTATION_STRATEGY,
) -> dict:
    """
    Generate a candidate replacement persona.
    
    Strategy: "predefined_pool"
    - Pick the persona from REPLACEMENT_POOL that is not already in existing_persona_ids
    - If all pool entries are already active, fall back to llm_generated
    
    Strategy: "llm_generated"  
    - Call Ollama with a meta-prompt asking it to generate a new analytical persona
      that is meaningfully different from all existing personas
    - The meta-prompt must include the list of all existing persona descriptions
      so the LLM can avoid redundancy
    - Parse the LLM response to extract: persona_id (slug), display_name, system_prompt
    
    Strategy: "hybrid"
    - Try predefined_pool first
    - If exhausted, fall back to llm_generated
    
    Returns:
    {
        "persona_id": "historical-comparativist",
        "display_name": "Historical Comparativist",
        "system_prompt": "You are the Historical Comparativist...",
        "generated_by": "predefined_pool",  # or "llm_generated"
        "generated_at": "2026-07-04T22:00:00Z"
    }
    """

LLM_MUTATION_PROMPT = """
You are designing an analytical persona for a multi-perspective reasoning system.

The system currently has these personas (do NOT duplicate any of these):
{existing_personas_list}

The persona being replaced was:
ID: {weak_persona_id}
Description: {weak_persona_prompt}

Design ONE new analytical persona that:
1. Is meaningfully different from all existing personas listed above
2. Fills a genuine analytical gap not covered by the existing set
3. Would produce different answers than the existing personas on the same question

Return ONLY valid JSON with this exact structure:
{{
    "persona_id": "slug-case-id-max-3-words",
    "display_name": "Display Name",
    "system_prompt": "You are the [Name]. Your job is..."
}}
"""
```

---

### replacer.py — Execute replacement and version history

```python
"""
Executes persona replacement and maintains a complete version history
in persona_versions.json so every experiment is reproducible.
"""

PERSONA_VERSIONS_PATH = "persona_evolution/persona_versions.json"

def load_persona_versions() -> dict:
    """Load persona_versions.json. Returns default structure if file doesn't exist."""

def save_persona_versions(versions: dict) -> None:
    """Save persona_versions.json with file locking."""

def execute_replacement(
    weak_persona_id: str,
    candidate: dict,
    reason: str,
    personas_module,  # reference to the module where PERSONAS dict lives
    logger,
) -> dict:
    """
    Replace weak_persona_id with the candidate persona in the active PERSONAS dict.
    Record the full replacement event in persona_versions.json.
    
    persona_versions.json structure:
    {
        "active_personas": ["analytical-critical", "empirical-evidential", ...],
        "version_counter": 3,
        "replacements": [
            {
                "version": 1,
                "timestamp": "2026-07-04T22:00:00Z",
                "replaced_persona_id": "creative-lateral",
                "replaced_persona_prompt": "You are the Creative...",
                "replacement_persona_id": "historical-comparativist",
                "replacement_persona_prompt": "You are the Historical...",
                "reason": "avg_score 4.1 < threshold 5.0, top_k_rate 10% < minimum 20%",
                "avg_score_at_replacement": 4.1,
                "queries_evaluated": 10
            }
        ]
    }
    
    After replacement:
    - Update PERSONAS dict in memory (remove weak, add candidate)
    - Write the event to persona_versions.json
    - Log clearly: which persona was replaced, by what, and why
    - Return the replacement event dict
    """

def get_replacement_history() -> list[dict]:
    """Return the full list of past replacement events from persona_versions.json."""
```

---

### Integration — Where to call the evolution cycle

Add an evolution check at the END of every `/query/consensus` request, after the score store has been written. This runs after the user already has their answer so it adds zero latency to the response.

In `api.py` (or wherever `/query/consensus` is handled), after `write_score_store_entry(...)`:

```python
# Run evolution check after every query — async, after response is sent
async def run_evolution_check():
    from persona_evolution.tracker import load_score_history, compute_rolling_averages
    from persona_evolution.detector import detect_underperformers, should_trigger_replacement
    from persona_evolution.mutator import generate_replacement_persona
    from persona_evolution.replacer import execute_replacement
    import personas  # your personas module

    history = load_score_history()
    averages = compute_rolling_averages(history)
    underperformers = detect_underperformers(averages)

    if should_trigger_replacement(underperformers):
        worst = min(underperformers, key=lambda x: x["avg_score"])
        candidate = generate_replacement_persona(
            weak_persona_id=worst["persona_id"],
            weak_persona_prompt=personas.PERSONAS[worst["persona_id"]],
            existing_persona_ids=list(personas.PERSONAS.keys()),
        )
        execute_replacement(
            weak_persona_id=worst["persona_id"],
            candidate=candidate,
            reason=worst["reason"],
            personas_module=personas,
            logger=logger,
        )

# Call it as a background task (FastAPI example):
background_tasks.add_task(run_evolution_check)
```

---

### New API endpoint — /persona/status

Add this endpoint to `api.py` so the frontend can display persona health and replacement history:

```python
GET /persona/status

Response:
{
    "active_personas": [
        {
            "id": "analytical-critical",
            "display_name": "Analytical Critic",
            "avg_score": 7.2,
            "query_count": 12,
            "top_k_rate": 0.58,
            "status": "healthy",       # "healthy" | "at_risk" | "replaced"
            "version": 1               # increments each time this slot is replaced
        },
        ...
    ],
    "replacement_history": [
        {
            "version": 1,
            "timestamp": "2026-07-04T22:00:00Z",
            "replaced": "creative-lateral",
            "replacement": "historical-comparativist",
            "reason": "avg_score 4.1 < threshold 5.0"
        }
    ],
    "total_replacements": 1,
    "queries_since_last_replacement": 3
}
```

Status rules:
- `"healthy"` — avg_score >= REPLACEMENT_THRESHOLD
- `"at_risk"` — avg_score < REPLACEMENT_THRESHOLD but not enough queries yet to replace
- `"replaced"` — this ID has been replaced at some point (show in history only, not active)

---

## IMPLEMENTATION CONSTRAINTS

1. The evolution check must NEVER block the API response — always run as a background task
2. File locking required on `persona_versions.json` (same pattern as `score_store.json`)
3. Never replace more than one persona per query cycle
4. Never replace a persona with fewer than `MIN_QUERIES_BEFORE_REPLACEMENT` data points
5. The `PERSONAS` dict in memory must be updated after replacement — new queries use the new persona immediately
6. All replacement events must be logged at INFO level with this format:
   ```
   [EVOLUTION] Replaced 'creative-lateral' (avg=4.1) → 'historical-comparativist' (v2)
   Reason: avg_score 4.1 < threshold 5.0, top_k_rate 10% < minimum 20%
   ```
7. Full type hints and docstrings on every function
8. If `score_store.json` has fewer than `MIN_QUERIES_BEFORE_REPLACEMENT` total entries, skip the evolution check entirely and log:
   ```
   [EVOLUTION] Skipping check — only N queries in history (need MIN_QUERIES_BEFORE_REPLACEMENT)
   ```

---

## WHAT NOT TO DO

- Do not run the evolution check synchronously inside the request/response cycle
- Do not replace a persona mid-query (only between queries)
- Do not use retraining or fine-tuning — mutation is prompt-level only
- Do not delete old persona prompts — always archive them in `persona_versions.json`
- Do not implement user adaptation yet — that is a separate future task
- Do not modify the score store schema — read it as-is

---

## DELIVERABLE ORDER

Produce in this exact order:

1. **embedder.py** — add the restart warning (Fix A)
2. **validate_phase5.py** — increase timeout to 1800s (Fix B Part 1)
3. **config.py / utils.py** — change OLLAMA_MODEL to phi3:mini (Fix B Part 2)
4. **persona_evolution/__init__.py** — empty init
5. **persona_evolution/tracker.py** — complete implementation
6. **persona_evolution/detector.py** — complete implementation
7. **persona_evolution/mutator.py** — complete implementation
8. **persona_evolution/replacer.py** — complete implementation
9. **api.py** — add background evolution check + /persona/status endpoint

After all files, write a section "How to verify persona replacement is working" with:
- Exactly what to put in `score_store.json` manually to trigger a replacement on the next query (so you can test without waiting for real data)
- What log lines to expect when a replacement fires
- What `persona_versions.json` should look like after one replacement

---

## FINAL TASKS — DO NOT IMPLEMENT NOW

```
╔══════════════════════════════════════════════════════════════╗
║  FINAL TASKS — IMPLEMENT ONLY AFTER REPLACEMENT IS WORKING  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Fix 3 — Full corpus rescrape                             ║
║     - 500-800 chunks across focused AI topics                ║
║     - 4-chunk-per-article cap (already built)                ║
║     - Add "topic" field to chunk metadata                    ║
║     - Rebuild FAISS index + restart API                      ║
║     - Re-run all 5 validation scripts                        ║
║                                                              ║
║  2. User adaptation layer                                    ║
║     - Track implicit feedback signals per query              ║
║       (follow-up questions, regenerate clicks, dwell time)   ║
║     - Build per-user style profile                           ║
║     - Upweight preferred persona types in scoring            ║
║     - Store user profiles separately from score_store.json   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```