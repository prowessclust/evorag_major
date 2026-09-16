"""
validate_phase4.py — EvoRAG Phase 4 sanity check.

Runs in FAST MODE by default (2 personas only) to avoid a 16-min CPU wait.
Full 8-persona run: python validate_phase4.py --full

Prerequisites (must all be running before this):
    Terminal 1:  ollama serve
    Terminal 2:  python api.py   ← restart after Phase 4 changes

Exit code 0 = PASSED, 1 = FAILED.
"""

import argparse
import json
import sys

import httpx

API_URL = "http://127.0.0.1:8000"

# Fast mode: just 2 personas; full mode: None = all 8
FAST_PERSONA_IDS = ["analytical-critical", "empirical-evidential"]


def fail(msg: str):
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str):
    print(f"[OK]   {msg}")


def report_http_error(resp: httpx.Response) -> None:
    """Print structured API error details, especially 502 empty-persona payloads."""
    print(f"\n[FAIL] /query/personas returned HTTP {resp.status_code}")
    try:
        body = resp.json()
    except json.JSONDecodeError:
        print(resp.text[:2000])
        fail(f"/query/personas returned HTTP {resp.status_code}: {resp.text[:500]}")
        return

    detail = body.get("detail")
    if isinstance(detail, dict):
        print(f"  message: {detail.get('message', detail)}")
        empty_personas = detail.get("empty_personas") or []
        for item in empty_personas:
            print(
                f"  - [{item.get('id', '?')}] {item.get('name', '?')}: "
                f"{item.get('error', 'empty response')}"
            )
        fail(detail.get("message") or f"HTTP {resp.status_code} from /query/personas")
    elif detail:
        fail(str(detail))
    else:
        fail(f"/query/personas returned HTTP {resp.status_code}: {resp.text[:500]}")


# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Run all 8 personas (slow on CPU)")
args = parser.parse_args()

persona_ids = None if args.full else FAST_PERSONA_IDS
mode_label  = "FULL (8 personas)" if args.full else "FAST (2 personas)"
est_time    = "8–16 min" if args.full else "2–4 min"

print("\n" + "=" * 60)
print(f"  EvoRAG Phase 4 — Validation  [{mode_label}]")
print(f"  Estimated time: {est_time} on CPU")
print("  (Make sure 'python api.py' is running in another terminal)")
print("=" * 60)

# ── Step 1: Health check ──────────────────────────────────────────────────────
print("\n[...] GET /health")
try:
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_URL}/health")
except httpx.ConnectError:
    fail(
        "Cannot connect to API at http://127.0.0.1:8000\n"
        "       → Restart with: python api.py  (Phase 4 changes need a reload)"
    )
if resp.status_code != 200:
    fail(f"/health returned HTTP {resp.status_code}: {resp.text}")
ok(f"Health check passed — {resp.json().get('index_vectors')} vectors in index")

# ── Step 2: POST /query/personas ──────────────────────────────────────────────
PHASE4_QUERY = "Why are AI agents starting to act like family members, and what does that reveal about human psychology?"
query = PHASE4_QUERY
print(f"\n[...] POST /query/personas")
print(f"      Query   : '{query}'")
print(f"      Personas: {persona_ids or 'all 8'}")
print(f"      Waiting for phi3 to generate {len(persona_ids or [0]*8)} responses...")
print(f"      (This may take {est_time} — do not interrupt)\n")

payload = {"query": query, "top_k": 2}
if persona_ids:
    payload["persona_ids"] = persona_ids

try:
    with httpx.Client(timeout=600) as client:   # 10-min ceiling
        resp = client.post(f"{API_URL}/query/personas", json=payload)
except httpx.ConnectError:
    fail("Lost connection to API server during request.")
except httpx.ReadTimeout:
    fail("Request timed out after 10 minutes. Try restarting Ollama.")

if resp.status_code != 200:
    report_http_error(resp)

data = resp.json()

# ── Step 3: Validate schema ───────────────────────────────────────────────────
if "personas" not in data:
    fail(f"Response missing 'personas' key: {list(data.keys())}")
if "sources" not in data:
    fail(f"Response missing 'sources' key: {list(data.keys())}")

personas = data["personas"]
expected_count = len(persona_ids) if persona_ids else 8
if len(personas) != expected_count:
    fail(f"Expected {expected_count} persona results, got {len(personas)}")

ok(f"Received {len(personas)} persona responses")
ok(f"Sources: {len(data['sources'])} chunks")

# ── Step 4: Validate each persona result ─────────────────────────────────────
required_keys = {"id", "name", "response", "error"}
errors_found = []
empty_found = []

print("\n  --- Persona Responses ---")
for p in personas:
    missing = required_keys - set(p.keys())
    if missing:
        fail(f"Persona result missing keys {missing}: {p}")
    if p["error"]:
        errors_found.append(f"[{p['id']}] ERROR: {p['error']}")
    response = p.get("response") or ""
    if not response.strip():
        err_note = p.get("error") or "(no error field set)"
        empty_found.append(
            f"[{p['id']}] EMPTY response with {len(response)} chars — error: {err_note}"
        )
    response_preview = response[:200].strip().replace("\n", " ")
    print(f"\n  [{p['id']}] {p['name']}")
    print(f"  Response ({len(response)} chars): {response_preview}…")
    if not response.strip() and p.get("error"):
        print(f"  Error: {p['error']}")

if errors_found:
    print("\n[WARN] Some personas returned errors (Ollama issue, not code bug):")
    for e in errors_found:
        print(f"{e}")
    fail("Persona validation found errors. Check API logs for timeout, Ollama, or generation details.")
else:
    ok("No persona errors reported.")

if empty_found:
    print("\n[WARN] Some personas returned empty responses:")
    for e in empty_found:
        print(f"{e}")
    print(
        "\n  Tip: If you recently changed api.py/personas.py, restart the API server "
        "(python api.py). An old process may return HTTP 200 with empty bodies."
    )
    fail("Persona validation found empty responses. Check Ollama load, timeout, or retry behavior.")
else:
    ok("All persona results contain non-empty responses.")

# ── Step 5: Schema check on first persona ─────────────────────────────────────
ok(f"All required keys present in persona results: {sorted(required_keys)}")

print("\n" + "=" * 60)
print("  === Phase 4 PASSED ===")
print("=" * 60 + "\n")
