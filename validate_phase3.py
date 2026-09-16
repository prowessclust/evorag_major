"""
validate_phase3.py — EvoRAG Phase 3 sanity check.

IMPORTANT: Start the API server FIRST in a separate terminal, then run this.

    Terminal 1:  python api.py
    Terminal 2:  python validate_phase3.py

Exit code 0 = PASSED, 1 = FAILED.
"""

import sys
import httpx

API_URL = "http://127.0.0.1:8000"


def fail(msg: str):
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str):
    print(f"[OK]   {msg}")


def source_key(source: dict) -> str:
    """Return article-level identity for API source duplicate checks."""
    title = " ".join((source.get("title") or "").lower().split())
    return title or source.get("url") or source.get("source", "")


print("\n" + "=" * 60)
print("  EvoRAG Phase 3 — Validation")
print("  (Make sure 'python api.py' is running in another terminal)")
print("=" * 60)

# ── Step 1: GET /health ───────────────────────────────────────────────────────
print("\n[...] GET /health")
try:
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_URL}/health")
except httpx.ConnectError:
    fail(
        "Cannot connect to API server at http://127.0.0.1:8000\n"
        "       → Start it first with:  python api.py"
    )

if resp.status_code != 200:
    fail(f"/health returned HTTP {resp.status_code}: {resp.text}")

health = resp.json()
if health.get("status") != "ok":
    fail(f"/health status is not 'ok': {health}")

n_vectors = health.get("index_vectors", 0)
model_name = health.get("model", "?")
ok(f"Health check passed — {n_vectors} vectors in index, model: {model_name}")

# ── Step 2: POST /query ───────────────────────────────────────────────────────
print("\n[...] POST /query — 'What is the key to a successful AI strategy according to recent reports?'")
print("      phi3 on CPU is slow — this may take 60–120 seconds. Please wait...\n")

try:
    with httpx.Client(timeout=300) as client:
        resp = client.post(
            f"{API_URL}/query",
            json={"query": "What is the key to a successful AI strategy according to recent reports?", "top_k": 2},
        )
except httpx.ReadTimeout:
    fail(
        "phi3 took longer than 300 seconds.\n"
        "       → Run: ollama run phi3 'hello' to warm up the model, then retry."
    )

if resp.status_code != 200:
    fail(f"/query returned HTTP {resp.status_code}: {resp.text}")

data = resp.json()

if not data.get("answer", "").strip():
    fail("'answer' field is empty.")
if not isinstance(data.get("sources"), list):
    fail("'sources' is not a list.")

ok(f"POST /query returned answer ({len(data['answer'])} chars)")
ok(f"Sources returned: {len(data['sources'])}")

print("\n  --- Answer ---")
print(f"  {data['answer'][:400].strip()}")
print()
print("  --- Sources ---")
for i, s in enumerate(data["sources"]):
    print(f"  [{i}] score={s.get('score')} | {s.get('source')} | {(s.get('title') or '')[:55]}")

# ── Step 3: Validate source schema ────────────────────────────────────────────
if data["sources"]:
    required = {"title", "url", "source", "score"}
    missing = required - set(data["sources"][0].keys())
    if missing:
        fail(f"Source dict missing keys: {missing}")
ok("Source schema correct.")

if len({source_key(s) for s in data["sources"]}) != len(data["sources"]):
    fail("Duplicate source articles returned by /query. Restart api.py after dedupe changes.")
ok("Source list contains unique article titles.")

# ── Step 4: Second query ──────────────────────────────────────────────────────
print("\n[...] POST /query — 'Why are stocks and oil prices changing amid tensions between the US and Iran?'")
try:
    with httpx.Client(timeout=300) as client:
        resp2 = client.post(
            f"{API_URL}/query",
            json={"query": "Why are stocks and oil prices changing amid tensions between the US and Iran?", "top_k": 2},
        )
except httpx.ReadTimeout:
    fail("Second query timed out after 300 seconds.")

if resp2.status_code != 200:
    fail(f"Second /query returned HTTP {resp2.status_code}: {resp2.text}")
if not resp2.json().get("answer", "").strip():
    fail("Second query returned empty answer.")
ok(f"Second query answered ({len(resp2.json()['answer'])} chars).")

print("\n" + "=" * 60)
print("  === Phase 3 PASSED ===")
print("=" * 60 + "\n")
