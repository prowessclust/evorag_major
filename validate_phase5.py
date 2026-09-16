"""
validate_phase5.py - EvoRAG Phase 5 consensus sanity check.

Prerequisites:
    Terminal 1:  ollama serve
    Terminal 2:  python api.py

Exit code 0 = PASSED, 1 = FAILED.
"""

import sys
from pathlib import Path

import httpx

API_URL = "http://127.0.0.1:8000"
SCORE_STORE = Path("score_store.json")


def fail(msg: str) -> None:
    """Print a failure and exit non-zero."""
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    """Print a successful validation step."""
    print(f"[OK]   {msg}")


print("\n" + "=" * 60)
print("  EvoRAG Phase 5 - Consensus Voting Validation")
print("  (Make sure 'python api.py' and Ollama are running)")
print("=" * 60)

print("\n[...] GET /health")
try:
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_URL}/health")
except httpx.ConnectError:
    fail("Cannot connect to API server at http://127.0.0.1:8000")

if resp.status_code != 200:
    fail(f"/health returned HTTP {resp.status_code}: {resp.text}")
ok(f"Health check passed - {resp.json().get('index_vectors')} vectors in index")

before_size = SCORE_STORE.stat().st_size if SCORE_STORE.exists() else 0
PHASE5_QUERY = "Should AI be used in judicial judgment writing, and what are the risks of doing so?"
query = PHASE5_QUERY

CONSENSUS_TIMEOUT_SECONDS = 1800  # 30 minutes — covers worst-case 17 calls × 120s on CPU

print("\n[...] POST /query/consensus")
print(f"      Query: {query}")
print("      This can take up to 30 minutes on CPU (17 LLM calls).")
print("      Do not interrupt. Progress is logged in api.py terminal.\n")

try:
    with httpx.Client(timeout=CONSENSUS_TIMEOUT_SECONDS) as client:
        resp = client.post(f"{API_URL}/query/consensus", json={"query": query, "top_k": 2})
except httpx.ReadTimeout:
    fail(f"Consensus query timed out after {CONSENSUS_TIMEOUT_SECONDS // 60} minutes.")

if resp.status_code != 200:
    fail(f"/query/consensus returned HTTP {resp.status_code}: {resp.text}")

data = resp.json()
required = {"final_answer", "top_personas", "scores", "all_responses", "score_matrix", "query_id"}
missing = required - set(data.keys())
if missing:
    fail(f"Consensus response missing keys: {sorted(missing)}")

if not data["final_answer"].strip():
    fail("final_answer is empty.")
if len(data["all_responses"]) != 8:
    fail(f"Expected 8 persona responses, got {len(data['all_responses'])}.")
if not data["top_personas"]:
    fail("No top personas selected.")
if len(data["score_matrix"]) != 8:
    fail(f"Expected 8 score matrix rows, got {len(data['score_matrix'])}.")

after_size = SCORE_STORE.stat().st_size if SCORE_STORE.exists() else 0
if after_size <= before_size:
    fail("score_store.json was not appended.")

ok(f"Final answer returned ({len(data['final_answer'])} chars)")
ok(f"Top personas: {', '.join(data['top_personas'])}")
ok(f"Query ID stored: {data['query_id']}")

print("\n" + "=" * 60)
print("  === Phase 5 PASSED ===")
print("=" * 60 + "\n")
