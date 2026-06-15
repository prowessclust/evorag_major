"""
api.py — EvoRAG Phase 3+4: FastAPI Application

Endpoints:
    GET  /health            → {"status": "ok", "index_vectors": N, "model": "..."}
    POST /query             → single RAG answer (Phase 3)
    POST /query/personas    → 8 parallel persona responses (Phase 4)

Run:
    python api.py
    -- or --
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    API_HOST,
    API_PORT,
    OLLAMA_MODEL,
    RETRIEVAL_TOP_K,
)
from embedder import load_index
from retriever import answer_async
from personas import run_all_personas
from pipeline import evorag_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Shared state loaded once at startup ───────────────────────────────────────
_state: dict = {}


def _preview(text: str, limit: int = 100) -> str:
    """Return a compact single-line preview for logs."""
    return text[:limit].strip().replace("\n", " ")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index once when the server starts."""
    log.info("EvoRAG API starting — loading FAISS index…")
    try:
        index, metadata = load_index()
        _state["index"] = index
        _state["metadata"] = metadata
        log.info(f"Index ready: {index.ntotal} vectors, dim={index.d}")
    except FileNotFoundError as e:
        log.error(str(e))
        log.error("Run 'python embedder.py --build' to create the index first.")
        raise RuntimeError(str(e))
    yield
    log.info("EvoRAG API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EvoRAG API",
    description="Evolutionary Multi-Personality RAG — retrieval, personas, and consensus voting",
    version="0.5.0",
    lifespan=lifespan,
)

# Allow all origins for local development (Phase 8 React frontend will need this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    top_k: int = Field(RETRIEVAL_TOP_K, ge=1, le=20, description="Chunks to retrieve")


class SourceItem(BaseModel):
    title:  str
    url:    str
    source: str
    score:  float


class QueryResponse(BaseModel):
    query:   str
    answer:  str
    sources: List[SourceItem]


# Phase 4 schemas
class PersonaQueryRequest(BaseModel):
    query:       str           = Field(..., min_length=1, max_length=1000)
    top_k:       int           = Field(RETRIEVAL_TOP_K, ge=1, le=20)
    persona_ids: Optional[List[str]] = Field(
        None,
        description="Subset of persona IDs to run. Omit for all 8."
    )


class PersonaResult(BaseModel):
    id:       str
    name:     str
    response: str
    error:    str


class PersonaQueryResponse(BaseModel):
    query:    str
    sources:  List[SourceItem]
    personas: List[PersonaResult]


class ConsensusQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(RETRIEVAL_TOP_K, ge=1, le=20)


class ConsensusQueryResponse(BaseModel):
    final_answer:  str
    top_personas:  List[str]
    scores:        Dict[str, float]
    all_responses: Dict[str, Optional[str]]
    score_matrix:  Dict[str, Dict[str, Optional[Dict[str, float]]]]
    query_id:      str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    """Returns server status and the number of vectors in the loaded index."""
    index = _state.get("index")
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded yet.")
    return {
        "status": "ok",
        "index_vectors": index.ntotal,
        "embedding_dim": index.d,
        "model": OLLAMA_MODEL,
    }


@app.post("/query", response_model=QueryResponse, summary="RAG query")
async def query_endpoint(req: QueryRequest):
    """
    Full retrieval-augmented generation pipeline:
      1. Embed the query
      2. Retrieve top-k chunks from FAISS
      3. Assemble RAG prompt
      4. Send to Ollama (phi3:latest)
      5. Return answer + source attribution
    """
    index    = _state.get("index")
    metadata = _state.get("metadata")

    if index is None or metadata is None:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    log.info(f"POST /query — '{req.query[:60]}' (top_k={req.top_k})")

    try:
        result = await answer_async(req.query, top_k=req.top_k)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error during query processing")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
    )


@app.post("/query/personas", response_model=PersonaQueryResponse, summary="Multi-persona RAG")
async def personas_endpoint(req: PersonaQueryRequest):
    """
    Phase 4 — Multi-Persona Engine:
      1. Retrieve top-k chunks (once, shared)
      2. Build shared RAG context prompt
      3. Run each persona concurrently via asyncio.gather()
      4. Return all persona responses + source attribution

    Use persona_ids=["analytical-critical", "empirical-evidential"] for fast testing.
    Omit persona_ids to run all 8 personas.
    """
    if _state.get("index") is None:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    log.info(
        f"POST /query/personas — '{req.query[:60]}' "
        f"(top_k={req.top_k}, personas={req.persona_ids or 'all'})"
    )

    try:
        result = await run_all_personas(
            query=req.query,
            top_k=req.top_k,
            persona_ids=req.persona_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error during persona processing")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    empty_personas = []
    for persona in result["personas"]:
        response = (persona.get("response") or "").strip()
        log.info(
            "Persona endpoint result: name=%s length=%s preview='%s' error=%s",
            persona.get("name", ""),
            len(response),
            _preview(response),
            persona.get("error") or "",
        )
        if not response:
            empty_personas.append({
                "id": persona.get("id", ""),
                "name": persona.get("name", ""),
                "error": persona.get("error") or "Empty response returned.",
            })

    if empty_personas:
        log.error("Persona endpoint refusing success because empty responses remain: %s", empty_personas)
        raise HTTPException(
            status_code=502,
            detail={
                "message": "One or more personas returned empty responses after retries.",
                "empty_personas": empty_personas,
            },
        )

    return PersonaQueryResponse(
        query=result["query"],
        sources=[SourceItem(**s) for s in result["sources"]],
        personas=[PersonaResult(**p) for p in result["personas"]],
    )


@app.post("/query/consensus", response_model=ConsensusQueryResponse, summary="EvoRAG consensus query")
async def consensus_endpoint(req: ConsensusQueryRequest):
    """
    Phase 5 — Consensus Voting:
      1. Run all 8 personas on shared retrieved context
      2. Ask personas to score peer responses
      3. Select the top 3 personas by peer score
      4. Synthesize a final answer and append score_store.json
    """
    if _state.get("index") is None:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    log.info(f"POST /query/consensus — '{req.query[:60]}' (top_k={req.top_k})")

    try:
        result = await evorag_query(req.query, top_k=req.top_k)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error during consensus processing")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return ConsensusQueryResponse(**result)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
