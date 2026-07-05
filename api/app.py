"""
api/app.py — FastAPI application exposing the RAG pipeline as HTTP endpoints.

Endpoints:
  POST /query  — Run the full pipeline (retrieval → rerank → generate)
  POST /ingest — Trigger document ingestion
  GET  /health — Service health check
  GET  /       — Serve the Web Chat UI (index.html)

Run: uvicorn api.app:app --reload --port 8000
"""

import sys
import os
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Run: pip install fastapi uvicorn[standard]")

from retrieval.hybrid_search import search
from llm.reranker import rerank, get_skip_stats
from llm.gemini_client import generate
from utils.helpers import run_ingest

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hybrid RAG API",
    description="Dense + BM25 + RRF retrieval with Gemini generation",
    version="1.2.0",
)

# Allow all origins so the web UI can call the API from any port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (web UI) from api/static/
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 20


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    skip_stats: dict
    latency_ms: float


class IngestResponse(BaseModel):
    status: str
    message: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the Web Chat UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>RAG API is running. UI not found.</h1>")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Hybrid RAG API",
        "version": "1.2.0",
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """Run the full RAG pipeline for a given question."""
    t_start = time.time()
    try:
        candidates = search(request.question, top_k=request.top_k)
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="No relevant chunks found. Run /ingest first.",
            )

        top_chunks = rerank(request.question, candidates)
        result = generate(request.question, top_chunks)
        latency = round((time.time() - t_start) * 1000, 1)

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            skip_stats=get_skip_stats(),
            latency_ms=latency,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint():
    """Trigger document ingestion pipeline."""
    try:
        run_ingest()
        return IngestResponse(status="success", message="Ingestion completed successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
