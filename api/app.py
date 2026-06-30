"""
api/app.py — FastAPI application exposing the RAG pipeline as HTTP endpoints.

Endpoints:
  POST /query  — Run the full pipeline (retrieval → rerank → generate)
  POST /ingest — Trigger document ingestion
  GET  /health — Service health check

Run: uvicorn api.app:app --reload
"""

import sys
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI not installed. Run: pip install fastapi uvicorn"
    )

from retrieval.hybrid_search import search
from llm.reranker import rerank, get_skip_stats
from llm.gemini_client import generate
from utils.helpers import run_ingest

app = FastAPI(
    title="Hybrid RAG API",
    description="Dense + BM25 + RRF retrieval with Gemini generation",
    version="1.1.0",
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 20


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    skip_stats: dict


class IngestResponse(BaseModel):
    status: str
    message: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "Hybrid RAG API v1.1"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """Run the full RAG pipeline for a given question."""
    try:
        candidates = search(request.question, top_k=request.top_k)
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="No relevant chunks found. Run /ingest first.",
            )

        top_chunks = rerank(request.question, candidates)
        result = generate(request.question, top_chunks)

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            skip_stats=get_skip_stats(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint():
    """Trigger document ingestion pipeline."""
    try:
        run_ingest()
        return IngestResponse(status="success", message="Ingestion completed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
