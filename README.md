# Hybrid RAG Pipeline

> **v1.2 — Modular Architecture** | Dense + BM25 + RRF + Reranking + Gemini Generation

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini-blueviolet?logo=google)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A production-inspired **Hybrid RAG (Retrieval-Augmented Generation)** system combining dense vector search, sparse BM25 retrieval, and Reciprocal Rank Fusion — now refactored into a clean, modular architecture.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  ingestion/ │ →  │  chunking/  │ →  │ embeddings/ │
│  loader.py  │    │ splitter.py │    │ encoder.py  │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                                      ┌──────▼──────┐
                                      │  vectordb/  │
                                      │chroma_store │
                                      └──────┬──────┘
                                             │
User Query ─────────────────────────► retrieval/
                                      hybrid_search.py
                                      (Dense + BM25 + RRF)
                                             │
                                      ┌──────▼──────┐
                                      │    llm/     │
                                      │ reranker.py │ ← Skip-rerank optimisation
                                      └──────┬──────┘
                                             │
                                      ┌──────▼──────┐
                                      │  prompts/   │
                                      │templates.py │
                                      └──────┬──────┘
                                             │
                                      ┌──────▼──────┐
                                      │    llm/     │
                                      │gemini_client│
                                      └──────┬──────┘
                                             │
                                      ┌──────▼──────┐
                                      │   Answer    │
                                      │ + Sources   │
                                      └─────────────┘
```

---

## Project Structure

```
rag-project/
├── config.yaml          # Central configuration (models, thresholds, paths)
├── main.py              # CLI entry point
├── requirements.txt
├── .env                 # API keys (not committed)
├── .gitignore
│
├── ingestion/           # File loading (txt, pdf)
│   └── loader.py
│
├── chunking/            # Sentence-aware document splitting
│   └── splitter.py
│
├── embeddings/          # SentenceTransformer encoder
│   └── encoder.py
│
├── vectordb/            # ChromaDB store & query
│   └── chroma_store.py
│
├── retrieval/           # Hybrid search: Dense + BM25 + RRF
│   └── hybrid_search.py
│
├── prompts/             # Prompt templates (versioned, reusable)
│   └── templates.py
│
├── llm/                 # LLM clients (Gemini) + reranker
│   ├── gemini_client.py
│   └── reranker.py
│
├── api/                 # FastAPI HTTP endpoints
│   └── app.py
│
├── utils/               # Pipeline orchestration helpers
│   └── helpers.py
│
├── tests/               # Evaluation harness + unit tests
│   ├── eval.py
│   └── test_skip_rerank.py
│
├── logs/                # Runtime logs (gitignored except .gitkeep)
├── data/belgeler/       # Source documents (.txt, .pdf)
└── db/                  # ChromaDB + chunks.json (gitignored)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API key

```bash
# .env
GEMINI_API_KEY=your_key_here
```

### 3. Add documents

Place `.txt` or `.pdf` files in `data/belgeler/`.

### 4. Ingest documents

```bash
python utils/helpers.py
```

### 5. Run Q&A

```bash
python main.py
```

### 6. (Optional) Run as API

```bash
uvicorn api.app:app --reload
# → POST http://localhost:8000/query  {"question": "..."}
# → POST http://localhost:8000/ingest
# → GET  http://localhost:8000/health
```

---

## Configuration

All settings are in `config.yaml`:

```yaml
embedding:
  model: "all-MiniLM-L6-v2"

chunking:
  chunk_size: 500
  chunk_overlap: 50

reranker:
  mode: "local"          # "local" | "gemini"
  skip_threshold: 0.03   # skip reranker when top-1 dominates

llm:
  model: "gemini-flash-latest"
  temperature: 0.2
```

---

## Key Features

| Feature | Module | Description |
|---|---|---|
| **Sentence-aware chunking** | `chunking/splitter.py` | Never cuts mid-sentence; table-line detection |
| **Hybrid retrieval** | `retrieval/hybrid_search.py` | Dense (ChromaDB) + Sparse (BM25) + RRF |
| **Skip-rerank** | `llm/reranker.py` | Bypass reranker for high-confidence queries |
| **Local reranker** | `llm/reranker.py` | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Prompt templates** | `prompts/templates.py` | Versioned, separated from model logic |
| **FastAPI** | `api/app.py` | REST endpoints for integration |
| **Eval harness** | `tests/eval.py` | Citation check + keyword relevance + latency |

---

## Evaluation

```bash
# Standard eval
python tests/eval.py

# Compare rerankers (Gemini vs Local latency)
python tests/eval.py --compare-rerankers

# Skip-rerank unit tests
python tests/test_skip_rerank.py
```

---

## Community Improvements (v1.1)

| Contributor | Improvement |
|---|---|
| **Ahmet Özel** | Sentence-aware chunking replacing fixed 500-char splits |
| **Gunjan Tailor** | Skip-rerank optimisation — bypass reranker for high-confidence queries |
| **Tae Kim** | Local cross-encoder fallback — eliminates Gemini API dependency for reranking |

---

## License

MIT
