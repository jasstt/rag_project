"""
retrieval/hybrid_search.py — Hybrid search: Dense + Sparse (BM25) fused via RRF.

Pipeline:
  1. Dense search: ChromaDB cosine similarity on query embedding.
  2. Sparse search: BM25 lexical matching on raw chunk text.
  3. RRF fusion: Reciprocal Rank Fusion combines both ranked lists.

Returns top-K chunks with rrf_score and rank metadata.
"""

import os
import json
import yaml
import chromadb
from rank_bm25 import BM25Okapi

from embeddings.encoder import encode_query


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg
    except Exception:
        return {}


def _get_db_settings(cfg: dict) -> tuple[str, str, str]:
    vdb = cfg.get("vectordb", {})
    paths = cfg.get("paths", {})
    db_dir = os.path.join(os.path.dirname(__file__), "..", vdb.get("db_dir", "db"))
    collection_name = vdb.get("collection_name", "rag_collection")
    chunks_json = os.path.join(os.path.dirname(__file__), "..", paths.get("chunks_json", "db/chunks.json"))
    return db_dir, collection_name, chunks_json


def load_bm25(chunks_json_path: str) -> tuple:
    """Build BM25 index from chunks.json."""
    if not os.path.exists(chunks_json_path):
        raise FileNotFoundError(
            f"'{chunks_json_path}' not found. Run ingest first: python utils/helpers.py"
        )
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    corpus = [item["text"].lower().split() for item in chunks_data]
    bm25 = BM25Okapi(corpus)
    return bm25, chunks_data


def rrf_fusion(dense_ids: list[str], sparse_ids: list[str], k: int = 60) -> list[tuple]:
    """
    Reciprocal Rank Fusion (RRF).

    Args:
        dense_ids: Ordered chunk IDs from dense search.
        sparse_ids: Ordered chunk IDs from sparse search.
        k: RRF constant (default 60).

    Returns:
        List of (chunk_id, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, chunk_id in enumerate(sparse_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search(query: str, top_k: int = None) -> list[dict]:
    """
    Hybrid search: Dense (ChromaDB) + Sparse (BM25) → RRF.

    Args:
        query: User query string.
        top_k: Number of results to return (default from config.yaml).

    Returns:
        List of dicts: {id, text, metadata, rrf_score, rank}
    """
    cfg = _load_config()
    retrieval_cfg = cfg.get("retrieval", {})
    if top_k is None:
        top_k = retrieval_cfg.get("top_k", 20)
    rrf_k = retrieval_cfg.get("rrf_k", 60)

    db_dir, collection_name, chunks_json = _get_db_settings(cfg)

    # --- Dense Search ---
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(collection_name)

    query_embedding = encode_query(query)
    n_results = min(top_k, collection.count())

    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = dense_results["ids"][0]

    # --- Sparse Search (BM25) ---
    bm25, chunks_data = load_bm25(chunks_json)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    ranked_bm25 = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )
    sparse_ids = [chunks_data[i]["id"] for i in ranked_bm25[:top_k]]

    # --- RRF Fusion ---
    fused = rrf_fusion(dense_ids, sparse_ids, k=rrf_k)

    id_to_chunk = {item["id"]: item for item in chunks_data}

    results = []
    for rank, (chunk_id, score) in enumerate(fused[:top_k]):
        chunk = id_to_chunk.get(chunk_id)
        if chunk:
            results.append({
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "rrf_score": score,
                "rank": rank + 1,
            })

    return results


if __name__ == "__main__":
    query = input("Enter search query: ")
    results = search(query)
    print(f"\nTop {len(results)} results:")
    for r in results:
        print(f"  [{r['rank']}] RRF={r['rrf_score']:.4f} | {r['metadata']['source']} — {r['text'][:100]}...")
