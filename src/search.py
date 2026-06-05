"""
search.py — Gelen soruyu hem ChromaDB'de dense search hem BM25 ile sparse search yaparak arar,
iki sonucu RRF (Reciprocal Rank Fusion) ile birleştirir, top 20 döner.
"""

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "db")
CHUNKS_JSON = os.path.join(DB_DIR, "chunks.json")
COLLECTION_NAME = "rag_collection"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 20
RRF_K = 60  # RRF sabitesi


def load_bm25():
    """chunks.json'dan BM25 indeksini yükler."""
    if not os.path.exists(CHUNKS_JSON):
        raise FileNotFoundError(
            f"'{CHUNKS_JSON}' bulunamadı. Lütfen önce 'python src/ingest.py' çalıştırın."
        )
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    corpus = [item["text"].lower().split() for item in chunks_data]
    bm25 = BM25Okapi(corpus)
    return bm25, chunks_data


def rrf_fusion(dense_ids: list, sparse_ids: list, k: int = RRF_K) -> list:
    """
    Reciprocal Rank Fusion (RRF) ile iki sıralı listeyi birleştirir.
    Dönen liste: [(chunk_id, score), ...] şeklinde sıralıdır.
    """
    scores: dict[str, float] = {}

    for rank, chunk_id in enumerate(dense_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, chunk_id in enumerate(sparse_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_ids


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Hibrit arama: Dense (ChromaDB) + Sparse (BM25) → RRF birleştirme.
    Returns: top_k adet {"id", "text", "metadata", "score"} sözlükleri.
    """
    # --- Dense Search ---
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBED_MODEL)
    query_embedding = model.encode([query])[0].tolist()

    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    dense_ids = dense_results["ids"][0]

    # --- Sparse Search (BM25) ---
    bm25, chunks_data = load_bm25()
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # En yüksek BM25 skorlarına göre sırala
    ranked_bm25 = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )
    sparse_ids = [chunks_data[i]["id"] for i in ranked_bm25[:top_k]]

    # --- RRF Birleştirme ---
    fused = rrf_fusion(dense_ids, sparse_ids)
    top_ids = [chunk_id for chunk_id, _ in fused[:top_k]]

    # Chunk verilerini id'ye göre eşle
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
                "rank": rank + 1
            })

    return results


if __name__ == "__main__":
    query = input("Arama sorgusu girin: ")
    results = search(query)
    print(f"\nTop {len(results)} sonuç:")
    for r in results:
        print(f"  [{r['rank']}] (RRF={r['rrf_score']:.4f}) {r['metadata']['source']} — {r['text'][:100]}...")
