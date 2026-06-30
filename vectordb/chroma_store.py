"""
vectordb/chroma_store.py — Manages ChromaDB storage and similarity search.

Responsibility:
  - Create / reset the collection.
  - Store chunks + embeddings + metadata.
  - Query the collection and return matching document IDs.
"""

import os
import yaml
import chromadb


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("vectordb", {})
    except Exception:
        return {}


def _get_client_and_collection(db_dir: str = None, collection_name: str = None):
    """Return (client, collection) for the persistent ChromaDB store."""
    cfg = _load_config()
    if db_dir is None:
        db_dir = os.path.join(
            os.path.dirname(__file__), "..", cfg.get("db_dir", "db")
        )
    if collection_name is None:
        collection_name = cfg.get("collection_name", "rag_collection")

    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(collection_name)
    return client, collection


def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    db_dir: str = None,
    collection_name: str = None,
) -> int:
    """
    Store chunks, embeddings, and metadata into ChromaDB.
    Deletes the existing collection before writing (fresh ingest).

    Returns:
        Number of chunks stored.
    """
    cfg = _load_config()
    if db_dir is None:
        db_dir = os.path.join(
            os.path.dirname(__file__), "..", cfg.get("db_dir", "db")
        )
    if collection_name is None:
        collection_name = cfg.get("collection_name", "rag_collection")

    os.makedirs(db_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=db_dir)

    # Drop existing collection for clean re-ingest
    try:
        client.delete_collection(collection_name)
        print(f"[VECTORDB] Existing collection '{collection_name}' deleted.")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"[VECTORDB] {len(chunks)} chunks stored in '{collection_name}'.")
    return len(chunks)


def query_collection(
    query_embedding: list[float],
    top_k: int = 20,
    db_dir: str = None,
    collection_name: str = None,
) -> dict:
    """
    Query the ChromaDB collection for nearest neighbors.

    Returns:
        Raw ChromaDB query result dict with keys: ids, documents, metadatas, distances.
    """
    _, collection = _get_client_and_collection(db_dir, collection_name)

    n = min(top_k, collection.count())
    if n == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
