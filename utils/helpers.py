"""
utils/helpers.py — Ingest pipeline orchestration.

Coordinates: ingestion/ → chunking/ → embeddings/ → vectordb/
Run directly: python utils/helpers.py
"""

import os
import sys
import glob
import json
import yaml

# Allow imports from project root
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from ingestion.loader import load_file
from chunking.splitter import chunk_text
from embeddings.encoder import encode
from vectordb.chroma_store import store_chunks


def _load_config() -> dict:
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_ingest(data_dir: str = None, db_dir: str = None):
    """
    Full ingest pipeline:
      1. Load all .txt and .pdf files from data_dir.
      2. Chunk each document (sentence-aware).
      3. Encode chunks into embeddings.
      4. Store in ChromaDB.
      5. Write chunks.json for BM25.

    Args:
        data_dir: Override data directory (default: config.yaml paths.data_dir).
        db_dir:   Override DB directory (default: config.yaml vectordb.db_dir).
    """
    cfg = _load_config()
    paths_cfg = cfg.get("paths", {})
    vdb_cfg = cfg.get("vectordb", {})
    chunking_cfg = cfg.get("chunking", {})

    if data_dir is None:
        data_dir = os.path.join(ROOT, paths_cfg.get("data_dir", "data/belgeler"))
    if db_dir is None:
        db_dir = os.path.join(ROOT, vdb_cfg.get("db_dir", "db"))

    chunks_json_path = os.path.join(ROOT, paths_cfg.get("chunks_json", "db/chunks.json"))

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)

    # 1. Discover files
    all_files = (
        glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)
    )

    if not all_files:
        print(f"[INGEST] No .txt or .pdf files found in: {data_dir}")
        return

    print(f"[INGEST] Found {len(all_files)} file(s).")

    all_chunks: list[str] = []
    all_metadata: list[dict] = []

    # 2. Load + Chunk
    for filepath in all_files:
        print(f"[INGEST] Reading: {filepath}")
        try:
            text = load_file(filepath)
            chunks = chunk_text(
                text,
                chunk_size=chunking_cfg.get("chunk_size", 500),
                overlap=chunking_cfg.get("chunk_overlap", 50),
            )
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": os.path.basename(filepath),
                    "chunk_index": i,
                    "filepath": filepath,
                })
        except Exception as e:
            print(f"[INGEST] Error reading {filepath}: {e}")

    print(f"[INGEST] Total chunks created: {len(all_chunks)}")

    # 3. Encode
    print("[INGEST] Encoding chunks...")
    embeddings = encode(all_chunks, show_progress=True)

    # 4. Store in ChromaDB
    store_chunks(
        chunks=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata,
        db_dir=db_dir,
        collection_name=vdb_cfg.get("collection_name", "rag_collection"),
    )

    # 5. Write chunks.json for BM25
    chunks_data = [
        {"id": f"chunk_{i}", "text": all_chunks[i], "metadata": all_metadata[i]}
        for i in range(len(all_chunks))
    ]
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    print(f"[INGEST] Chunks written to: {chunks_json_path}")
    print(f"[INGEST] ✓ Done. {len(all_chunks)} chunks indexed.")


if __name__ == "__main__":
    run_ingest()
