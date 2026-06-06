"""
compare_test.py — Dense arama ile Hybrid (Dense + Sparse) aramayı karşılaştıran test scripti.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from sentence_transformers import SentenceTransformer
from search import search, DB_DIR, COLLECTION_NAME, EMBED_MODEL

def test_compare(query: str):
    print("=" * 60)
    print(f"KARŞILAŞTIRMA TESTİ: {query}")
    print("=" * 60)
    
    # Sadece Dense Arama
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL)
    query_embedding = model.encode([query])[0].tolist()
    
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    print("\n[1] SADECE DENSE ARAMA (ChromaDB) TOP 3:")
    print("-" * 50)
    for i in range(len(dense_results["ids"][0])):
        chunk_id = dense_results["ids"][0][i]
        text = dense_results["documents"][0][i][:100].replace("\n", " ")
        source = dense_results["metadatas"][0][i]["source"]
        print(f"{i+1}. [ID: {chunk_id}] {source} => {text}...")

    # Hybrid Arama (Dense + BM25 + RRF) - `search.py` içindeki search fonksiyonu
    hybrid_results = search(query, top_k=3)
    
    print("\n[2] HİBRİT ARAMA (Dense + BM25 + RRF) TOP 3:")
    print("-" * 50)
    for i, res in enumerate(hybrid_results[:3]):
        chunk_id = res["id"]
        text = res["text"][:100].replace("\n", " ")
        source = res["metadata"]["source"]
        print(f"{i+1}. [ID: {chunk_id}] {source} => {text}...")

if __name__ == "__main__":
    test_query = "Transformer mimarisinin avantajları nelerdir?"
    test_compare(test_query)
