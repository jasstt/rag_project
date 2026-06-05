"""
ingest.py — data/belgeler/ klasöründeki .txt ve .pdf dosyalarını okur,
500 karakterlik parçalara böler (50 karakter örtüşme),
sentence-transformers ile embed eder, ChromaDB'ye db/ klasörüne kaydeder,
BM25 için chunk listesini db/chunks.json'a yazar.
"""

import os
import json
import glob
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# --- Sabitler ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "belgeler")
DB_DIR   = os.path.join(os.path.dirname(__file__), "..", "db")
CHUNKS_JSON = os.path.join(DB_DIR, "chunks.json")
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 50
COLLECTION_NAME = "rag_collection"
EMBED_MODEL = "all-MiniLM-L6-v2"


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Metni sabit boyutlu ve örtüşmeli parçalara böler."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += chunk_size - overlap
    return chunks


def ingest():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Dosyaları topla
    txt_files = glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
    all_files = txt_files + pdf_files

    if not all_files:
        print("[UYARI] data/belgeler/ klasöründe hiç .txt veya .pdf dosyası bulunamadı.")
        return

    # Metinleri ve kaynak bilgilerini topla
    all_chunks = []
    all_metadata = []

    for filepath in all_files:
        ext = os.path.splitext(filepath)[1].lower()
        print(f"[OKUMA] {filepath}")
        if ext == ".txt":
            text = read_txt(filepath)
        elif ext == ".pdf":
            text = read_pdf(filepath)
        else:
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": os.path.basename(filepath),
                "chunk_index": i,
                "filepath": filepath
            })

    print(f"[BİLGİ] Toplam {len(all_chunks)} chunk oluşturuldu.")

    # Embedding modeli yükle
    print(f"[BİLGİ] '{EMBED_MODEL}' modeli yükleniyor...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True)

    # ChromaDB'ye kaydet
    print("[BİLGİ] ChromaDB'ye kaydediliyor...")
    client = chromadb.PersistentClient(path=DB_DIR)

    # Mevcut koleksiyonu sil (yeniden ingest için)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    collection.add(
        ids=ids,
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        metadatas=all_metadata
    )
    print(f"[BAŞARILI] {len(all_chunks)} chunk ChromaDB'ye eklendi.")

    # BM25 için chunks.json'a kaydet
    chunks_data = [
        {"id": f"chunk_{i}", "text": all_chunks[i], "metadata": all_metadata[i]}
        for i in range(len(all_chunks))
    ]
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    print(f"[BAŞARILI] Chunk listesi '{CHUNKS_JSON}' dosyasına yazıldı.")


if __name__ == "__main__":
    ingest()
