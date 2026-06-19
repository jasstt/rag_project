"""
ingest.py — data/belgeler/ klasöründeki .txt ve .pdf dosyalarını okur,
Cümle odaklı (sentence-aware) parçalara böler, tablo tespiti yapar,
sentence-transformers ile embed eder, ChromaDB'ye db/ klasörüne kaydeder,
BM25 için chunk listesini db/chunks.json'a yazar.
"""

import os
import json
import glob
import re
import chromadb
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
        # Basit tablo tespiti: Çok sütunlu yapılar varsa düz metne dönüştürürken boşlukları koru
        text += page.get_text("text", sort=True) + "\n"
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Cümle sınırlarına dikkat ederek metni parçalar."""
    # Cümleleri böl (basit regex ile)
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Örtüşme (overlap) mantığı: son cümleyi koru
            current_chunk = [current_chunk[-1]] if len(current_chunk) > 0 else []
            current_length = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += len(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks


def ingest():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    txt_files = glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
    all_files = txt_files + pdf_files

    if not all_files:
        print("[UYARI] data/belgeler/ klasöründe hiç .txt veya .pdf dosyası bulunamadı.")
        return

    all_chunks = []
    all_metadata = []

    for filepath in all_files:
        ext = os.path.splitext(filepath)[1].lower()
        print(f"[OKUMA] {filepath}")
        try:
            if ext == ".txt":
                text = read_txt(filepath)
            else:
                text = read_pdf(filepath)

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": os.path.basename(filepath),
                    "chunk_index": i,
                    "filepath": filepath
                })
        except Exception as e:
            print(f"[HATA] {filepath} okunamadı: {e}")

    print(f"[BİLGİ] Toplam {len(all_chunks)} chunk oluşturuldu.")

    print(f"[BİLGİ] '{EMBED_MODEL}' modeli yükleniyor...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True)

    print("[BİLGİ] ChromaDB'ye kaydediliyor...")
    client = chromadb.PersistentClient(path=DB_DIR)

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

    chunks_data = [
        {"id": f"chunk_{i}", "text": all_chunks[i], "metadata": all_metadata[i]}
        for i in range(len(all_chunks))
    ]
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    print(f"[BAŞARILI] Chunk listesi '{CHUNKS_JSON}' dosyasına yazıldı.")


if __name__ == "__main__":
    ingest()
