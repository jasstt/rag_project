"""
main.py — Kullanıcıdan soru alır, sırayla search → rerank → generate adımlarını çağırır,
yanıtı ve kaynak parçalarını ekrana basar.
"""

import sys
import os

# Windows konsolunda UTF-8 zorla
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# src/ dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from search import search
from rerank import rerank
from generate import generate


def print_separator(char="=", width=60):
    print(char * width)


def run_pipeline(query: str):
    print_separator()
    print(f"❓ SORU: {query}")
    print_separator()

    # 1. Arama
    print("\n🔍 [1/3] Hibrit arama yapılıyor (Dense + BM25 + RRF)...")
    candidates = search(query)
    print(f"   → {len(candidates)} aday chunk bulundu.")

    if not candidates:
        print("[UYARI] Hiçbir sonuç bulunamadı. Lütfen önce 'python src/ingest.py' çalıştırın.")
        return

    # 2. Reranking
    print("\n🎯 [2/3] Reranking yapılıyor (Gemini)...")
    top5 = rerank(query, candidates)
    print(f"   → Top {len(top5)} parça seçildi.")

    # 3. Yanıt üretimi
    print("\n✍️  [3/3] Yanıt üretiliyor (Gemini)...")
    result = generate(query, top5)

    # Sonuçları göster
    print_separator("=")
    print("📝 YANIT:")
    print_separator("-")
    print(result["answer"])

    print_separator("=")
    print("📚 KULLANILAN KAYNAKLAR:")
    print_separator("-")
    for source in result["sources"]:
        print(f"  [{source['number']}] {source['source']}")
        print(f"      {source['text_preview']}...")
        print()

    print_separator()


def main():
    print("=" * 50)
    print("   RAG Pipeline -- Soru-Cevap")
    print("=" * 50)
    print("  Cikmak icin 'q' veya 'exit' yazin.\n")

    while True:
        try:
            query = input("Sorunuzu girin: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkılıyor...")
            break

        if not query:
            continue
        if query.lower() in ("q", "exit", "quit", "çık"):
            print("Çıkılıyor...")
            break

        try:
            run_pipeline(query)
        except Exception as e:
            print(f"\n[HATA] {e}")
            import traceback
            traceback.print_exc()

        print()


if __name__ == "__main__":
    main()
