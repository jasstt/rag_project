"""
rerank.py — search.py'den gelen 20 sonucu Google Gemini API üzerinden rerank eder,
en alakalı 5 parçayı sıralı döner.
"""

import os
import json
from google import genai
from dotenv import load_dotenv

# .env dosyasından API anahtarını yükle
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TOP_RERANK = 5
GEMINI_MODEL = "gemini-flash-latest"


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Gemini ile reranking.
    candidates: search.py'den gelen {id, text, metadata, rrf_score} listesi
    Returns: en alakalı TOP_RERANK adet chunk (sıralı)
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY ortam değişkeni ayarlanmamış. "
            "Lütfen .env dosyasına 'GEMINI_API_KEY=...' ekleyin."
        )

    if not candidates:
        return []

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Prompt için chunk listesini hazırla
    chunks_text = ""
    for i, chunk in enumerate(candidates):
        preview = chunk["text"][:300].replace("\n", " ")
        chunks_text += f"\n[{i+1}] (Kaynak: {chunk['metadata']['source']})\n{preview}\n"

    prompt = (
        f"Aşağıdaki kullanıcı sorusuna en alakalı {TOP_RERANK} metin parçasını seç ve "
        f"sıralı indeks numaralarını JSON formatında döndür.\n\n"
        f"SORU: {query}\n\n"
        f"ADAY PARÇALAR ({len(candidates)} adet):\n{chunks_text}\n\n"
        f"SADECE şu JSON formatında yanıt ver, başka hiçbir şey yazma:\n"
        f'{{\"ranking\": [idx1, idx2, idx3, idx4, idx5]}}\n\n'
        f"İndeksler 1-tabanlı olmalı (1 ile {len(candidates)} arasında)."
    )

    print(f"[RERANK] Gemini'ye {len(candidates)} aday gönderiliyor...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    raw = response.text.strip()

    # JSON parse
    try:
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        ranking = parsed.get("ranking", [])
    except json.JSONDecodeError as e:
        print(f"[UYARI] JSON parse hatası: {e}. Ham yanıt: {raw}")
        return candidates[:TOP_RERANK]

    # Geçerli indeksleri filtrele (1-tabanlı → 0-tabanlı)
    reranked = []
    seen_ids = set()
    for idx in ranking:
        zero_idx = idx - 1
        if 0 <= zero_idx < len(candidates):
            chunk = candidates[zero_idx]
            if chunk["id"] not in seen_ids:
                reranked.append(chunk)
                seen_ids.add(chunk["id"])
        if len(reranked) >= TOP_RERANK:
            break

    # Yeterli sonuç yoksa kalan adaylarla doldur
    if len(reranked) < TOP_RERANK:
        for chunk in candidates:
            if chunk["id"] not in seen_ids:
                reranked.append(chunk)
                seen_ids.add(chunk["id"])
            if len(reranked) >= TOP_RERANK:
                break

    print(f"[RERANK] Top {len(reranked)} parça seçildi.")
    return reranked
