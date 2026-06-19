"""
rerank.py — search.py'den gelen 20 sonucu rerank eder.

İki mod:
  - "gemini" (varsayılan): Google Gemini API ile reranking
  - "local": sentence-transformers cross-encoder ile yerel reranking

Skip-rerank optimizasyonu: RRF top-1 skoru top-2'den %40+ yüksekse
reranker'a gitmeden direkt top-3 döner.
"""

import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TOP_RERANK = 5
GEMINI_MODEL = "gemini-flash-latest"

# Skip-rerank eşiği: top1/top2 oran farkı bu değeri geçerse rerank atla
SKIP_RERANK_THRESHOLD = 0.40

# --- Skip-rerank istatistikleri (session genelinde) ---
_skip_stats = {"skipped": 0, "total": 0}


def _check_skip_rerank(candidates: list[dict]) -> bool:
    """
    RRF fusion top-1 skoru top-2'den SKIP_RERANK_THRESHOLD kadar
    yüksekse True döner (rerank atlanabilir).
    """
    if len(candidates) < 2:
        return True
    score1 = candidates[0].get("rrf_score", 0)
    score2 = candidates[1].get("rrf_score", 0)
    if score2 == 0:
        return True
    gap = (score1 - score2) / score2
    return gap >= SKIP_RERANK_THRESHOLD


def rerank_local(query: str, candidates: list[dict]) -> list[dict]:
    """
    Cross-encoder (ms-marco-MiniLM-L-6-v2) ile yerel reranking.
    sentence-transformers kütüphanesi gerekli.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("[UYARI] sentence-transformers yüklü değil. 'pip install sentence-transformers' çalıştırın.")
        return candidates[:TOP_RERANK]

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"[RERANK-LOCAL] Cross-encoder yükleniyor: {model_name}")
    model = CrossEncoder(model_name, max_length=512)

    pairs = [(query, c["text"][:512]) for c in candidates]
    scores = model.predict(pairs)

    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    reranked = [c for _, c in scored[:TOP_RERANK]]
    print(f"[RERANK-LOCAL] Top {len(reranked)} parça seçildi.")
    return reranked


def rerank_gemini(query: str, candidates: list[dict]) -> list[dict]:
    """
    Gemini API ile reranking (mevcut implementasyon).
    """
    from google import genai

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY ortam değişkeni ayarlanmamış. "
            "Lütfen .env dosyasına 'GEMINI_API_KEY=...' ekleyin."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

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

    print(f"[RERANK-GEMINI] {len(candidates)} aday gönderiliyor...")

    max_retries = 3
    retry_delay = 5
    raw = ""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            raw = response.text.strip()
            break
        except Exception as e:
            print(f"[UYARI] Gemini API hatası ({attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"        {retry_delay} saniye sonra tekrar deneniyor...")
                time.sleep(retry_delay)
            else:
                print("[HATA] Reranking başarısız. İlk 5 sonuç dönülüyor.")
                return candidates[:TOP_RERANK]

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

    if len(reranked) < TOP_RERANK:
        for chunk in candidates:
            if chunk["id"] not in seen_ids:
                reranked.append(chunk)
                seen_ids.add(chunk["id"])
            if len(reranked) >= TOP_RERANK:
                break

    print(f"[RERANK-GEMINI] Top {len(reranked)} parça seçildi.")
    return reranked


def rerank(query: str, candidates: list[dict], mode: str = None) -> list[dict]:
    """
    Ana rerank fonksiyonu.
    mode: "gemini" | "local" | None (config'den okur)
    
    Skip-rerank optimizasyonu: top-1, top-2'den %40+ yüksekse
    reranker atlanır, top-3 doğrudan döner.
    """
    # Config'den mode oku
    if mode is None:
        try:
            from config import RERANK_MODE
            mode = RERANK_MODE
        except ImportError:
            mode = "gemini"

    if not candidates:
        return []

    _skip_stats["total"] += 1

    # --- Skip-rerank kontrolü ---
    if _check_skip_rerank(candidates):
        _skip_stats["skipped"] += 1
        skipped = _skip_stats["skipped"]
        total = _skip_stats["total"]
        print(
            f"[SKIP-RERANK] ✓ Yüksek güven: top-1 skoru baskın. "
            f"Reranker atlandı. ({skipped}/{total} sorgu reranker'ı atladı)"
        )
        # Top-1 + sonraki 2 → toplam 3 (veya mevcut kadar)
        return candidates[:min(3, len(candidates))]

    # --- Normal reranking ---
    if mode == "local":
        return rerank_local(query, candidates)
    else:
        return rerank_gemini(query, candidates)


def get_skip_stats() -> dict:
    """Session boyunca kaç sorgu reranker'ı atladı."""
    return {
        "skipped": _skip_stats["skipped"],
        "total": _skip_stats["total"],
        "skip_rate": (
            f"{_skip_stats['skipped']}/{_skip_stats['total']} sorgu reranker'ı atladı"
            if _skip_stats["total"] > 0 else "Henüz sorgu yok"
        )
    }
