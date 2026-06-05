"""
generate.py — rerank.py'den gelen 5 parçayı ve soruyu Google Gemini API'ye gönderir,
her iddiayı [1], [2] gibi kaynak numarasıyla destekleyen bir yanıt üretir.
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# .env dosyasından API anahtarını yükle
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-flash-latest"


def generate(query: str, context_chunks: list[dict]) -> dict:
    """
    Gemini ile kaynak atıflı yanıt üretir.
    Returns: {"answer": str, "sources": list[dict]}
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY ortam değişkeni ayarlanmamış. "
            "Lütfen .env dosyasına 'GEMINI_API_KEY=...' ekleyin."
        )

    if not context_chunks:
        return {"answer": "İlgili bilgi bulunamadı.", "sources": []}

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Bağlam metnini ve kaynak listesini hazırla
    context_parts = []
    sources = []
    for i, chunk in enumerate(context_chunks, start=1):
        source_name = chunk["metadata"].get("source", "bilinmeyen")
        context_parts.append(f"[{i}] (Kaynak: {source_name})\n{chunk['text']}")
        sources.append({
            "number": i,
            "id": chunk["id"],
            "source": source_name,
            "text_preview": chunk["text"][:200]
        })

    context_text = "\n\n".join(context_parts)

    system_instruction = (
        "Sen yardımcı bir Türkçe asistansın. "
        "Verilen bağlam parçalarını kullanarak soruyu yanıtla. "
        "Her bilgiyi [1], [2] gibi kaynak numarasıyla destekle. "
        "Bağlamda olmayan bilgileri uydurma, 'bu bilgi mevcut değil' de. "
        "Yanıtın sonunda 'Kaynaklar:' bölümü ekle."
    )

    user_prompt = (
        f"BAĞLAM PARÇALARI:\n{context_text}\n\n"
        f"SORU: {query}\n\n"
        "Yukarıdaki bağlam parçalarını kullanarak soruyu Türkçe yanıtla. "
        "Her bilgiye [N] formatında kaynak numarası ekle."
    )

    print(f"[GENERATE] Gemini'ye yanıt üretme isteği gönderiliyor...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=1024,
            temperature=0.2,
        )
    )

    answer = response.text.strip()
    print(f"[GENERATE] Yanıt üretildi ({len(answer)} karakter).")

    return {
        "answer": answer,
        "sources": sources
    }
