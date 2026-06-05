"""
eval.py — eval_set.json içindeki soru-cevap çiftlerini pipeline'dan geçirir,
her yanıt için citation doğrulaması yapar, doğruluk oranını raporlar.
"""

import os
import json
import re
import sys

# src/ dizinini path'e ekle
sys.path.insert(0, os.path.dirname(__file__))

from search import search
from rerank import rerank
from generate import generate

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "eval_set.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "eval_report.json")


def extract_citations(text: str) -> list[int]:
    """Yanıt metnindeki [N] formatındaki atıfları döndürür."""
    return [int(m) for m in re.findall(r'\[(\d+)\]', text)]


def check_citation_validity(answer: str, sources: list[dict]) -> dict:
    """
    Atıf doğrulaması:
    - citation_count: toplam atıf sayısı
    - valid_citations: geçerli atıf sayısı (var olan kaynak numaralarına işaret eden)
    - invalid_citations: geçersiz atıf numaraları
    - citation_coverage: atıfla desteklenen cümle oranı
    """
    citations = extract_citations(answer)
    valid_source_nums = {s["number"] for s in sources}
    valid = [c for c in citations if c in valid_source_nums]
    invalid = [c for c in citations if c not in valid_source_nums]

    # Cümle bazlı kapsam
    sentences = [s.strip() for s in answer.replace(".", ".\n").split("\n") if s.strip()]
    cited_sentences = sum(1 for s in sentences if re.search(r'\[\d+\]', s))
    coverage = cited_sentences / len(sentences) if sentences else 0.0

    return {
        "citation_count": len(citations),
        "valid_citations": len(valid),
        "invalid_citations": invalid,
        "citation_coverage": round(coverage, 3)
    }


def has_relevant_answer(answer: str, expected_keywords: list[str]) -> bool:
    """Beklenen anahtar kelimelerin yanıtta geçip geçmediğini kontrol eder."""
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in expected_keywords)


def run_eval():
    # eval_set.json yükle
    if not os.path.exists(EVAL_SET_PATH):
        print(f"[HATA] '{EVAL_SET_PATH}' bulunamadı.")
        return

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    if not eval_set:
        print("[UYARI] eval_set.json boş. Lütfen soru-cevap çiftleri ekleyin.")
        print("Örnek format:")
        example = [
            {
                "question": "Yapay zeka nedir?",
                "expected_keywords": ["öğrenme", "yapay"],
                "notes": "Temel tanım sorusu"
            }
        ]
        print(json.dumps(example, ensure_ascii=False, indent=2))
        return

    results = []
    total = len(eval_set)
    passed = 0

    print(f"\n[EVAL] {total} soru değerlendiriliyor...\n{'='*50}")

    for i, item in enumerate(eval_set, start=1):
        question = item.get("question", "")
        expected_keywords = item.get("expected_keywords", [])

        print(f"\n[{i}/{total}] Soru: {question}")

        try:
            # Pipeline çalıştır
            candidates = search(question)
            top5 = rerank(question, candidates)
            result = generate(question, top5)

            answer = result["answer"]
            sources = result["sources"]

            # Doğrulama
            citation_check = check_citation_validity(answer, sources)
            relevance = has_relevant_answer(answer, expected_keywords) if expected_keywords else None

            item_result = {
                "question": question,
                "answer_preview": answer[:300],
                "source_count": len(sources),
                "citation_check": citation_check,
                "keyword_relevance": relevance,
                "passed": citation_check["valid_citations"] > 0 and (relevance is None or relevance)
            }

            if item_result["passed"]:
                passed += 1
                print(f"  ✓ BAŞARILI — Atıf: {citation_check['valid_citations']}/{citation_check['citation_count']}")
            else:
                print(f"  ✗ BAŞARISIZ — Atıf: {citation_check['valid_citations']}/{citation_check['citation_count']}, Keyword: {relevance}")

        except Exception as e:
            print(f"  [HATA] {e}")
            item_result = {
                "question": question,
                "error": str(e),
                "passed": False
            }

        results.append(item_result)

    # Rapor
    accuracy = passed / total if total > 0 else 0.0
    report = {
        "total_questions": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(accuracy, 3),
        "results": results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"[SONUÇ] Doğruluk oranı: {accuracy:.1%} ({passed}/{total})")
    print(f"[RAPOR] '{REPORT_PATH}' dosyasına kaydedildi.")


if __name__ == "__main__":
    run_eval()
