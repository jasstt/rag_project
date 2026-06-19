"""
eval.py — eval_set.json içindeki soru-cevap çiftlerini pipeline'dan geçirir.

Yeni özellikler:
  - local vs Gemini reranker latency karşılaştırması (--compare-rerankers)
  - Skip-rerank tetiklenme oranı raporu
"""

import os
import json
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from search import search
from rerank import rerank, get_skip_stats
from generate import generate

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "eval_set.json")
REPORT_PATH   = os.path.join(os.path.dirname(__file__), "..", "eval_report.json")


def extract_citations(text: str) -> list[int]:
    return [int(m) for m in re.findall(r'\[(\d+)\]', text)]


def check_citation_validity(answer: str, sources: list[dict]) -> dict:
    citations = extract_citations(answer)
    valid_source_nums = {s["number"] for s in sources}
    valid   = [c for c in citations if c in valid_source_nums]
    invalid = [c for c in citations if c not in valid_source_nums]

    sentences = [s.strip() for s in answer.replace(".", ".\n").split("\n") if s.strip()]
    cited_sentences = sum(1 for s in sentences if re.search(r'\[\d+\]', s))
    coverage = cited_sentences / len(sentences) if sentences else 0.0

    return {
        "citation_count":    len(citations),
        "valid_citations":   len(valid),
        "invalid_citations": invalid,
        "citation_coverage": round(coverage, 3)
    }


def has_relevant_answer(answer: str, expected_keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in expected_keywords)


def run_single_mode(eval_set: list[dict], mode: str) -> tuple[list[dict], float]:
    """Belirli reranker moduyla tüm eval setini çalıştırır, ortalama latency döner."""
    results = []
    total_latency = 0.0

    for i, item in enumerate(eval_set, start=1):
        question = item.get("question", "")
        expected_keywords = item.get("expected_keywords", [])

        try:
            candidates = search(question)

            t0 = time.time()
            top_chunks = rerank(question, candidates, mode=mode)
            rerank_latency = (time.time() - t0) * 1000  # ms

            result = generate(question, top_chunks)
            answer  = result["answer"]
            sources = result["sources"]

            citation_check = check_citation_validity(answer, sources)
            relevance = has_relevant_answer(answer, expected_keywords) if expected_keywords else None
            passed = citation_check["valid_citations"] > 0 and (relevance is None or relevance)

            results.append({
                "question":         question,
                "answer_preview":   answer[:200],
                "rerank_latency_ms": round(rerank_latency, 1),
                "citation_check":   citation_check,
                "keyword_relevance": relevance,
                "passed":           passed
            })
            total_latency += rerank_latency

        except Exception as e:
            results.append({"question": question, "error": str(e), "passed": False, "rerank_latency_ms": 0})

    avg_latency = total_latency / len(eval_set) if eval_set else 0
    return results, avg_latency


def run_compare_rerankers(eval_set: list[dict]):
    """Gemini vs Local reranker latency karşılaştırması."""
    print("\n" + "="*60)
    print("  RERANKER KARŞILAŞTIRMA: Gemini vs Local Cross-Encoder")
    print("="*60)

    print("\n[1/2] Gemini reranker çalışıyor...")
    gemini_results, gemini_avg = run_single_mode(eval_set, mode="gemini")

    print("\n[2/2] Local cross-encoder çalışıyor...")
    local_results, local_avg = run_single_mode(eval_set, mode="local")

    print("\n" + "="*60)
    print("  SONUÇLAR")
    print("="*60)
    print(f"  {'Soru':<40} {'Gemini':>10} {'Local':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    for i, (g, l) in enumerate(zip(gemini_results, local_results), 1):
        q = g["question"][:38]
        g_lat = f"{g.get('rerank_latency_ms', 0):.0f}ms"
        l_lat = f"{l.get('rerank_latency_ms', 0):.0f}ms"
        print(f"  {i}. {q:<38} {g_lat:>10} {l_lat:>10}")

    print(f"\n  Ortalama Latency:")
    print(f"  Gemini reranker : {gemini_avg:.0f} ms")
    print(f"  Local reranker  : {local_avg:.0f} ms")
    faster = "Local" if local_avg < gemini_avg else "Gemini"
    diff = abs(gemini_avg - local_avg)
    print(f"  → {faster} daha hızlı ({diff:.0f} ms fark)")
    print("="*60)

    # Rapor kaydet
    comparison = {
        "gemini_avg_latency_ms": round(gemini_avg, 1),
        "local_avg_latency_ms":  round(local_avg, 1),
        "faster": faster,
        "diff_ms": round(diff, 1),
        "gemini_results": gemini_results,
        "local_results":  local_results
    }
    compare_path = os.path.join(os.path.dirname(__file__), "..", "eval_compare_report.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n[RAPOR] Karşılaştırma raporu kaydedildi: {compare_path}")


def run_eval():
    if not os.path.exists(EVAL_SET_PATH):
        print(f"[HATA] '{EVAL_SET_PATH}' bulunamadı.")
        return

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    if not eval_set:
        print("[UYARI] eval_set.json boş.")
        return

    # Karşılaştırma modu
    compare_mode = "--compare-rerankers" in sys.argv

    if compare_mode:
        run_compare_rerankers(eval_set)
        return

    # --- Normal eval ---
    results = []
    total   = len(eval_set)
    passed  = 0

    print(f"\n[EVAL] {total} soru değerlendiriliyor...\n{'='*50}")

    for i, item in enumerate(eval_set, start=1):
        question         = item.get("question", "")
        expected_keywords = item.get("expected_keywords", [])

        print(f"\n[{i}/{total}] Soru: {question}")

        try:
            candidates = search(question)
            t0 = time.time()
            top_chunks = rerank(question, candidates)
            rerank_ms  = (time.time() - t0) * 1000

            result  = generate(question, top_chunks)
            answer  = result["answer"]
            sources = result["sources"]

            citation_check = check_citation_validity(answer, sources)
            relevance = has_relevant_answer(answer, expected_keywords) if expected_keywords else None

            item_result = {
                "question":          question,
                "answer_preview":    answer[:300],
                "source_count":      len(sources),
                "rerank_latency_ms": round(rerank_ms, 1),
                "citation_check":    citation_check,
                "keyword_relevance": relevance,
                "passed": citation_check["valid_citations"] > 0 and (relevance is None or relevance)
            }

            if item_result["passed"]:
                passed += 1
                print(f"  ✓ BAŞARILI — Atıf: {citation_check['valid_citations']}/{citation_check['citation_count']} | Rerank: {rerank_ms:.0f}ms")
            else:
                print(f"  ✗ BAŞARISIZ — Atıf: {citation_check['valid_citations']}/{citation_check['citation_count']} | Keyword: {relevance}")

        except Exception as e:
            print(f"  [HATA] {e}")
            item_result = {"question": question, "error": str(e), "passed": False}

        results.append(item_result)

    # Skip-rerank istatistikleri
    skip_stats = get_skip_stats()

    accuracy = passed / total if total > 0 else 0.0
    report = {
        "total_questions": total,
        "passed":          passed,
        "failed":          total - passed,
        "accuracy":        round(accuracy, 3),
        "skip_rerank_stats": skip_stats,
        "results":         results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"[SONUÇ] Doğruluk oranı: {accuracy:.1%} ({passed}/{total})")
    print(f"[SKIP]  {skip_stats['skip_rate']}")
    print(f"[RAPOR] '{REPORT_PATH}' dosyasına kaydedildi.")


if __name__ == "__main__":
    run_eval()
