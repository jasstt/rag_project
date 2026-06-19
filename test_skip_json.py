import sys
import json
import time
sys.path.insert(0, 'src')
from search import search
from rerank import rerank, get_skip_stats
from config import SKIP_RERANK_THRESHOLD

print("=== SKIP-RERANK OPTİMİZASYON TESTİ (JSON'dan) ===")
print(f"Eşik (Threshold): {SKIP_RERANK_THRESHOLD*100}%\n")

with open('eval_set.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

for i, item in enumerate(eval_data, 1):
    q = item['question']
    candidates = search(q)
    s1 = candidates[0]['rrf_score'] if candidates else 0
    s2 = candidates[1]['rrf_score'] if len(candidates) > 1 else 0
    gap = (s1 - s2) / s2 if s2 > 0 else 999
    gap_percent = gap * 100
    
    print(f"\n[{i}] Soru: {q}")
    print(f"    Top 1 RRF: {s1:.4f} | Top 2 RRF: {s2:.4f} -> Fark: %{gap_percent:.1f}")
    
    start_t = time.time()
    top = rerank(q, candidates, mode='local')
    t_ms = (time.time() - start_t) * 1000
    
    if gap >= SKIP_RERANK_THRESHOLD:
        print(f"    --> Eşik aşıldı! Reranker ATLANDI. ({t_ms:.1f} ms)")
    else:
        print(f"    --> Reranker ÇALIŞTI. ({t_ms:.1f} ms)")

print("\n")
s = get_skip_stats()
print(f"SONUÇ: {s['skip_rate']}")
print(f"       Toplam {s['total']} sorgu, {s['skipped']} tanesi reranker atladı ({int(s['skipped']/s['total']*100) if s['total'] else 0}%)")
