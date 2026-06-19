import sys
sys.path.insert(0, 'src')
from search import search
from rerank import rerank, get_skip_stats
import json

questions = [
    'Tokenization nedir?',
    'Pekiştirmeli ogrenme nasil calisir?',
    'Transformer mimarisi nedir?',
    'Ozellik muhendisligi nedir?',
    'Siniflandirma problemleri nasil cozulur?',
]

print('=== SKIP-RERANK OPTİMİZASYON TESTİ ===')
for i, q in enumerate(questions, 1):
    candidates = search(q)
    s1 = candidates[0]['rrf_score'] if candidates else 0
    s2 = candidates[1]['rrf_score'] if len(candidates) > 1 else 0
    gap = (s1 - s2) / s2 * 100 if s2 > 0 else 999
    top = rerank(q, candidates, mode='local')
    print(f'[{i}] top1={s1:.4f} top2={s2:.4f} gap={gap:.1f}% chunks_returned={len(top)}')

print()
s = get_skip_stats()
print(f"SONUÇ: {s['skip_rate']}")
print(f"       Toplam {s['total']} sorgu, {s['skipped']} tanesi reranker atladı ({int(s['skipped']/s['total']*100) if s['total'] else 0}%)")
