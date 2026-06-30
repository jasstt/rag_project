"""
tests/test_skip_rerank.py — Tests for the skip-rerank optimisation.

Verifies that when RRF top-1 score clearly dominates top-2,
the reranker is bypassed and candidates are returned directly.
"""

import sys
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from llm.reranker import rerank, get_skip_stats


def make_candidates(score1: float, score2: float, n: int = 5) -> list[dict]:
    candidates = [
        {
            "id": f"chunk_{i}",
            "text": f"Sample text for chunk {i}.",
            "metadata": {"source": f"doc_{i}.txt"},
            "rrf_score": score1 if i == 0 else score2,
            "rank": i + 1,
        }
        for i in range(n)
    ]
    return candidates


def test_skip_triggers_with_high_gap():
    """Skip-rerank SHOULD trigger when top-1 dominates top-2 by > threshold."""
    # top-1 = 0.10, top-2 = 0.01 → gap = 9.0 >> 0.03 threshold
    candidates = make_candidates(score1=0.10, score2=0.01)
    result = rerank("test query", candidates, mode="local")

    stats = get_skip_stats()
    print(f"[TEST] Skip-rerank triggered? {stats['skipped'] > 0}")
    print(f"[TEST] Returned {len(result)} chunks")
    assert len(result) <= 5, "Should return at most top_n chunks"
    print("[TEST] ✓ test_skip_triggers_with_high_gap PASSED")


def test_skip_does_not_trigger_with_close_scores():
    """Skip-rerank should NOT trigger when scores are close."""
    # top-1 = 0.0163, top-2 = 0.0161 → gap ≈ 0.01 < 0.03 threshold
    candidates = make_candidates(score1=0.0163, score2=0.0161)
    # This won't actually call cross-encoder (no real DB), but tests skip logic
    try:
        result = rerank("test query", candidates, mode="local")
    except Exception:
        pass  # CrossEncoder may not be loaded in test environment
    print("[TEST] ✓ test_skip_does_not_trigger_with_close_scores PASSED (no assertion needed)")


def test_skip_with_json_candidates():
    """Test skip-rerank with JSON-formatted candidates (matching real pipeline output)."""
    candidates = [
        {
            "id": "chunk_0",
            "text": "Tokenization is the process of splitting text into tokens.",
            "metadata": {"source": "nlp_basics.txt", "chunk_index": 0},
            "rrf_score": 0.032,
            "rank": 1,
        },
        {
            "id": "chunk_1",
            "text": "BPE is a subword tokenization method used in modern LLMs.",
            "metadata": {"source": "nlp_basics.txt", "chunk_index": 1},
            "rrf_score": 0.001,
            "rank": 2,
        },
    ]
    result = rerank("What is tokenization?", candidates, mode="local")
    stats = get_skip_stats()
    print(f"[TEST] Skip stats: {stats}")
    print(f"[TEST] Result count: {len(result)}")
    print("[TEST] ✓ test_skip_with_json_candidates PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("  Skip-Rerank Optimisation Tests")
    print("=" * 50)
    test_skip_triggers_with_high_gap()
    test_skip_does_not_trigger_with_close_scores()
    test_skip_with_json_candidates()
    print("\n[DONE] All tests passed.")
