"""
llm/reranker.py — Reranks retrieval candidates before generation.

Two modes (set in config.yaml → reranker.mode):
  - "local": Cross-encoder (ms-marco-MiniLM-L-6-v2) — offline, free.
  - "gemini": Google Gemini API reranking — requires API key.

Skip-rerank optimisation (Gunjan Tailor's suggestion):
  If RRF top-1 score dominates top-2 by >= skip_threshold,
  the reranker is bypassed and top-3 chunks go directly to generation.
  This reduces latency for highly-confident queries.
"""

import json
import os
import time
import yaml
from dotenv import load_dotenv

from prompts.templates import build_rerank_prompt

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Session-level skip-rerank statistics
_skip_stats = {"skipped": 0, "total": 0}


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("reranker", {})
    except Exception:
        return {}


def _check_skip_rerank(candidates: list[dict], threshold: float) -> bool:
    """Return True if top-1 RRF score dominates top-2 by >= threshold."""
    if len(candidates) < 2:
        return True
    s1 = candidates[0].get("rrf_score", 0)
    s2 = candidates[1].get("rrf_score", 0)
    if s2 == 0:
        return True
    return (s1 - s2) / s2 >= threshold


def rerank_local(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """
    Local cross-encoder reranking using sentence-transformers.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("[RERANKER] sentence-transformers not installed. Returning top candidates.")
        return candidates[:top_n]

    cfg = _load_config()
    model_name = cfg.get("local_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"[RERANKER-LOCAL] Loading cross-encoder: {model_name}")
    model = CrossEncoder(model_name, max_length=512)

    pairs = [(query, c["text"][:512]) for c in candidates]
    scores = model.predict(pairs)

    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    reranked = [c for _, c in scored[:top_n]]
    print(f"[RERANKER-LOCAL] Selected top {len(reranked)} chunks.")
    return reranked


def rerank_gemini(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """
    Gemini API reranking. Uses prompts/templates.py for prompt construction.
    """
    from google import genai

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env")

    cfg_llm_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(cfg_llm_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
        model_name = full_cfg.get("llm", {}).get("model", "gemini-flash-latest")
        max_retries = full_cfg.get("llm", {}).get("max_retries", 3)
        retry_delay = full_cfg.get("llm", {}).get("retry_delay_seconds", 5)
    except Exception:
        model_name = "gemini-flash-latest"
        max_retries = 3
        retry_delay = 5

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = build_rerank_prompt(query, candidates, top_n)

    print(f"[RERANKER-GEMINI] Sending {len(candidates)} candidates to Gemini...")

    raw = ""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            raw = response.text.strip()
            break
        except Exception as e:
            print(f"[RERANKER-GEMINI] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("[RERANKER-GEMINI] All retries failed. Returning top candidates.")
                return candidates[:top_n]

    # Parse JSON response
    try:
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        ranking = parsed.get("ranking", [])
    except json.JSONDecodeError as e:
        print(f"[RERANKER-GEMINI] JSON parse error: {e}. Returning top candidates.")
        return candidates[:top_n]

    reranked = []
    seen = set()
    for idx in ranking:
        zero_idx = idx - 1
        if 0 <= zero_idx < len(candidates):
            chunk = candidates[zero_idx]
            if chunk["id"] not in seen:
                reranked.append(chunk)
                seen.add(chunk["id"])
        if len(reranked) >= top_n:
            break

    # Fill remaining slots if Gemini returned fewer than top_n
    for chunk in candidates:
        if len(reranked) >= top_n:
            break
        if chunk["id"] not in seen:
            reranked.append(chunk)
            seen.add(chunk["id"])

    print(f"[RERANKER-GEMINI] Selected top {len(reranked)} chunks.")
    return reranked


def rerank(query: str, candidates: list[dict], mode: str = None) -> list[dict]:
    """
    Main rerank function. Reads mode from config.yaml if not specified.

    Args:
        query: User question string.
        candidates: Retrieved chunks from hybrid_search.
        mode: "local" | "gemini" | None (reads config.yaml).

    Returns:
        Top-N reranked chunks.
    """
    cfg = _load_config()
    if mode is None:
        mode = cfg.get("mode", "local")
    top_n = cfg.get("top_rerank", 5)
    threshold = cfg.get("skip_threshold", 0.03)

    if not candidates:
        return []

    _skip_stats["total"] += 1

    # --- Skip-rerank check ---
    if _check_skip_rerank(candidates, threshold):
        _skip_stats["skipped"] += 1
        skipped = _skip_stats["skipped"]
        total = _skip_stats["total"]
        print(
            f"[SKIP-RERANK] ✓ High confidence: top-1 dominant. "
            f"Reranker bypassed. ({skipped}/{total} queries skipped reranker)"
        )
        return candidates[:min(top_n, len(candidates))]

    # --- Normal reranking ---
    if mode == "local":
        return rerank_local(query, candidates, top_n)
    else:
        return rerank_gemini(query, candidates, top_n)


def get_skip_stats() -> dict:
    """Return session-level skip-rerank statistics."""
    total = _skip_stats["total"]
    skipped = _skip_stats["skipped"]
    return {
        "skipped": skipped,
        "total": total,
        "skip_rate": (
            f"{skipped}/{total} queries bypassed the reranker"
            if total > 0 else "No queries yet"
        ),
    }
