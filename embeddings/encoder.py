"""
embeddings/encoder.py — Converts text chunks into vector representations.

Uses sentence-transformers (SentenceTransformer) for dense embeddings.
Model is loaded once and reused to avoid repeated disk reads.
"""

import yaml
import os
from sentence_transformers import SentenceTransformer

# Module-level singleton to avoid reloading on every call
_model: SentenceTransformer | None = None
_loaded_model_name: str = ""


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("embedding", {})
    except Exception:
        return {}


def _get_model(model_name: str = None) -> SentenceTransformer:
    """Load (or reuse) the SentenceTransformer model."""
    global _model, _loaded_model_name

    if model_name is None:
        cfg = _load_config()
        model_name = cfg.get("model", "all-MiniLM-L6-v2")

    if _model is None or _loaded_model_name != model_name:
        print(f"[EMBEDDINGS] Loading model: {model_name}")
        _model = SentenceTransformer(model_name)
        _loaded_model_name = model_name

    return _model


def encode(texts: list[str], model_name: str = None, show_progress: bool = True) -> list[list[float]]:
    """
    Encode a list of text chunks into embedding vectors.

    Args:
        texts: List of strings to encode.
        model_name: Override model from config.yaml.
        show_progress: Show tqdm progress bar.

    Returns:
        List of float vectors (one per input text).
    """
    model = _get_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=show_progress)
    return embeddings.tolist()


def encode_query(query: str, model_name: str = None) -> list[float]:
    """
    Encode a single query string.

    Args:
        query: User question string.
        model_name: Override model from config.yaml.

    Returns:
        Single float vector.
    """
    model = _get_model(model_name)
    return model.encode([query])[0].tolist()
