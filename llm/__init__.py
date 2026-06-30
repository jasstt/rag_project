"""
llm/ — Model-provider logic: Gemini client + reranker.
Separated from prompt templates for clean provider boundaries.
"""
from .gemini_client import generate
from .reranker import rerank, get_skip_stats

__all__ = ["generate", "rerank", "get_skip_stats"]
