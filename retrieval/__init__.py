"""
retrieval/ — Finds the most relevant chunks for each query.
Uses hybrid search: Dense (ChromaDB) + Sparse (BM25) fused via RRF.
"""
from .hybrid_search import search

__all__ = ["search"]
