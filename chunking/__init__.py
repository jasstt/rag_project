"""
chunking/ — Breaks large documents into retrieval-friendly sections.
"""
from .splitter import chunk_text

__all__ = ["chunk_text"]
