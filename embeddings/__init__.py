"""
embeddings/ — Converts chunks into vector representations.
"""
from .encoder import encode, encode_query

__all__ = ["encode", "encode_query"]
