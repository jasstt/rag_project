"""
ingestion/ — Loads content from PDFs, text files, and other sources.
"""
from .loader import read_txt, read_pdf

__all__ = ["read_txt", "read_pdf"]
