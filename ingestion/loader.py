"""
ingestion/loader.py — Loads raw content from .txt and .pdf files.

Responsibility: File reading only. No chunking, no embedding.
Supported formats: .txt, .pdf (via PyMuPDF)
"""

import fitz  # PyMuPDF


def read_txt(path: str) -> str:
    """Read plain text file and return content as string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(path: str) -> str:
    """
    Read PDF and return text content as string.
    Uses PyMuPDF with sort=True for better multi-column handling.
    """
    doc = fitz.open(path)
    text = ""
    for page in doc:
        # sort=True preserves reading order for multi-column layouts
        text += page.get_text("text", sort=True) + "\n"
    doc.close()
    return text


def load_file(path: str) -> str:
    """
    Auto-detect file type and load content.
    Raises ValueError for unsupported formats.
    """
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "txt":
        return read_txt(path)
    elif ext == "pdf":
        return read_pdf(path)
    else:
        raise ValueError(f"Unsupported file format: .{ext} ({path})")
