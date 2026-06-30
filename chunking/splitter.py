"""
chunking/splitter.py — Breaks large documents into retrieval-friendly sections.

Strategy: Sentence-aware chunking (Ahmet Özel's suggestion).
- Detects sentence boundaries via regex instead of fixed character splits.
- Never cuts mid-sentence: if adding a sentence exceeds chunk_size,
  the current chunk is saved and a new one starts.
- Table-like line detection: lines containing '%', ':', or multiple
  whitespace/tabs are kept together with their preceding header line.
"""

import re
import yaml
import os


def _load_config() -> dict:
    """Load chunking settings from config.yaml, fall back to defaults."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("chunking", {})
    except Exception:
        return {}


def _is_table_like(line: str) -> bool:
    """
    Detect table-like rows: lines with '%', ':', or multiple spaces/tabs.
    These should stay attached to their preceding header line.
    """
    has_colon = ":" in line
    has_percent = "%" in line
    has_multi_space = bool(re.search(r"[ \t]{2,}", line.strip()))
    return has_colon or has_percent or has_multi_space


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> list[str]:
    """
    Split text into sentence-aware chunks.

    Args:
        text: Raw document text.
        chunk_size: Max characters per chunk (default from config.yaml or 500).
        overlap: Characters of overlap between chunks via sentence carry-over.

    Returns:
        List of non-empty chunk strings.
    """
    cfg = _load_config()
    if chunk_size is None:
        chunk_size = cfg.get("chunk_size", 500)
    if overlap is None:
        overlap = cfg.get("chunk_overlap", 50)

    # --- Table-line pre-processing ---
    # Join table-like lines to their preceding line so they stay together
    lines = text.splitlines()
    merged_lines = []
    i = 0
    while i < len(lines):
        current = lines[i]
        # If next line looks like a table row, attach it to current line
        if i + 1 < len(lines) and _is_table_like(lines[i + 1]):
            combined = current + " " + lines[i + 1]
            merged_lines.append(combined)
            i += 2
        else:
            merged_lines.append(current)
            i += 1
    text = "\n".join(merged_lines)

    # --- Sentence splitting ---
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if current_length + len(sentence) > chunk_size and current_sentences:
            # Save current chunk
            chunks.append(" ".join(current_sentences))

            # Carry-over: keep last sentence for context overlap
            carry = [current_sentences[-1]] if current_sentences else []
            current_sentences = carry
            current_length = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_length += len(sentence)

    # Don't forget the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return [c for c in chunks if c.strip()]


if __name__ == "__main__":
    sample = (
        "Tokenization is the process of breaking text into tokens. "
        "It is a fundamental step in NLP. "
        "Each token represents a word or subword. "
        "Modern LLMs use subword tokenization methods like BPE. "
        "This allows them to handle rare words efficiently."
    )
    result = chunk_text(sample, chunk_size=100)
    for i, c in enumerate(result, 1):
        print(f"[{i}] ({len(c)} chars) {c}")
