"""
text_ingestor.py — Handles plain-text input normalization.

Splits long documents into overlapping chunks so large articles can be
processed without exceeding LLM context windows.
"""
from __future__ import annotations

import re

# Characters per chunk (approx — actual split is on sentence boundary)
DEFAULT_CHUNK_SIZE = 1500
OVERLAP = 200


def ingest_text(
    raw_text: str,
    source_name: str = "text_input",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict]:
    """
    Normalize a raw text string into one or more ingestion records.

    Each record follows the unified intermediate representation::

        {
            "source_type": "text",
            "content": "<chunk>",
            "metadata": {
                "source": "<source_name>",
                "chunk_index": <int>,
                "total_chunks": <int>,
                "char_count": <int>
            }
        }

    Args:
        raw_text:    Raw input string (article, paragraph, document).
        source_name: Human-readable name for the source.
        chunk_size:  Approximate maximum characters per chunk.

    Returns:
        List of normalized ingestion records.
    """
    cleaned = _clean_text(raw_text)
    chunks = _split_into_chunks(cleaned, chunk_size)

    records = []
    for idx, chunk in enumerate(chunks):
        records.append(
            {
                "source_type": "text",
                "content": chunk,
                "metadata": {
                    "source": source_name,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk),
                },
            }
        )
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Remove control characters, normalize whitespace."""
    text = re.sub(r"[\r\n]+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """
    Split text into sentence-boundary-aware chunks of ≤ max_chars characters.
    Uses a simple overlapping window so context is not lost at boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    # Split on sentence endings
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current))
            # Overlap: keep last few sentences
            overlap_sentences = current[-2:] if len(current) >= 2 else current[-1:]
            current = overlap_sentences
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]
