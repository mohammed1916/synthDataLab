"""
pdf_ingestor.py — PDF ingestion with math expression preservation.

Extracts text from PDF files (NCERT textbooks, previous year papers) while
keeping as much mathematical notation intact as possible.

Dependencies
------------
  Primary:   pymupdf (``pip install pymupdf``) — fast, accurate, no external binary
  Fallback:  pdfminer.six (``pip install pdfminer.six``) — pure-Python, slower
  Fallback2: Plain text read (for .txt exports)

If neither PDF library is installed, the module raises a helpful ImportError
with install instructions rather than failing silently.

Usage
-----
    from cbse_math.pdf_ingestor import ingest_pdf

    chunks = ingest_pdf("ncert_class12_maths.pdf", source_name="NCERT Class 12")
    # Returns list[IngestionResult] with math-aware text chunks
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Maximum characters per chunk (balances prompt length vs context)
_CHUNK_SIZE = 3000
# Overlap between chunks so questions split across pages stay intact
_CHUNK_OVERLAP = 300
# Minimum chars for a chunk to be worth keeping
_MIN_CHUNK_CHARS = 80


# ─────────────────────────────────────────────────────────────────────────────
# Math-preservation helpers
# ─────────────────────────────────────────────────────────────────────────────

# These Unicode characters appear in PDF math text; we normalise them to ASCII
# LaTeX equivalents so the LLM can reason about them correctly.
_UNICODE_MATH_MAP = str.maketrans({
    "−": "-",
    "×": r"\times",
    "÷": r"\div",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "√": r"\sqrt",
    "∞": r"\infty",
    "∫": r"\int",
    "∑": r"\sum",
    "∏": r"\prod",
    "∂": r"\partial",
    "Δ": r"\Delta",
    "δ": r"\delta",
    "θ": r"\theta",
    "π": r"\pi",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "λ": r"\lambda",
    "μ": r"\mu",
    "σ": r"\sigma",
    "ω": r"\omega",
    "∈": r"\in",
    "∉": r"\notin",
    "⊆": r"\subseteq",
    "⊂": r"\subset",
    "∪": r"\cup",
    "∩": r"\cap",
    "⌊": r"\lfloor",
    "⌋": r"\rfloor",
    "⌈": r"\lceil",
    "⌉": r"\rceil",
    "\u00b2": "^2",   # superscript 2
    "\u00b3": "^3",   # superscript 3
    "\u00b9": "^1",   # superscript 1
    "\u2070": "^0",
    "\u2074": "^4",
    "\u2075": "^5",
    "\u2076": "^6",
    "\u2077": "^7",
    "\u2078": "^8",
    "\u2079": "^9",
})

# Remove PDF artefacts (page numbers alone on a line, headers/footers)
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)
_HEADER_RE = re.compile(
    r"^(NCERT|Chapter\s+\d+|Exercise\s+\d+[\.\d]*|MATHEMATICS)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _clean_pdf_text(raw: str) -> str:
    """Normalise raw PDF text for LLM consumption."""
    text = raw.translate(_UNICODE_MATH_MAP)
    text = _PAGE_NUM_RE.sub("", text)
    text = _HEADER_RE.sub("", text)
    # Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove hyphenation artefacts (word-\nnewline)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks at paragraph boundaries where possible.
    Falls back to hard split if no paragraph boundary is found.
    """
    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            # Try to break at a paragraph boundary (double newline) within ±200 chars
            boundary = text.rfind("\n\n", start + chunk_size - 200, end)
            if boundary == -1:
                # Fall back to last sentence end
                boundary = text.rfind(". ", start + chunk_size - 300, end)
            if boundary != -1:
                end = boundary + 2

        chunk = text[start:end].strip()
        if len(chunk) >= _MIN_CHUNK_CHARS:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Ingestor
# ─────────────────────────────────────────────────────────────────────────────

def ingest_pdf(
    file_path: str,
    source_name: str | None = None,
    password: str = "",
    page_range: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Extract and chunk text from a PDF file.

    Args:
        file_path:   Path to the PDF.
        source_name: Human-readable label (defaults to filename).
        password:    PDF password if encrypted.
        page_range:  (start_page, end_page) 0-indexed inclusive; None = all pages.

    Returns:
        List of dicts compatible with ``IngestionResult.from_dict()``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError:       If no PDF library is installed.
        ValueError:        If the PDF cannot be opened.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    name = source_name or path.stem
    raw_text = _extract_text(path, password=password, page_range=page_range)

    if not raw_text.strip():
        logger.warning("PDF '%s' yielded no extractable text (may be image-only).", path.name)
        return []

    cleaned = _clean_pdf_text(raw_text)
    chunks = _chunk_text(cleaned)

    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "source_type": "pdf",
            "content": chunk,
            "metadata": {
                "source": name,
                "file": str(path),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk),
                "page_range": list(page_range) if page_range else None,
            },
        })

    logger.info("Ingested PDF '%s' → %d chunk(s) from %d pages.", path.name, len(results),
                _count_pages(path, password))
    return results


def _extract_text(path: Path, password: str = "", page_range: tuple[int, int] | None = None) -> str:
    """Try pymupdf first, fall back to pdfminer.six."""
    try:
        return _extract_pymupdf(path, password, page_range)
    except ImportError:
        pass

    try:
        return _extract_pdfminer(path, page_range)
    except ImportError:
        pass

    raise ImportError(
        "No PDF library found. Install one of:\n"
        "  pip install pymupdf          # recommended — fast & accurate\n"
        "  pip install pdfminer.six     # pure Python fallback\n"
        "Or export your PDF as plain text and pass the .txt file instead."
    )


def _count_pages(path: Path, password: str = "") -> int:
    try:
        import fitz  # pymupdf
        with fitz.open(str(path)) as doc:
            if password:
                doc.authenticate(password)
            return doc.page_count
    except Exception:
        return -1


def _extract_pymupdf(
    path: Path, password: str = "", page_range: tuple[int, int] | None = None
) -> str:
    import fitz  # type: ignore[import]  # pymupdf

    pages_text: list[str] = []
    with fitz.open(str(path)) as doc:
        if password and not doc.authenticate(password):
            raise ValueError(f"Incorrect password for PDF: {path.name}")

        start = page_range[0] if page_range else 0
        end = (page_range[1] + 1) if page_range else doc.page_count

        for page_num in range(start, min(end, doc.page_count)):
            page = doc[page_num]
            text = page.get_text("text")   # plain text extraction
            pages_text.append(text)

    return "\n\n".join(pages_text)


def _extract_pdfminer(
    path: Path, page_range: tuple[int, int] | None = None
) -> str:
    from io import StringIO

    from pdfminer.high_level import extract_text_to_fp  # type: ignore[import]
    from pdfminer.layout import LAParams  # type: ignore[import]

    output = StringIO()
    kwargs: dict[str, Any] = {"laparams": LAParams()}
    if page_range:
        kwargs["page_numbers"] = list(range(page_range[0], page_range[1] + 1))

    with open(path, "rb") as f:
        extract_text_to_fp(f, output, **kwargs)

    return output.getvalue()
