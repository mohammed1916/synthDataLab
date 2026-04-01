"""
image_ingestor.py — Handles image input via OCR (pytesseract) or
a simple filename-based fallback description.

pytesseract and Pillow are optional; if unavailable the module degrades
gracefully by returning a placeholder description so the rest of the
pipeline can still run.
"""
from __future__ import annotations

from pathlib import Path


def ingest_image(image_path: str) -> list[dict]:
    """
    Attempt OCR on *image_path* and return a normalized ingestion record.

    Falls back to a descriptive placeholder if OCR dependencies are missing.

    Args:
        image_path: Absolute or relative path to a PNG/JPEG/TIFF image.

    Returns:
        A single-element list containing one ingestion record.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    extracted_text = _ocr_image(path)

    return [
        {
            "source_type": "image",
            "content": extracted_text,
            "metadata": {
                "source": path.name,
                "file_path": str(path.resolve()),
                "char_count": len(extracted_text),
                "ocr_used": True,
            },
        }
    ]


# ──────────────────────────────────────────────────────────────────────────────
# OCR logic
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_image(path: Path) -> str:
    """Run Tesseract OCR; degrade gracefully if libs are missing."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            # Convert to greyscale for better recognition
            grey = img.convert("L")
            text: str = pytesseract.image_to_string(grey)
        return text.strip() or f"[Image: {path.name} — no text detected by OCR]"
    except ImportError:
        return (
            f"[Image: {path.name} — OCR unavailable. "
            "Install pytesseract + Pillow to enable text extraction.]"
        )
    except Exception as exc:  # pragma: no cover — runtime OCR errors
        return f"[Image: {path.name} — OCR failed: {exc}]"
