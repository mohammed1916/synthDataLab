"""
ingestor.py — Orchestrates the ingestion layer.

Accepts text strings, text files, image files, or JSON article collections
and returns a list of normalized IngestionResult dicts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .image_ingestor import ingest_image
from .text_ingestor import ingest_text

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


@dataclass
class IngestionResult:
    """
    Unified intermediate representation produced by every ingestor.

    Fields:
        source_type: One of "text", "image", "json_article".
        content:     Normalised plain-text content.
        metadata:    Provenance and bookkeeping info.
    """

    source_type: str
    content: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IngestionResult:
        return cls(
            source_type=d["source_type"],
            content=d["content"],
            metadata=d.get("metadata", {}),
        )


class Ingestor:
    """
    Entry point for the ingestion layer.

    Usage::

        ingestor = Ingestor()
        results = ingestor.ingest_text("Some article text...")
        results += ingestor.ingest_file("path/to/article.txt")
        results += ingestor.ingest_image("path/to/chart.png")
        results += ingestor.ingest_json("path/to/articles.json")
    """

    def ingest_text(
        self, raw_text: str, source_name: str = "inline_text"
    ) -> list[IngestionResult]:
        """Ingest a raw text string."""
        records = ingest_text(raw_text, source_name=source_name)
        results = [IngestionResult(**r) for r in records]
        logger.info(
            "Ingested inline text → %d chunk(s) (%d chars total)",
            len(results),
            sum(r.metadata.get("char_count", 0) for r in results),
        )
        return results

    def ingest_file(self, file_path: str) -> list[IngestionResult]:
        """Ingest a plain-text file or an image file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Guard against unreasonably large files (>50 MB)
        max_bytes = 50 * 1024 * 1024
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"File too large ({size / 1_048_576:.1f} MB > 50 MB limit): {file_path}. "
                "Split into smaller chunks before ingestion."
            )

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            raw_records = ingest_image(file_path)
        else:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            raw_records = ingest_text(raw_text, source_name=path.name)

        results = [IngestionResult(**r) for r in raw_records]
        logger.info("Ingested '%s' → %d chunk(s)", path.name, len(results))
        return results

    def ingest_image(self, image_path: str) -> list[IngestionResult]:
        """Ingest an image file (OCR)."""
        raw_records = ingest_image(image_path)
        results = [IngestionResult(**r) for r in raw_records]
        logger.info("Ingested image '%s'", Path(image_path).name)
        return results

    def ingest_json(self, json_path: str) -> list[IngestionResult]:
        """
        Ingest a JSON file containing an array of article objects.

        Expected format (each element)::

            {
                "title": "...",
                "content": "...",
                "source": "..."   # optional
            }
        """
        path = Path(json_path)
        max_bytes = 50 * 1024 * 1024
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"JSON file too large ({size / 1_048_576:.1f} MB > 50 MB limit): {json_path}"
            )
        articles: list[dict] = json.loads(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if not isinstance(articles, list):
            raise ValueError(
                f"JSON file must contain a list of article objects: {json_path}"
            )

        results: list[IngestionResult] = []
        for article in articles:
            content = article.get("content", "")
            title = article.get("title", "Untitled")
            source = article.get("source", path.name)
            if not content.strip():
                continue
            raw_records = ingest_text(
                content, source_name=f"{source} / {title}"
            )
            results.extend([IngestionResult(**r) for r in raw_records])

        logger.info(
            "Ingested JSON '%s' → %d chunk(s) from %d article(s)",
            path.name,
            len(results),
            len(articles),
        )
        return results

    def ingest_batch(self, file_paths: list[str]) -> list[IngestionResult]:
        """Convenience: ingest a list of file paths in one call."""
        results: list[IngestionResult] = []
        for fp in file_paths:
            results.extend(self.ingest_file(fp))
        return results
