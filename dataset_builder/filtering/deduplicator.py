"""
deduplicator.py — Content-based deduplication of dataset samples.

Two samples are duplicates if the Jaccard similarity of their normalised
input-text token sets exceeds a configurable threshold.  The first-seen
sample is kept; the rest are dropped.

For very large datasets a MinHash / LSH approach would be appropriate.
This implementation uses an O(n²) pairwise comparison which is fine for
datasets up to ~10 000 samples.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Remove near-duplicate samples based on Jaccard token similarity.

    Deduplication is performed **within each task type** only.  Two samples
    from the same passage but different task types (qa vs extraction vs
    reasoning) are intentionally kept — they represent distinct annotation
    tasks on the same source material.

    Args:
        threshold: Jaccard similarity threshold above which two samples are
                   considered duplicates (default 0.85).
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def deduplicate(
        self, samples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Return (unique_samples, removed_duplicates).

        Samples with the same input text but *different* task types are NOT
        considered duplicates and are always kept.
        """
        kept: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []

        # Track seen token sets per task type  {task_type: [set, ...]}
        seen_by_task: dict[str, list[set[str]]] = {}

        for sample in samples:
            task_type = sample.get("task_type", "unknown")
            tokens = _tokenise(sample.get("input", ""))

            existing_sets = seen_by_task.setdefault(task_type, [])
            is_dup = any(
                _jaccard(tokens, existing) >= self.threshold
                for existing in existing_sets
            )

            if is_dup:
                removed.append(sample)
            else:
                kept.append(sample)
                existing_sets.append(tokens)

        logger.info(
            "Deduplication: kept %d, removed %d duplicate(s) (threshold=%.2f)",
            len(kept),
            len(removed),
            self.threshold,
        )
        return kept, removed


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    """Lowercase word tokenisation with stop-word stripping."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "of", "and", "or", "but", "it", "its", "that", "this",
        "for", "with", "as", "be", "by", "from",
    }
    words = re.findall(r"\b\w{2,}\b", text.lower())
    return {w for w in words if w not in stop_words}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
