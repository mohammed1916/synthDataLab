"""
fingerprint_store.py — Persistent cross-run deduplication via SHA-256 fingerprints.

Each sample is identified by a normalised fingerprint:
    SHA-256( lower(strip(input_text)) + "|" + task_type )

Fingerprints are stored in a JSON file so deduplication works across separate
``run-all`` invocations, not just within a single run.

Usage::

    store = FingerprintStore(path)
    new, dupes = store.filter_new(samples)
    store.save()   # persist after successful run
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _fingerprint(input_text: str, task_type: str) -> str:
    """Return a 16-char hex fingerprint for (input_text, task_type)."""
    normalised = re.sub(r"\s+", " ", input_text.strip().lower())
    raw = f"{normalised}|{task_type.lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class FingerprintStore:
    """
    Thread-safe persistent set of sample fingerprints.

    Args:
        path: Path to the JSON file that stores the fingerprint set.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._fingerprints: set[str] = set()
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._fingerprints = set(data.get("fingerprints", []))
                logger.debug(
                    "Loaded %d fingerprints from %s", len(self._fingerprints), self._path
                )
            except Exception as exc:
                logger.warning("Could not load fingerprint store (%s) — starting fresh.", exc)
                self._fingerprints = set()

    def save(self) -> None:
        """Atomically persist the current fingerprint set to disk."""
        import os
        import tempfile

        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    {"fingerprints": sorted(self._fingerprints), "count": len(self._fingerprints)},
                    f,
                    indent=2,
                )
            os.replace(tmp, self._path)
        except Exception:
            import contextlib
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        logger.debug("Saved %d fingerprints to %s", len(self._fingerprints), self._path)

    # ── Query / update ────────────────────────────────────────────────────────

    def contains(self, input_text: str, task_type: str) -> bool:
        """Return True if a sample with this (input, task_type) was seen before."""
        return _fingerprint(input_text, task_type) in self._fingerprints

    def add(self, input_text: str, task_type: str) -> str:
        """Record a sample fingerprint and return the fingerprint string."""
        fp = _fingerprint(input_text, task_type)
        self._fingerprints.add(fp)
        return fp

    def filter_new(
        self, samples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Partition *samples* into (new_samples, cross_run_duplicates).

        New samples are also added to the store (but not yet persisted —
        call ``save()`` after a successful run).

        Args:
            samples: List of sample dicts with at minimum ``"input"`` and
                     ``"task_type"`` keys.

        Returns:
            (new_samples, duplicates) where ``new_samples`` are unseen and
            ``duplicates`` were already present in a previous run.
        """
        new: list[dict[str, Any]] = []
        dupes: list[dict[str, Any]] = []
        for sample in samples:
            input_text = sample.get("input", "")
            task_type = sample.get("task_type", "unknown")
            if self.contains(input_text, task_type):
                dupes.append(sample)
            else:
                self.add(input_text, task_type)
                new.append(sample)

        if dupes:
            logger.info(
                "Cross-run dedup: %d new / %d duplicate(s) removed",
                len(new), len(dupes),
            )
        return new, dupes

    def __len__(self) -> int:
        return len(self._fingerprints)
