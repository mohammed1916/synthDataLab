"""
pipeline.py — Multi-stage quality filtering pipeline.

Stages (in order)
-----------------
1. Schema-violation removal  — drop samples marked REJECT due to SCHEMA_INVALID
2. Deduplication             — drop near-duplicate input × task_type pairs
3. Low-confidence removal    — drop samples below min_confidence threshold
4. Output-length filtering   — drop samples where output text is suspiciously
                               short or excessively long
5. Final ACCEPT-only pass    — keep only samples with label == ACCEPT

Each stage records the number of removed samples and the reason so the
evaluation module can report a per-stage funnel.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config import FilteringConfig
from filtering.deduplicator import Deduplicator
from validation.annotation import AnnotatedSample, AnnotationLabel, RejectionCode

logger = logging.getLogger(__name__)


@dataclass
class StageStats:
    stage_name: str
    input_count: int
    output_count: int
    removed_count: int
    removal_reasons: list[str] = field(default_factory=list)

    @property
    def removal_rate(self) -> float:
        if self.input_count == 0:
            return 0.0
        return self.removed_count / self.input_count


@dataclass
class FilteringReport:
    """Summary statistics for the complete filtering run."""

    total_input: int = 0
    total_output: int = 0
    stages: list[StageStats] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return self.total_input - self.total_output

    @property
    def overall_retention_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.total_output / self.total_input

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "total_removed": self.total_removed,
            "overall_retention_rate": round(self.overall_retention_rate, 4),
            "stages": [
                {
                    "stage": s.stage_name,
                    "input": s.input_count,
                    "output": s.output_count,
                    "removed": s.removed_count,
                    "removal_rate": round(s.removal_rate, 4),
                    "sample_reasons": s.removal_reasons[:5],  # first 5 examples
                }
                for s in self.stages
            ],
        }


class FilteringPipeline:
    """
    Applies a multi-stage quality filter to a list of AnnotatedSamples.

    Args:
        config: FilteringConfig with thresholds.
    """

    def __init__(self, config: FilteringConfig):
        self.config = config
        self._dedup = Deduplicator(threshold=config.max_duplicate_similarity)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self, annotated: list[AnnotatedSample]
    ) -> tuple[list[AnnotatedSample], FilteringReport]:
        """
        Run the full pipeline.

        Returns:
            (filtered_samples, report)  where filtered_samples contains only
            ACCEPT-labelled samples that survived all stages.
        """
        report = FilteringReport(total_input=len(annotated))
        current = list(annotated)

        # ── Stage 1: remove samples with hard schema violations ───────────────
        current, stage1 = self._filter_schema_violations(current)
        report.stages.append(stage1)

        # ── Stage 2: deduplication ────────────────────────────────────────────
        current, stage2 = self._filter_duplicates(current)
        report.stages.append(stage2)

        # ── Stage 3: low-confidence removal ───────────────────────────────────
        current, stage3 = self._filter_low_confidence(current)
        report.stages.append(stage3)

        # ── Stage 4: output-length filtering ──────────────────────────────────
        current, stage4 = self._filter_output_length(current)
        report.stages.append(stage4)

        # ── Stage 5: final ACCEPT-only pass ───────────────────────────────────
        current, stage5 = self._filter_non_accepted(current)
        report.stages.append(stage5)

        report.total_output = len(current)
        logger.info(
            "Filtering complete: %d → %d samples (%.1f%% retained).",
            report.total_input,
            report.total_output,
            100 * report.overall_retention_rate,
        )
        return current, report

    # ─────────────────────────────────────────────────────────────────────────
    # Stages
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _filter_schema_violations(
        samples: list[AnnotatedSample],
    ) -> tuple[list[AnnotatedSample], StageStats]:
        kept = []
        reasons: list[str] = []
        for s in samples:
            has_schema_error = any(
                r.code == RejectionCode.SCHEMA_INVALID
                for r in s.rejection_reasons
            )
            if has_schema_error:
                reasons.append(
                    f"[{s.sample.get('id', '?')}] {s.rejection_reasons[0].message[:80]}"
                )
            else:
                kept.append(s)

        stat = StageStats(
            stage_name="schema_violation_removal",
            input_count=len(samples),
            output_count=len(kept),
            removed_count=len(samples) - len(kept),
            removal_reasons=reasons,
        )
        return kept, stat

    def _filter_duplicates(
        self, samples: list[AnnotatedSample]
    ) -> tuple[list[AnnotatedSample], StageStats]:
        raw_dicts = [s.sample for s in samples]
        unique_dicts, removed_dicts = self._dedup.deduplicate(raw_dicts)

        removed_ids = {d.get("id") for d in removed_dicts}
        kept = [s for s in samples if s.sample.get("id") not in removed_ids]
        reasons = [
            f"Duplicate of earlier sample (id={d.get('id', '?')})"
            for d in removed_dicts
        ]

        stat = StageStats(
            stage_name="deduplication",
            input_count=len(samples),
            output_count=len(kept),
            removed_count=len(removed_dicts),
            removal_reasons=reasons,
        )
        return kept, stat

    def _filter_low_confidence(
        self, samples: list[AnnotatedSample]
    ) -> tuple[list[AnnotatedSample], StageStats]:
        kept, reasons = [], []
        for s in samples:
            conf = float(s.sample.get("metadata", {}).get("confidence", 1))
            if conf < self.config.min_confidence:
                reasons.append(
                    f"[{s.sample.get('id', '?')}] confidence={conf:.2f}"
                )
            else:
                kept.append(s)

        stat = StageStats(
            stage_name="low_confidence_removal",
            input_count=len(samples),
            output_count=len(kept),
            removed_count=len(samples) - len(kept),
            removal_reasons=reasons,
        )
        return kept, stat

    def _filter_output_length(
        self, samples: list[AnnotatedSample]
    ) -> tuple[list[AnnotatedSample], StageStats]:
        kept, reasons = [], []
        for s in samples:
            output_str = str(s.sample.get("output", ""))
            length = len(output_str)
            if length < self.config.min_output_length:
                reasons.append(
                    f"[{s.sample.get('id', '?')}] output length={length} chars (min={self.config.min_output_length})"
                )
            elif length > self.config.max_output_length:
                reasons.append(
                    f"[{s.sample.get('id', '?')}] output length={length} chars (max={self.config.max_output_length})"
                )
            else:
                kept.append(s)

        stat = StageStats(
            stage_name="output_length_filter",
            input_count=len(samples),
            output_count=len(kept),
            removed_count=len(samples) - len(kept),
            removal_reasons=reasons,
        )
        return kept, stat

    @staticmethod
    def _filter_non_accepted(
        samples: list[AnnotatedSample],
    ) -> tuple[list[AnnotatedSample], StageStats]:
        kept = [s for s in samples if s.label == AnnotationLabel.ACCEPT]
        removed = len(samples) - len(kept)
        reasons = [
            f"[{s.sample.get('id', '?')}] label={s.label.value}"
            for s in samples
            if s.label != AnnotationLabel.ACCEPT
        ]

        stat = StageStats(
            stage_name="non_accepted_removal",
            input_count=len(samples),
            output_count=len(kept),
            removed_count=removed,
            removal_reasons=reasons,
        )
        return kept, stat
