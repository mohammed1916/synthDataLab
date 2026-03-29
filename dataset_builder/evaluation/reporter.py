"""
reporter.py — Generates human-readable and machine-readable metrics reports.

Produces:
* A rich console table (via the `rich` library if installed)
* A JSON report written to disk
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.metrics import DatasetMetrics


class MetricsReporter:
    """
    Renders before/after metrics comparison and writes JSON reports.

    Args:
        output_path: Where to write the JSON metrics report (optional).
    """

    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = output_path

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def report(
        self,
        raw_metrics: DatasetMetrics,
        filtered_metrics: DatasetMetrics,
        filtering_report_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build and optionally save a full metrics report.

        Returns the report as a dict (also printed to console).
        """
        report = self._build_report(
            raw_metrics, filtered_metrics, filtering_report_dict
        )

        self._print_report(raw_metrics, filtered_metrics)

        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _console_print(f"\n[Metrics] Report saved → {self.output_path}")

        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_report(
        raw: DatasetMetrics,
        filtered: DatasetMetrics,
        filtering_report: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _delta(before: float, after: float) -> str:
            diff = after - before
            sign = "+" if diff >= 0 else ""
            return f"{sign}{diff:+.2%}"

        return {
            "summary": {
                "raw_samples": raw.total_samples,
                "filtered_samples": filtered.total_samples,
                "samples_removed": raw.total_samples - filtered.total_samples,
            },
            "metrics": {
                "schema_validity_rate": {
                    "raw": raw.schema_validity_rate,
                    "filtered": filtered.schema_validity_rate,
                    "delta": _delta(raw.schema_validity_rate, filtered.schema_validity_rate),
                },
                "task_consistency_score": {
                    "raw": raw.task_consistency_score,
                    "filtered": filtered.task_consistency_score,
                    "delta": _delta(raw.task_consistency_score, filtered.task_consistency_score),
                },
                "completeness_score": {
                    "raw": raw.completeness_score,
                    "filtered": filtered.completeness_score,
                    "delta": _delta(raw.completeness_score, filtered.completeness_score),
                },
                "hallucination_rate": {
                    "raw": raw.hallucination_rate,
                    "filtered": filtered.hallucination_rate,
                    "delta": _delta(raw.hallucination_rate, filtered.hallucination_rate),
                },
                "diversity_score": {
                    "raw": raw.diversity_score,
                    "filtered": filtered.diversity_score,
                    "delta": _delta(raw.diversity_score, filtered.diversity_score),
                },
                "mean_confidence": {
                    "raw": raw.mean_confidence,
                    "filtered": filtered.mean_confidence,
                    "delta": _delta(raw.mean_confidence, filtered.mean_confidence),
                },
            },
            "task_distribution": {
                "raw": raw.task_type_distribution,
                "filtered": filtered.task_type_distribution,
            },
            "filtering_pipeline": filtering_report or {},
        }

    @staticmethod
    def _print_report(raw: DatasetMetrics, filtered: DatasetMetrics) -> None:
        try:
            from rich.table import Table
            from rich.console import Console
            from rich import box
            _rich_table(raw, filtered)
        except ImportError:
            _plain_table(raw, filtered)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rich_table(raw: DatasetMetrics, filtered: DatasetMetrics) -> None:
    from rich.table import Table
    from rich.console import Console
    from rich import box

    console = Console()

    table = Table(
        title="[bold cyan]Dataset Quality Metrics: Raw vs Filtered[/bold cyan]",
        box=box.ROUNDED,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Raw Dataset", justify="right")
    table.add_column("Filtered Dataset", justify="right")
    table.add_column("Δ Improvement", justify="right")

    rows = [
        ("Total Samples", raw.total_samples, filtered.total_samples, None),
        ("Schema Validity Rate", raw.schema_validity_rate, filtered.schema_validity_rate, True),
        ("Task Consistency Score", raw.task_consistency_score, filtered.task_consistency_score, True),
        ("Completeness Score", raw.completeness_score, filtered.completeness_score, True),
        ("Hallucination Rate ↓", raw.hallucination_rate, filtered.hallucination_rate, False),
        ("Diversity Score", raw.diversity_score, filtered.diversity_score, True),
        ("Mean Confidence", raw.mean_confidence, filtered.mean_confidence, True),
    ]

    for name, raw_val, filt_val, higher_is_better in rows:
        if isinstance(raw_val, int):
            rv = str(raw_val)
            fv = str(filt_val)
            delta = f"{filt_val - raw_val:+d}"
            color = "green" if filt_val < raw_val else "yellow"
        else:
            rv = f"{raw_val:.1%}"
            fv = f"{filt_val:.1%}"
            diff = filt_val - raw_val
            if higher_is_better:
                color = "green" if diff >= 0 else "red"
            else:
                color = "green" if diff <= 0 else "red"
            delta = f"[{color}]{diff:+.1%}[/{color}]"
        table.add_row(name, rv, fv, delta)

    console.print()
    console.print(table)
    console.print()


def _plain_table(raw: DatasetMetrics, filtered: DatasetMetrics) -> None:
    lines = [
        "",
        "=" * 65,
        "  Dataset Quality Metrics: Raw vs Filtered",
        "=" * 65,
        f"  {'Metric':<30} {'Raw':>10} {'Filtered':>10} {'Delta':>10}",
        "-" * 65,
        f"  {'Total Samples':<30} {raw.total_samples:>10d} {filtered.total_samples:>10d}",
        f"  {'Schema Validity Rate':<30} {raw.schema_validity_rate:>10.1%} {filtered.schema_validity_rate:>10.1%} {filtered.schema_validity_rate - raw.schema_validity_rate:>+10.1%}",
        f"  {'Task Consistency Score':<30} {raw.task_consistency_score:>10.1%} {filtered.task_consistency_score:>10.1%} {filtered.task_consistency_score - raw.task_consistency_score:>+10.1%}",
        f"  {'Completeness Score':<30} {raw.completeness_score:>10.1%} {filtered.completeness_score:>10.1%} {filtered.completeness_score - raw.completeness_score:>+10.1%}",
        f"  {'Hallucination Rate':<30} {raw.hallucination_rate:>10.1%} {filtered.hallucination_rate:>10.1%} {filtered.hallucination_rate - raw.hallucination_rate:>+10.1%}",
        f"  {'Diversity Score':<30} {raw.diversity_score:>10.1%} {filtered.diversity_score:>10.1%} {filtered.diversity_score - raw.diversity_score:>+10.1%}",
        f"  {'Mean Confidence':<30} {raw.mean_confidence:>10.2f} {filtered.mean_confidence:>10.2f} {filtered.mean_confidence - raw.mean_confidence:>+10.2f}",
        "=" * 65,
        "",
    ]
    print("\n".join(lines))


def _console_print(msg: str) -> None:
    try:
        from rich.console import Console
        Console().print(msg)
    except ImportError:
        print(msg)
