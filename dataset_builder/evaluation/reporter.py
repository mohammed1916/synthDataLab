"""
reporter.py — Generates human-readable and machine-readable metrics reports.

Produces:
* A rich console table (via the `rich` library if installed)
* A JSON report written to disk
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.metrics import DatasetMetrics


class MetricsReporter:
    """
    Renders before/after metrics comparison and writes JSON reports.

    Args:
        output_path: Where to write the JSON metrics report (optional).
    """

    def __init__(self, output_path: Path | None = None):
        self.output_path = output_path

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def report(
        self,
        raw_metrics: DatasetMetrics,
        filtered_metrics: DatasetMetrics,
        filtering_report_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        filtering_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
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
                "vocabulary_entropy_bits": {
                    "raw": raw.vocabulary_entropy,
                    "filtered": filtered.vocabulary_entropy,
                    "delta": _delta(raw.vocabulary_entropy, filtered.vocabulary_entropy),
                },
                "bigram_entropy_bits": {
                    "raw": raw.bigram_entropy,
                    "filtered": filtered.bigram_entropy,
                    "delta": _delta(raw.bigram_entropy, filtered.bigram_entropy),
                },
                "collapse_risk_score": {
                    "raw": raw.collapse_risk_score,
                    "filtered": filtered.collapse_risk_score,
                    "alert": filtered.collapse_warning or "OK",
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
            from rich import box
            from rich.console import Console
            from rich.table import Table
            _rich_table(raw, filtered)
        except ImportError:
            _plain_table(raw, filtered)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rich_table(raw: DatasetMetrics, filtered: DatasetMetrics) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

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

    # Each row: (label, raw_val, filt_val, higher_is_better, format_mode)
    # format_mode: "pct" = percentage, "float2" = 2-decimal plain float, "int" = integer
    rows = [
        ("Total Samples",          raw.total_samples,          filtered.total_samples,          None,  "int"),
        ("Schema Validity Rate",   raw.schema_validity_rate,   filtered.schema_validity_rate,   True,  "pct"),
        ("Task Consistency Score", raw.task_consistency_score, filtered.task_consistency_score, True,  "pct"),
        ("Completeness Score",     raw.completeness_score,     filtered.completeness_score,     True,  "pct"),
        ("Hallucination Rate ↓",   raw.hallucination_rate,     filtered.hallucination_rate,     False, "pct"),
        ("Diversity Score",        raw.diversity_score,        filtered.diversity_score,        True,  "pct"),
        ("Mean Confidence",        raw.mean_confidence,        filtered.mean_confidence,        True,  "pct"),
        ("Vocab Entropy (bits)",   raw.vocabulary_entropy,     filtered.vocabulary_entropy,     True,  "float2"),
        ("Bigram Entropy (bits)",  raw.bigram_entropy,         filtered.bigram_entropy,         True,  "float2"),
        ("Collapse Risk Score ↓",  raw.collapse_risk_score,    filtered.collapse_risk_score,    False, "pct"),
    ]

    for name, raw_val, filt_val, higher_is_better, fmt in rows:
        diff = filt_val - raw_val if not isinstance(raw_val, int) else filt_val - raw_val
        if fmt == "int":
            rv = str(raw_val)
            fv = str(filt_val)
            delta = f"{diff:+d}"
            color = "green" if diff <= 0 else "yellow"
        elif fmt == "float2":
            rv = f"{raw_val:.2f}"
            fv = f"{filt_val:.2f}"
            color = "green" if (higher_is_better and diff >= 0) or (not higher_is_better and diff <= 0) else "red"
            delta = f"[{color}]{diff:+.2f}[/{color}]"
        else:  # "pct"
            rv = f"{raw_val:.1%}"
            fv = f"{filt_val:.1%}"
            color = "green" if (higher_is_better and diff >= 0) or (not higher_is_better and diff <= 0) else "red"
            delta = f"[{color}]{diff:+.1%}[/{color}]"
        table.add_row(name, rv, fv, delta)

    console.print()
    console.print(table)

    # Print collapse warning if present
    if filtered.collapse_warning:
        severity = "bold red" if "CRITICAL" in filtered.collapse_warning else "bold yellow"
        console.print(f"[{severity}]⚠️  {filtered.collapse_warning}[/{severity}]")
    elif filtered.collapse_risk_score < 0.30:
        console.print("[bold green]✓  Collapse risk LOW — diversity is healthy.[/bold green]")
    console.print()


def _plain_table(raw: DatasetMetrics, filtered: DatasetMetrics) -> None:
    lines = [
        "",
        "=" * 70,
        "  Dataset Quality Metrics: Raw vs Filtered",
        "=" * 70,
        f"  {'Metric':<32} {'Raw':>10} {'Filtered':>10} {'Delta':>10}",
        "-" * 70,
        f"  {'Total Samples':<32} {raw.total_samples:>10d} {filtered.total_samples:>10d}",
        f"  {'Schema Validity Rate':<32} {raw.schema_validity_rate:>10.1%} {filtered.schema_validity_rate:>10.1%} {filtered.schema_validity_rate - raw.schema_validity_rate:>+10.1%}",
        f"  {'Task Consistency Score':<32} {raw.task_consistency_score:>10.1%} {filtered.task_consistency_score:>10.1%} {filtered.task_consistency_score - raw.task_consistency_score:>+10.1%}",
        f"  {'Completeness Score':<32} {raw.completeness_score:>10.1%} {filtered.completeness_score:>10.1%} {filtered.completeness_score - raw.completeness_score:>+10.1%}",
        f"  {'Hallucination Rate':<32} {raw.hallucination_rate:>10.1%} {filtered.hallucination_rate:>10.1%} {filtered.hallucination_rate - raw.hallucination_rate:>+10.1%}",
        f"  {'Diversity Score':<32} {raw.diversity_score:>10.1%} {filtered.diversity_score:>10.1%} {filtered.diversity_score - raw.diversity_score:>+10.1%}",
        f"  {'Mean Confidence':<32} {raw.mean_confidence:>10.2f} {filtered.mean_confidence:>10.2f} {filtered.mean_confidence - raw.mean_confidence:>+10.2f}",
        f"  {'Vocab Entropy (bits)':<32} {raw.vocabulary_entropy:>10.2f} {filtered.vocabulary_entropy:>10.2f} {filtered.vocabulary_entropy - raw.vocabulary_entropy:>+10.2f}",
        f"  {'Bigram Entropy (bits)':<32} {raw.bigram_entropy:>10.2f} {filtered.bigram_entropy:>10.2f} {filtered.bigram_entropy - raw.bigram_entropy:>+10.2f}",
        f"  {'Collapse Risk Score':<32} {raw.collapse_risk_score:>10.2f} {filtered.collapse_risk_score:>10.2f} {filtered.collapse_risk_score - raw.collapse_risk_score:>+10.2f}",
        "=" * 70,
    ]
    if filtered.collapse_warning:
        lines.append(f"  !! {filtered.collapse_warning}")
    elif filtered.collapse_risk_score < 0.30:
        lines.append("  OK Collapse risk LOW — diversity is healthy.")
    lines.append("")
    print("\n".join(lines))


def _console_print(msg: str) -> None:
    try:
        from rich.console import Console
        Console().print(msg)
    except ImportError:
        print(msg)
