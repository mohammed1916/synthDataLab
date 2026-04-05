"""
error_analyzer.py — Analyses rejected / flagged samples to surface patterns.

Error categories
----------------
MISSING_FIELD               — A required JSON field is absent
EMPTY_FIELD                 — Field exists but holds empty string / list / dict
WRONG_FORMAT                — Field present but wrong type or structure
LOW_CONFIDENCE              — Model assigned low confidence score
SCHEMA_INVALID              — Fails JSON schema validation
INSUFFICIENT_REASONING      — Reasoning chain too short
ANSWER_NOT_GROUNDED         — QA evidence not traceable to input text
LLM_REVIEWER_REJECTED       — Second-pass reviewer downgraded the sample
PARSE_ERROR                 — LLM returned non-JSON output
OTHER                       — Uncategorised

Output
------
* Per-category error counts and percentage of total rejects
* Per-task-type breakdown
* Concrete before/after correction examples (auto-fix demo)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from validation.annotation import AnnotatedSample, AnnotationLabel, RejectionCode

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ErrorExample:
    """A before/after comparison illustrating an error and its correction."""

    sample_id: str
    task_type: str
    error_code: str
    error_message: str
    before: dict[str, Any]     # original output
    after: dict[str, Any]      # auto-corrected output (if applicable)
    correction_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorReport:
    """Complete error analysis report."""

    total_rejected: int = 0
    total_fix_required: int = 0
    error_counts: dict[str, int] = field(default_factory=dict)
    error_rates: dict[str, float] = field(default_factory=dict)
    per_task_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    examples: list[ErrorExample] = field(default_factory=list)
    auto_corrections_applied: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rejected": self.total_rejected,
            "total_fix_required": self.total_fix_required,
            "error_counts": self.error_counts,
            "error_rates": {k: round(v, 4) for k, v in self.error_rates.items()},
            "per_task_breakdown": self.per_task_breakdown,
            "examples": [e.to_dict() for e in self.examples],
            "auto_corrections_applied": self.auto_corrections_applied,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Analyser
# ─────────────────────────────────────────────────────────────────────────────

class ErrorAnalyzer:
    """
    Analyses a list of AnnotatedSamples to produce an ErrorReport.

    Usage::

        analyzer = ErrorAnalyzer()
        report = analyzer.analyze(annotated_samples)
        analyzer.save_report(report, path="data/error_analysis.json")
        analyzer.print_summary(report)
    """

    # Maximum examples to include per error code
    MAX_EXAMPLES_PER_CODE = 2

    def analyze(self, annotated: list[AnnotatedSample]) -> ErrorReport:
        """Build a full ErrorReport from a list of annotated samples."""
        report = ErrorReport()

        rejected = [s for s in annotated if s.label == AnnotationLabel.REJECT]
        fix_required = [s for s in annotated if s.label == AnnotationLabel.FIX_REQUIRED]

        report.total_rejected = len(rejected)
        report.total_fix_required = len(fix_required)

        non_accepted = rejected + fix_required
        total_problems = len(non_accepted)

        if not non_accepted:
            return report

        # ── Count error codes ─────────────────────────────────────────────────
        code_counts: dict[str, int] = defaultdict(int)
        task_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        examples_per_code: dict[str, int] = defaultdict(int)

        for ann in non_accepted:
            task_type = ann.sample.get("task_type", "unknown")
            for reason in ann.rejection_reasons:
                code = reason.code if isinstance(reason.code, str) else reason.code.value
                code_counts[code] += 1
                task_breakdown[task_type][code] += 1

                # Collect illustrative examples
                if examples_per_code[code] < self.MAX_EXAMPLES_PER_CODE:
                    before_output = ann.sample.get("output", {})
                    after_output, corrected = _auto_correct(
                        ann.sample, code, reason.message
                    )
                    report.examples.append(
                        ErrorExample(
                            sample_id=ann.sample.get("id", "?"),
                            task_type=task_type,
                            error_code=code,
                            error_message=reason.message,
                            before=before_output,
                            after=after_output,
                            correction_applied=corrected,
                        )
                    )
                    if corrected:
                        report.auto_corrections_applied += 1
                    examples_per_code[code] += 1

        report.error_counts = dict(code_counts)
        report.error_rates = {
            code: count / total_problems
            for code, count in code_counts.items()
        }
        report.per_task_breakdown = {
            task: dict(codes) for task, codes in task_breakdown.items()
        }

        return report

    @staticmethod
    def save_report(report: ErrorReport, path: Path) -> None:
        """Write the error report to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def print_summary(report: ErrorReport) -> None:
        """Print a concise error summary to the console."""
        try:
            _rich_summary(report)
        except ImportError:
            _plain_summary(report)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-correction logic (best-effort fixes for common issues)
# ─────────────────────────────────────────────────────────────────────────────

def _auto_correct(
    sample: dict[str, Any], error_code: str, error_message: str
) -> tuple[dict[str, Any], bool]:
    """
    Attempt a deterministic fix for common error types.

    Returns:
        (corrected_output_dict, was_correction_applied)
    """
    import copy
    output = copy.deepcopy(sample.get("output", {}))
    task_type = sample.get("task_type", "")

    if not isinstance(output, dict):
        return output, False

    corrected = False

    if error_code == RejectionCode.WRONG_FORMAT and task_type == "qa":
        # Add missing "?" to question
        q = output.get("question", "")
        if q and not q.strip().endswith("?"):
            output["question"] = q.strip() + "?"
            corrected = True

    elif error_code == RejectionCode.MISSING_FIELD and task_type == "qa":
        # Add empty evidence from input text snippet
        if "evidence" not in output:
            input_text = sample.get("input", "")
            output["evidence"] = input_text[:200] + "..." if len(input_text) > 200 else input_text
            corrected = True

    elif error_code == RejectionCode.EXTRACTION_ENTITY_LIST_EMPTY:
        # Insert a generic placeholder entity
        if "entities" not in output or not output["entities"]:
            words = re.findall(r"\b[A-Z][a-z]{2,}\b", sample.get("input", ""))
            if words:
                output["entities"] = [{"text": words[0], "type": "CONCEPT"}]
                corrected = True

    elif error_code == RejectionCode.INSUFFICIENT_REASONING_STEPS:
        # Add a generic synthesis step
        steps = output.get("reasoning_steps", [])
        if isinstance(steps, list) and len(steps) < 2:
            output["reasoning_steps"] = steps + [
                "Step (auto) — Further analysis is required to strengthen this reasoning chain."
            ]
            corrected = True

    return output, corrected


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rich_summary(report: ErrorReport) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    total = report.total_rejected + report.total_fix_required

    console.print("\n[bold cyan]Error Analysis Summary[/bold cyan]")
    console.print(
        f"  Rejected: [red]{report.total_rejected}[/red]  |  "
        f"Fix Required: [yellow]{report.total_fix_required}[/yellow]  |  "
        f"Total Problematic: [bold]{total}[/bold]"
    )

    if not report.error_counts:
        console.print("  No errors to display.\n")
        return

    table = Table(box=box.SIMPLE, header_style="bold")
    table.add_column("Error Code", style="bold red")
    table.add_column("Count", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Affected Task Types")

    for code, count in sorted(report.error_counts.items(), key=lambda x: -x[1]):
        rate = report.error_rates.get(code, 0)
        tasks = [
            task for task, codes in report.per_task_breakdown.items() if code in codes
        ]
        table.add_row(code, str(count), f"{rate:.1%}", ", ".join(tasks) or "—")

    console.print(table)

    if report.auto_corrections_applied:
        console.print(
            f"  [green]Auto-corrections applied: {report.auto_corrections_applied}[/green]\n"
        )


def _plain_summary(report: ErrorReport) -> None:
    total = report.total_rejected + report.total_fix_required
    lines = [
        "",
        "Error Analysis Summary",
        "=" * 55,
        f"  Rejected: {report.total_rejected}  |  "
        f"Fix Required: {report.total_fix_required}  |  "
        f"Total Problematic: {total}",
        "",
        f"  {'Error Code':<40} {'Count':>6} {'Rate':>8}",
        "-" * 55,
    ]
    for code, count in sorted(report.error_counts.items(), key=lambda x: -x[1]):
        rate = report.error_rates.get(code, 0)
        lines.append(f"  {code:<40} {count:>6} {rate:>8.1%}")
    lines.append("=" * 55)
    if report.auto_corrections_applied:
        lines.append(f"  Auto-corrections applied: {report.auto_corrections_applied}")
    lines.append("")
    print("\n".join(lines))
