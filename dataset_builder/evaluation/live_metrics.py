"""
live_metrics.py — Thread-safe real-time generation metrics tracker.

Provides a Rich Live dashboard that updates every time a sample is emitted,
showing the user exactly what is happening during generation/critic passes:

  ┌ SynthDataLab · Live Generation Dashboard ─────────────────────────────────┐
  │  Progress  ████████████░░░░  45/100  45%   ⚡ 3.2 samples/sec  ETA 17s   │
  │                                                                            │
  │  Quality Decisions        Task Types          Quality / Collapse           │
  │  ✓ ACCEPT   38  (84%)  ██  qa      18 (40%)  Confidence   0.82            │
  │  ✗ REJECT    4   (9%)  █   extract 14 (31%)  Critic score 0.71            │
  │  ~ FIX       3   (7%)  █   reason  13 (29%)  Collapse risk  LOW 🟢        │
  │                                                                            │
  │  Recent samples ─────────────────────────────────────────────────────────  │
  │  ✓ qa             conf=0.85 critic=0.79  s_abc123…                        │
  │  ✗ reasoning      conf=0.32 critic=0.28  s_def456…                        │
  └────────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Risk level → Rich colour tag
_RISK_COLOURS: Dict[str, str] = {
    "LOW": "green",
    "MEDIUM": "yellow",
    "HIGH": "red",
    "CRITICAL": "red bold",
    "UNKNOWN": "dim",
}

# Bar fill characters
_BAR_FULL = "█"
_BAR_EMPTY = "░"


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleEvent:
    """One recorded generation event."""

    sample_id: str
    task_type: str
    confidence: float
    critic_score: float
    status: str          # "ACCEPT" | "REJECT" | "FIX_REQUIRED" | "ERROR"
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class LiveSnapshot:
    """Point-in-time snapshot of accumulated metrics (safe to read off-lock)."""

    total: int = 0
    generated: int = 0
    accepted: int = 0
    rejected: int = 0
    fix_required: int = 0
    errors: int = 0
    task_counts: Dict[str, int] = field(default_factory=dict)
    confidence_scores: List[float] = field(default_factory=list)
    critic_scores: List[float] = field(default_factory=list)
    collapse_risk_label: str = "UNKNOWN"
    collapse_risk_score: float = 0.0
    start_time: float = field(default_factory=time.monotonic)

    # ── Derived stats ─────────────────────────────────────────────────────────

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def throughput(self) -> float:
        e = self.elapsed
        return self.generated / e if e > 0 else 0.0

    @property
    def eta_seconds(self) -> Optional[float]:
        remaining = self.total - self.generated
        thr = self.throughput
        return remaining / thr if (thr > 0 and remaining > 0) else None

    @property
    def mean_confidence(self) -> float:
        return (
            sum(self.confidence_scores) / len(self.confidence_scores)
            if self.confidence_scores
            else 0.0
        )

    @property
    def mean_critic(self) -> float:
        return (
            sum(self.critic_scores) / len(self.critic_scores)
            if self.critic_scores
            else 0.0
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main tracker
# ─────────────────────────────────────────────────────────────────────────────

class LiveMetricsTracker:
    """
    Thread-safe real-time tracker with an optional Rich Live dashboard.

    Usage (context-manager form — preferred)::

        with LiveMetricsTracker(total=100) as tracker:
            for sample in pipeline:
                tracker.record(sample.id, sample.task_type, 0.9, 0.8, "ACCEPT")

        tracker.print_final_report()

    The live panel auto-updates each time ``record()`` is called.
    """

    _RECENT_MAX = 10    # lines shown in the "Recent Samples" section

    def __init__(
        self,
        total: int,
        show_dashboard: bool = True,
        refresh_per_second: float = 4.0,
    ):
        self.total = total
        self.show_dashboard = show_dashboard
        self.refresh_per_second = refresh_per_second

        self._lock = threading.Lock()
        self._snapshot = LiveSnapshot(total=total, start_time=time.monotonic())
        self._recent: Deque[SampleEvent] = deque(maxlen=self._RECENT_MAX)
        self._live = None    # Rich Live instance, set in __enter__

    # ── Context-manager interface ─────────────────────────────────────────────

    def __enter__(self) -> "LiveMetricsTracker":
        if self.show_dashboard:
            try:
                from rich.live import Live
                from rich.console import Console

                self._live = Live(
                    self._render(),
                    refresh_per_second=self.refresh_per_second,
                    console=Console(stderr=True),
                )
                self._live.__enter__()
            except ImportError:
                self._live = None
        return self

    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)
            self._live = None

    # ── Record events ─────────────────────────────────────────────────────────

    def record(
        self,
        sample_id: str,
        task_type: str,
        confidence: float,
        critic_score: float,
        status: str,
    ) -> None:
        """Record a completed generation event. Thread-safe."""
        event = SampleEvent(
            sample_id=sample_id,
            task_type=task_type,
            confidence=confidence,
            critic_score=critic_score,
            status=status,
        )
        with self._lock:
            snap = self._snapshot
            snap.generated += 1
            snap.task_counts[task_type] = snap.task_counts.get(task_type, 0) + 1
            snap.confidence_scores.append(confidence)
            snap.critic_scores.append(critic_score)
            if status == "ACCEPT":
                snap.accepted += 1
            elif status == "REJECT":
                snap.rejected += 1
            elif status == "FIX_REQUIRED":
                snap.fix_required += 1
            else:
                snap.errors += 1
            self._recent.append(event)

        if self._live:
            self._live.update(self._render())

    def update_collapse_risk(self, score: float, label: str) -> None:
        """Push the latest collapse risk reading (called periodically)."""
        with self._lock:
            self._snapshot.collapse_risk_score = score
            self._snapshot.collapse_risk_label = label
        if self._live:
            self._live.update(self._render())

    def get_snapshot(self) -> LiveSnapshot:
        """Return a thread-safe deep-copy snapshot."""
        with self._lock:
            s = self._snapshot
            return LiveSnapshot(
                total=s.total,
                generated=s.generated,
                accepted=s.accepted,
                rejected=s.rejected,
                fix_required=s.fix_required,
                errors=s.errors,
                task_counts=dict(s.task_counts),
                confidence_scores=list(s.confidence_scores),
                critic_scores=list(s.critic_scores),
                collapse_risk_label=s.collapse_risk_label,
                collapse_risk_score=s.collapse_risk_score,
                start_time=s.start_time,
            )

    # ── Final summary ─────────────────────────────────────────────────────────

    def print_final_report(self) -> None:
        """Print a formatted summary table after generation completes."""
        snap = self.get_snapshot()
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.rule import Rule

            console = Console()
            console.print(Rule("[bold cyan]Generation Complete — Driver Summary[/bold cyan]"))

            t = Table(show_header=False, box=None, padding=(0, 2), expand=False)
            t.add_column(style="dim", justify="right", min_width=22)
            t.add_column(min_width=30)

            gen = snap.generated or 1
            a_pct = snap.accepted / gen * 100
            r_pct = snap.rejected / gen * 100
            f_pct = snap.fix_required / gen * 100

            t.add_row("Total generated", f"[bold]{snap.generated}[/bold] / {snap.total}")
            t.add_row(
                "Accepted",
                f"[green]{snap.accepted}[/green]  ({a_pct:.1f}%)"
                f"  [green]{'█' * max(0, int(a_pct / 5))}[/green]",
            )
            t.add_row(
                "Rejected",
                f"[red]{snap.rejected}[/red]  ({r_pct:.1f}%)"
                f"  [red]{'█' * max(0, int(r_pct / 5))}[/red]",
            )
            t.add_row(
                "Fix required",
                f"[yellow]{snap.fix_required}[/yellow]  ({f_pct:.1f}%)"
                f"  [yellow]{'█' * max(0, int(f_pct / 5))}[/yellow]",
            )
            if snap.errors:
                t.add_row("Errors", f"[red]{snap.errors}[/red]")
            t.add_row("Elapsed", f"{snap.elapsed:.1f}s")
            t.add_row("Throughput", f"{snap.throughput:.2f} samples/sec")
            t.add_row("Mean confidence", f"{snap.mean_confidence:.3f}")
            t.add_row("Mean critic score", f"{snap.mean_critic:.3f}")

            col = _RISK_COLOURS.get(snap.collapse_risk_label, "white")
            t.add_row(
                "Collapse risk",
                f"[{col}]{snap.collapse_risk_label}[/{col}]"
                f"  (score={snap.collapse_risk_score:.3f})",
            )

            console.print(t)

            # Task type breakdown
            if snap.task_counts:
                console.print()
                console.print("[dim]Task type breakdown:[/dim]")
                for task, count in sorted(snap.task_counts.items(), key=lambda x: -x[1]):
                    bar = "█" * max(1, int(count / gen * 20))
                    console.print(
                        f"  [cyan]{task:<18}[/cyan] {count:>4}  ({count/gen:.0%})  [cyan]{bar}[/cyan]"
                    )
        except ImportError:
            # Fallback plain-text
            print(
                f"Generated {snap.generated}/{snap.total}  "
                f"Accept={snap.accepted}  Reject={snap.rejected}  "
                f"Fix={snap.fix_required}  {snap.elapsed:.1f}s  "
                f"{snap.throughput:.1f} samp/s"
            )

    # ── Rich rendering ────────────────────────────────────────────────────────

    def _render(self):
        """Build the Rich renderable for the Live panel."""
        try:
            from rich.columns import Columns
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            snap = self.get_snapshot()
            gen = max(snap.generated, 1)
            pct = snap.generated / snap.total if snap.total > 0 else 0.0
            bar_w = 28
            filled = int(pct * bar_w)
            bar = f"[cyan]{_BAR_FULL * filled}[/cyan][dim]{_BAR_EMPTY * (bar_w - filled)}[/dim]"

            eta_str = ""
            if snap.eta_seconds is not None:
                eta_sec = snap.eta_seconds
                eta_str = (
                    f"  ETA {int(eta_sec)}s"
                    if eta_sec < 3600
                    else f"  ETA {int(eta_sec / 60)}m"
                )

            progress_line = Text.from_markup(
                f"[bold]Progress[/bold]  {bar}  "
                f"[bold]{snap.generated}[/bold][dim]/{snap.total}[/dim]  {pct:.0%}"
                f"    [dim]⚡ {snap.throughput:.1f} samp/sec{eta_str}[/dim]"
            )

            # ── Decisions ─────────────────────────────────────────────────────
            d = Table(title="[bold]Decisions[/bold]", box=None, show_header=False, padding=(0, 1))
            d.add_column(no_wrap=True)
            d.add_column(justify="right")
            d.add_column(justify="right", style="dim")
            d.add_column()
            _mk_bar = lambda n: ("█" * max(0, int(n / gen * 10))) if n > 0 else ""
            d.add_row(
                "[green]✓ ACCEPT[/green]", str(snap.accepted),
                f"({snap.accepted / gen:.0%})", f"[green]{_mk_bar(snap.accepted)}[/green]",
            )
            d.add_row(
                "[red]✗ REJECT[/red]", str(snap.rejected),
                f"({snap.rejected / gen:.0%})", f"[red]{_mk_bar(snap.rejected)}[/red]",
            )
            d.add_row(
                "[yellow]~ FIX[/yellow]", str(snap.fix_required),
                f"({snap.fix_required / gen:.0%})", f"[yellow]{_mk_bar(snap.fix_required)}[/yellow]",
            )
            if snap.errors:
                d.add_row("[dim]⚠ ERR[/dim]", str(snap.errors), f"({snap.errors / gen:.0%})", "")

            # ── Task types ────────────────────────────────────────────────────
            tt = Table(title="[bold]Task Types[/bold]", box=None, show_header=False, padding=(0, 1))
            tt.add_column(no_wrap=True)
            tt.add_column(justify="right")
            tt.add_column(justify="right", style="dim")
            for task, count in sorted(snap.task_counts.items(), key=lambda x: -x[1]):
                tt.add_row(f"[cyan]{task}[/cyan]", str(count), f"({count / gen:.0%})")
            if not snap.task_counts:
                tt.add_row("[dim]— waiting —[/dim]", "", "")

            # ── Quality & collapse ────────────────────────────────────────────
            col = _RISK_COLOURS.get(snap.collapse_risk_label, "white")
            qs = Table(title="[bold]Quality / Health[/bold]", box=None, show_header=False, padding=(0, 1))
            qs.add_column(style="dim", no_wrap=True)
            qs.add_column(justify="right")
            qs.add_row("Confidence", f"[cyan]{snap.mean_confidence:.3f}[/cyan]")
            qs.add_row("Critic score", f"[cyan]{snap.mean_critic:.3f}[/cyan]")
            qs.add_row(
                "Collapse risk",
                f"[{col}]{snap.collapse_risk_label}[/{col}] ({snap.collapse_risk_score:.2f})",
            )

            # ── Recent events ─────────────────────────────────────────────────
            with self._lock:
                recent = list(self._recent)
            event_lines: List[str] = []
            for ev in reversed(recent):
                icon = {
                    "ACCEPT": "[green]✓[/green]",
                    "REJECT": "[red]✗[/red]",
                    "FIX_REQUIRED": "[yellow]~[/yellow]",
                }.get(ev.status, "[dim]?[/dim]")
                event_lines.append(
                    f"  {icon} [dim]{ev.task_type:<14}[/dim]"
                    f" conf=[cyan]{ev.confidence:.2f}[/cyan]"
                    f" critic=[cyan]{ev.critic_score:.2f}[/cyan]"
                    f"  [dim]{ev.sample_id[-14:]}[/dim]"
                )
            recent_str = (
                "\n".join(event_lines)
                if event_lines
                else "  [dim]— waiting for first sample —[/dim]"
            )

            return Panel(
                Group(
                    progress_line,
                    Text(""),
                    Columns([d, tt, qs], equal=False, expand=True),
                    Text(""),
                    Panel(
                        Text.from_markup(recent_str),
                        title="[dim]Recent Samples[/dim]",
                        border_style="dim",
                        padding=(0, 1),
                    ),
                ),
                title="[bold cyan]SynthDataLab · Live Generation Dashboard[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )
        except Exception as exc:
            logger.debug("Live render error: %s", exc)
            snap = self.get_snapshot()
            return (
                f"Generating {snap.generated}/{snap.total}  "
                f"✓{snap.accepted} ✗{snap.rejected} ~{snap.fix_required}"
            )
