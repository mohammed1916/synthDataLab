"""
orchestrator.py — Multi-agent orchestrator with human steering.

Architecture
------------

  ┌────────────────────────────────────────────────────────────────────┐
  │                    MultiAgentOrchestrator                          │
  │                                                                    │
  │  ┌────────────────┐   sample    ┌──────────────────────────────┐  │
  │  │ Generator      │ ──────────► │  CriticAgent                 │  │
  │  │ Agent          │             │  (4-axis heuristic scoring)  │  │
  │  └────────────────┘             └──────────────────────────────┘  │
  │                                           │ critic_score           │
  │                                           ▼                        │
  │                               ┌─────────────────────┐             │
  │                               │  Steering Gate       │             │
  │                               │  AUTO   → threshold  │             │
  │                               │  REVIEW → human TTY  │             │
  │                               └─────────────────────┘             │
  │                                           │ accept/reject/rewrite  │
  │                                           ▼                        │
  │                               ┌─────────────────────┐             │
  │                               │  LiveMetricsTracker  │             │
  │                               │  (Rich dashboard)    │             │
  │                               └─────────────────────┘             │
  └────────────────────────────────────────────────────────────────────┘

Steering modes (``--steering`` CLI flag)
-----------------------------------------
  auto         Critic makes ALL decisions based on threshold. No human input.
  review-low   Human reviews samples whose critic score < threshold.
  review-all   Human reviews EVERY sample.

Human steering keys (when in a review prompt):
  a  → approve (accept sample as-is)
  r  → reject  (discard sample)
  f  → fix     (mark FIX_REQUIRED — downstream can attempt repair)
  s  → skip    (pass sample through without critic evaluation)
  q  → quit    (stop generation and save what was collected up to now)
  ?  → show the sample's full content

Collapse risk is re-evaluated every ``collapse_check_interval`` samples and
pushed to the LiveMetricsTracker for the dashboard gauge.
"""
from __future__ import annotations

import logging
import random
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from config import Config, DEFAULT_CONFIG
from ingestion.ingestor import IngestionResult
from generation.generator import DatasetGenerator
from generation.critic_agent import CriticAgent, CriticScore
from schema.dataset_schema import DatasetSample
from evaluation.live_metrics import LiveMetricsTracker
from evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Steering configuration
# ─────────────────────────────────────────────────────────────────────────────

class SteeringMode(str, Enum):
    AUTO = "auto"               # critic decides everything
    REVIEW_LOW = "review-low"   # human reviews score < threshold
    REVIEW_ALL = "review-all"   # human reviews every sample


@dataclass
class OrchestratorConfig:
    """Tuning knobs for the multi-agent run."""

    steering_mode: SteeringMode = SteeringMode.AUTO
    critic_pass_threshold: float = 0.70     # PASS if composite ≥ this
    critic_review_threshold: float = 0.45   # REVIEW if composite ≥ this
    auto_reject_below: float = 0.30         # hard-reject if composite < this
    collapse_check_interval: int = 10       # re-compute collapse risk every N samples
    show_dashboard: bool = True
    save_critic_metadata: bool = True       # attach critic scores to sample metadata


# ─────────────────────────────────────────────────────────────────────────────
# Human steering
# ─────────────────────────────────────────────────────────────────────────────

_REVIEW_HELP = (
    "  [a] approve   [r] reject   [f] fix-required   [s] skip   [q] quit   [?] show sample"
)


class HumanSteering:
    """
    Interactive TTY gate that presents a sample to the human and waits for
    an action.  Falls back to AUTO-approve when stdin is not a TTY (e.g. CI).
    """

    def __init__(self, is_tty: bool = True):
        self.is_tty = is_tty and sys.stdin.isatty()

    def review(
        self,
        sample: Dict[str, Any],
        critic_score: CriticScore,
        index: int,
        total: int,
    ) -> str:
        """
        Show the sample to the human and return one of:
        'ACCEPT' | 'REJECT' | 'FIX_REQUIRED' | 'SKIP' | 'QUIT'
        """
        if not self.is_tty:
            # Non-interactive environment — silently approve
            return "ACCEPT"

        # Try to use rich for pretty printing
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            console = Console()
            console.print()
            console.rule(
                f"[bold yellow]Human Review ({index}/{total})[/bold yellow]"
            )

            # Score breakdown panel
            st = Table.grid(padding=(0, 2))
            st.add_column(style="dim", justify="right")
            st.add_column()
            st.add_row("Task type", f"[cyan]{sample.get('task_type', '?')}[/cyan]")
            st.add_row("Relevance",    f"{critic_score.relevance:.2f}")
            st.add_row("Coherence",    f"{critic_score.coherence:.2f}")
            st.add_row("Groundedness", f"{critic_score.groundedness:.2f}")
            st.add_row("Fluency",      f"{critic_score.fluency:.2f}")
            verdict_col = {"PASS": "green", "REVIEW": "yellow", "FAIL": "red"}.get(
                critic_score.verdict, "white"
            )
            st.add_row(
                "Composite",
                f"[bold {verdict_col}]{critic_score.composite:.3f} → {critic_score.verdict}[/bold {verdict_col}]",
            )
            console.print(
                Panel(st, title="[bold cyan]Critics Scores[/bold cyan]", border_style="cyan")
            )
        except ImportError:
            console = None
            print(f"\n--- Human Review ({index}/{total}) ---")
            print(f"  Task:    {sample.get('task_type', '?')}")
            print(f"  Critic:  {critic_score.composite:.3f} ({critic_score.verdict})")

        print(_REVIEW_HELP)

        while True:
            try:
                choice = input("  Action: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  [Interrupted — treating as QUIT]")
                return "QUIT"

            if choice in ("a", "approve", ""):
                return "ACCEPT"
            elif choice in ("r", "reject"):
                return "REJECT"
            elif choice in ("f", "fix"):
                return "FIX_REQUIRED"
            elif choice in ("s", "skip"):
                return "ACCEPT"    # skip → let it through unchanged
            elif choice in ("q", "quit"):
                return "QUIT"
            elif choice == "?":
                _show_sample(sample)
            else:
                print(f"  Unknown choice '{choice}'. {_REVIEW_HELP}")


def _show_sample(sample: Dict[str, Any]) -> None:
    """Print the full sample content for human inspection."""
    try:
        import json
        from rich.console import Console
        from rich.syntax import Syntax

        Console().print(
            Syntax(
                json.dumps(sample, indent=2, ensure_ascii=False),
                "json",
                theme="monokai",
                line_numbers=False,
            )
        )
    except Exception:
        import json as _json
        print(_json.dumps(sample, indent=2, ensure_ascii=False)[:2000])


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrchestrationResult:
    """Summary returned by ``MultiAgentOrchestrator.run()``."""

    accepted: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    fix_required: List[Dict[str, Any]] = field(default_factory=list)
    total_generated: int = 0
    critic_scores: List[CriticScore] = field(default_factory=list)
    metrics_snapshot: Optional[Dict[str, Any]] = None
    aborted: bool = False

    @property
    def acceptance_rate(self) -> float:
        return len(self.accepted) / max(self.total_generated, 1)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_generated": self.total_generated,
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "fix_required": len(self.fix_required),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "aborted": self.aborted,
        }


class MultiAgentOrchestrator:
    """
    Coordinates Generator Agent → CriticAgent → Steering Gate → Tracker.

    Usage::

        orch = MultiAgentOrchestrator(config, orch_config)
        result = orch.run(ingestion_results)
        # result.accepted  — list of approved sample dicts
        # result.rejected  — list of discarded sample dicts
    """

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        orch_config: Optional[OrchestratorConfig] = None,
    ):
        self.config = config
        self.orch = orch_config or OrchestratorConfig()
        self.generator = DatasetGenerator(config)
        self.critic = CriticAgent(
            pass_threshold=self.orch.critic_pass_threshold,
            review_threshold=self.orch.critic_review_threshold,
        )
        self.steering = HumanSteering()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        ingestion_results: List[IngestionResult],
        on_sample: Optional[Any] = None,   # optional callback(sample, score, status)
    ) -> OrchestrationResult:
        """
        Run the full multi-agent pipeline.

        Args:
            ingestion_results: Normalised input records from the ingestion layer.
            on_sample: Optional callback called after each sample is processed.
                       Signature: ``on_sample(sample_dict, critic_score, status)``

        Returns:
            ``OrchestrationResult`` with accepted/rejected/fix_required lists.
        """
        work = [
            (chunk, task_type)
            for chunk in ingestion_results
            for task_type in self.config.generation.task_types
        ]
        total = len(work)
        result = OrchestrationResult()

        logger.info(
            "Multi-agent run: %d work items  steering=%s  pass_threshold=%.2f",
            total,
            self.orch.steering_mode,
            self.orch.critic_pass_threshold,
        )

        with LiveMetricsTracker(
            total=total,
            show_dashboard=self.orch.show_dashboard,
        ) as tracker:
            for idx, (chunk, task_type) in enumerate(work, start=1):
                # ── 1. Generate ───────────────────────────────────────────────
                sample = self.generator._generate_one(chunk, task_type)
                if sample is None:
                    result.total_generated += 1
                    tracker.record("—", task_type, 0.0, 0.0, "ERROR")
                    continue

                result.total_generated += 1
                sample_dict = sample.to_dict()

                # ── 2. Critic scoring ─────────────────────────────────────────
                critic_score = self.critic.score(sample_dict)
                result.critic_scores.append(critic_score)

                if self.orch.save_critic_metadata:
                    sample_dict.setdefault("metadata", {})["critic"] = (
                        critic_score.to_dict()
                    )

                # ── 3. Steering gate ──────────────────────────────────────────
                status = self._apply_steering(
                    sample_dict, critic_score, idx, total
                )

                if status == "QUIT":
                    result.aborted = True
                    logger.warning("User aborted multi-agent run at sample %d/%d", idx, total)
                    break

                # ── 4. Bucket sample ──────────────────────────────────────────
                if status == "ACCEPT":
                    result.accepted.append(sample_dict)
                elif status == "REJECT":
                    result.rejected.append(sample_dict)
                else:   # FIX_REQUIRED
                    result.fix_required.append(sample_dict)

                # ── 5. Record to tracker ──────────────────────────────────────
                tracker.record(
                    sample_id=sample_dict.get("id", ""),
                    task_type=task_type,
                    confidence=float(
                        sample_dict.get("metadata", {}).get("confidence", 0.0)
                    ),
                    critic_score=critic_score.composite,
                    status=status,
                )

                # ── 6. Periodic collapse check ────────────────────────────────
                if idx % self.orch.collapse_check_interval == 0:
                    self._update_collapse_risk(tracker, result.accepted)

                # ── 7. External callback ──────────────────────────────────────
                if on_sample:
                    try:
                        on_sample(sample_dict, critic_score, status)
                    except Exception as cb_exc:
                        logger.debug("on_sample callback error: %s", cb_exc)

        # Final collapse check
        self._update_collapse_risk(tracker, result.accepted)
        tracker.print_final_report()

        # Attach final metrics snapshot
        if result.accepted:
            try:
                m = compute_metrics(result.accepted)
                result.metrics_snapshot = m.to_dict()
            except Exception as exc:
                logger.warning("Could not compute final metrics: %s", exc)

        logger.info(
            "Orchestration complete: %d accepted / %d rejected / %d fix-required  "
            "(acceptance rate=%.1f%%)",
            len(result.accepted),
            len(result.rejected),
            len(result.fix_required),
            result.acceptance_rate * 100,
        )
        return result

    # ── Steering logic ────────────────────────────────────────────────────────

    def _apply_steering(
        self,
        sample_dict: Dict[str, Any],
        critic_score: CriticScore,
        idx: int,
        total: int,
    ) -> str:
        """
        Apply the configured steering policy.

        Returns: 'ACCEPT' | 'REJECT' | 'FIX_REQUIRED' | 'QUIT'
        """
        composite = critic_score.composite

        if self.orch.steering_mode == SteeringMode.AUTO:
            return self._auto_decision(composite)

        if self.orch.steering_mode == SteeringMode.REVIEW_ALL:
            return self.steering.review(sample_dict, critic_score, idx, total)

        # REVIEW_LOW — human reviews only below-threshold samples
        if composite < self.orch.critic_pass_threshold:
            return self.steering.review(sample_dict, critic_score, idx, total)

        return "ACCEPT"

    def _auto_decision(self, composite: float) -> str:
        if composite >= self.orch.critic_pass_threshold:
            return "ACCEPT"
        if composite >= self.orch.critic_review_threshold:
            return "FIX_REQUIRED"
        return "REJECT"

    # ── Collapse monitoring ───────────────────────────────────────────────────

    def _update_collapse_risk(
        self, tracker: LiveMetricsTracker, accepted: List[Dict[str, Any]]
    ) -> None:
        if not accepted:
            return
        try:
            m = compute_metrics(accepted)
            score = m.collapse_risk_score
            if score >= 0.70:
                label = "CRITICAL"
            elif score >= 0.50:
                label = "HIGH"
            elif score >= 0.30:
                label = "MEDIUM"
            else:
                label = "LOW"
            tracker.update_collapse_risk(score, label)
            if score >= 0.70:
                logger.warning(
                    "COLLAPSE RISK CRITICAL (score=%.3f) after %d accepted samples. "
                    "Consider diversifying input sources.",
                    score,
                    len(accepted),
                )
        except Exception as exc:
            logger.debug("Collapse risk update failed: %s", exc)
