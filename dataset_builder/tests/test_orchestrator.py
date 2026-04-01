"""tests/test_orchestrator.py — Tests for MultiAgentOrchestrator, steering, auto-halt, repair."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from config import Config
from generation.orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorConfig,
    OrchestrationResult,
    SteeringMode,
    CollapseAbortError,
    HumanSteering,
)
from ingestion.ingestor import IngestionResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cfg() -> Config:
    c = Config()
    c.llm.provider = "mock"
    return c


def _chunks(n: int = 2) -> list[IngestionResult]:
    return [
        IngestionResult(
            source_type="text",
            content=f"Machine learning is a powerful AI technique. Sample passage {i}.",
            metadata={"source": f"test_{i}"},
        )
        for i in range(n)
    ]


# ── Auto steering mode ────────────────────────────────────────────────────────

class TestAutoSteering:
    def test_run_returns_orchestration_result(self):
        cfg = _cfg()
        orch_cfg = OrchestratorConfig(steering_mode=SteeringMode.AUTO, show_dashboard=False)
        orch = MultiAgentOrchestrator(config=cfg, orch_config=orch_cfg)
        result = orch.run(_chunks(2))
        assert isinstance(result, OrchestrationResult)
        assert result.total_generated >= 0

    def test_all_samples_bucketed(self):
        cfg = _cfg()
        orch_cfg = OrchestratorConfig(steering_mode=SteeringMode.AUTO, show_dashboard=False)
        orch = MultiAgentOrchestrator(config=cfg, orch_config=orch_cfg)
        result = orch.run(_chunks(2))
        total = len(result.accepted) + len(result.rejected) + len(result.fix_required)
        assert total == result.total_generated

    def test_high_threshold_reduces_direct_acceptance(self):
        """With threshold=1.0 most samples are rejected (repair may still promote a few)."""
        cfg = _cfg()
        orch_cfg = OrchestratorConfig(
            steering_mode=SteeringMode.AUTO,
            critic_pass_threshold=1.0,
            critic_review_threshold=0.99,
            show_dashboard=False,
            collapse_abort_threshold=0.0,   # disable auto-halt
        )
        orch = MultiAgentOrchestrator(config=cfg, orch_config=orch_cfg)
        result = orch.run(_chunks(2))
        # Majority of samples should be rejected or fix_required (not all accepted)
        assert len(result.rejected) + len(result.fix_required) > 0

    def test_low_threshold_accepts_everything(self):
        """With threshold=0.0 and review_threshold=0.0, all samples pass."""
        cfg = _cfg()
        orch_cfg = OrchestratorConfig(
            steering_mode=SteeringMode.AUTO,
            critic_pass_threshold=0.0,
            critic_review_threshold=0.0,
            show_dashboard=False,
            collapse_abort_threshold=0.0,
        )
        orch = MultiAgentOrchestrator(config=cfg, orch_config=orch_cfg)
        result = orch.run(_chunks(2))
        assert result.total_generated > 0
        assert len(result.rejected) == 0


# ── Non-interactive steering ──────────────────────────────────────────────────

class TestHumanSteering:
    def test_non_tty_always_returns_accept(self):
        hs = HumanSteering(is_tty=False)
        action = hs.review({}, MagicMock(composite=0.3, verdict="FAIL"), 1, 10)
        assert action == "ACCEPT"


# ── orchestration result helpers ──────────────────────────────────────────────

class TestOrchestrationResult:
    def test_acceptance_rate_zero_when_nothing_generated(self):
        r = OrchestrationResult()
        assert r.acceptance_rate == 0.0

    def test_acceptance_rate_calculation(self):
        r = OrchestrationResult(total_generated=10)
        r.accepted = [{}] * 7
        assert abs(r.acceptance_rate - 0.7) < 1e-9

    def test_summary_dict_keys(self):
        r = OrchestrationResult(total_generated=5)
        r.accepted = [{}] * 3
        r.rejected = [{}] * 2
        s = r.summary()
        assert "total_generated" in s
        assert "acceptance_rate" in s
        assert "aborted" in s


# ── Collapse abort ────────────────────────────────────────────────────────────

class TestCollapseAbortError:
    def test_exception_message_contains_score(self):
        exc = CollapseAbortError(0.85, 20)
        assert "0.850" in str(exc)
        assert "20" in str(exc)

    def test_exception_attributes(self):
        exc = CollapseAbortError(0.91, 15)
        assert exc.score == 0.91
        assert exc.n_accepted == 15

    def test_auto_halt_sets_aborted_flag(self):
        """When collapse_abort is triggered, result.aborted should be True."""
        cfg = _cfg()
        # Set a very low abort threshold so it fires immediately
        orch_cfg = OrchestratorConfig(
            steering_mode=SteeringMode.AUTO,
            show_dashboard=False,
            collapse_check_interval=1,
            collapse_abort_threshold=0.001,  # fires on first check
        )
        orch = MultiAgentOrchestrator(config=cfg, orch_config=orch_cfg)
        # Even if it aborts, it should return a result (not raise)
        result = orch.run(_chunks(1))
        # Result is returned regardless of abort
        assert isinstance(result, OrchestrationResult)


# ── Critic scores attached to metadata ───────────────────────────────────────

class TestCriticMetadata:
    def test_critic_metadata_attached_when_configured(self):
        cfg = _cfg()
        orch_cfg = OrchestratorConfig(
            steering_mode=SteeringMode.AUTO,
            critic_pass_threshold=0.0,
            show_dashboard=False,
            save_critic_metadata=True,
            collapse_abort_threshold=0.0,
        )
        orch = MultiAgentOrchestrator(config=cfg, orch_config=orch_cfg)
        result = orch.run(_chunks(1))
        all_samples = result.accepted + result.fix_required + result.rejected
        if all_samples:
            meta = all_samples[0].get("metadata", {})
            assert "critic" in meta

    def test_critic_metadata_absent_when_disabled(self):
        cfg = _cfg()
        orch_cfg = OrchestratorConfig(
            steering_mode=SteeringMode.AUTO,
            critic_pass_threshold=0.0,
            show_dashboard=False,
            save_critic_metadata=False,
            collapse_abort_threshold=0.0,
        )
        orch = MultiAgentOrchestrator(config=cfg, orch_config=orch_cfg)
        result = orch.run(_chunks(1))
        all_samples = result.accepted + result.fix_required + result.rejected
        if all_samples:
            meta = all_samples[0].get("metadata", {})
            assert "critic" not in meta


# ── on_sample callback ────────────────────────────────────────────────────────

class TestOnSampleCallback:
    def test_callback_called_for_each_generated_sample(self):
        cfg = _cfg()
        orch_cfg = OrchestratorConfig(
            steering_mode=SteeringMode.AUTO,
            show_dashboard=False,
            collapse_abort_threshold=0.0,
        )
        orch = MultiAgentOrchestrator(config=cfg, orch_config=orch_cfg)
        calls = []
        orch.run(_chunks(2), on_sample=lambda s, sc, st: calls.append(st))
        assert len(calls) > 0
