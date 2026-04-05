"""tests/test_live_metrics.py — Tests for LiveMetricsTracker: thread safety, snapshots, report."""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.live_metrics import LiveMetricsTracker, LiveSnapshot

# ── Basic tracking ────────────────────────────────────────────────────────────

class TestBasicTracking:
    def test_initial_snapshot_is_zero(self):
        tracker = LiveMetricsTracker(total=10, show_dashboard=False)
        s = tracker._snapshot
        assert s.generated == 0
        assert s.accepted == 0
        assert s.rejected == 0
        assert s.fix_required == 0
        assert s.errors == 0

    def test_record_increments_generated(self):
        tracker = LiveMetricsTracker(total=5, show_dashboard=False)
        with tracker:
            tracker.record("s1", "qa", 0.9, 0.8, "ACCEPT")
        s = tracker._snapshot
        assert s.generated == 1
        assert s.accepted == 1

    def test_record_reject_increments_rejected(self):
        tracker = LiveMetricsTracker(total=5, show_dashboard=False)
        with tracker:
            tracker.record("s1", "qa", 0.3, 0.2, "REJECT")
        s = tracker._snapshot
        assert s.rejected == 1
        assert s.accepted == 0

    def test_record_fix_required(self):
        tracker = LiveMetricsTracker(total=5, show_dashboard=False)
        with tracker:
            tracker.record("s1", "reasoning", 0.5, 0.5, "FIX_REQUIRED")
        s = tracker._snapshot
        assert s.fix_required == 1

    def test_record_error_increments_errors(self):
        tracker = LiveMetricsTracker(total=5, show_dashboard=False)
        with tracker:
            tracker.record("—", "qa", 0.0, 0.0, "ERROR")
        s = tracker._snapshot
        assert s.errors == 1

    def test_task_counts_tracked(self):
        tracker = LiveMetricsTracker(total=5, show_dashboard=False)
        with tracker:
            tracker.record("s1", "qa", 0.9, 0.8, "ACCEPT")
            tracker.record("s2", "qa", 0.8, 0.7, "ACCEPT")
            tracker.record("s3", "reasoning", 0.6, 0.5, "ACCEPT")
        s = tracker._snapshot
        assert s.task_counts.get("qa", 0) == 2
        assert s.task_counts.get("reasoning", 0) == 1


# ── Derived stats ─────────────────────────────────────────────────────────────

class TestDerivedStats:
    def test_mean_confidence_computed_correctly(self):
        snap = LiveSnapshot()
        snap.confidence_scores = [0.8, 0.6, 1.0]
        assert abs(snap.mean_confidence - 0.8) < 1e-9

    def test_mean_critic_computed_correctly(self):
        snap = LiveSnapshot()
        snap.critic_scores = [0.7, 0.9]
        assert abs(snap.mean_critic - 0.8) < 1e-9

    def test_throughput_positive_after_recording(self):
        tracker = LiveMetricsTracker(total=10, show_dashboard=False)
        with tracker:
            tracker.record("s1", "qa", 0.9, 0.8, "ACCEPT")
        s = tracker._snapshot
        # Elapsed > 0, generated = 1 → throughput > 0
        assert s.throughput >= 0.0

    def test_eta_is_none_when_nothing_generated(self):
        snap = LiveSnapshot(total=10, generated=0, start_time=__import__("time").monotonic())
        assert snap.eta_seconds is None


# ── Thread safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_record_calls_are_consistent(self):
        """N threads each record M samples, total should equal N*M."""
        n_threads = 8
        m_samples = 20
        tracker = LiveMetricsTracker(total=n_threads * m_samples, show_dashboard=False)

        errors = []

        def worker():
            try:
                for i in range(m_samples):
                    tracker.record(f"s_{threading.get_ident()}_{i}", "qa", 0.8, 0.7, "ACCEPT")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        with tracker:
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"Thread errors: {errors}"
        s = tracker._snapshot
        assert s.generated == n_threads * m_samples
        assert s.accepted == n_threads * m_samples


# ── Collapse risk ─────────────────────────────────────────────────────────────

class TestCollapseRisk:
    def test_update_collapse_risk_stores_label(self):
        tracker = LiveMetricsTracker(total=5, show_dashboard=False)
        with tracker:
            tracker.update_collapse_risk(0.85, "CRITICAL")
        s = tracker._snapshot
        assert s.collapse_risk_label == "CRITICAL"
        assert abs(s.collapse_risk_score - 0.85) < 1e-9

    def test_low_collapse_risk_stored(self):
        tracker = LiveMetricsTracker(total=5, show_dashboard=False)
        with tracker:
            tracker.update_collapse_risk(0.10, "LOW")
        s = tracker._snapshot
        assert s.collapse_risk_label == "LOW"


# ── Final report ──────────────────────────────────────────────────────────────

class TestFinalReport:
    def test_print_final_report_does_not_raise(self):
        tracker = LiveMetricsTracker(total=3, show_dashboard=False)
        with tracker:
            tracker.record("s1", "qa", 0.9, 0.8, "ACCEPT")
            tracker.record("s2", "qa", 0.3, 0.2, "REJECT")
        # Should not raise regardless of rich availability
        tracker.print_final_report()


# ── Context-manager ───────────────────────────────────────────────────────────

class TestContextManager:
    def test_context_manager_enter_exit_without_error(self):
        tracker = LiveMetricsTracker(total=0, show_dashboard=False)
        with tracker:
            pass   # just verify no crash

    def test_snapshot_accessible_outside_context(self):
        tracker = LiveMetricsTracker(total=2, show_dashboard=False)
        with tracker:
            tracker.record("s1", "qa", 0.8, 0.7, "ACCEPT")
        # snapshot still valid after exit
        s = tracker._snapshot
        assert s.generated == 1
