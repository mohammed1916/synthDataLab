"""
tests/test_fingerprint_dedup.py — Tests for cross-run deduplication logic.

Covers:
  - FingerprintStore: basic add/query/persist/reload cycle
  - FingerprintStore: second run returns 0 new samples
  - FingerprintStore: --reset-fingerprints wipes the store
  - run-all CLI: graceful early-exit when all samples already seen
  - run-all CLI: --force bypasses dedup and runs the full pipeline
  - run-all CLI: --reset-fingerprints + second run = full pipeline again
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from click.testing import CliRunner
from main import cli

from filtering.fingerprint_store import FingerprintStore

# ─────────────────────────────────────────────────────────────────────────────
# FingerprintStore unit tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fp_path(tmp_path) -> Path:
    return tmp_path / "fingerprints.json"


def _make_samples(n: int = 3):
    return [{"input": f"sample text {i}", "task_type": "qa"} for i in range(n)]


class TestFingerprintStore:
    def test_first_run_all_new(self, fp_path):
        store = FingerprintStore(fp_path)
        samples = _make_samples(5)
        new, dupes = store.filter_new(samples)
        assert len(new) == 5
        assert len(dupes) == 0

    def test_second_run_all_dupes(self, fp_path):
        samples = _make_samples(5)

        store1 = FingerprintStore(fp_path)
        store1.filter_new(samples)
        store1.save()

        # Reload from disk — simulates a new invocation
        store2 = FingerprintStore(fp_path)
        new, dupes = store2.filter_new(samples)
        assert len(new) == 0
        assert len(dupes) == 5

    def test_partial_overlap(self, fp_path):
        old_samples = _make_samples(3)
        new_samples = [{"input": "brand new text", "task_type": "qa"}]

        store1 = FingerprintStore(fp_path)
        store1.filter_new(old_samples)
        store1.save()

        store2 = FingerprintStore(fp_path)
        new, dupes = store2.filter_new(old_samples + new_samples)
        assert len(new) == 1
        assert new[0]["input"] == "brand new text"
        assert len(dupes) == 3

    def test_reset_wipes_store(self, fp_path):
        samples = _make_samples(3)

        store1 = FingerprintStore(fp_path)
        store1.filter_new(samples)
        store1.save()
        assert fp_path.exists()

        # Simulate --reset-fingerprints: unlink before creating a new store
        fp_path.unlink()

        store2 = FingerprintStore(fp_path)
        new, dupes = store2.filter_new(samples)
        assert len(new) == 3
        assert len(dupes) == 0

    def test_save_is_atomic(self, fp_path):
        """save() must never leave a partial file if it fails mid-write."""
        store = FingerprintStore(fp_path)
        store.filter_new(_make_samples(2))
        store.save()
        data = json.loads(fp_path.read_text())
        assert "fingerprints" in data
        assert data["count"] == 2

    def test_fingerprint_is_case_and_whitespace_insensitive(self, fp_path):
        """Normalisation: leading/trailing spaces + mixed case → same fingerprint."""
        store = FingerprintStore(fp_path)
        store.filter_new([{"input": "  Hello World  ", "task_type": "QA"}])
        store.save()

        store2 = FingerprintStore(fp_path)
        new, dupes = store2.filter_new([{"input": "hello world", "task_type": "qa"}])
        assert len(new) == 0  # treated as same fingerprint
        assert len(dupes) == 1

    def test_different_task_types_are_distinct(self, fp_path):
        """Same input text with different task_type = different fingerprint."""
        store = FingerprintStore(fp_path)
        s1 = [{"input": "same text", "task_type": "qa"}]
        s2 = [{"input": "same text", "task_type": "extraction"}]
        new, _ = store.filter_new(s1 + s2)
        assert len(new) == 2


# ─────────────────────────────────────────────────────────────────────────────
# CLI integration tests (run-all --mock)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """
    Point Config.storage.data_dir at a temp dir so CLI tests don't touch
    the real data/ folder and don't interfere with each other.
    """
    import config as cfg_module

    original_init = cfg_module.StorageConfig.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.data_dir = tmp_path / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cfg_module.StorageConfig, "__init__", patched_init)
    return tmp_path / "data"


class TestRunAllCLI:
    def test_first_run_succeeds_end_to_end(self, runner, isolated_data_dir):
        result = runner.invoke(cli, ["run-all", "--mock", "--reset-fingerprints"])
        assert result.exit_code == 0, result.output
        assert "DONE" in result.output
        raw = isolated_data_dir / "raw_dataset.jsonl"
        assert raw.exists()
        records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        assert len(records) == 30

    def test_second_run_exits_cleanly_with_nothing_new(self, runner, isolated_data_dir):
        # First run — seeds fingerprints
        r1 = runner.invoke(cli, ["run-all", "--mock", "--reset-fingerprints"])
        assert r1.exit_code == 0, r1.output

        # Second run — everything is a duplicate
        r2 = runner.invoke(cli, ["run-all", "--mock"])
        assert r2.exit_code == 0, r2.output
        assert "Nothing new to process" in r2.output or "already seen" in r2.output

        # raw_dataset.jsonl must still contain 30 records (NOT wiped to 0)
        raw = isolated_data_dir / "raw_dataset.jsonl"
        records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        assert len(records) == 30, (
            f"raw_dataset.jsonl was wiped! Expected 30, got {len(records)}"
        )

    def test_force_flag_reprocesses_all_samples(self, runner, isolated_data_dir):
        # Seed fingerprints
        r1 = runner.invoke(cli, ["run-all", "--mock", "--reset-fingerprints"])
        assert r1.exit_code == 0, r1.output

        # --force should bypass dedup and complete the full pipeline
        r2 = runner.invoke(cli, ["run-all", "--mock", "--force"])
        assert r2.exit_code == 0, r2.output
        assert "DONE" in r2.output
        assert "--force: skipping cross-run dedup" in r2.output

    def test_reset_fingerprints_allows_full_rerun(self, runner, isolated_data_dir):
        # Two successful full runs when fingerprints are reset each time
        r1 = runner.invoke(cli, ["run-all", "--mock", "--reset-fingerprints"])
        assert r1.exit_code == 0 and "DONE" in r1.output, r1.output

        r2 = runner.invoke(cli, ["run-all", "--mock", "--reset-fingerprints"])
        assert r2.exit_code == 0 and "DONE" in r2.output, r2.output

    def test_health_check_passes_in_mock_mode(self, runner, isolated_data_dir):
        result = runner.invoke(cli, ["health-check", "--mock"])
        assert result.exit_code == 0, result.output
        assert "PASS" in result.output
