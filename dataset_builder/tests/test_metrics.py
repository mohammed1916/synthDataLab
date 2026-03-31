"""tests/test_metrics.py — Unit tests for evaluation metrics and collapse detection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from schema.dataset_schema import DatasetSample
from evaluation.metrics import compute_metrics, DatasetMetrics, _shannon_entropy
from collections import Counter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _qa_sample(i: int, answer: str = "The answer.", question: str = "What?") -> dict:
    return DatasetSample.create(
        "qa",
        f"Passage {i} discusses topic {i % 5} and covers area {i}.",
        "Answer the question.",
        {"question": question, "answer": answer, "evidence": f"Evidence {i}."},
        "test", 0.85, "mock", i,
    ).to_dict()


def _diverse_samples(n: int) -> list:
    topics = [
        "quantum entanglement enables faster-than-light correlation between particles",
        "machine learning models are trained using gradient descent optimisation",
        "climate change accelerates ice-sheet melting at Antarctic poles",
        "CRISPR gene editing can correct hereditary disease mutations",
        "transformer attention mechanism revolutionised natural language processing",
    ]
    return [_qa_sample(i, topics[i % len(topics)], f"Question about {topics[i % len(topics)][:30]}?") for i in range(n)]


def _repetitive_samples(n: int) -> list:
    return [_qa_sample(i, "Yes.", "What?") for i in range(n)]


# ── Basic metrics ─────────────────────────────────────────────────────────────

def test_empty_dataset_returns_zero_metrics():
    m = compute_metrics([])
    assert m.total_samples == 0
    assert m.schema_validity_rate == 0.0
    assert m.diversity_score == 0.0


def test_single_sample_metrics():
    samples = [_qa_sample(0)]
    m = compute_metrics(samples)
    assert m.total_samples == 1
    assert m.schema_validity_rate == 1.0


def test_perfect_schema_validity():
    samples = _diverse_samples(10)
    m = compute_metrics(samples)
    assert m.schema_validity_rate == 1.0


def test_completeness_score_for_full_outputs():
    samples = _diverse_samples(5)
    m = compute_metrics(samples)
    assert m.completeness_score > 0.9


# ── Entropy metrics ───────────────────────────────────────────────────────────

def test_shannon_entropy_empty_counter_is_zero():
    assert _shannon_entropy(Counter()) == 0.0


def test_shannon_entropy_uniform_is_max():
    c = Counter({"a": 5, "b": 5, "c": 5, "d": 5})
    h = _shannon_entropy(c)
    assert abs(h - 2.0) < 1e-9   # log2(4) == 2.0


def test_shannon_entropy_deterministic_is_zero():
    c = Counter({"a": 100})
    assert _shannon_entropy(c) == 0.0


def test_diverse_samples_have_higher_entropy_than_repetitive():
    m_diverse = compute_metrics(_diverse_samples(20))
    m_repetitive = compute_metrics(_repetitive_samples(20))
    assert m_diverse.vocabulary_entropy > m_repetitive.vocabulary_entropy


# ── Collapse risk ─────────────────────────────────────────────────────────────

def test_repetitive_samples_trigger_collapse_warning():
    m = compute_metrics(_repetitive_samples(30))
    # Low vocab = high collapse risk
    assert m.collapse_risk_score > 0.5


def test_collapse_warning_string_present_when_risk_high():
    m = compute_metrics(_repetitive_samples(40))
    assert m.collapse_warning is not None
    assert "Collapse" in m.collapse_warning


def test_diverse_samples_have_lower_risk_than_repetitive():
    # Diverse text has higher entropy → collapse risk is no worse than pure repetition.
    m_diverse = compute_metrics(_diverse_samples(30))
    m_repetitive = compute_metrics(_repetitive_samples(30))
    assert m_diverse.collapse_risk_score <= m_repetitive.collapse_risk_score


def test_collapse_risk_score_is_in_unit_interval():
    for samples in [_diverse_samples(10), _repetitive_samples(10)]:
        m = compute_metrics(samples)
        assert 0.0 <= m.collapse_risk_score <= 1.0


# ── Task distribution ─────────────────────────────────────────────────────────

def test_task_distribution_counts_correctly():
    samples = _diverse_samples(9)
    m = compute_metrics(samples)
    assert m.task_type_distribution.get("qa", 0) == 9
