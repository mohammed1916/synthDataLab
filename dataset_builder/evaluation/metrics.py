"""n
metrics.py — Quantitative dataset quality metrics.

Metrics computed
----------------
1.  Schema Validity Rate       — fraction of samples passing JSON schema
2.  Task Consistency Score     — fraction with task-type-consistent output keys
3.  Completeness Score         — mean fraction of required fields present
4.  Hallucination Rate         — heuristic: QA answers with < 25 % word overlap
                                  with their input text
5.  Diversity Score            — type-token ratio of output vocabulary (lexical diversity)
6.  Mean Confidence            — mean model confidence across all samples

Collapse Early-Warning Metrics (new)
--------------------------------------
7.  Vocabulary Entropy         — Shannon entropy of token frequency distribution
                                  (lower = distribution collapse)
8.  Bigram Entropy             — Shannon entropy of bigram frequency distribution
9.  Collapse Risk Score        — composite 0-1 score; ≥ 0.7 triggers RED alert
10. Collapse Warning           — human-readable warning string or None

Metrics are computed on both the raw and filtered datasets so the report
can show before-vs-after improvement.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from schema.dataset_schema import validate_sample

# Required output keys per task type
_REQUIRED_OUTPUT_KEYS: dict[str, list[str]] = {
    "qa": ["question", "answer", "evidence"],
    "extraction": ["entities", "relations", "key_facts"],
    "reasoning": ["reasoning_steps", "conclusion", "confidence_explanation"],
    "reasoning_trace": ["think", "answer", "verification", "confidence"],
    "preference": ["prompt", "chosen", "rejected", "preference_margin"],
}

_HALLUCINATION_OVERLAP_THRESHOLD = 0.20

# Collapse alert thresholds
_COLLAPSE_ENTROPY_FLOOR = 6.0      # bits — below this vocabulary is suspiciously narrow
_COLLAPSE_TTR_FLOOR = 0.35         # type-token ratio — below this lexical diversity is critical
_COLLAPSE_RISK_WARN = 0.50         # score >= this → WARNING
_COLLAPSE_RISK_CRITICAL = 0.70     # score >= this → CRITICAL (halt recommended)


@dataclass
class DatasetMetrics:
    """Container for all computed quality metrics."""

    # Counts
    total_samples: int = 0
    task_type_distribution: dict[str, int] = None  # type: ignore

    # Quality rates  (all in [0, 1])
    schema_validity_rate: float = 0.0
    task_consistency_score: float = 0.0
    completeness_score: float = 0.0
    hallucination_rate: float = 0.0   # lower is better
    diversity_score: float = 0.0

    # Confidence stats
    mean_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0

    # ── Collapse early-warning metrics ──────────────────────────────────────
    vocabulary_entropy: float = 0.0    # Shannon entropy bits; higher = more diverse
    bigram_entropy: float = 0.0        # Bigram entropy bits
    collapse_risk_score: float = 0.0   # Composite 0-1; >= 0.7 = CRITICAL
    collapse_warning: str | None = None   # Human-readable warning or None

    def __post_init__(self):
        if self.task_type_distribution is None:
            self.task_type_distribution = {}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Round floats for readability
        for key in list(d.keys()):
            if isinstance(d[key], float):
                d[key] = round(d[key], 4)
        return d


def compute_metrics(samples: list[dict[str, Any]]) -> DatasetMetrics:
    """
    Compute all quality metrics for a list of sample dicts.

    Args:
        samples: List of DatasetSample-compatible dicts.

    Returns:
        Populated DatasetMetrics dataclass.
    """
    if not samples:
        return DatasetMetrics()

    m = DatasetMetrics(total_samples=len(samples))

    # Task type distribution
    dist: dict[str, int] = {}
    for s in samples:
        tt = s.get("task_type", "unknown")
        dist[tt] = dist.get(tt, 0) + 1
    m.task_type_distribution = dist

    # Schema validity
    schema_valid = sum(1 for s in samples if validate_sample(s)[0])
    m.schema_validity_rate = schema_valid / len(samples)

    # Task consistency
    consistent = sum(1 for s in samples if _is_task_consistent(s))
    m.task_consistency_score = consistent / len(samples)

    # Completeness
    completeness_scores = [_completeness(s) for s in samples]
    m.completeness_score = sum(completeness_scores) / len(completeness_scores)

    # Hallucination rate (QA only)
    qa_samples = [s for s in samples if s.get("task_type") == "qa"]
    if qa_samples:
        hallucinated = sum(1 for s in qa_samples if _is_hallucinated(s))
        m.hallucination_rate = hallucinated / len(qa_samples)
    else:
        m.hallucination_rate = 0.0

    # Diversity score
    m.diversity_score = _diversity_score(samples)

    # Confidence stats
    confidences = [
        float(s.get("metadata", {}).get("confidence", 0)) for s in samples
    ]
    if confidences:
        m.mean_confidence = sum(confidences) / len(confidences)
        m.min_confidence = min(confidences)
        m.max_confidence = max(confidences)

    # ── Collapse early-warning metrics ──────────────────────────────────────
    all_tokens = _collect_tokens(samples)
    m.vocabulary_entropy = _shannon_entropy(Counter(all_tokens))
    bigrams = _collect_bigrams(all_tokens)
    m.bigram_entropy = _shannon_entropy(Counter(bigrams))
    m.collapse_risk_score, m.collapse_warning = _collapse_risk(
        m.diversity_score, m.vocabulary_entropy, m.hallucination_rate
    )

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Per-metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_task_consistent(sample: dict[str, Any]) -> bool:
    """True if the output contains the expected keys for the task type."""
    task_type = sample.get("task_type", "")
    output = sample.get("output", {})
    required = _REQUIRED_OUTPUT_KEYS.get(task_type, [])
    if not required or not isinstance(output, dict):
        return False
    return all(k in output and output[k] not in ("", None, [], {}) for k in required)


def _completeness(sample: dict[str, Any]) -> float:
    """Fraction of required fields present and non-empty in the output."""
    task_type = sample.get("task_type", "")
    output = sample.get("output", {})
    required = _REQUIRED_OUTPUT_KEYS.get(task_type, [])
    if not required or not isinstance(output, dict):
        return 0.0
    present = sum(
        1 for k in required if output.get(k) not in (None, "", [], {})
    )
    return present / len(required)


def _is_hallucinated(sample: dict[str, Any]) -> bool:
    """
    Heuristic: mark a QA sample as potentially hallucinated if the answer
    words have < 20 % overlap with the input text.
    """
    input_text = sample.get("input", "")
    output = sample.get("output", {})
    if not isinstance(output, dict):
        return False
    answer = str(output.get("answer", ""))
    if not answer:
        return False
    words_answer = _words(answer)
    words_input = _words(input_text)
    if not words_answer:
        return True
    overlap = len(words_answer & words_input) / len(words_answer)
    return overlap < _HALLUCINATION_OVERLAP_THRESHOLD


def _diversity_score(samples: list[dict[str, Any]]) -> float:
    """
    Lexical diversity (type-token ratio) of output text across all samples.

    TTR = unique_tokens / total_tokens — ranges [0, 1]; higher is more diverse.
    """
    all_tokens: list[str] = []
    for s in samples:
        output_str = str(s.get("output", ""))
        all_tokens.extend(re.findall(r"\b\w{3,}\b", output_str.lower()))

    if not all_tokens:
        return 0.0
    unique = len(set(all_tokens))
    return unique / len(all_tokens)


def _words(text: str) -> set[str]:
    return set(re.findall(r"\b\w{2,}\b", text.lower()))


# ─────────────────────────────────────────────────────────────────────────────
# Collapse early-warning helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_tokens(samples: list[dict[str, Any]]) -> list[str]:
    """Collect all lowercase word tokens from all sample outputs."""
    tokens: list[str] = []
    for s in samples:
        text = str(s.get("output", ""))
        tokens.extend(re.findall(r"\b\w{3,}\b", text.lower()))
    return tokens


def _collect_bigrams(tokens: list[str]) -> list[str]:
    """Return consecutive bigrams as 'w1_w2' strings."""
    if len(tokens) < 2:
        return []
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def _shannon_entropy(counter: Counter) -> float:
    """
    Compute Shannon entropy (bits) for a frequency distribution.
    H = -Σ p(x) log2 p(x)
    Returns 0.0 for empty distributions.
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counter.values()
        if c > 0
    )


def _collapse_risk(
    diversity_score: float,
    vocab_entropy: float,
    hallucination_rate: float,
) -> tuple:
    """
    Compute a composite collapse risk score ∈ [0, 1].

    Weights:
      40% — vocabulary entropy (normalised; floor at 6 bits, full at 14 bits)
      40% — diversity score (TTR; below 0.35 = danger zone)
      20% — hallucination rate (higher hallucination = higher collapse risk)

    Returns (risk_score, warning_string | None).
    """
    # Entropy component: 0 when at or below floor, 1 when fully healthy (≥ 14 bits)
    max_entropy = 14.0
    entropy_health = min(max(vocab_entropy - _COLLAPSE_ENTROPY_FLOOR, 0) /
                         (max_entropy - _COLLAPSE_ENTROPY_FLOOR), 1.0)
    entropy_risk = 1.0 - entropy_health

    # Diversity component: 0 when TTR ≥ 0.65 (healthy), 1 when TTR ≤ 0.35 (critical)
    diversity_risk = max(0.0, min(1.0, (0.65 - diversity_score) / (0.65 - _COLLAPSE_TTR_FLOOR)))

    # Hallucination component: direct proportion (rate already ∈ [0, 1])
    hallucination_risk = min(hallucination_rate * 2.0, 1.0)  # doubled weight

    risk = 0.40 * entropy_risk + 0.40 * diversity_risk + 0.20 * hallucination_risk
    risk = round(risk, 4)

    warning: str | None = None
    if risk >= _COLLAPSE_RISK_CRITICAL:
        warning = (
            f"CRITICAL — Collapse risk {risk:.2f}. "
            f"Vocab entropy={vocab_entropy:.2f} bits, TTR={diversity_score:.3f}. "
            "HALT generation and diversify input sources immediately."
        )
    elif risk >= _COLLAPSE_RISK_WARN:
        warning = (
            f"WARNING — Collapse risk {risk:.2f}. "
            f"Vocab entropy={vocab_entropy:.2f} bits, TTR={diversity_score:.3f}. "
            "Increase generation temperature and broaden input topics."
        )

    return risk, warning
