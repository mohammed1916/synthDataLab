"""
metrics.py — Quantitative dataset quality metrics.

Metrics computed
----------------
1. Schema Validity Rate       — fraction of samples passing JSON schema
2. Task Consistency Score     — fraction with task-type-consistent output keys
3. Completeness Score         — mean fraction of required fields present
4. Hallucination Rate         — heuristic: QA answers with < 25 % word overlap
                                 with their input text
5. Diversity Score            — type-token ratio of output vocabulary (lexical diversity)

Metrics are computed on both the raw and filtered datasets so the report
can show before-vs-after improvement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Set

from schema.dataset_schema import validate_sample

# Required output keys per task type
_REQUIRED_OUTPUT_KEYS: Dict[str, List[str]] = {
    "qa": ["question", "answer", "evidence"],
    "extraction": ["entities", "relations", "key_facts"],
    "reasoning": ["reasoning_steps", "conclusion", "confidence_explanation"],
}

_HALLUCINATION_OVERLAP_THRESHOLD = 0.20


@dataclass
class DatasetMetrics:
    """Container for all computed quality metrics."""

    # Counts
    total_samples: int = 0
    task_type_distribution: Dict[str, int] = None  # type: ignore

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

    def __post_init__(self):
        if self.task_type_distribution is None:
            self.task_type_distribution = {}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Round floats for readability
        for key in list(d.keys()):
            if isinstance(d[key], float):
                d[key] = round(d[key], 4)
        return d


def compute_metrics(samples: List[Dict[str, Any]]) -> DatasetMetrics:
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
    dist: Dict[str, int] = {}
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

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Per-metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_task_consistent(sample: Dict[str, Any]) -> bool:
    """True if the output contains the expected keys for the task type."""
    task_type = sample.get("task_type", "")
    output = sample.get("output", {})
    required = _REQUIRED_OUTPUT_KEYS.get(task_type, [])
    if not required or not isinstance(output, dict):
        return False
    return all(k in output and output[k] not in ("", None, [], {}) for k in required)


def _completeness(sample: Dict[str, Any]) -> float:
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


def _is_hallucinated(sample: Dict[str, Any]) -> bool:
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


def _diversity_score(samples: List[Dict[str, Any]]) -> float:
    """
    Lexical diversity (type-token ratio) of output text across all samples.

    TTR = unique_tokens / total_tokens — ranges [0, 1]; higher is more diverse.
    """
    all_tokens: List[str] = []
    for s in samples:
        output_str = str(s.get("output", ""))
        all_tokens.extend(re.findall(r"\b\w{3,}\b", output_str.lower()))

    if not all_tokens:
        return 0.0
    unique = len(set(all_tokens))
    return unique / len(all_tokens)


def _words(text: str) -> Set[str]:
    return set(re.findall(r"\b\w{2,}\b", text.lower()))
