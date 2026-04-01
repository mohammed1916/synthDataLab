"""
critic_agent.py — Heuristic quality-critic agent for multi-agent generation.

The CriticAgent scores each generated DatasetSample on four orthogonal axes
*without* issuing any additional LLM calls, making it fast and suitable for
real-time inline use:

  1. Relevance    — output is semantically related to the input text
                    (word-overlap Jaccard coefficient, 0–1)
  2. Coherence    — output is internally well-formed: all required keys
                    present, values are non-trivial strings
                    (structural completeness check, 0–1)
  3. Groundedness — factual claims appear to be anchored in the source text
                    (evidence / reasoning steps overlap, 0–1)
  4. Fluency      — output is fluent: no raw JSON, no error markers, length
                    is within acceptable bounds
                    (surface-form heuristics, 0–1)

The composite ``critic_score`` is the arithmetic mean of the four axes.
Samples are then bucketed:

  ≥ 0.70  → PASS      (auto-accept in AUTO steering mode)
  ≥ 0.45  → REVIEW    (surface for human steering if REVIEW_LOW mode)
  < 0.45  → FAIL      (auto-reject in AUTO steering mode)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Required output keys per task type (mirrors evaluation/metrics.py)
_REQUIRED_KEYS: Dict[str, List[str]] = {
    "qa": ["question", "answer", "evidence"],
    "extraction": ["entities", "relations", "key_facts"],
    "reasoning": ["reasoning_steps", "conclusion", "confidence_explanation"],
    "reasoning_trace": ["think", "answer", "verification", "confidence"],
    "preference": ["prompt", "chosen", "rejected", "preference_margin"],
}

# Critic verdict thresholds
PASS_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.45

# Fluency: these strings in an output signal a broken generation
_ERROR_MARKERS = (
    "_raw_response",
    "_parse_error",
    "JSONDecodeError",
    "Traceback",
    "<error>",
)


# ─────────────────────────────────────────────────────────────────────────────
# Score container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CriticScore:
    """Axis-level scores and composite verdict for one sample."""

    relevance: float = 0.0       # output↔input overlap
    coherence: float = 0.0       # structural completeness
    groundedness: float = 0.0    # claims anchored in source
    fluency: float = 0.0         # surface-form quality

    @property
    def composite(self) -> float:
        return (self.relevance + self.coherence + self.groundedness + self.fluency) / 4.0

    @property
    def verdict(self) -> str:
        c = self.composite
        if c >= PASS_THRESHOLD:
            return "PASS"
        if c >= REVIEW_THRESHOLD:
            return "REVIEW"
        return "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["composite"] = round(self.composite, 4)
        d["verdict"] = self.verdict
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Critic agent
# ─────────────────────────────────────────────────────────────────────────────

class CriticAgent:
    """
    Heuristic quality-critic that scores ``DatasetSample`` objects.

    Args:
        pass_threshold:   Composite score at or above which a sample is PASS.
        review_threshold: Composite score at or above which a sample is REVIEW
                          (below pass_threshold).  Anything lower is FAIL.
        min_output_chars: Minimum character count for output fields.
        max_output_chars: Maximum character count for a single output field.
    """

    def __init__(
        self,
        pass_threshold: float = PASS_THRESHOLD,
        review_threshold: float = REVIEW_THRESHOLD,
        min_output_chars: int = 15,
        max_output_chars: int = 10_000,
    ):
        self.pass_threshold = pass_threshold
        self.review_threshold = review_threshold
        self.min_output_chars = min_output_chars
        self.max_output_chars = max_output_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, sample: Dict[str, Any]) -> CriticScore:
        """
        Score a single sample dict and return a ``CriticScore``.

        Args:
            sample: A DatasetSample-compatible dict (from ``sample.to_dict()``).

        Returns:
            ``CriticScore`` with per-axis scores, composite, and verdict.
        """
        task_type: str = sample.get("task_type", "")
        input_text: str = str(sample.get("input", ""))
        output: Any = sample.get("output", {})

        # Ensure output is always a dict (graceful handling of bad parses)
        if not isinstance(output, dict):
            output = {}

        return CriticScore(
            relevance=self._score_relevance(input_text, output),
            coherence=self._score_coherence(task_type, output),
            groundedness=self._score_groundedness(task_type, input_text, output),
            fluency=self._score_fluency(output),
        )

    def score_batch(self, samples: List[Dict[str, Any]]) -> List[CriticScore]:
        """Score a list of samples, returning one CriticScore per sample."""
        return [self.score(s) for s in samples]

    def score_with_llm(
        self,
        sample: Dict[str, Any],
        llm_client: Any,
    ) -> CriticScore:
        """
        LLM-as-Judge scoring using a G-Eval style JSON prompt.

        Sends the sample to the LLM and asks it to return a JSON object with
        four float scores (0–1) for relevance, coherence, groundedness, fluency.
        Falls back to heuristic ``score()`` on any parse failure.

        Args:
            sample: A DatasetSample-compatible dict.
            llm_client: Any client with a ``generate(prompt: str) -> str`` method.

        Returns:
            ``CriticScore`` from LLM judgment, or heuristic fallback.
        """
        import json as _json

        task_type = sample.get("task_type", "unknown")
        input_snippet = str(sample.get("input", ""))[:400]
        output_snippet = _json.dumps(sample.get("output", {}), ensure_ascii=False)[:600]

        prompt = (
            "You are a strict quality judge for a synthetic NLP dataset.\n"
            "Score the following sample on four axes, returning ONLY a JSON object.\n\n"
            f"Task type: {task_type}\n"
            f"Input: {input_snippet}\n"
            f"Output: {output_snippet}\n\n"
            "Return exactly this JSON (no markdown, no explanation):\n"
            '{"relevance": 0.0, "coherence": 0.0, "groundedness": 0.0, "fluency": 0.0}\n\n'
            "Scores must be floats in [0.0, 1.0] where 1.0 = excellent."
        )

        try:
            raw = llm_client.generate(prompt)
            # Strip markdown fences if present
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
            # Extract first JSON object
            match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object in LLM response")
            scores = _json.loads(match.group())

            def _clamp(v: Any) -> float:
                return max(0.0, min(1.0, float(v)))

            return CriticScore(
                relevance=_clamp(scores.get("relevance", 0.0)),
                coherence=_clamp(scores.get("coherence", 0.0)),
                groundedness=_clamp(scores.get("groundedness", 0.0)),
                fluency=_clamp(scores.get("fluency", 0.0)),
            )
        except Exception as exc:
            logger.debug("LLM-as-Judge parse failed (%s) — falling back to heuristic scoring.", exc)
            return self.score(sample)



    # ── Axis scorers ──────────────────────────────────────────────────────────

    def _score_relevance(self, input_text: str, output: Dict[str, Any]) -> float:
        """
        Jaccard word-overlap between *all* output string values and the input.
        A score of 0 means the output shares nothing with the source passage.
        """
        input_words = _word_set(input_text)
        if not input_words:
            return 0.5   # nothing to compare — be neutral

        output_text = _flatten_output(output)
        output_words = _word_set(output_text)
        if not output_words:
            return 0.0

        intersection = len(input_words & output_words)
        union = len(input_words | output_words)
        raw_jaccard = intersection / union if union else 0.0

        # Jaccard tends to be low for genuine summaries; rescale [0, 0.5] → [0, 1]
        return min(1.0, raw_jaccard * 2.5)

    def _score_coherence(self, task_type: str, output: Dict[str, Any]) -> float:
        """
        Fraction of required output keys that are present and non-trivial.
        Bonus for extra non-empty optional keys.
        """
        required = _REQUIRED_KEYS.get(task_type, [])
        if not required:
            # Unknown task type — score based on whether output is non-empty
            return 0.5 if output else 0.0

        filled = sum(1 for k in required if _is_non_trivial(output.get(k)))
        base_score = filled / len(required)

        # Small bonus for additional populated fields (capped at 0.15)
        extra = sum(
            1 for k, v in output.items()
            if k not in required and _is_non_trivial(v)
        )
        bonus = min(0.15, extra * 0.05)
        return min(1.0, base_score + bonus)

    def _score_groundedness(
        self, task_type: str, input_text: str, output: Dict[str, Any]
    ) -> float:
        """
        Check whether the key evidence / reasoning field shares substantial
        content with the input, indicating claims are source-grounded.
        """
        input_words = _word_set(input_text)
        if not input_words:
            return 0.5

        # Extract the "evidence field" per task type
        evidence_keys = {
            "qa": ["evidence"],
            "extraction": ["key_facts"],
            "reasoning": ["reasoning_steps"],
            "reasoning_trace": ["think", "verification"],
            "preference": ["chosen"],
        }
        keys = evidence_keys.get(task_type, list(output.keys()))
        evidence_parts: List[str] = []
        for k in keys:
            val = output.get(k, "")
            if isinstance(val, list):
                evidence_parts.extend(str(v) for v in val)
            elif isinstance(val, str):
                evidence_parts.append(val)

        evidence_text = " ".join(evidence_parts)
        ev_words = _word_set(evidence_text)
        if not ev_words:
            return 0.2   # no evidence text at all → low groundedness

        overlap = len(input_words & ev_words) / len(ev_words)
        return min(1.0, overlap * 2.0)   # rescale: 50 % overlap → score 1.0

    def _score_fluency(self, output: Dict[str, Any]) -> float:
        """
        Surface-form quality check:
          - Penalise outputs with error markers (_raw_response, _parse_error …)
          - Penalise outputs that are empty or contain only whitespace
          - Penalise outputs where individual fields are too short or too long
        """
        flat = _flatten_output(output)

        # Hard fail for error markers
        for marker in _ERROR_MARKERS:
            if marker in flat:
                return 0.0

        if not flat.strip():
            return 0.0

        # Check individual field lengths
        field_scores: List[float] = []
        for v in output.values():
            text = _stringify(v)
            if not text:
                field_scores.append(0.2)
            elif len(text) < self.min_output_chars:
                field_scores.append(0.5)
            elif len(text) > self.max_output_chars:
                field_scores.append(0.6)    # long but present — mild penalty
            else:
                field_scores.append(1.0)

        if not field_scores:
            return 0.3

        return sum(field_scores) / len(field_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _word_set(text: str) -> set:
    """Lowercase word set, stripping punctuation, min length 3."""
    return {
        w.lower()
        for w in re.findall(r"\b[a-zA-Z]{3,}\b", text)
    }


def _flatten_output(output: Dict[str, Any]) -> str:
    """Join all string values (recursively) in an output dict."""
    parts: List[str] = []
    for v in output.values():
        parts.append(_stringify(v))
    return " ".join(parts)


def _stringify(value: Any) -> str:
    """Convert a value to a flat string regardless of type."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(i) for i in value)
    if isinstance(value, dict):
        return " ".join(str(i) for i in value.values())
    return str(value) if value is not None else ""


def _is_non_trivial(value: Any) -> bool:
    """Return True if the value is meaningfully populated."""
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) >= 3
    if isinstance(value, (list, dict)):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return True
    return bool(value)
