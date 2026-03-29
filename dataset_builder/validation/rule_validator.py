"""
rule_validator.py — Deterministic rule-based validation layer.

Implements the annotation guidelines referenced by human annotators in a
real HITL workflow.  Every check is independent so the error analysis
module can attribute failures to specific rule codes.

Guideline summary (also printed in `annotation_guidelines()`)
------------------------------------------------------------
VALID sample:
  * Passes JSON schema (all required fields present, correct types)
  * confidence ∈ [0.6, 1.0]
  * Input text ≥ 30 characters
  * Output is a non-empty dict without parse-error keys
  * Task-specific output constraints satisfied (see below)

REJECT sample:
  * Missing required field(s)
  * confidence < 0.6
  * Output contains _parse_error key (LLM returned invalid JSON)
  * Output field value is empty string / empty list

FIX_REQUIRED:
  * Output is suspiciously short but not empty
  * QA evidence does not appear in input text  ← groundedness check
  * Reasoning has < 3 steps

Task-specific rules
-------------------
QA:
  - question must end with "?"
  - answer ≥ 5 chars
  - evidence must be a substring (≥ 50 % overlap) of input

Extraction:
  - entities list must have ≥ 1 item
  - each entity must have "text" and "type" string keys
  - key_facts list must have ≥ 1 item

Reasoning:
  - reasoning_steps must have ≥ 2 items (< 3 → FIX_REQUIRED)
  - conclusion ≥ 20 chars
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from schema.dataset_schema import validate_sample
from validation.annotation import (
    AnnotatedSample,
    AnnotationLabel,
    RejectionCode,
)

logger = logging.getLogger(__name__)

# Minimum word-overlap fraction for groundedness check
_GROUNDEDNESS_THRESHOLD = 0.25


class RuleValidator:
    """
    Apply deterministic rule checks to a collection of raw samples.

    Usage::

        validator = RuleValidator(min_confidence=0.6)
        annotated = validator.validate_batch(raw_samples)
    """

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def validate_batch(
        self, samples: List[Dict[str, Any]]
    ) -> List[AnnotatedSample]:
        """Validate every sample and return annotated results."""
        results = [self.validate_one(s) for s in samples]
        accepted = sum(1 for r in results if r.is_accepted)
        logger.info(
            "Rule validation: %d/%d accepted (%.1f%%)",
            accepted,
            len(results),
            100 * accepted / max(len(results), 1),
        )
        return results

    def validate_one(self, sample: Dict[str, Any]) -> AnnotatedSample:
        """Apply all rule checks to a single sample dict."""
        annotated = AnnotatedSample.from_sample_dict(sample)

        # 1. JSON schema validation
        is_valid, schema_errors = validate_sample(sample)
        if not is_valid:
            annotated.is_valid_schema = False
            for err in schema_errors:
                annotated.reject(RejectionCode.SCHEMA_INVALID, err)

        # Continue checking even if schema failed — gather all errors
        output = sample.get("output", {})
        metadata = sample.get("metadata", {})
        task_type = sample.get("task_type", "")

        # 2. Parse error check
        if isinstance(output, dict) and "_parse_error" in output:
            annotated.reject(
                RejectionCode.PARSE_ERROR,
                f"LLM returned non-JSON response: {output.get('_parse_error', '')[:100]}",
            )
            annotated.rule_checks_passed = False
            return annotated   # nothing else to check

        # 3. Confidence threshold
        confidence = float(metadata.get("confidence", 0))
        if confidence < self.min_confidence:
            annotated.reject(
                RejectionCode.LOW_CONFIDENCE,
                f"confidence={confidence:.2f} is below threshold={self.min_confidence}",
            )

        # 4. Input length
        input_text = sample.get("input", "")
        if len(input_text.strip()) < 30:
            annotated.reject(
                RejectionCode.OUTPUT_TOO_SHORT,
                "Input text is too short (< 30 chars) to generate meaningful samples.",
            )

        # 5. Output non-empty
        if not output:
            annotated.reject(
                RejectionCode.MISSING_FIELD,
                "output is empty or null",
            )
            annotated.rule_checks_passed = False
            return annotated

        # 6. Task-specific rules
        if task_type == "qa":
            self._check_qa(annotated, sample, output)
        elif task_type == "extraction":
            self._check_extraction(annotated, output)
        elif task_type == "reasoning":
            self._check_reasoning(annotated, output)

        if annotated.rejection_reasons:
            annotated.rule_checks_passed = False

        return annotated

    # ─────────────────────────────────────────────────────────────────────────
    # Task-specific checks
    # ─────────────────────────────────────────────────────────────────────────

    def _check_qa(
        self,
        annotated: AnnotatedSample,
        sample: Dict[str, Any],
        output: Dict[str, Any],
    ) -> None:
        question = output.get("question", "")
        answer = output.get("answer", "")
        evidence = output.get("evidence", "")

        if not question.strip():
            annotated.reject(RejectionCode.MISSING_FIELD, "QA: question is empty")
        elif not question.strip().endswith("?"):
            annotated.flag_for_fix(
                RejectionCode.WRONG_FORMAT, "QA: question does not end with '?'"
            )

        if not answer.strip():
            annotated.reject(RejectionCode.EMPTY_FIELD, "QA: answer is empty")
        elif len(answer.strip()) < 5:
            annotated.flag_for_fix(
                RejectionCode.OUTPUT_TOO_SHORT,
                f"QA: answer is too short ({len(answer.strip())} chars)",
            )

        if "evidence" not in output:
            annotated.reject(RejectionCode.MISSING_FIELD, "QA: evidence field is missing")
        elif not evidence.strip():
            annotated.flag_for_fix(RejectionCode.EMPTY_FIELD, "QA: evidence is empty")
        else:
            # Groundedness: check word overlap between evidence and input
            overlap = _word_overlap(evidence, sample.get("input", ""))
            if overlap < _GROUNDEDNESS_THRESHOLD:
                annotated.flag_for_fix(
                    RejectionCode.QA_ANSWER_NOT_GROUNDED,
                    f"QA: evidence words overlap with input is only {overlap:.0%} "
                    f"(threshold {_GROUNDEDNESS_THRESHOLD:.0%})",
                )

    @staticmethod
    def _check_extraction(
        annotated: AnnotatedSample, output: Dict[str, Any]
    ) -> None:
        entities = output.get("entities")
        key_facts = output.get("key_facts")

        if entities is None:
            annotated.reject(
                RejectionCode.MISSING_FIELD, "Extraction: entities field is missing"
            )
        elif not isinstance(entities, list) or len(entities) == 0:
            annotated.reject(
                RejectionCode.EXTRACTION_ENTITY_LIST_EMPTY,
                "Extraction: entities list is empty",
            )
        else:
            for i, ent in enumerate(entities):
                if not isinstance(ent, dict) or "text" not in ent or "type" not in ent:
                    annotated.reject(
                        RejectionCode.WRONG_FORMAT,
                        f"Extraction: entity[{i}] missing 'text' or 'type' key",
                    )
                    break

        if key_facts is None:
            annotated.reject(
                RejectionCode.MISSING_FIELD, "Extraction: key_facts field is missing"
            )
        elif not isinstance(key_facts, list) or len(key_facts) == 0:
            annotated.reject(
                RejectionCode.EMPTY_FIELD, "Extraction: key_facts list is empty"
            )

    @staticmethod
    def _check_reasoning(
        annotated: AnnotatedSample, output: Dict[str, Any]
    ) -> None:
        steps = output.get("reasoning_steps", [])
        conclusion = output.get("conclusion", "")

        if not isinstance(steps, list) or len(steps) == 0:
            annotated.reject(
                RejectionCode.MISSING_FIELD,
                "Reasoning: reasoning_steps is missing or empty",
            )
        elif len(steps) < 2:
            annotated.reject(
                RejectionCode.INSUFFICIENT_REASONING_STEPS,
                f"Reasoning: only {len(steps)} step(s) provided (minimum 2 required)",
            )
        elif len(steps) < 3:
            annotated.flag_for_fix(
                RejectionCode.INSUFFICIENT_REASONING_STEPS,
                f"Reasoning: only {len(steps)} step(s) provided; 3+ recommended",
            )

        if not conclusion.strip():
            annotated.reject(RejectionCode.EMPTY_FIELD, "Reasoning: conclusion is empty")
        elif len(conclusion.strip()) < 20:
            annotated.flag_for_fix(
                RejectionCode.OUTPUT_TOO_SHORT,
                f"Reasoning: conclusion is very short ({len(conclusion.strip())} chars)",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Annotation guidelines — printed on --show-guidelines
# ─────────────────────────────────────────────────────────────────────────────

def annotation_guidelines() -> str:
    return """
╔══════════════════════════════════════════════════════════════╗
║              ANNOTATION GUIDELINES (HITL Simulation)        ║
╠══════════════════════════════════════════════════════════════╣
║  ACCEPT  — All the following are true:                       ║
║    • Passes JSON schema validation                           ║
║    • confidence ≥ 0.6                                        ║
║    • Input text ≥ 30 characters                              ║
║    • Output dict is non-empty, no parse errors               ║
║    • Task-specific constraints met (see below)               ║
║                                                              ║
║  REJECT  — Any of the following are true:                    ║
║    • Missing or null required field                          ║
║    • confidence < 0.6                                        ║
║    • Output contains _parse_error                            ║
║    • Empty answer (QA) / empty entities (Extraction)         ║
║    • < 2 reasoning steps (Reasoning)                         ║
║                                                              ║
║  FIX_REQUIRED — Minor issues fixable by editor:              ║
║    • Question missing "?" suffix (QA)                        ║
║    • Answer < 5 chars but present (QA)                       ║
║    • Evidence-input overlap < 25 % (QA)                      ║
║    • Only 2 reasoning steps; 3+ preferred (Reasoning)        ║
║    • Conclusion < 20 chars (Reasoning)                       ║
║                                                              ║
║  GOOD SAMPLE: clear question, grounded answer, ≥ 100 chars  ║
║  BAD SAMPLE:  copied question, empty output, low confidence  ║
╚══════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _word_overlap(a: str, b: str) -> float:
    """Jaccard word-level overlap between two strings."""
    words_a = set(re.findall(r"\w+", a.lower()))
    words_b = set(re.findall(r"\w+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
