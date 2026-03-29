"""
annotation.py — Annotation labels and the AnnotatedSample wrapper.

The annotation schema mirrors human-in-the-loop (HITL) workflows used in
large-scale data labelling pipelines.  Every sample that passes through the
validation layer receives an ``AnnotatedSample`` record with:

* A final **label** (ACCEPT / REJECT / FIX_REQUIRED)
* A list of **rejection reasons** (structured codes + human-readable messages)
* Per-check boolean flags for the rule and schema validators
* Optional reviewer notes from the LLM second-pass reviewer
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class AnnotationLabel(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    FIX_REQUIRED = "FIX_REQUIRED"


# ─────────────────────────────────────────────────────────────────────────────
# Rejection reason codes (machine-readable)
# ─────────────────────────────────────────────────────────────────────────────

class RejectionCode(str, Enum):
    SCHEMA_INVALID = "SCHEMA_INVALID"
    MISSING_FIELD = "MISSING_FIELD"
    EMPTY_FIELD = "EMPTY_FIELD"
    WRONG_FORMAT = "WRONG_FORMAT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    OUTPUT_TOO_SHORT = "OUTPUT_TOO_SHORT"
    OUTPUT_TOO_LONG = "OUTPUT_TOO_LONG"
    TASK_TYPE_MISMATCH = "TASK_TYPE_MISMATCH"
    INSUFFICIENT_REASONING_STEPS = "INSUFFICIENT_REASONING_STEPS"
    QA_ANSWER_NOT_GROUNDED = "QA_ANSWER_NOT_GROUNDED"
    EXTRACTION_ENTITY_LIST_EMPTY = "EXTRACTION_ENTITY_LIST_EMPTY"
    LLM_REVIEWER_REJECTED = "LLM_REVIEWER_REJECTED"
    PARSE_ERROR = "PARSE_ERROR"


@dataclass
class RejectionReason:
    code: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.message}


# ─────────────────────────────────────────────────────────────────────────────
# AnnotatedSample
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnnotatedSample:
    """
    Wraps a raw DatasetSample dict together with annotation metadata.

    Consumers should read the ``label`` field first; ``rejection_reasons``
    explains why a sample was not accepted.
    """

    sample: Dict[str, Any]                             # original sample dict
    label: AnnotationLabel = AnnotationLabel.ACCEPT
    rejection_reasons: List[RejectionReason] = field(default_factory=list)

    # Per-check flags
    is_valid_schema: bool = True
    rule_checks_passed: bool = True
    llm_review_passed: Optional[bool] = None           # None = not run

    reviewer_notes: str = ""

    # ── Convenience ───────────────────────────────────────────────────────────

    def reject(self, code: str, message: str) -> None:
        """Add a rejection reason and downgrade label to REJECT."""
        self.rejection_reasons.append(RejectionReason(code=code, message=message))
        self.label = AnnotationLabel.REJECT

    def flag_for_fix(self, code: str, message: str) -> None:
        """Add a reason and downgrade to FIX_REQUIRED (if not already REJECT)."""
        self.rejection_reasons.append(RejectionReason(code=code, message=message))
        if self.label == AnnotationLabel.ACCEPT:
            self.label = AnnotationLabel.FIX_REQUIRED

    @property
    def is_accepted(self) -> bool:
        return self.label == AnnotationLabel.ACCEPT

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.sample,
            "annotation": {
                "label": self.label.value,
                "rejection_reasons": [r.to_dict() for r in self.rejection_reasons],
                "is_valid_schema": self.is_valid_schema,
                "rule_checks_passed": self.rule_checks_passed,
                "llm_review_passed": self.llm_review_passed,
                "reviewer_notes": self.reviewer_notes,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_sample_dict(cls, sample: Dict[str, Any]) -> "AnnotatedSample":
        """Create a fresh AnnotatedSample from a raw sample dict."""
        return cls(sample=sample)
