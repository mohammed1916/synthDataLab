"""
llm_reviewer.py — Second-pass LLM-based sample reviewer.

In production HITL pipelines a second model (or a larger, more capable model)
reviews borderline samples.  Here we simulate that with:

* Real mode  — calls the same LLM with a "critique" prompt.
* Mock mode  — deterministic rule-based heuristics that mimic a reviewer.

The reviewer upgrades REJECT → FIX_REQUIRED or ACCEPT where appropriate,
and can also downgrade ACCEPT → REJECT if it detects hallucination signals.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from validation.annotation import AnnotatedSample, AnnotationLabel, RejectionCode

logger = logging.getLogger(__name__)

# System prompt used for the LLM reviewer
_REVIEWER_SYSTEM_PROMPT = """You are a rigorous dataset quality reviewer.
Given a dataset sample, evaluate it on three criteria:
1. Faithfulness — is the output grounded in the input text?
2. Completeness — are all required fields present and substantive?
3. Coherence     — is the output internally consistent?

Return a JSON object with exactly these keys:
{
  "verdict": "ACCEPT" | "FIX_REQUIRED" | "REJECT",
  "issues": ["<issue 1>", ...],
  "notes": "<brief explanation>"
}
Return ONLY the JSON object — no prose, no code fences."""


class LLMReviewer:
    """
    Optional second-pass reviewer that applies an LLM critique to samples.

    Args:
        llm_client: A BaseLLMClient (real or mock).
        run_on:     Which labels to review; defaults to FIX_REQUIRED only.
    """

    def __init__(
        self,
        llm_client=None,
        run_on: list[str] | None = None,
    ):
        self._client = llm_client
        self._run_on: list[str] = run_on or [AnnotationLabel.FIX_REQUIRED.value]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def review_batch(
        self, annotated_samples: list[AnnotatedSample]
    ) -> list[AnnotatedSample]:
        """Run the reviewer on samples whose current label is in self._run_on."""
        reviewed = 0
        for ann in annotated_samples:
            if ann.label.value in self._run_on:
                self._review_one(ann)
                reviewed += 1
        logger.info("LLM reviewer: reviewed %d borderline sample(s).", reviewed)
        return annotated_samples

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _review_one(self, ann: AnnotatedSample) -> None:
        """Run a single review and mutate the AnnotatedSample in place."""
        verdict, issues, notes = self._call_reviewer(ann.sample)
        ann.llm_review_passed = verdict == "ACCEPT"
        ann.reviewer_notes = notes

        if verdict == "REJECT":
            ann.label = AnnotationLabel.REJECT
            for issue in issues:
                ann.rejection_reasons.append(
                    type(
                        "RejectionReason",
                        (),
                        {
                            "code": RejectionCode.LLM_REVIEWER_REJECTED,
                            "message": issue,
                            "to_dict": lambda self: {
                                "code": self.code,
                                "message": self.message,
                            },
                        },
                    )()
                )
        elif verdict == "ACCEPT":
            # Upgrade only if there are no hard schema failures
            hard_fail = any(
                r.code == RejectionCode.SCHEMA_INVALID
                for r in ann.rejection_reasons
            )
            if not hard_fail:
                ann.label = AnnotationLabel.ACCEPT

    def _call_reviewer(self, sample: dict[str, Any]) -> tuple[str, list[str], str]:
        """
        Return (verdict, issues, notes) for a sample.

        Uses mock heuristics if no LLM client is configured.
        """
        if self._client is None:
            return self._mock_review(sample)

        sample_str = json.dumps(sample, ensure_ascii=False)
        user_prompt = (
            f"Review the following dataset sample and return your verdict.\n\n"
            f"SAMPLE:\n{sample_str}\n\nVERDICT JSON:"
        )
        try:
            raw = self._client.complete(
                system_prompt=_REVIEWER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=512,
            )
            parsed = json.loads(raw.strip())
            return (
                parsed.get("verdict", "FIX_REQUIRED"),
                parsed.get("issues", []),
                parsed.get("notes", ""),
            )
        except Exception as exc:
            logger.warning("LLM reviewer call failed: %s", exc)
            return "FIX_REQUIRED", [], f"reviewer_error: {exc}"

    @staticmethod
    def _mock_review(sample: dict[str, Any]) -> tuple[str, list[str], str]:
        """
        Heuristic mock reviewer logic.

        Upgrades FIX_REQUIRED to ACCEPT if the output looks substantive.
        """
        output = sample.get("output", {})
        task_type = sample.get("task_type", "")
        issues: List[str] = []

        # Per task-type quality signals
        if task_type == "qa":
            answer = str(output.get("answer", ""))
            evidence = str(output.get("evidence", ""))
            if len(answer) >= 20 and len(evidence) >= 20:
                return "ACCEPT", [], "Mock reviewer: answer and evidence are substantive."
            issues.append("QA answer or evidence is too brief for a high-quality sample.")

        elif task_type == "extraction":
            entities = output.get("entities", [])
            key_facts = output.get("key_facts", [])
            if isinstance(entities, list) and len(entities) >= 1 and isinstance(key_facts, list) and len(key_facts) >= 1:
                return "ACCEPT", [], "Mock reviewer: extraction looks complete."
            issues.append("Extraction is missing entities or key_facts.")

        elif task_type == "reasoning":
            steps = output.get("reasoning_steps", [])
            conclusion = str(output.get("conclusion", ""))
            if isinstance(steps, list) and len(steps) >= 2 and len(conclusion) >= 20:
                return "ACCEPT", [], "Mock reviewer: reasoning chain is adequate."
            issues.append("Reasoning steps or conclusion insufficient.")

        if issues:
            return "FIX_REQUIRED", issues, "Mock reviewer flagged minor issues."
        return "ACCEPT", [], "Mock reviewer: no issues found."
