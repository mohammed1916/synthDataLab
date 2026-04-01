"""tests/test_validator.py — Unit tests for rule-based validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema.dataset_schema import DatasetSample
from validation.annotation import AnnotationLabel
from validation.rule_validator import RuleValidator

# ── Fixtures ──────────────────────────────────────────────────────────────────

rv = RuleValidator(min_confidence=0.60)

def _sample(task_type: str, output: dict, confidence: float = 0.85) -> dict:
    return DatasetSample.create(
        task_type, "A sufficiently long passage about an interesting scientific topic.",
        "Instruction.", output, "test", confidence, "mock"
    ).to_dict()


# ── QA ────────────────────────────────────────────────────────────────────────

def test_valid_qa_is_accepted():
    # Use a properly grounded evidence that shares words with the input passage
    s = _sample("qa", {
        "question": "What is a scientific topic?",
        "answer": "It is an interesting scientific topic.",
        "evidence": "a sufficiently long passage about an interesting scientific topic",
    })
    ann = rv.validate_one(s)
    assert ann.label == AnnotationLabel.ACCEPT


def test_qa_empty_answer_is_rejected():
    s = _sample("qa", {"question": "What?", "answer": "", "evidence": "See it."})
    ann = rv.validate_one(s)
    assert ann.label != AnnotationLabel.ACCEPT


def test_qa_short_answer_under_three_chars_gets_fix_required():
    s = _sample("qa", {"question": "What is it?", "answer": "No", "evidence": "Text here."})
    ann = rv.validate_one(s)
    # Very short answer triggers FIX_REQUIRED
    assert ann.label in (AnnotationLabel.FIX_REQUIRED, AnnotationLabel.REJECT)


def test_qa_low_confidence_is_rejected():
    s = _sample("qa", {"question": "Q?", "answer": "Answer here.", "evidence": "Evidence."}, confidence=0.30)
    ann = rv.validate_one(s)
    assert ann.label == AnnotationLabel.REJECT


# ── Extraction ────────────────────────────────────────────────────────────────

def test_valid_extraction_is_accepted():
    s = _sample("extraction", {
        "entities": [{"text": "Python", "type": "LANGUAGE"}],
        "relations": [],
        "key_facts": ["Python is a language."],
    })
    ann = rv.validate_one(s)
    assert ann.label == AnnotationLabel.ACCEPT


def test_extraction_empty_entities_is_fix_required():
    s = _sample("extraction", {"entities": [], "relations": [], "key_facts": ["fact"]})
    ann = rv.validate_one(s)
    assert ann.label in (AnnotationLabel.FIX_REQUIRED, AnnotationLabel.REJECT)


# ── Reasoning ─────────────────────────────────────────────────────────────────

def test_valid_reasoning_is_accepted():
    s = _sample("reasoning", {
        "reasoning_steps": ["Step 1: read.", "Step 2: analyse.", "Step 3: conclude."],
        "conclusion": "The passage supports the claim.",
        "confidence_explanation": "High — direct textual evidence.",
    })
    ann = rv.validate_one(s)
    assert ann.label == AnnotationLabel.ACCEPT


def test_reasoning_single_step_is_fix_required():
    s = _sample("reasoning", {
        "reasoning_steps": ["Only one step."],
        "conclusion": "Some conclusion here.",
        "confidence_explanation": "Some explanation.",
    })
    ann = rv.validate_one(s)
    assert ann.label in (AnnotationLabel.FIX_REQUIRED, AnnotationLabel.REJECT)


# ── Reasoning trace (R1-style) ────────────────────────────────────────────────

def test_valid_reasoning_trace_is_accepted():
    think = "<think>" + "Step-by-step reasoning, self-correcting halfway through. " * 4 + "</think>"
    s = _sample("reasoning_trace", {
        "think": think,
        "answer": "The answer is clearly stated in the passage.",
        "verification": "Verified against passage text.",
        "confidence": 0.88,
    })
    ann = rv.validate_one(s)
    assert ann.label == AnnotationLabel.ACCEPT


def test_reasoning_trace_missing_think_tags_is_rejected():
    s = _sample("reasoning_trace", {
        "think": "Just raw text with no think tags and it is long enough text here.",
        "answer": "Some answer.",
        "verification": "Verified.",
        "confidence": 0.80,
    })
    ann = rv.validate_one(s)
    assert ann.label in (AnnotationLabel.FIX_REQUIRED, AnnotationLabel.REJECT)


def test_reasoning_trace_confidence_out_of_range_is_rejected():
    think = "<think>" + "Reasoning. " * 15 + "</think>"
    s = _sample("reasoning_trace", {
        "think": think,
        "answer": "Answer.",
        "verification": "Verified.",
        "confidence": 1.5,  # invalid
    })
    ann = rv.validate_one(s)
    assert ann.label in (AnnotationLabel.FIX_REQUIRED, AnnotationLabel.REJECT)


# ── Preference (DPO-ready) ────────────────────────────────────────────────────

def test_valid_preference_pair_is_accepted():
    s = _sample("preference", {
        "prompt": "What is photosynthesis?",
        "chosen": {"response": "Photosynthesis converts sunlight into glucose using chlorophyll.", "quality_score": 0.93},
        "rejected": {"response": "Plants eat light.", "quality_score": 0.20},
        "preference_margin": 0.73,
    })
    ann = rv.validate_one(s)
    assert ann.label == AnnotationLabel.ACCEPT


def test_preference_negative_margin_is_rejected():
    s = _sample("preference", {
        "prompt": "Question?",
        "chosen": {"response": "Good answer here with detail.", "quality_score": 0.90},
        "rejected": {"response": "Bad answer.", "quality_score": 0.30},
        "preference_margin": -0.10,   # inverted — chosen is actually worse
    })
    ann = rv.validate_one(s)
    assert ann.label == AnnotationLabel.REJECT


def test_preference_small_margin_is_fix_required():
    s = _sample("preference", {
        "prompt": "Question?",
        "chosen": {"response": "Good answer.", "quality_score": 0.75},
        "rejected": {"response": "Slightly worse.", "quality_score": 0.70},
        "preference_margin": 0.05,   # below 0.15 threshold
    })
    ann = rv.validate_one(s)
    assert ann.label == AnnotationLabel.FIX_REQUIRED


def test_preference_low_chosen_quality_score_is_fix_required():
    s = _sample("preference", {
        "prompt": "Question?",
        "chosen": {"response": "This is the chosen response.", "quality_score": 0.50},  # below 0.60 floor
        "rejected": {"response": "Rejected.", "quality_score": 0.10},
        "preference_margin": 0.40,
    })
    ann = rv.validate_one(s)
    assert ann.label in (AnnotationLabel.FIX_REQUIRED, AnnotationLabel.REJECT)


# ── Batch validation ──────────────────────────────────────────────────────────

def test_batch_validate_returns_same_count():
    samples = [
        _sample("qa", {"question": "Q?", "answer": "A.", "evidence": "E."}),
        _sample("reasoning", {
            "reasoning_steps": ["S1.", "S2."],
            "conclusion": "C.",
            "confidence_explanation": "CE.",
        }),
    ]
    annotated = rv.validate_batch(samples)
    assert len(annotated) == len(samples)
