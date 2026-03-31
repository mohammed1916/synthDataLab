"""tests/test_schema.py — Unit tests for schema validation and DatasetSample."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from schema.dataset_schema import validate_sample, DatasetSample, TASK_OUTPUT_SCHEMAS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_sample(task_type: str, output: dict) -> dict:
    return DatasetSample.create(
        task_type=task_type,
        input_text="A test passage about an interesting topic in science.",
        instruction="Test instruction for validation.",
        output=output,
        source="test",
        confidence=0.85,
        model="mock",
    ).to_dict()


QA_OUT = {"question": "What is the topic?", "answer": "Science.", "evidence": "See text."}
EX_OUT = {
    "entities": [{"text": "Science", "type": "FIELD"}],
    "relations": [],
    "key_facts": ["Science is interesting."],
}
RS_OUT = {
    "reasoning_steps": ["Step 1: read.", "Step 2: analyse."],
    "conclusion": "The passage is about science.",
    "confidence_explanation": "High confidence based on direct statement.",
}
RT_OUT = {
    "think": "<think>Step 1: identify the topic. Step 2: verify claim. Let me re-check…</think>",
    "answer": "The topic is science.",
    "verification": "Confirmed by explicit text reference.",
    "confidence": 0.91,
}
PR_OUT = {
    "prompt": "What is science?",
    "chosen": {"response": "Science is systematic empirical investigation.", "quality_score": 0.92},
    "rejected": {"response": "Science is magic.", "quality_score": 0.15},
    "preference_margin": 0.77,
}


# ── Valid sample tests ────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_type,output", [
    ("qa", QA_OUT),
    ("extraction", EX_OUT),
    ("reasoning", RS_OUT),
    ("reasoning_trace", RT_OUT),
    ("preference", PR_OUT),
])
def test_valid_sample_passes_schema(task_type, output):
    sample = _make_sample(task_type, output)
    ok, errs = validate_sample(sample)
    assert ok, f"{task_type} validation failed: {errs}"


# ── Invalid sample tests ──────────────────────────────────────────────────────

def test_qa_missing_answer_fails():
    out = {"question": "What?", "evidence": "Some evidence here."}  # missing answer
    sample = _make_sample("qa", out)
    ok, errs = validate_sample(sample)
    assert not ok
    assert any("answer" in e for e in errs)


def test_extraction_empty_entities_fails():
    out = {**EX_OUT, "entities": []}  # violates minItems:1
    sample = _make_sample("extraction", out)
    ok, errs = validate_sample(sample)
    assert not ok


def test_reasoning_single_step_fails():
    out = {**RS_OUT, "reasoning_steps": ["Only one step."]}  # violates minItems:2
    sample = _make_sample("reasoning", out)
    ok, errs = validate_sample(sample)
    assert not ok


def test_reasoning_trace_short_think_fails():
    out = {**RT_OUT, "think": "short"}  # violates minLength:50
    sample = _make_sample("reasoning_trace", out)
    ok, errs = validate_sample(sample)
    assert not ok


def test_preference_confidence_out_of_range_fails():
    out = {**PR_OUT, "preference_margin": 1.5}  # violates maximum:1.0
    sample = _make_sample("preference", out)
    ok, errs = validate_sample(sample)
    assert not ok


def test_unknown_task_type_fails_enum():
    sample = _make_sample("qa", QA_OUT)
    sample["task_type"] = "unknown_type"
    ok, errs = validate_sample(sample)
    assert not ok


def test_missing_top_level_field_fails():
    sample = _make_sample("qa", QA_OUT)
    del sample["input"]
    ok, errs = validate_sample(sample)
    assert not ok


# ── ID format tests ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_type,prefix", [
    ("qa", "QA"), ("extraction", "EX"), ("reasoning", "RS"),
    ("reasoning_trace", "RT"), ("preference", "PR"), ("unknown", "XX"),
])
def test_sample_id_has_correct_prefix(task_type, prefix):
    s = DatasetSample.create(task_type, "input text " * 5, "instruction", {}, "src", 0.5, "m")
    assert s.id.startswith(prefix + "_"), f"Expected prefix {prefix}_, got {s.id}"


# ── Schema coverage ───────────────────────────────────────────────────────────

def test_all_five_task_types_have_output_schemas():
    expected = {"qa", "extraction", "reasoning", "reasoning_trace", "preference"}
    assert expected.issubset(set(TASK_OUTPUT_SCHEMAS.keys()))
