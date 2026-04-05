"""
math_schema.py — JSON Schema + dataclass for CBSE Mathematics samples.

Every generated math item has one of three ``item_type`` values:

  problem      — A LaTeX-formatted question with a full step-by-step solution.
  explanation  — A concept explanation with key formulas and worked examples.
  fill_gap     — A problem targeting a specific gap in the student's syllabus
                  coverage (identified by the GapAnalyzer).

LaTeX conventions used throughout
----------------------------------
  - Inline math  : ``$...$``
  - Display math : ``\\[...\\]``
  - Align env    : ``\\begin{align*}...\\end{align*}``
  - All generated content is UTF-8 + LaTeX; no Unicode math symbols in formulas.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import jsonschema

# ─────────────────────────────────────────────────────────────────────────────
# JSON Schema
# ─────────────────────────────────────────────────────────────────────────────

MATH_SAMPLE_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "MathSample",
    "type": "object",
    "required": ["id", "item_type", "class_level", "chapter_id", "chapter_title",
                 "subtopic", "difficulty", "marks", "content", "metadata"],
    "additionalProperties": False,
    "properties": {
        "id":            {"type": "string", "minLength": 1},
        "item_type":     {"type": "string", "enum": ["problem", "explanation", "fill_gap"]},
        "class_level":   {"type": "integer", "enum": [9, 10, 11, 12]},
        "chapter_id":    {"type": "string", "minLength": 1},
        "chapter_title": {"type": "string", "minLength": 1},
        "subtopic":      {"type": "string", "minLength": 1},
        "difficulty":    {"type": "string", "enum": ["easy", "medium", "hard"]},
        "marks":         {"type": "integer", "minimum": 0, "maximum": 10},
        "content":       {"type": "object", "minProperties": 1},
        "metadata": {
            "type": "object",
            "required": ["source", "confidence", "generation_model", "timestamp"],
            "additionalProperties": True,
            "properties": {
                "source":           {"type": "string", "minLength": 1},
                "confidence":       {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "generation_model": {"type": "string", "minLength": 1},
                "timestamp":        {"type": "string", "minLength": 1},
                "problem_type":     {"type": "string"},
                "bloom_level":      {"type": "string"},
                "is_gap_fill":      {"type": "boolean"},
                "source_problems":  {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

# ── Content sub-schemas ───────────────────────────────────────────────────────

PROBLEM_CONTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["question_latex", "solution_latex", "answer_latex", "hints"],
    "properties": {
        "question_latex": {"type": "string", "minLength": 10},
        "solution_latex": {"type": "string", "minLength": 20},
        "answer_latex":   {"type": "string", "minLength": 1},
        "hints": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 5},
        },
        "common_mistakes": {"type": "array", "items": {"type": "string"}},
        "mcq_options": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "string"},
        },
        "correct_option": {"type": "string", "enum": ["A", "B", "C", "D"]},
    },
}

EXPLANATION_CONTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["concept_latex", "key_formulas", "worked_example_latex", "summary"],
    "properties": {
        "concept_latex":        {"type": "string", "minLength": 30},
        "key_formulas":         {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "worked_example_latex": {"type": "string", "minLength": 30},
        "summary":              {"type": "string", "minLength": 10},
        "common_misconceptions": {"type": "array", "items": {"type": "string"}},
    },
}

FILL_GAP_CONTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["gap_description", "question_latex", "solution_latex", "answer_latex",
                 "why_this_gap_matters"],
    "properties": {
        "gap_description":      {"type": "string", "minLength": 10},
        "question_latex":       {"type": "string", "minLength": 10},
        "solution_latex":       {"type": "string", "minLength": 20},
        "answer_latex":         {"type": "string", "minLength": 1},
        "why_this_gap_matters": {"type": "string", "minLength": 10},
        "hints":                {"type": "array", "items": {"type": "string"}},
    },
}

_CONTENT_SCHEMAS = {
    "problem":     PROBLEM_CONTENT_SCHEMA,
    "explanation": EXPLANATION_CONTENT_SCHEMA,
    "fill_gap":    FILL_GAP_CONTENT_SCHEMA,
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MathMetadata:
    source: str
    confidence: float
    generation_model: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    problem_type: str = ""
    bloom_level: str = "apply"          # remember | understand | apply | analyze | evaluate | create
    is_gap_fill: bool = False
    source_problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v or v == 0 or isinstance(v, bool)}


@dataclass
class MathSample:
    item_type: Literal["problem", "explanation", "fill_gap"]
    class_level: int
    chapter_id: str
    chapter_title: str
    subtopic: str
    difficulty: Literal["easy", "medium", "hard"]
    marks: int
    content: dict[str, Any]
    metadata: MathMetadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":            self.id,
            "item_type":     self.item_type,
            "class_level":   self.class_level,
            "chapter_id":    self.chapter_id,
            "chapter_title": self.chapter_title,
            "subtopic":      self.subtopic,
            "difficulty":    self.difficulty,
            "marks":         self.marks,
            "content":       self.content,
            "metadata":      self.metadata.to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

_top_validator = jsonschema.Draft7Validator(MATH_SAMPLE_SCHEMA)


def validate_math_sample(sample: dict[str, Any]) -> list[str]:
    """
    Validate a math sample dict against the schema.

    Returns a list of error messages (empty = valid).
    """
    errors = [e.message for e in _top_validator.iter_errors(sample)]
    if errors:
        return errors

    # Validate content sub-schema based on item_type
    item_type = sample.get("item_type")
    sub_schema = _CONTENT_SCHEMAS.get(item_type)
    if sub_schema:
        sub_validator = jsonschema.Draft7Validator(sub_schema)
        errors = [e.message for e in sub_validator.iter_errors(sample.get("content", {}))]

    return errors
