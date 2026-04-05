"""
tests/test_cbse_math.py — Targeted tests for CBSE math generation modules.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import cli

from cbse_math.cbse_syllabus import CLASS_12_CHAPTERS, Chapter
from cbse_math.gap_analyzer import ChapterCoverage, CoverageReport
from cbse_math.math_generator import MathGenerator
from cbse_math.math_schema import MathMetadata, MathSample, validate_math_sample


def test_mock_generation_routes_item_types_without_cross_schema_mismatch(monkeypatch):
    # Keep mock output valid so we specifically test item-type routing.
    monkeypatch.setattr("cbse_math.math_generator.random.random", lambda: 0.99)

    gen = MathGenerator(mock=True)
    chapter = CLASS_12_CHAPTERS[0]
    source_text = "relation function domain range bijective"

    p = gen._generate_problem(chapter, chapter.subtopics[0], source_text, "unit-test")
    e = gen._generate_explanation(chapter, chapter.subtopics[0], source_text, "unit-test")
    g = gen._generate_fill_gap(
        chapter,
        chapter.subtopics[0],
        "No examples found for this topic",
        source_text,
        "unit-test",
    )

    assert p is not None and p["item_type"] == "problem"
    assert e is not None and e["item_type"] == "explanation"
    assert g is not None and g["item_type"] == "fill_gap"
    assert validate_math_sample(p) == []
    assert validate_math_sample(e) == []
    assert validate_math_sample(g) == []


def test_prioritised_gaps_falls_back_to_chapter_subtopics():
    chapter = Chapter(
        chapter_id="c12_demo",
        title="Demo",
        unit="Demo Unit",
        class_level=12,
        marks=5,
        subtopics=["Subtopic A", "Subtopic B"],
        problem_types=["short_answer_2m"],
        keywords=["demo"],
    )
    gap_ch = ChapterCoverage(
        chapter=chapter,
        score=0.0,
        status="gap",
        uncovered_subtopics=[],
    )
    report = CoverageReport(
        class_level=12,
        total_chapters=1,
        covered_count=0,
        partial_count=0,
        gap_count=1,
        chapter_coverages=[gap_ch],
    )
    gaps = report.prioritised_gaps()
    assert [sub for _, sub in gaps] == ["Subtopic A", "Subtopic B"]


def test_math_schema_rejects_unsupported_class_level():
    sample = MathSample(
        item_type="problem",
        class_level=11,  # unsupported for cbse_math registry
        chapter_id="x",
        chapter_title="x",
        subtopic="x",
        difficulty="easy",
        marks=1,
        content={
            "question_latex": "Find $x$ if $x+1=2$.",
            "solution_latex": "Subtract 1 from both sides to get $x=1$.",
            "answer_latex": "$x=1$",
            "hints": ["Subtract 1 from both sides."],
        },
        metadata=MathMetadata(source="t", confidence=0.8, generation_model="mock"),
    ).to_dict()
    errors = validate_math_sample(sample)
    assert any("is not one of" in err for err in errors)


def test_math_generate_rejects_unsupported_cli_class_level():
    runner = CliRunner()
    result = runner.invoke(cli, ["math-generate", "--mock", "--class-level", "11"])
    assert result.exit_code != 0
    assert "Invalid value for '--class-level'" in result.output


def test_math_gap_analysis_parses_json_input(tmp_path):
    src = tmp_path / "math_input.json"
    src.write_text(
        json.dumps(
            [
                {"content": "matrix determinant inverse matrix cramer rule solved example"},
                {"text": "conditional probability bayes theorem binomial distribution"},
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["math-gap-analysis", str(src), "--class-level", "12"])
    assert result.exit_code == 0, result.output
    assert "Ingested:" in result.output
    assert "Coverage Report — Class 12" in result.output


def test_math_generate_valid_only_and_summary_output(tmp_path, monkeypatch):
    # Force invalid mock payloads so valid-only mode writes zero records.
    monkeypatch.setattr("cbse_math.math_generator.random.random", lambda: 0.0)

    out_path = tmp_path / "math_valid_only.jsonl"
    summary_path = tmp_path / "math_summary.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "math-generate",
            "--mock",
            "--class-level",
            "12",
            "--problems-per-subtopic",
            "1",
            "--gap-fills",
            "0",
            "--valid-only",
            "--output",
            str(out_path),
            "--summary-output",
            str(summary_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").strip() == ""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["valid_only"] is True
    assert summary["saved_count"] == 0
    assert summary["invalid_count"] > 0


def test_math_gap_analysis_output_writes_json_report(tmp_path):
    src = tmp_path / "gap_input.txt"
    src.write_text("matrix determinant inverse matrix solved example", encoding="utf-8")
    out_path = tmp_path / "gap_report.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["math-gap-analysis", str(src), "--class-level", "12", "--output", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["class_level"] == 12
    assert "chapter_coverages" in payload
    assert "prioritised_gaps" in payload
