"""
math_prompts.py — LaTeX-output prompt templates for CBSE math generation.

Three generation modes
-----------------------
  problem      — Generate a new math problem (with full solution) from a source passage.
  explanation  — Explain a concept / subtopic with key formulas and a worked example.
  fill_gap     — Generate a problem that specifically addresses an identified gap in
                 syllabus coverage.

All outputs are valid JSON with LaTeX inside string fields.
LaTeX convention: inline = $...$, display = \\[...\\], multi-step = align* env.
"""
from __future__ import annotations

from .cbse_syllabus import PROBLEM_TYPE_DESCRIPTIONS, Chapter

# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROBLEM = """\
You are an expert CBSE Mathematics teacher and question-paper setter for \
Classes 10 and 12. Your task is to create a **new, original** mathematics problem \
that is closely related to the provided source material but is NOT a copy of \
any problem in it.

RULES
-----
1. The problem must be solvable using concepts from the specified chapter and subtopic.
2. Write ALL mathematical expressions in LaTeX:
   - Inline math  : $...$
   - Display math : \\[...\\]
   - Multi-step   : \\begin{{align*}}...\\end{{align*}}
3. The solution must show every step with brief English commentary between LaTeX lines.
4. Provide 1–3 hints (progressive: each hint gives slightly more information).
5. List common mistakes students make on this type of problem.
6. For MCQ problems, provide exactly 4 plausible options (one correct).
7. Return ONLY a valid JSON object — no prose, no markdown fences.
8. JSON schema:
{{
  "question_latex":   "<LaTeX question statement>",
  "solution_latex":   "<step-by-step LaTeX solution>",
  "answer_latex":     "<final answer in LaTeX>",
  "hints":            ["<hint 1>", "<hint 2>"],
  "common_mistakes":  ["<mistake 1>", "<mistake 2>"],
  "mcq_options":      ["A. ...", "B. ...", "C. ...", "D. ..."],
  "correct_option":   "A"|"B"|"C"|"D"
}}
Omit "mcq_options" and "correct_option" if the problem is not MCQ."""

_SYSTEM_EXPLANATION = """\
You are an expert CBSE Mathematics teacher preparing study material for \
Class {class_level} students.

Your task is to write a clear, concise explanation of a mathematical concept \
with key formulas and a fully worked example — entirely in LaTeX.

RULES
-----
1. Write ALL formulas in LaTeX ($...$ for inline, \\[...\\] for display).
2. Keep language simple and targeted at a Class {class_level} student.
3. Include at least 2 key formulas.
4. The worked example must show every calculation step.
5. End with a one-paragraph summary a student can memorise.
6. List common misconceptions.
7. Return ONLY a valid JSON object — no prose, no markdown fences.
8. JSON schema:
{{
  "concept_latex":         "<explanation text with embedded LaTeX>",
  "key_formulas":          ["<formula 1 in LaTeX>", "<formula 2 in LaTeX>"],
  "worked_example_latex":  "<fully worked example with steps>",
  "summary":               "<one-paragraph revision summary>",
  "common_misconceptions": ["<misconception 1>", "<misconception 2>"]
}}"""

_SYSTEM_FILL_GAP = """\
You are an expert CBSE Mathematics teacher identifying and filling gaps in a \
student's understanding.

A gap analysis has revealed that the student has insufficient practice with the \
specified subtopic. Your task is to generate a targeted problem that directly \
addresses this gap, with a full solution and an explanation of why this concept \
is important.

RULES
-----
1. The problem must squarely address the identified gap — not adjacent topics.
2. Write ALL mathematical expressions in LaTeX.
3. Include a brief "gap description" explaining what the student is likely missing.
4. Include "why_this_gap_matters" — real-world or exam consequence.
5. Return ONLY a valid JSON object — no prose, no markdown fences.
6. JSON schema:
{{
  "gap_description":      "<what concept is missing and why>",
  "question_latex":       "<LaTeX question>",
  "solution_latex":       "<step-by-step LaTeX solution>",
  "answer_latex":         "<final answer>",
  "why_this_gap_matters": "<board-exam or conceptual consequence>",
  "hints":                ["<hint 1>"]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# User prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_problem_prompt(
    source_text: str,
    chapter: Chapter,
    subtopic: str,
    difficulty: str,
    problem_type: str,
    marks: int,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for math problem generation.

    Args:
        source_text:  Extracted text from NCERT / past paper (already sanitised).
        chapter:      Chapter object from the CBSE syllabus registry.
        subtopic:     Specific subtopic to target.
        difficulty:   "easy" | "medium" | "hard"
        problem_type: Key from PROBLEM_TYPE_DESCRIPTIONS.
        marks:        Marks the problem should be worth.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    system = _SYSTEM_PROBLEM
    pt_desc = PROBLEM_TYPE_DESCRIPTIONS.get(problem_type, problem_type)

    user = f"""\
CLASS    : {chapter.class_level}
CHAPTER  : {chapter.title}
SUBTOPIC : {subtopic}
DIFFICULTY: {difficulty}
PROBLEM TYPE: {pt_desc}  ({marks} marks)

SOURCE MATERIAL (extracted from NCERT / past papers — use as context, do NOT copy verbatim):
---
{source_text[:3500]}
---

Generate ONE original {difficulty} {pt_desc} worth {marks} mark(s) on "{subtopic}".
The problem must require genuine mathematical reasoning, not just recall.
Return the JSON object only."""

    return system, user


def build_explanation_prompt(
    chapter: Chapter,
    subtopic: str,
    source_text: str = "",
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for concept explanation generation."""
    system = _SYSTEM_EXPLANATION.format(class_level=chapter.class_level)

    context_block = ""
    if source_text:
        context_block = f"""
REFERENCE MATERIAL (use as context for accuracy):
---
{source_text[:2000]}
---
"""

    user = f"""\
CLASS   : {chapter.class_level}
CHAPTER : {chapter.title}
SUBTOPIC: {subtopic}
{context_block}
Write a clear explanation of "{subtopic}" for a Class {chapter.class_level} student.
Include key formulas, a worked example, and a summary.
Return the JSON object only."""

    return system, user


def build_fill_gap_prompt(
    chapter: Chapter,
    subtopic: str,
    gap_reason: str,
    source_text: str = "",
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for gap-fill problem generation."""
    system = _SYSTEM_FILL_GAP

    context_block = ""
    if source_text:
        context_block = f"""
REFERENCE MATERIAL:
---
{source_text[:2000]}
---
"""

    user = f"""\
CLASS    : {chapter.class_level}
CHAPTER  : {chapter.title}
SUBTOPIC : {subtopic}
GAP REASON: {gap_reason}
{context_block}
Generate ONE problem that fills this specific gap for the student.
Return the JSON object only."""

    return system, user
