"""
templates.py — Prompt templates for all supported task types.

Design principles
-----------------
* Structured output enforcement  — the system prompt explicitly forbids prose
  and demands valid JSON.
* Hallucination reduction         — instructions restrict answers to evidence
  present in the input text.
* Schema adherence               — the exact required JSON shape is embedded in
  every prompt.
* Diversity encouragement        — the user prompt asks for varied phrasing so
  consecutive samples generated from the same passage differ.
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List

from .few_shot_examples import FEW_SHOT_EXAMPLES


class TaskType(str, Enum):
    QA = "qa"
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    REASONING_TRACE = "reasoning_trace"   # o1/R1-style extended scratchpad
    PREFERENCE = "preference"             # DPO-ready (chosen, rejected) pairs


# ─────────────────────────────────────────────────────────────────────────────
# System prompts (one per task type)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPTS: Dict[str, str] = {
    TaskType.QA: """You are a high-quality dataset curation assistant specialised in \
Question Answering (QA).

RULES:
1. Generate exactly ONE question-answer pair from the provided passage.
2. The answer MUST be directly supported by text in the passage — do NOT invent facts.
3. The question should require genuine comprehension, not simple word-matching.
4. Return ONLY a valid JSON object — no prose, no markdown code fences.
5. The JSON must match this schema exactly:
{
  "question": "<question string>",
  "answer":   "<concise, faithful answer>",
  "evidence": "<verbatim or near-verbatim excerpt that supports the answer>"
}""",

    TaskType.EXTRACTION: """You are a high-quality dataset curation assistant specialised in \
Information Extraction (IE).

RULES:
1. Extract named entities (persons, organisations, locations, products, dates, …).
2. Identify subject–predicate–object relations expressed in the text.
3. List 3–5 key facts as complete, standalone sentences.
4. Base ALL extraction strictly on the provided text — no external knowledge.
5. Return ONLY a valid JSON object — no prose, no markdown code fences.
6. The JSON must match this schema exactly:
{
  "entities": [{"text": "...", "type": "..."}],
  "relations": [{"subject": "...", "predicate": "...", "object": "..."}],
  "key_facts": ["<fact 1>", "<fact 2>", ...]
}""",

    TaskType.REASONING: """You are a high-quality dataset curation assistant specialised in \
Multi-Step Reasoning.

RULES:
1. Analyse the passage using explicit chain-of-thought reasoning.
2. Produce 3–5 discrete, numbered reasoning steps that build toward a conclusion.
3. The conclusion must follow logically from the steps — no unsupported leaps.
4. Rate your confidence and briefly explain it.
5. Return ONLY a valid JSON object — no prose, no markdown code fences.
6. The JSON must match this schema exactly:
{
  "reasoning_steps": ["Step 1 — ...", "Step 2 — ...", ...],
  "conclusion": "<final conclusion>",
  "confidence_explanation": "<why you are confident or uncertain>"
}""",

    TaskType.REASONING_TRACE: """You are a high-quality dataset curation assistant that generates \
extended reasoning traces in the style of OpenAI o1 and DeepSeek-R1.

RULES:
1. Produce a full inner monologue inside a <think>...</think> block.
2. The <think> block MUST show genuine exploration: consider approaches, hit dead ends,
   backtrack, self-correct at least ONCE, and converge on the best answer.
3. After </think>, write a clean, concise final answer with no redundancy.
4. Optionally add a one-sentence verification confirming the answer is consistent.
5. Return ONLY a valid JSON object — no prose, no markdown code fences.
6. The JSON must match this schema exactly:
{
  "think": "<think>\\nFull inner reasoning monologue...\\n</think>",
  "answer": "<clean final answer>",
  "verification": "<one-sentence self-check>",
  "confidence": <float 0.0–1.0>
}""",

    TaskType.PREFERENCE: """You are a high-quality dataset curation assistant that generates \
DPO (Direct Preference Optimisation) training pairs.

RULES:
1. From the provided passage and question, generate TWO responses:
   - 'chosen': the ideal response — accurate, grounded, complete, well-structured.
   - 'rejected': a flawed response — may be over-confident, partially hallucinated,
     missing key nuance, or poorly formatted. Make it plausibly wrong, not obviously wrong.
2. The 'chosen' response must be clearly superior to 'rejected'.
3. Include per-response quality scores so training frameworks can weight pairs.
4. Return ONLY a valid JSON object — no prose, no markdown code fences.
5. The JSON must match this schema exactly:
{
  "prompt": "<the question or instruction>",
  "chosen": {"response": "<ideal response>", "quality_score": <float 0.7–1.0>},
  "rejected": {"response": "<flawed response>", "quality_score": <float 0.0–0.6>},
  "preference_margin": <chosen.quality_score - rejected.quality_score>
}""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot block builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_few_shot_block(task_type: str) -> str:
    """Format few-shot examples as numbered EXAMPLE blocks."""
    examples: List[Dict[str, Any]] = FEW_SHOT_EXAMPLES.get(task_type, [])
    if not examples:
        return ""

    lines = ["--- FEW-SHOT EXAMPLES ---"]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"\nEXAMPLE {i}:")
        lines.append(f"INPUT:\n{ex['input']}")
        lines.append(f"OUTPUT:\n{json.dumps(ex['output'], indent=2)}")
    lines.append("\n--- END EXAMPLES ---\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class PromptTemplates:
    """
    Factory that builds (system_prompt, user_prompt) pairs for any task type.

    Usage::

        system_msg, user_msg = PromptTemplates.build("qa", input_text="...")
    """

    @staticmethod
    def build(task_type: str, input_text: str) -> tuple[str, str]:
        """
        Build a (system_prompt, user_prompt) tuple for *task_type*.

        Args:
            task_type:  One of "qa", "extraction", "reasoning".
            input_text: The passage the LLM must process.

        Returns:
            (system_prompt, user_prompt) strings.
        """
        t = task_type.lower()
        # Normalise alias
        if t not in _SYSTEM_PROMPTS:
            raise ValueError(
                f"Unknown task type '{task_type}'. "
                f"Valid options: {list(_SYSTEM_PROMPTS.keys())}"
            )

        system_prompt = _SYSTEM_PROMPTS[t]
        few_shot_block = _build_few_shot_block(t)

        user_prompt = (
            f"{few_shot_block}"
            f"Now process the following passage and produce ONE {t.upper()} sample.\n\n"
            f"PASSAGE:\n{input_text}\n\n"
            "OUTPUT (valid JSON only):"
        )

        return system_prompt, user_prompt

    @staticmethod
    def system_prompt(task_type: str) -> str:
        """Return just the system prompt for a task type."""
        return _SYSTEM_PROMPTS[task_type.lower()]

    @staticmethod
    def task_instruction(task_type: str) -> str:
        """Return a short human-readable instruction string for the sample record."""
        _instructions = {
            TaskType.QA: (
                "Based on the provided passage, answer the question accurately "
                "and concisely, citing evidence from the text."
            ),
            TaskType.EXTRACTION: (
                "Extract all named entities, subject-predicate-object relations, "
                "and key facts from the provided text."
            ),
            TaskType.REASONING: (
                "Analyse the provided passage using step-by-step reasoning and "
                "arrive at a well-supported conclusion."
            ),
            TaskType.REASONING_TRACE: (
                "Produce an extended inner-monologue reasoning trace (o1/R1 style) "
                "that explores, backtracks, self-corrects, and converges on the answer."
            ),
            TaskType.PREFERENCE: (
                "Generate a DPO-ready preference pair: one ideal (chosen) response "
                "and one flawed (rejected) response for the given passage and question."
            ),
        }
        return _instructions.get(task_type.lower(), "Process the following text.")
