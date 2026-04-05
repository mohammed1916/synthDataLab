"""
math_generator.py — CBSE Math content generation orchestrator.

Workflow
--------

  1.  Ingest PDFs / text files (NCERT chapters, past papers).
  2.  Run GapAnalyzer → CoverageReport.
  3.  For each chapter/subtopic determine what to generate:
        - "problem"     → new practice questions with solutions
        - "explanation" → concept note with formulas + worked example
        - "fill_gap"    → targeted problem for uncovered subtopics
  4.  Call the LLM (Ollama or mock) using math-specific prompts.
  5.  Parse, validate (math_schema), and save JSONL output.

Usage (CLI will wrap this; direct usage shown below)
------------------------------------------------------
    from cbse_math.math_generator import MathGenerator, MathGenConfig

    gen = MathGenerator()
    results = gen.run(
        inputs=["ncert_class12.pdf", "prev_papers_2023.pdf"],
        class_level=12,
        problems_per_subtopic=3,
        mock=True,       # offline mode
    )
    # results → list[MathSample]
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MathGenConfig:
    class_level: int = 12
    problems_per_subtopic: int = 2      # practice problems per subtopic
    explanations_per_gap: int = 1       # concept notes for uncovered subtopics
    gap_fills_per_gap: int = 2          # fill_gap problems per uncovered subtopic
    difficulty_mix: list[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    problem_type_mix: list[str] = field(default_factory=lambda: [
        "short_answer_2m", "long_answer_3m", "long_answer_5m", "case_study",
    ])
    # Whether to include MCQ problems
    include_mcq: bool = True
    temperature: float = 0.8           # higher → more varied problems
    max_tokens: int = 2048
    # Max source text chars fed per prompt
    max_source_chars: int = 3000


# ─────────────────────────────────────────────────────────────────────────────
# Mock Math LLM (offline development / testing)
# ─────────────────────────────────────────────────────────────────────────────

class MockMathLLM:
    """
    Deterministic mock that returns plausible LaTeX math JSON without any LLM call.
    Produces valid samples 80 % of the time; invalid 20 % (for validation testing).
    """

    _MOCK_PROBLEMS = {
        "problem": {
            "question_latex": r"If $f(x) = x^2 - 3x + 2$, find all values of $x$ for which $f(x) = 0$.",
            "solution_latex": (
                r"We need to solve $x^2 - 3x + 2 = 0$." + "\n\n"
                r"Factorising: $(x-1)(x-2) = 0$" + "\n\n"
                r"Therefore $x = 1$ or $x = 2$."
            ),
            "answer_latex": r"$x = 1$ or $x = 2$",
            "hints": [
                "Try factorising the quadratic expression.",
                "Look for two numbers that multiply to $+2$ and add to $-3$.",
            ],
            "common_mistakes": [
                "Forgetting to check both roots.",
                "Sign errors when factorising.",
            ],
        },
        "explanation": {
            "concept_latex": (
                r"**The Quadratic Formula** allows us to solve any equation of the form $ax^2 + bx + c = 0$." + "\n\n"
                r"The discriminant $D = b^2 - 4ac$ determines the nature of roots."
            ),
            "key_formulas": [
                r"$x = \dfrac{-b \pm \sqrt{b^2 - 4ac}}{2a}$",
                r"$D = b^2 - 4ac$: if $D > 0$ two real roots, $D = 0$ equal roots, $D < 0$ no real roots",
            ],
            "worked_example_latex": (
                r"Solve $2x^2 - 5x + 3 = 0$." + "\n\n"
                r"Here $a=2,\ b=-5,\ c=3$." + "\n\n"
                r"\[D = (-5)^2 - 4(2)(3) = 25 - 24 = 1\]" + "\n\n"
                r"\[x = \frac{5 \pm 1}{4} \implies x = \frac{3}{2} \text{ or } x = 1\]"
            ),
            "summary": (
                "The quadratic formula solves any quadratic. "
                "Always compute the discriminant first to know how many solutions exist."
            ),
            "common_misconceptions": [
                "Students often forget the $\\pm$ and compute only one root.",
                "Dividing by $2a$ is divided into BOTH terms of the numerator.",
            ],
        },
        "fill_gap": {
            "gap_description": "Student has not practised problems involving the sum and product of roots.",
            "question_latex": (
                r"If one root of the equation $3x^2 - kx + 6 = 0$ is $2$, "
                r"find the value of $k$ and the other root."
            ),
            "solution_latex": (
                r"Since $x = 2$ is a root: $3(4) - 2k + 6 = 0 \implies k = 9$." + "\n\n"
                r"Sum of roots $= \dfrac{k}{3} = 3$, so other root $= 3 - 2 = 1$."
            ),
            "answer_latex": r"$k = 9$, other root $= 1$",
            "why_this_gap_matters": (
                "Sum/product of roots questions appear in CBSE board exams almost every year. "
                "Missing this concept directly costs 2–3 marks."
            ),
            "hints": [
                "If $\\alpha$ is a root, substitute directly to find the unknown.",
            ],
        },
    }

    def complete(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        # Prefer explicit routing when caller provides expected item type.
        item_type = kwargs.get("expected_item_type")
        if item_type not in {"problem", "explanation", "fill_gap"}:
            # Fallback inference from prompts.
            if "fill_gap" in system_prompt.lower() or "gap" in user_prompt.lower():
                item_type = "fill_gap"
            elif "concept explanation" in system_prompt.lower():
                item_type = "explanation"
            else:
                item_type = "problem"

        template = dict(self._MOCK_PROBLEMS[item_type])

        # 20 % chance of deliberately broken output to feed the validator
        if random.random() < 0.20:
            required_fields = {
                "problem": "solution_latex",
                "explanation": "worked_example_latex",
                "fill_gap": "solution_latex",
            }
            template.pop(required_fields[item_type], None)   # break schema on purpose

        return json.dumps(template, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class MathGenerator:
    """
    Generates CBSE mathematics problems, explanations, and gap-fill items.

    Args:
        mock:   Use the offline mock LLM (no Ollama required).
        model:  Ollama model name (ignored when mock=True).
        config: Generation configuration knobs.
    """

    def __init__(
        self,
        mock: bool = False,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        config: MathGenConfig | None = None,
    ):
        self.config = config or MathGenConfig()

        if mock:
            self._llm = MockMathLLM()
            self._model_name = "mock-math-llm"
        else:
            from generation.llm_client import OllamaClient
            self._llm = OllamaClient(model=model, base_url=base_url)
            self._model_name = model

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        inputs: list[str],
        class_level: int | None = None,
        output_path: Path | str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Full generation run.

        Args:
            inputs:       List of file paths (.pdf, .txt, or .json).
            class_level:  Override the config class level.
            output_path:  If set, write JSONL here as well as returning.

        Returns:
            List of validated MathSample dicts.
        """
        from cbse_math.cbse_syllabus import chapters_for_class
        from cbse_math.gap_analyzer import GapAnalyzer
        from cbse_math.math_prompts import (
            build_explanation_prompt,
            build_fill_gap_prompt,
            build_problem_prompt,
        )
        from cbse_math.math_schema import MathMetadata, MathSample, validate_math_sample
        from cbse_math.pdf_ingestor import ingest_pdf

        cl = class_level or self.config.class_level
        chapters = chapters_for_class(cl)  # type: ignore[arg-type]
        if not chapters:
            raise ValueError(
                f"No CBSE syllabus chapters registered for class {cl}. "
                "Supported classes are 10 and 12."
            )

        # ── 1. Ingest all inputs ──────────────────────────────────────────────
        all_chunks: list[str] = []
        source_label = "user_input"
        for inp in inputs:
            path = Path(inp)
            source_label = path.stem
            try:
                if path.suffix.lower() == ".pdf":
                    records = ingest_pdf(str(path))
                    all_chunks.extend(r["content"] for r in records)
                elif path.suffix.lower() in (".txt", ".md"):
                    all_chunks.append(path.read_text(encoding="utf-8", errors="replace"))
                elif path.suffix.lower() == ".json":
                    import json as _json
                    data = _json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                text = item.get("content") or item.get("text") or str(item)
                            else:
                                text = str(item)
                            all_chunks.append(text)
                    else:
                        all_chunks.append(str(data))
                else:
                    logger.warning("Unsupported file type: %s — skipping.", path.suffix)
            except Exception as exc:
                logger.error("Failed to ingest '%s': %s", inp, exc)

        if not all_chunks:
            logger.warning("No content ingested from inputs. Using chapter descriptions as source.")
            all_chunks = [ch.title + ": " + "; ".join(ch.subtopics) for ch in chapters]

        # ── 2. Gap analysis ───────────────────────────────────────────────────
        analyzer = GapAnalyzer(class_level=cl)  # type: ignore[arg-type]
        report = analyzer.analyse(all_chunks)
        logger.info("\n%s", report.summary())

        # ── 3. Generation ─────────────────────────────────────────────────────
        samples: list[dict[str, Any]] = []

        for ch in chapters:
            ch_chunks = self._relevant_chunks(all_chunks, ch.keywords)
            source_text = "\n\n".join(ch_chunks[:3]) if ch_chunks else (
                ch.title + "\n" + "\n".join(ch.subtopics)
            )
            source_text = source_text[:self.config.max_source_chars]

            # Determine status for this chapter
            cc = next((c for c in report.chapter_coverages if c.chapter.chapter_id == ch.chapter_id), None)
            status = cc.status if cc else "gap"

            for subtopic in ch.subtopics:
                # Always generate practice problems
                for _ in range(self.config.problems_per_subtopic):
                    sample = self._generate_problem(ch, subtopic, source_text, source_label)
                    if sample:
                        errors = validate_math_sample(sample)
                        sample["_validation_errors"] = errors
                        samples.append(sample)

                # Generate explanation if partial or gap
                if status in ("partial", "gap"):
                    for _ in range(self.config.explanations_per_gap):
                        sample = self._generate_explanation(ch, subtopic, source_text, source_label)
                        if sample:
                            errors = validate_math_sample(sample)
                            sample["_validation_errors"] = errors
                            samples.append(sample)

            # Fill gaps for uncovered subtopics
            if cc:
                for gap_sub in cc.uncovered_subtopics:
                    gap_reason = f"No examples found in provided materials for: {gap_sub}"
                    for _ in range(self.config.gap_fills_per_gap):
                        sample = self._generate_fill_gap(ch, gap_sub, gap_reason, source_text, source_label)
                        if sample:
                            errors = validate_math_sample(sample)
                            sample["_validation_errors"] = errors
                            samples.append(sample)

        logger.info("Generated %d math items total.", len(samples))

        # ── 4. Save ───────────────────────────────────────────────────────────
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            logger.info("Saved %d items → %s", len(samples), out)

        return samples

    # ─────────────────────────────────────────────────────────────────────────
    # Private generation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _relevant_chunks(self, chunks: list[str], keywords: list[str]) -> list[str]:
        """Return chunks that contain at least one chapter keyword."""
        hits = []
        for chunk in chunks:
            cl = chunk.lower()
            if any(kw.lower() in cl for kw in keywords if len(kw) >= 3):
                hits.append(chunk)
        return hits

    def _pick_difficulty(self) -> str:
        return random.choice(self.config.difficulty_mix)

    def _pick_problem_type(self, chapter_types: list[str]) -> tuple[str, int]:
        """Pick a problem type and its default marks."""
        _marks_map = {
            "mcq_1m": 1,
            "assertion_reason": 1,
            "short_answer_2m": 2,
            "long_answer_3m": 3,
            "long_answer_4m": 4,
            "long_answer_5m": 5,
            "case_study": 4,
        }
        candidates = [t for t in chapter_types if t in self.config.problem_type_mix]
        if self.config.include_mcq and "mcq_1m" in chapter_types:
            candidates.append("mcq_1m")
        if not candidates:
            candidates = self.config.problem_type_mix[:2]
        pt = random.choice(candidates)
        return pt, _marks_map.get(pt, 3)

    def _call_llm_json(
        self, system: str, user: str, expected_item_type: str | None = None
    ) -> dict[str, Any] | None:
        """Call the LLM and parse the JSON response."""
        try:
            raw = self._llm.complete(
                system_prompt=system,
                user_prompt=user,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                expected_item_type=expected_item_type,
            )
            # Strip markdown fences if present
            raw = re.sub(r"```(?:json)?\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw)
            return json.loads(raw.strip())
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("LLM response parse error: %s", exc)
            return None

    def _generate_problem(
        self,
        chapter: Any,
        subtopic: str,
        source_text: str,
        source_label: str,
    ) -> dict[str, Any] | None:
        from cbse_math.math_prompts import build_problem_prompt
        from cbse_math.math_schema import MathMetadata, MathSample

        difficulty = self._pick_difficulty()
        pt, marks = self._pick_problem_type(chapter.problem_types)

        system, user = build_problem_prompt(source_text, chapter, subtopic, difficulty, pt, marks)
        content = self._call_llm_json(system, user, expected_item_type="problem")
        if content is None:
            return None

        meta = MathMetadata(
            source=source_label,
            confidence=0.75,
            generation_model=self._model_name,
            problem_type=pt,
            bloom_level="apply",
        )
        sample = MathSample(
            item_type="problem",
            class_level=chapter.class_level,
            chapter_id=chapter.chapter_id,
            chapter_title=chapter.title,
            subtopic=subtopic,
            difficulty=difficulty,
            marks=marks,
            content=content,
            metadata=meta,
        )
        return sample.to_dict()

    def _generate_explanation(
        self,
        chapter: Any,
        subtopic: str,
        source_text: str,
        source_label: str,
    ) -> dict[str, Any] | None:
        from cbse_math.math_prompts import build_explanation_prompt
        from cbse_math.math_schema import MathMetadata, MathSample

        system, user = build_explanation_prompt(chapter, subtopic, source_text)
        content = self._call_llm_json(system, user, expected_item_type="explanation")
        if content is None:
            return None

        meta = MathMetadata(
            source=source_label,
            confidence=0.80,
            generation_model=self._model_name,
            bloom_level="understand",
        )
        sample = MathSample(
            item_type="explanation",
            class_level=chapter.class_level,
            chapter_id=chapter.chapter_id,
            chapter_title=chapter.title,
            subtopic=subtopic,
            difficulty="medium",
            marks=0,
            content=content,
            metadata=meta,
        )
        d = sample.to_dict()
        return d

    def _generate_fill_gap(
        self,
        chapter: Any,
        subtopic: str,
        gap_reason: str,
        source_text: str,
        source_label: str,
    ) -> dict[str, Any] | None:
        from cbse_math.math_prompts import build_fill_gap_prompt
        from cbse_math.math_schema import MathMetadata, MathSample

        system, user = build_fill_gap_prompt(chapter, subtopic, gap_reason, source_text)
        content = self._call_llm_json(system, user, expected_item_type="fill_gap")
        if content is None:
            return None

        meta = MathMetadata(
            source=source_label,
            confidence=0.70,
            generation_model=self._model_name,
            is_gap_fill=True,
            bloom_level="analyze",
        )
        sample = MathSample(
            item_type="fill_gap",
            class_level=chapter.class_level,
            chapter_id=chapter.chapter_id,
            chapter_title=chapter.title,
            subtopic=subtopic,
            difficulty="medium",
            marks=3,
            content=content,
            metadata=meta,
        )
        return sample.to_dict()
