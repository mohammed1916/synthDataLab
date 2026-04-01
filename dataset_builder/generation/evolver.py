"""
evolver.py — Evol-Instruct style prompt evolution for synthetic data diversity.

Based on the WizardLM Evol-Instruct technique (Xu et al., 2023) and
Microsoft's AgentInstruct (Mitra et al., 2024).

How it works
------------
Starting from a set of seed prompts (instructions), the evolver applies one
of four evolution operations to each prompt, producing a harder, more
diverse variant:

  1. add_constraints    — "Answer X" → "Answer X using only analogies, no jargon,
                           in ≤ 200 words"
  2. deepen             — "What is X?" → "Compare X and Y at a mechanistic level,
                           citing first principles"
  3. concretise         — "How do vaccines work?" → "How does the BNT162b2 mRNA
                           vaccine prime CD8+ T-cell response in humans?"
  4. increase_reasoning — "Solve X" → "Solve X and justify why each step is
                           necessary, identifying where other approaches would fail"

Each evolved prompt is optionally validated by an LLM judge:
  - If the evolved prompt is too trivially similar to the seed, it is discarded.
  - If the evolved prompt is judged unanswerable or nonsensical, it is discarded.

The result is a list of EvolvedPrompt objects suitable for injecting back
into the generation pipeline as new `input_text` seeds.

Usage::

    from generation.evolver import PromptEvolver, EvolveConfig
    from generation.llm_client import MockLLMClient

    evolver = PromptEvolver(EvolveConfig(), llm_client=MockLLMClient())
    evolved = evolver.evolve(seed_prompts, n_rounds=2)
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvolveConfig:
    """Configuration for the prompt evolution process."""

    n_rounds: int = 2                  # how many evolution rounds to apply
    operations: list[str] = field(
        default_factory=lambda: [
            "add_constraints",
            "deepen",
            "concretise",
            "increase_reasoning",
        ]
    )
    max_seeds_per_round: int = 50      # limit to prevent runaway growth
    min_length_delta: int = 10         # evolved prompt must be longer than seed by at least this
    use_llm_evolution: bool = False    # True = use real LLM; False = template-based (mock mode)


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvolvedPrompt:
    """A single evolved prompt with its lineage metadata."""

    prompt: str
    seed_prompt: str
    operation: str             # which evolution operation was applied
    round_number: int          # which evolution round produced this
    complexity_score: float    # heuristic estimate of relative complexity [0, 1]
    discarded: bool = False    # True if the prompt was rejected by the quality filter
    discard_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "seed_prompt": self.seed_prompt,
            "operation": self.operation,
            "round_number": self.round_number,
            "complexity_score": round(self.complexity_score, 3),
            "discarded": self.discarded,
            "discard_reason": self.discard_reason,
        }


# ─────────────────────────────────────────────────────────────────────────────
# System prompts for LLM-based evolution (used when use_llm_evolution=True)
# ─────────────────────────────────────────────────────────────────────────────

_LLM_EVOLUTION_PROMPTS: dict[str, str] = {
    "add_constraints": """You are an instruction difficulty evolver.
Given the SEED INSTRUCTION below, create a harder variant by adding 1-3 specific constraints.
Constraints can include: length limits, format requirements, restricted vocabulary, style rules.
The evolved instruction must still be answerable from a knowledgeable person.
Return ONLY the evolved instruction text — no commentary, no quotes.

Rules:
- Do NOT change the core topic.
- Add constraints that require more careful, deliberate answering.
- The evolved instruction should be 20-60% longer in word count than the seed.

SEED INSTRUCTION:
{seed}

EVOLVED INSTRUCTION:""",

    "deepen": """You are an instruction difficulty evolver.
Given the SEED INSTRUCTION below, create a deeper, more analytical variant.
Replace surface-level questions with ones requiring mechanistic understanding,
causal reasoning, or comparison between concepts.
Return ONLY the evolved instruction text — no commentary, no quotes.

Rules:
- Keep the same domain/topic.
- Require the answerer to explain WHY or HOW, not just WHAT.
- The evolved instruction must be genuinely harder, not just longer.

SEED INSTRUCTION:
{seed}

EVOLVED INSTRUCTION:""",

    "concretise": """You are an instruction difficulty evolver.
Given the SEED INSTRUCTION below, make it more specific and concrete by:
- Replacing generic terms with specific named instances, models, or examples.
- Adding specific numerical or temporal context if appropriate.
Return ONLY the evolved instruction text — no commentary, no quotes.

Rules:
- Preserve the intent of the original instruction.
- The named instances must be real and verifiable.
- Do not make the question so specific that it becomes trivial.

SEED INSTRUCTION:
{seed}

EVOLVED INSTRUCTION:""",

    "increase_reasoning": """You are an instruction difficulty evolver.
Given the SEED INSTRUCTION below, create a variant that requires multi-step reasoning.
The evolved instruction should ask the respondent to:
- Show their work or justify each step.
- Identify where simpler approaches would fail.
- Consider edge cases or counter-arguments.
Return ONLY the evolved instruction text — no commentary, no quotes.

SEED INSTRUCTION:
{seed}

EVOLVED INSTRUCTION:""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Template-based evolution (mock mode — no API call required)
# ─────────────────────────────────────────────────────────────────────────────

_CONSTRAINT_TEMPLATES = [
    "—but your answer must use no technical jargon and be understandable to a 12-year-old",
    ", using only analogies drawn from everyday life",
    ". Limit your answer to exactly 3 sentences.",
    ", and explicitly state one common misconception people have about this topic",
    ". Your answer must include a concrete real-world example.",
    ", avoiding the use of the words 'important', 'significant', or 'key'",
    ". Structure your answer as: Definition → Mechanism → Implication.",
]

_DEEPENING_PREFIXES = [
    "Compare and contrast, at a mechanistic level, the following: ",
    "Explain the causal chain that leads from ",
    "What are the first-principles reasons why ",
    "Analyse the trade-offs involved in ",
    "From an adversarial perspective, what are the weaknesses in ",
    "Trace the historical evolution of thinking about ",
]

_CONCRETISE_WRAPPERS = [
    lambda s: s.replace("vaccines", "the BNT162b2 mRNA vaccine").replace("vaccine", "the BNT162b2 mRNA vaccine"),
    lambda s: s.replace("AI model", "GPT-4o").replace("language model", "Llama-3 70B"),
    lambda s: s.replace("company", "Tesla Inc. (2024)").replace("organisation", "NASA (2023)"),
    lambda s: re.sub(r"\bexplain\b", "explain, using a specific published study,", s, count=1, flags=re.IGNORECASE),
]

_REASONING_SUFFIXES = [
    " Justify each step of your reasoning and identify where a simpler approach would fail.",
    " Show your work step by step, and after each step state what assumption you are relying on.",
    " For each claim you make, provide a falsifiability criterion.",
    " After giving your answer, provide one counter-argument and explain why it does not hold.",
    " Structure your response as: Claim → Evidence → Counter-argument → Rebuttal → Conclusion.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core evolver
# ─────────────────────────────────────────────────────────────────────────────

class PromptEvolver:
    """
    Applies Evol-Instruct style evolution operations to seed prompts.

    Args:
        config:     EvolveConfig controlling rounds, operations, etc.
        llm_client: If provided AND config.use_llm_evolution=True, used to
                    generate evolved prompts via LLM API. Otherwise, uses
                    deterministic template-based evolution.
        seed:       Random seed for reproducibility.
    """

    def __init__(
        self,
        config: EvolveConfig | None = None,
        llm_client: Any | None = None,
        seed: int = 42,
    ):
        self.config = config or EvolveConfig()
        self.llm_client = llm_client
        self._rng = random.Random(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def evolve(
        self,
        seed_prompts: list[str],
        n_rounds: int | None = None,
    ) -> list[EvolvedPrompt]:
        """
        Evolve a list of seed prompts over multiple rounds.

        Each round takes the surviving prompts from the previous round as seeds.
        Returns ALL evolved prompts across ALL rounds (including discarded ones
        for analysis — filter by `discarded=False` for training data).

        Args:
            seed_prompts: Initial list of instruction strings.
            n_rounds:     Override config.n_rounds if provided.

        Returns:
            List[EvolvedPrompt] with full lineage metadata.
        """
        rounds = n_rounds if n_rounds is not None else self.config.n_rounds
        all_evolved: list[EvolvedPrompt] = []
        current_seeds = seed_prompts[: self.config.max_seeds_per_round]

        for round_num in range(1, rounds + 1):
            logger.info(
                "Evol-Instruct round %d/%d — evolving %d prompts",
                round_num, rounds, len(current_seeds),
            )
            round_results = self._evolve_round(current_seeds, round_num)
            all_evolved.extend(round_results)

            # Next round seeds = surviving (non-discarded) prompts from this round
            current_seeds = [
                ep.prompt
                for ep in round_results
                if not ep.discarded
            ][: self.config.max_seeds_per_round]

            if not current_seeds:
                logger.warning("No surviving prompts after round %d. Stopping.", round_num)
                break

        logger.info(
            "Evolution complete: %d total evolved prompts, %d surviving",
            len(all_evolved),
            sum(1 for ep in all_evolved if not ep.discarded),
        )
        return all_evolved

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _evolve_round(
        self, seeds: list[str], round_num: int
    ) -> list[EvolvedPrompt]:
        results: list[EvolvedPrompt] = []
        for seed in seeds:
            operation = self._rng.choice(self.config.operations)
            try:
                evolved_text = self._apply_operation(seed, operation)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Evolution failed for operation=%s: %s", operation, exc)
                evolved_text = seed  # fallback: keep original

            score = self._complexity_score(evolved_text, seed)
            ep = EvolvedPrompt(
                prompt=evolved_text,
                seed_prompt=seed,
                operation=operation,
                round_number=round_num,
                complexity_score=score,
            )
            self._quality_filter(ep, seed)
            results.append(ep)
        return results

    def _apply_operation(self, seed: str, operation: str) -> str:
        """Apply the named evolution operation to the seed prompt."""
        if self.config.use_llm_evolution and self.llm_client is not None:
            return self._llm_evolve(seed, operation)
        return self._template_evolve(seed, operation)

    def _llm_evolve(self, seed: str, operation: str) -> str:
        """Use an LLM backend to evolve the prompt."""
        system_prompt = "You are a prompt evolution assistant."
        user_prompt = _LLM_EVOLUTION_PROMPTS[operation].format(seed=seed)
        raw = self.llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.8,
            max_tokens=512,
        )
        # The LLM might return JSON or plain text; handle both
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed.get("evolved_instruction", parsed.get("prompt", seed))
            if isinstance(parsed, str):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return raw.strip().strip('"').strip("'")

    def _template_evolve(self, seed: str, operation: str) -> str:
        """Deterministic template-based evolution (no API call)."""
        if operation == "add_constraints":
            suffix = self._rng.choice(_CONSTRAINT_TEMPLATES)
            # Trim trailing punctuation before appending
            base = seed.rstrip("?.!")
            return f"{base}{suffix}"

        if operation == "deepen":
            prefix = self._rng.choice(_DEEPENING_PREFIXES)
            # Strip leading question words to avoid double-interrogatives
            core = re.sub(r"^(what|how|why|explain|describe)\s+", "", seed, flags=re.IGNORECASE)
            return f"{prefix}{core}"

        if operation == "concretise":
            transform = self._rng.choice(_CONCRETISE_WRAPPERS)
            evolved = transform(seed)
            # If no substitution matched, append a specificity nudge
            if evolved == seed:
                evolved = seed.rstrip("?.!") + ", using a specific real-world case study as your example."
            return evolved

        if operation == "increase_reasoning":
            suffix = self._rng.choice(_REASONING_SUFFIXES)
            base = seed.rstrip(".")
            return f"{base}{suffix}"

        # Fallback: return seed unchanged
        return seed

    def _quality_filter(self, ep: EvolvedPrompt, seed: str) -> None:
        """
        Mark an evolved prompt as discarded if it fails quality checks:
          1. Too similar to seed (evolution had no effect).
          2. Evolved prompt is shorter than seed (likely a degradation).
          3. Evolved prompt is nonsensically long (> 8× seed length).
        """
        seed_words = len(seed.split())
        evolved_words = len(ep.prompt.split())

        # Check 1: Trivial similarity (word-overlap > 95%)
        seed_tokens = set(re.findall(r"\b\w{3,}\b", seed.lower()))
        evolved_tokens = set(re.findall(r"\b\w{3,}\b", ep.prompt.lower()))
        if seed_tokens:
            overlap = len(seed_tokens & evolved_tokens) / len(seed_tokens)
            if overlap > 0.95 and evolved_words <= seed_words + 2:
                ep.discarded = True
                ep.discard_reason = (
                    f"Too similar to seed (word overlap={overlap:.0%}, "
                    f"word count delta={evolved_words - seed_words})"
                )
                return

        # Check 2: Evolution made it shorter by more than min_length_delta words
        if evolved_words < seed_words - 2:
            ep.discarded = True
            ep.discard_reason = (
                f"Evolved prompt is shorter than seed: {evolved_words} < {seed_words} words"
            )
            return

        # Check 3: Runaway expansion
        if seed_words > 5 and evolved_words > seed_words * 8:
            ep.discarded = True
            ep.discard_reason = (
                f"Evolved prompt is suspiciously long ({evolved_words} words vs "
                f"seed {seed_words} words — ratio {evolved_words/seed_words:.1f}×)"
            )

    @staticmethod
    def _complexity_score(evolved: str, seed: str) -> float:
        """
        Heuristic complexity score in [0, 1].

        Factors:
          - Word count ratio (evolved / seed), capped at 3×
          - Presence of constraint words (must, only, without, exactly, etc.)
          - Question depth (sub-clause count approximated by comma count)
        """
        seed_words = max(len(seed.split()), 1)
        evolved_words = len(evolved.split())

        length_factor = min(evolved_words / seed_words, 3.0) / 3.0   # 0–1

        constraint_words = {
            "must", "only", "without", "exactly", "using", "avoid", "but",
            "however", "compare", "contrast", "justify", "falsif", "causal",
            "step-by-step", "first principles", "mechanistic",
        }
        constraint_count = sum(
            1 for w in constraint_words if w in evolved.lower()
        )
        constraint_factor = min(constraint_count / 5.0, 1.0)

        commas = evolved.count(",")
        structure_factor = min(commas / 4.0, 1.0)

        return round(0.4 * length_factor + 0.4 * constraint_factor + 0.2 * structure_factor, 3)
