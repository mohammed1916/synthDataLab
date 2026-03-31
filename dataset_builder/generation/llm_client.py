"""
llm_client.py — Unified LLM interface with Ollama backend and Mock fallback.

Architecture
------------
``OllamaClient``  – calls a locally running Ollama server (default model: qwen3:4b).
``MockLLMClient`` – deterministic synthetic generator that produces realistic
                    QA / Extraction / Reasoning samples WITHOUT any network call.
                    Used when provider="mock".

The mock backend deliberately introduces ~20 % defective samples so the
downstream validation and filtering stages have interesting data to process.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Return the raw text response from the model."""


# ─────────────────────────────────────────────────────────────────────────────
# Ollama client
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient(BaseLLMClient):
    """
    Client for a locally running Ollama server.

    Requires the `ollama` Python package and Ollama to be running locally.
    The model must already be pulled: ``ollama pull qwen3:4b``
    """

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        try:
            import ollama as _ollama  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ollama package not installed. Run: pip install ollama"
            ) from exc
        self._ollama = _ollama
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        import ollama as _ollama  # type: ignore

        # Build the messages list
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = _ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            format="json",
        )
        return response["message"]["content"] or ""


# ─────────────────────────────────────────────────────────────────────────────
# Mock client — rich synthetic data, no API key required
# ─────────────────────────────────────────────────────────────────────────────

class MockLLMClient(BaseLLMClient):
    """
    Deterministic mock LLM that synthesises realistic dataset samples.

    It parses the *user_prompt* to extract the passage, then applies
    templated generation logic per task type.  Roughly 20 % of responses
    contain intentional defects to stress-test validation layers.
    """

    MODEL_NAME = "mock-llm-v1"
    DEFECT_RATE = 0.22   # fraction of samples with deliberate flaws

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        task_type = self._detect_task_type(system_prompt)
        passage = self._extract_passage(user_prompt)
        introduce_defect = self._rng.random() < self.DEFECT_RATE

        generators = {
            "qa": self._gen_qa,
            "extraction": self._gen_extraction,
            "reasoning": self._gen_reasoning,
            "reasoning_trace": self._gen_reasoning_trace,
            "preference": self._gen_preference,
        }
        gen_fn = generators.get(task_type, self._gen_qa)
        output = gen_fn(passage, introduce_defect)

        # Simulate a tiny processing delay
        time.sleep(0.05)
        return json.dumps(output, ensure_ascii=False)

    # ── Task generators ───────────────────────────────────────────────────────

    def _gen_qa(self, passage: str, defect: bool) -> Dict[str, Any]:
        sentences = self._sentences(passage)
        key = sentences[self._rng.randint(0, len(sentences) - 1)] if sentences else passage[:120]

        question_starters = [
            "What is described as",
            "According to the passage, what",
            "How does the text explain",
            "What does the author state about",
            "Why, according to the passage,",
        ]
        starter = self._rng.choice(question_starters)
        key_fragment = " ".join(key.split()[:6]) if key.split() else "this topic"
        question = f"{starter} {key_fragment}?"

        answer = key[:300] if len(key) > 10 else passage[:200]

        result: Dict[str, Any] = {
            "question": question,
            "answer": answer,
        }

        if defect:
            defect_type = self._rng.choice(["missing_evidence", "empty_answer", "very_short"])
            if defect_type == "missing_evidence":
                # Missing required "evidence" field
                pass  # intentionally omit
            elif defect_type == "empty_answer":
                result["answer"] = ""
                result["evidence"] = key[:150]
            elif defect_type == "very_short":
                result["answer"] = "Yes."
                result["evidence"] = "."
        else:
            result["evidence"] = key[:250]

        return result

    def _gen_extraction(self, passage: str, defect: bool) -> Dict[str, Any]:
        entities = self._extract_entities(passage)
        relations = self._extract_relations(passage)
        sentences = self._sentences(passage)
        key_facts = [s for s in sentences if len(s) > 30][:5]
        if not key_facts:
            key_facts = [passage[:200]]

        result: Dict[str, Any] = {
            "entities": entities,
            "relations": relations,
            "key_facts": key_facts,
        }

        if defect:
            defect_type = self._rng.choice(
                ["empty_entities", "missing_key_facts", "bad_entity_format"]
            )
            if defect_type == "empty_entities":
                result["entities"] = []   # violates minItems:1
            elif defect_type == "missing_key_facts":
                del result["key_facts"]   # missing required field
            elif defect_type == "bad_entity_format":
                result["entities"] = ["plain string instead of object"]  # wrong type

        return result

    def _gen_reasoning(self, passage: str, defect: bool) -> Dict[str, Any]:
        sentences = self._sentences(passage)
        topic = " ".join(passage.split()[:8]) + "..."

        if len(sentences) >= 3:
            steps = [
                f"Step 1 — Identify the core claim: '{sentences[0][:120]}'",
                f"Step 2 — Examine the supporting evidence: '{sentences[min(1, len(sentences)-1)][:120]}'",
                f"Step 3 — Apply logical reasoning: The passage consistently supports the argument that {topic}",
                f"Step 4 — Consider counter-arguments or qualifications mentioned in the text.",
                f"Step 5 — Synthesise: The passage provides '{sentences[-1][:100]}' as a concluding point.",
            ]
        else:
            steps = [
                f"Step 1 — The passage introduces the concept of {topic}",
                f"Step 2 — Key evidence is found in the statement: '{passage[:150]}'",
                f"Step 3 — This leads to the conclusion that the passage demonstrates the stated claim.",
            ]

        conclusion = (
            f"Based on the evidence presented, the passage establishes that {topic} "
            f"This is supported by the explicit statements within the text."
        )
        confidence_explanation = (
            "High confidence — the conclusion follows directly from the textual evidence "
            "without requiring external assumptions."
        )

        result: Dict[str, Any] = {
            "reasoning_steps": steps,
            "conclusion": conclusion,
            "confidence_explanation": confidence_explanation,
        }

        if defect:
            defect_type = self._rng.choice(
                ["too_few_steps", "missing_conclusion", "empty_conclusion"]
            )
            if defect_type == "too_few_steps":
                result["reasoning_steps"] = steps[:1]  # violates minItems:2
            elif defect_type == "missing_conclusion":
                del result["conclusion"]               # missing required field
            elif defect_type == "empty_conclusion":
                result["conclusion"] = " "             # too short — minLength:10

        return result

    def _gen_reasoning_trace(self, passage: str, defect: bool) -> Dict[str, Any]:
        """Generate an o1/R1-style extended reasoning trace with self-correction."""
        sentences = self._sentences(passage)
        topic = " ".join(passage.split()[:8]) + "..."
        first_s = sentences[0][:150] if sentences else passage[:150]
        last_s = sentences[-1][:120] if len(sentences) > 1 else passage[-120:]
        words = passage.split()
        mid_word = words[len(words) // 2] if words else "concept"

        think_block = (
            "<think>\n"
            f"Let me carefully reason through this passage about {topic}\n\n"
            f"First pass: the key claim seems to be '{first_s}'\n\n"
            f"Wait — I should check whether I'm reading that correctly. "
            f"The phrase '{mid_word}' is important here. Let me re-examine.\n\n"
            "Actually, I overcomplicated my initial reading. The passage is making "
            f"a simpler, more direct point: {topic}\n\n"
            f"Supporting evidence: '{last_s}'\n\n"
            "This is directly stated in the text — I don't need to extrapolate. "
            "My conclusion is well-grounded.\n"
            "</think>"
        )
        answer = (
            f"The passage establishes that {topic} "
            f"This is directly evidenced by: '{first_s[:120]}'"
        )
        verification = (
            "Cross-check confirms: conclusion uses only information explicitly "
            "present in the passage without external assumptions."
        )
        confidence = round(self._rng.uniform(0.78, 0.97), 2)

        result: Dict[str, Any] = {
            "think": think_block,
            "answer": answer,
            "verification": verification,
            "confidence": confidence,
        }

        if defect:
            defect_type = self._rng.choice(
                ["missing_think_tags", "empty_answer", "no_self_correction"]
            )
            if defect_type == "missing_think_tags":
                result["think"] = think_block.replace("<think>", "").replace("</think>", "")
            elif defect_type == "empty_answer":
                result["answer"] = ""
            elif defect_type == "no_self_correction":
                # Think block with no backtracking — lower quality trace
                result["think"] = (
                    f"<think>\nThe passage is about {topic} "
                    f"The answer is {first_s[:80]}\n</think>"
                )
                result["confidence"] = round(self._rng.uniform(0.30, 0.55), 2)

        return result

    def _gen_preference(self, passage: str, defect: bool) -> Dict[str, Any]:
        """Generate a DPO-ready (chosen, rejected) preference pair."""
        sentences = self._sentences(passage)
        topic = " ".join(passage.split()[:6]) + "..."
        key_fact = sentences[0][:200] if sentences else passage[:200]
        last_fact = sentences[-1][:150] if len(sentences) > 1 else key_fact

        question_starters = [
            "What does the passage reveal about",
            "Explain the relationship between",
            "What are the key implications of",
        ]
        prompt_text = f"{self._rng.choice(question_starters)} {topic}?"

        chosen_response = (
            f"Based on the passage, {topic} "
            f"The text states: '{key_fact[:180]}'. "
            f"Furthermore, '{last_fact[:120]}'. "
            "This evidence directly supports a grounded, accurate answer "
            "without requiring external assumptions."
        )
        chosen_score = round(self._rng.uniform(0.80, 0.97), 2)

        # Rejected response: overconfident/vague/partially wrong
        rejected_patterns = [
            (
                f"{topic} is a well-known concept that has been studied extensively. "
                "Experts generally agree that this is an important area, and the evidence "
                "suggests there are many factors at play. The relationship is complex and "
                "multifaceted, requiring careful consideration of all perspectives."
            ),
            (
                f"The passage discusses {topic} It is widely understood that this leads to "
                "significant consequences. Research has consistently shown positive outcomes, "
                "though some debate remains. Overall the evidence is compelling."
            ),
        ]
        rejected_response = self._rng.choice(rejected_patterns)
        rejected_score = round(self._rng.uniform(0.15, 0.45), 2)

        result: Dict[str, Any] = {
            "prompt": prompt_text,
            "chosen": {
                "response": chosen_response,
                "quality_score": chosen_score,
            },
            "rejected": {
                "response": rejected_response,
                "quality_score": rejected_score,
            },
            "preference_margin": round(chosen_score - rejected_score, 2),
        }

        if defect:
            defect_type = self._rng.choice(
                ["margin_too_small", "chosen_empty", "inverted_scores"]
            )
            if defect_type == "margin_too_small":
                result["chosen"]["quality_score"] = 0.55
                result["rejected"]["quality_score"] = 0.50
                result["preference_margin"] = 0.05   # below useful threshold
            elif defect_type == "chosen_empty":
                result["chosen"]["response"] = ""
            elif defect_type == "inverted_scores":
                # Chosen scored lower than rejected — logic error
                result["chosen"]["quality_score"] = 0.30
                result["rejected"]["quality_score"] = 0.85
                result["preference_margin"] = -0.55

        return result


    @staticmethod
    def _detect_task_type(system_prompt: str) -> str:
        sp = system_prompt.lower()
        if "reasoning trace" in sp or "inner monologue" in sp or "scratchpad" in sp:
            return "reasoning_trace"
        if "preference" in sp or "dpo" in sp or "chosen" in sp:
            return "preference"
        if "question answering" in sp or "qa" in sp:
            return "qa"
        if "information extraction" in sp or "extract" in sp:
            return "extraction"
        if "reasoning" in sp or "chain-of-thought" in sp:
            return "reasoning"
        return "qa"

    @staticmethod
    def _extract_passage(user_prompt: str) -> str:
        """Pull the text between PASSAGE: and OUTPUT: markers."""
        match = re.search(r"PASSAGE:\s*(.*?)\s*OUTPUT", user_prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: last 1 000 chars of prompt
        return user_prompt[-1000:].strip()

    @staticmethod
    def _sentences(text: str) -> List[str]:
        """Simple sentence splitter."""
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if len(s.strip()) > 15]

    # Simple capitalised-word entity extraction heuristic
    _ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TECHNOLOGY", "CONCEPT"]

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        words = text.split()
        entities = []
        seen = set()
        for i, word in enumerate(words):
            clean = re.sub(r"[^\w]", "", word)
            if (
                len(clean) > 2
                and clean[0].isupper()
                and clean.lower() not in {"the", "a", "an", "in", "on", "at", "to", "of", "and"}
                and clean not in seen
            ):
                entity_type = self._rng.choice(self._ENTITY_TYPES)
                entities.append({"text": clean, "type": entity_type})
                seen.add(clean)
            if len(entities) >= 6:
                break
        if not entities:
            entities.append({"text": text.split()[0], "type": "CONCEPT"})
        return entities

    def _extract_relations(self, text: str) -> List[Dict[str, str]]:
        entities = self._extract_entities(text)
        if len(entities) < 2:
            return []
        predicates = ["is_related_to", "influences", "part_of", "developed_by", "located_in"]
        relations = []
        for i in range(min(3, len(entities) - 1)):
            relations.append(
                {
                    "subject": entities[i]["text"],
                    "predicate": self._rng.choice(predicates),
                    "object": entities[i + 1]["text"],
                }
            )
        return relations


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_client(
    provider: str,
    model: str,
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
) -> BaseLLMClient:
    """Return the appropriate LLM client based on provider setting."""
    if provider == "mock":
        logger.info("Using MockLLMClient (provider=mock).")
        return MockLLMClient()
    logger.info("Using OllamaClient (model=%s, url=%s).", model, base_url)
    return OllamaClient(model=model, base_url=base_url, timeout=timeout)
