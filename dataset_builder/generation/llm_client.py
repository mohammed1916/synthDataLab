"""
llm_client.py — Unified LLM interface with a full-fidelity Mock backend.

Architecture
------------
``LLMClient``     – thin wrapper around the OpenAI Chat Completions API.
``MockLLMClient`` – deterministic synthetic generator that produces realistic
                    QA / Extraction / Reasoning samples WITHOUT any API call.
                    Used when OPENAI_API_KEY is absent or provider="mock".

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
# OpenAI client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient(BaseLLMClient):
    """OpenAI Chat Completions client."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout: int = 30):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


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

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_task_type(system_prompt: str) -> str:
        sp = system_prompt.lower()
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
    api_key: str,
    model: str,
    timeout: int = 30,
) -> BaseLLMClient:
    """Return the appropriate LLM client based on provider setting."""
    if provider == "mock" or not api_key:
        logger.info("Using MockLLMClient (no API key configured or provider=mock).")
        return MockLLMClient()
    logger.info("Using OpenAI LLMClient (model=%s).", model)
    return LLMClient(api_key=api_key, model=model, timeout=timeout)
