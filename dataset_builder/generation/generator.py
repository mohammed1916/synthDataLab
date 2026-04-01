"""
generator.py — Orchestrates synthetic dataset generation.

For every ingested chunk the generator:
  1. Iterates over configured task types.
  2. Builds a prompt via PromptTemplates.
  3. Calls the LLM (real or mock).
  4. Parses the JSON response.
  5. Wraps it in a DatasetSample with metadata.
  6. Returns the full list of raw samples.

Errors (bad JSON, missing fields) are logged and the sample is still saved
with ``confidence = 0.0`` so the validation stage can handle them explicitly.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DEFAULT_CONFIG, Config
from generation.llm_client import BaseLLMClient, build_llm_client
from ingestion.ingestor import IngestionResult
from prompts.templates import PromptTemplates
from schema.dataset_schema import DatasetSample

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    Generates structured dataset samples from ingested content.

    Args:
        config:     Pipeline configuration.
        llm_client: Override the default LLM client (useful for testing).
    """

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        llm_client: BaseLLMClient | None = None,
    ):
        self.config = config
        self.llm: BaseLLMClient = llm_client or build_llm_client(
            provider=config.llm.provider,
            model=config.llm.model,
            base_url=config.llm.base_url,
            timeout=config.llm.request_timeout,
        )
        self._model_name = (
            "mock-llm-v1" if config.use_mock_llm else config.llm.model
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate_from_ingestion(
        self, ingestion_results: list[IngestionResult]
    ) -> list[DatasetSample]:
        """
        Generate samples from a list of IngestionResult objects.

        Each chunk × each enabled task type yields one sample attempt.
        When ``config.generation.max_workers > 1`` the calls are parallelised
        using a ``ThreadPoolExecutor`` (the LLM client is thread-safe).

        Args:
            ingestion_results: Normalised input records from the ingestion layer.

        Returns:
            Flat list of DatasetSample objects (valid and invalid).
        """
        # Build (chunk, task_type) work items
        work: list[tuple[IngestionResult, str]] = [
            (chunk, task_type)
            for chunk in ingestion_results
            for task_type in self.config.generation.task_types
        ]
        total = len(work)
        max_workers = max(1, self.config.generation.max_workers)

        samples: list[DatasetSample] = []

        def _run_one(item: tuple[IngestionResult, str]) -> DatasetSample | None:
            chunk, task_type = item
            return self._generate_one(chunk, task_type)

        try:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
            )
            _rich_available = True
        except ImportError:
            _rich_available = False

        if max_workers == 1:
            # Sequential path — keeps log output predictable
            if _rich_available:
                from rich.console import Console
                from rich.progress import (
                    BarColumn,
                    Progress,
                    SpinnerColumn,
                    TaskProgressColumn,
                    TextColumn,
                )
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]Generating[/bold cyan]"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=Console(stderr=True),
                    transient=True,
                ) as progress:
                    task = progress.add_task("", total=total)
                    for item in work:
                        s = _run_one(item)
                        if s:
                            samples.append(s)
                        progress.advance(task)
            else:
                for item in work:
                    s = _run_one(item)
                    if s:
                        samples.append(s)
        else:
            # Parallel path
            logger.info("Parallel generation: %d workers × %d tasks", max_workers, total)
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for item in work:
                    futures[pool.submit(_run_one, item)] = item
                completed = 0
                for fut in as_completed(futures):
                    completed += 1
                    try:
                        s = fut.result()
                        if s:
                            samples.append(s)
                    except Exception as exc:
                        chunk, task_type = futures[fut]
                        logger.error(
                            "Worker failed for task=%s source=%s: %s",
                            task_type,
                            chunk.metadata.get("source", "?"),
                            exc,
                        )
                    logger.debug("[%d/%d] parallel generation done", completed, total)

        logger.info(
            "Generation complete: %d samples from %d chunk(s), %d task type(s).",
            len(samples),
            len(ingestion_results),
            len(self.config.generation.task_types),
        )
        return samples

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def generate_stream(
        self, ingestion_results: list[IngestionResult]
    ) -> Generator[tuple[DatasetSample, IngestionResult, str], None, None]:
        """
        Lazily yield ``(sample, chunk, task_type)`` tuples one at a time
        (sequential, single-threaded).

        Used by the ``MultiAgentOrchestrator`` so it can run critic scoring
        and steering on each sample *before* the next one is produced.

        Args:
            ingestion_results: Normalised input records.

        Yields:
            Tuples of ``(DatasetSample, source_chunk, task_type)``.
            When generation fails, the sample is still returned (with
            ``confidence=0.0`` and a ``_parse_error`` key in its output).
        """
        for chunk in ingestion_results:
            for task_type in self.config.generation.task_types:
                sample = self._generate_one(chunk, task_type)
                if sample is not None:
                    yield sample, chunk, task_type

    def _generate_one(
        self, chunk: IngestionResult, task_type: str
    ) -> DatasetSample | None:
        """Generate a single sample; return None only on unrecoverable error."""
        system_prompt, user_prompt = PromptTemplates.build(
            task_type, chunk.content
        )
        instruction = PromptTemplates.task_instruction(task_type)

        raw_response = ""
        confidence = 0.0
        output: dict = {}

        for attempt in range(1, self.config.llm.max_retries + 1):
            try:
                raw_response = self.llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                )
                output = self._parse_json(raw_response)
                confidence = self._estimate_confidence(output, task_type)
                break  # success
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Attempt %d/%d — JSON parse error for %s: %s",
                    attempt,
                    self.config.llm.max_retries,
                    task_type,
                    exc,
                )
                if attempt == self.config.llm.max_retries:
                    # Keep a record with raw text so error analysis can inspect it
                    output = {"_raw_response": raw_response, "_parse_error": str(exc)}
                    confidence = 0.0

        return DatasetSample.create(
            task_type=task_type,
            input_text=chunk.content,
            instruction=instruction,
            output=output,
            source=chunk.metadata.get("source", "unknown"),
            confidence=confidence,
            model=self._model_name,
            chunk_index=chunk.metadata.get("chunk_index", 0),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        Robustly parse a JSON object from LLM response text.

        Strips markdown code fences if present before parsing.
        """
        # Strip ```json ... ``` or ``` ... ``` wrappers
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL)
        return json.loads(stripped)

    @staticmethod
    def _estimate_confidence(output: dict, task_type: str) -> float:
        """
        Heuristic confidence score based on output completeness.

        Rules:
        - 1.0  if all required fields are present and non-empty
        - 0.8  if most fields are present
        - 0.5  if some fields are missing
        - 0.0  if output is empty or contains a parse error
        """
        if not output or "_parse_error" in output:
            return 0.0

        required_fields = {
            "qa": ["question", "answer", "evidence"],
            "extraction": ["entities", "relations", "key_facts"],
            "reasoning": ["reasoning_steps", "conclusion", "confidence_explanation"],
        }

        required = required_fields.get(task_type, [])
        if not required:
            return 0.75

        present = sum(
            1 for f in required if output.get(f) not in (None, "", [], {})
        )
        ratio = present / len(required)

        if ratio == 1.0:
            # Check for obviously thin content
            if task_type == "qa":
                answer_len = len(str(output.get("answer", "")))
                if answer_len < 5:
                    return 0.45
            if task_type == "reasoning":
                steps = output.get("reasoning_steps", [])
                if len(steps) < 2:
                    return 0.50
            return 0.90 + (0.10 * ratio)   # ≈ 0.90–1.0
        elif ratio >= 0.67:
            return 0.60 + (0.20 * ratio)   # ≈ 0.73–0.80
        else:
            return 0.30 * ratio             # 0.0–0.30
