"""
config.py — Central configuration for the Dataset Builder pipeline.

All tuneable parameters live here so every module imports from one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Absolute path to the project root (this file's directory)
BASE_DIR: Path = Path(__file__).parent


# ──────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Settings for the language-model backend."""

    provider: str = "ollama"          # "ollama" | "mock"
    model: str = "qwen3:4b"
    temperature: float = 0.7
    max_tokens: int = 2048
    request_timeout: int = 120
    max_retries: int = 3
    base_url: str = "http://localhost:11434"


@dataclass
class GenerationConfig:
    """Controls how synthetic samples are produced."""

    samples_per_input: int = 3          # samples per ingested chunk
    task_types: List[str] = field(
        default_factory=lambda: ["qa", "extraction", "reasoning"]
    )
    batch_size: int = 10                # LLM call batch size


@dataclass
class FilteringConfig:
    """Quality thresholds for the filtering pipeline."""

    min_confidence: float = 0.60        # discard below this
    max_duplicate_similarity: float = 0.85   # Jaccard threshold
    min_output_length: int = 20         # characters
    max_output_length: int = 8000       # characters


@dataclass
class StorageConfig:
    """File paths for all pipeline outputs."""

    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    raw_output: str = "raw_dataset.jsonl"
    annotated_output: str = "annotated_dataset.jsonl"
    filtered_output: str = "filtered_dataset.jsonl"
    metrics_report: str = "metrics_report.json"
    error_analysis: str = "error_analysis.json"
    logs_dir: str = "logs"

    def raw_path(self) -> Path:
        return self.data_dir / self.raw_output

    def annotated_path(self) -> Path:
        return self.data_dir / self.annotated_output

    def filtered_path(self) -> Path:
        return self.data_dir / self.filtered_output

    def metrics_path(self) -> Path:
        return self.data_dir / self.metrics_report

    def error_path(self) -> Path:
        return self.data_dir / self.error_analysis

    def logs_path(self) -> Path:
        return self.data_dir / self.logs_dir


# ──────────────────────────────────────────────────────────────────────────────
# Root config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Top-level configuration object passed through the pipeline."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @property
    def use_mock_llm(self) -> bool:
        """True when provider is forced to mock."""
        return self.llm.provider == "mock"

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.storage.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage.logs_path().mkdir(parents=True, exist_ok=True)


# Module-level default instance — import and mutate as needed
DEFAULT_CONFIG = Config()
