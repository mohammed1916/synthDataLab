"""
config.py — Central configuration for the Dataset Builder pipeline.

All tuneable parameters live here so every module imports from one place.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Absolute path to the project root (this file's directory)
BASE_DIR: Path = Path(__file__).parent


def _current_git_sha() -> str:
    """Return the short git SHA of HEAD, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3, cwd=str(BASE_DIR)
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Settings for the language-model backend."""

    provider: str = "ollama"          # "ollama" | "mock"
    model: str = "qwen3:8b"
    temperature: float = 0.7
    max_tokens: int = 2048
    request_timeout: int = 120
    max_retries: int = 3
    base_url: str = "http://localhost:11434"


@dataclass
class GenerationConfig:
    """Controls how synthetic samples are produced."""

    samples_per_input: int = 3          # samples per ingested chunk
    task_types: list[str] = field(
        default_factory=lambda: ["qa", "extraction", "reasoning"]
    )
    batch_size: int = 10                # LLM call batch size
    max_workers: int = 1                # parallel LLM threads (1 = sequential)


@dataclass
class EvolutionConfig:
    """Controls the Evol-Instruct prompt evolution stage."""

    enabled: bool = False              # off by default; enable with --evolve flag
    n_rounds: int = 2                  # evolution rounds
    operations: list[str] = field(
        default_factory=lambda: [
            "add_constraints",
            "deepen",
            "concretise",
            "increase_reasoning",
        ]
    )
    max_seeds_per_round: int = 50
    use_llm_evolution: bool = False    # False = template mode (no extra API calls)


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
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    # ── Run identity ──────────────────────────────────────────────────────────
    # These are populated automatically; override only for testing/reproducibility.
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    git_sha: str = field(default_factory=_current_git_sha)
    config_hash: str = ""   # populated by __post_init__

    def __post_init__(self) -> None:
        if not self.config_hash:
            self.config_hash = self._compute_config_hash()

    def _compute_config_hash(self) -> str:
        """SHA-256 of the deterministic config (excludes run_id / git_sha)."""
        snapshot = {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "generation": {
                "samples_per_input": self.generation.samples_per_input,
                "task_types": sorted(self.generation.task_types),
                "max_workers": self.generation.max_workers,
            },
            "filtering": {
                "min_confidence": self.filtering.min_confidence,
                "max_duplicate_similarity": self.filtering.max_duplicate_similarity,
            },
        }
        raw = json.dumps(snapshot, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def use_mock_llm(self) -> bool:
        """True when provider is forced to mock."""
        return self.llm.provider == "mock"

    def validate(self) -> None:
        """
        Validate that all config values are internally consistent.

        Raises ``ValueError`` with a descriptive message on the first violation.
        """
        errs: list[str] = []

        # Threshold ordering
        if not (0.0 <= self.filtering.min_confidence <= 1.0):
            errs.append(
                f"filtering.min_confidence={self.filtering.min_confidence} must be in [0, 1]"
            )
        if not (0.0 < self.filtering.max_duplicate_similarity <= 1.0):
            errs.append(
                f"filtering.max_duplicate_similarity={self.filtering.max_duplicate_similarity} must be in (0, 1]"
            )
        if self.filtering.min_output_length < 0:
            errs.append("filtering.min_output_length must be ≥ 0")
        if self.filtering.max_output_length <= self.filtering.min_output_length:
            errs.append(
                "filtering.max_output_length must be > filtering.min_output_length"
            )

        # Workers
        if self.generation.max_workers < 1:
            errs.append("generation.max_workers must be ≥ 1")

        # Storage directory writable
        try:
            data_dir = Path(self.storage.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            test_file = data_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except OSError as exc:
            errs.append(f"storage.data_dir '{self.storage.data_dir}' is not writable: {exc}")

        # Disk space guard — require ≥ 500 MB free
        try:
            import shutil as _shutil
            free_mb = _shutil.disk_usage(self.storage.data_dir).free / (1024 ** 2)
            if free_mb < 500:
                errs.append(
                    f"Less than 500 MB free in '{self.storage.data_dir}' ({free_mb:.0f} MB). "
                    "Free up disk space before running the pipeline."
                )
        except OSError:
            pass  # if we can't stat, the writable check above will catch the real problem

        if errs:
            raise ValueError("Config validation errors:\n  " + "\n  ".join(errs))

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.storage.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage.logs_path().mkdir(parents=True, exist_ok=True)

    def run_dir(self) -> Path:
        """Versioned output directory for this specific run."""
        d = self.storage.data_dir / "runs" / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run_manifest(self) -> dict:
        """Metadata dict persisted alongside every versioned run."""
        return {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "provider": self.llm.provider,
            "model": self.llm.model,
            "task_types": self.generation.task_types,
        }


# Module-level default instance — import and mutate as needed
DEFAULT_CONFIG = Config()
