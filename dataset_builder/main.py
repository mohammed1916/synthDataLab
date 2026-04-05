"""
main.py — CLI entry point for the Dataset Builder pipeline.

Commands
--------
  run-all          Run the complete pipeline end-to-end (recommended for demo)
  ingest           Normalise raw inputs into intermediate representation
  generate         Synthesise dataset samples from ingested content
  generate-agent   Multi-agent generation with critic scoring + human steering
  validate         Apply schema + rule validation to a raw dataset file
  filter           Run quality filtering on a validated (annotated) dataset
  evaluate         Compute and compare metrics for raw vs filtered datasets
  analyze          Run error analysis on an annotated dataset
  evolve           Run Evol-Instruct prompt evolution on a seed file
  guidelines       Print human-in-the-loop annotation guidelines

Usage examples
--------------
  python main.py run-all
  python main.py run-all --input data/sample_inputs/sample_articles.json
  python main.py generate-agent --mock --steering review-low
  python main.py generate-agent --mock --steering auto --threshold 0.65
  python main.py evolve data/sample_inputs/sample_text.txt --rounds 2
  python main.py generate --help
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import click

# ── Ensure the project root is on sys.path ────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.error_analyzer import ErrorAnalyzer
from config import Config
from evaluation.metrics import compute_metrics
from evaluation.reporter import MetricsReporter
from filtering.fingerprint_store import FingerprintStore
from filtering.pipeline import FilteringPipeline
from generation.evolver import EvolveConfig, PromptEvolver
from generation.generator import DatasetGenerator
from ingestion.ingestor import IngestionResult, Ingestor
from schema.dataset_schema import DatasetSample
from validation.annotation import AnnotatedSample, AnnotationLabel
from validation.llm_reviewer import LLMReviewer
from validation.rule_validator import RuleValidator, annotation_guidelines

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset_builder")


def _setup_file_logging(log_dir: Path, session_id: str) -> None:
    """Add a rotating file handler so every run is persisted to disk."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline_{session_id}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logging.getLogger().addHandler(fh)
    logger.info("File logging active → %s", log_path)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_path(cfg: Config) -> Path:
    return cfg.storage.data_dir / "logs" / "checkpoint.json"


def _load_checkpoint(cfg: Config) -> dict[str, Any]:
    p = _checkpoint_path(cfg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"completed": [], "session_id": None}


def _save_checkpoint(cfg: Config, checkpoint: dict[str, Any]) -> None:
    p = _checkpoint_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")


def _clear_checkpoint(cfg: Config) -> None:
    p = _checkpoint_path(cfg)
    with contextlib.suppress(FileNotFoundError):
        p.unlink()


import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(mock: bool = False, skip_preflight: bool = False) -> Config:
    from datetime import timezone
    cfg = Config()
    if mock:
        cfg.llm.provider = "mock"
    cfg.ensure_dirs()
    session_id = datetime.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _setup_file_logging(cfg.storage.data_dir / "logs", session_id)

    # Fail fast: verify Ollama is reachable before the first LLM call
    # (skipped for health-check command which does its own checks)
    if not cfg.use_mock_llm and not skip_preflight:
        try:
            from generation.llm_client import OllamaClient
            _health_client = OllamaClient(
                model=cfg.llm.model,
                base_url=cfg.llm.base_url,
            )
            _health_client.health_check()
        except RuntimeError as exc:
            _echo(f"[bold red]\u26a0  Ollama pre-flight check failed:[/bold red] {exc}")
            _echo("[dim]Tip: pass --mock to use the offline mock LLM.[/dim]")
            import sys as _sys
            _sys.exit(1)

    return cfg


def _save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Atomically write records to a JSONL file (write-tmp → fsync → rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)   # atomic on POSIX; near-atomic on Windows
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    _echo(f"  Saved {len(records)} records → {path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _echo(msg: str) -> None:
    """Print with rich if available, otherwise plain."""
    try:
        from rich.console import Console
        Console().print(msg)
    except ImportError:
        click.echo(msg)


def _header(title: str) -> None:
    try:
        from rich.console import Console
        from rich.rule import Rule
        Console().print(Rule(f"[bold cyan]{title}[/bold cyan]"))
    except ImportError:
        click.echo(f"\n{'─' * 60}")
        click.echo(f"  {title}")
        click.echo(f"{'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps (reusable functions)
# ─────────────────────────────────────────────────────────────────────────────

def step_ingest(cfg: Config, input_path: str | None = None) -> list[IngestionResult]:
    """Ingest inputs and return IngestionResult list."""
    ingestor = Ingestor()
    results: list[IngestionResult] = []

    if input_path:
        p = Path(input_path)
        if p.suffix.lower() == ".json":
            results = ingestor.ingest_json(str(p))
        else:
            results = ingestor.ingest_file(str(p))
    else:
        # Fall back to bundled sample articles
        default_path = _PROJECT_ROOT / "data" / "sample_inputs" / "sample_articles.json"
        if default_path.exists():
            results = ingestor.ingest_json(str(default_path))
        else:
            _echo("[yellow]No input specified and no sample file found. Using inline demo text.[/yellow]")
            results = ingestor.ingest_text(DEMO_TEXT, source_name="inline_demo")

    _echo(f"  Ingested [bold]{len(results)}[/bold] chunk(s).")
    return results


def step_generate(
    cfg: Config, ingestion_results: list[IngestionResult]
) -> list[DatasetSample]:
    """Generate dataset samples from ingestion results. Does NOT save to disk."""
    generator = DatasetGenerator(cfg)
    samples = generator.generate_from_ingestion(ingestion_results)
    _echo(
        f"  Generated [bold]{len(samples)}[/bold] raw samples "
        f"({'mock LLM' if cfg.use_mock_llm else cfg.llm.model})"
    )
    return samples


def step_validate(
    cfg: Config, samples: list[DatasetSample]
) -> list[AnnotatedSample]:
    """Validate samples (rule-based + optional LLM review)."""
    rule_validator = RuleValidator(min_confidence=cfg.filtering.min_confidence)
    sample_dicts = [s.to_dict() for s in samples]
    annotated = rule_validator.validate_batch(sample_dicts)

    # LLM reviewer on FIX_REQUIRED samples
    # Pass a real OllamaClient when Ollama is available; fall back to mock heuristics.
    _reviewer_client = None
    if not cfg.use_mock_llm:
        try:
            from generation.llm_client import OllamaClient
            _reviewer_client = OllamaClient(
                model=cfg.llm.model,
                base_url=cfg.llm.base_url,
                timeout=cfg.llm.request_timeout,
                max_retries=cfg.llm.max_retries,
            )
        except Exception:
            pass  # fall back to heuristic mode silently
    llm_reviewer = LLMReviewer(llm_client=_reviewer_client)
    annotated = llm_reviewer.review_batch(annotated)

    accepted = sum(1 for a in annotated if a.is_accepted)
    fix_req = sum(1 for a in annotated if a.label == AnnotationLabel.FIX_REQUIRED)
    rejected = sum(1 for a in annotated if a.label == AnnotationLabel.REJECT)

    _echo(
        f"  Validation: [green]{accepted} ACCEPT[/green]  "
        f"[yellow]{fix_req} FIX_REQUIRED[/yellow]  "
        f"[red]{rejected} REJECT[/red]"
    )

    _save_jsonl(
        [a.to_dict() for a in annotated],
        cfg.storage.annotated_path(),
    )
    return annotated


def step_filter(
    cfg: Config, annotated: list[AnnotatedSample]
) -> tuple[list[AnnotatedSample], dict]:
    """Run quality filtering pipeline."""
    pipeline = FilteringPipeline(cfg.filtering)
    filtered, filter_report = pipeline.run(annotated)

    _save_jsonl(
        [a.sample for a in filtered],
        cfg.storage.filtered_path(),
    )
    _echo(
        f"  Filtering: [bold]{len(filtered)}[/bold] samples retained "
        f"({filter_report.overall_retention_rate:.1%} retention rate)"
    )
    for stage in filter_report.stages:
        if stage.removed_count > 0:
            _echo(
                f"    [{stage.stage_name}] removed {stage.removed_count} "
                f"({stage.removal_rate:.1%})"
            )
    return filtered, filter_report.to_dict()


def step_evaluate(
    cfg: Config,
    raw_samples: list[DatasetSample],
    filtered_samples: list[AnnotatedSample],
    filter_report: dict,
) -> None:
    """Compute and report quality metrics."""
    raw_dicts = [s.to_dict() for s in raw_samples]
    filtered_dicts = [a.sample for a in filtered_samples]

    raw_metrics = compute_metrics(raw_dicts)
    filtered_metrics = compute_metrics(filtered_dicts)

    reporter = MetricsReporter(output_path=cfg.storage.metrics_path())
    reporter.report(raw_metrics, filtered_metrics, filter_report)


def step_analyze(
    cfg: Config, annotated: list[AnnotatedSample]
) -> None:
    """Run error analysis and save report."""
    analyzer = ErrorAnalyzer()
    report = analyzer.analyze(annotated)
    analyzer.save_report(report, cfg.storage.error_path())
    analyzer.print_summary(report)
    _echo(f"  Error analysis saved → {cfg.storage.error_path()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI definition
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Dataset Builder — Multimodal Dataset Generation, Validation & Evaluation."""


@cli.command("run-all")
@click.option(
    "--input", "input_path",
    default=None,
    help="Path to a text file or JSON articles file. Defaults to bundled sample data.",
)
@click.option(
    "--mock/--no-mock",
    default=False,
    show_default=True,
    help="Use mock LLM instead of Ollama (no Ollama server needed).",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    show_default=True,
    help="Resume a previously interrupted run from the last completed step.",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=click.IntRange(min=1, max=16),
    help="Number of parallel LLM generation threads (default 1 = sequential).",
)
@click.option(
    "--agent/--no-agent",
    default=False,
    show_default=True,
    help="Use multi-agent orchestrator (CriticAgent + steering) for generation.",
)
@click.option(
    "--steering",
    type=click.Choice(["auto", "review-low", "review-all"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Human steering mode when --agent is set.",
)
@click.option(
    "--threshold",
    default=0.70,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Critic pass threshold for --agent mode.",
)
@click.option(
    "--force/--no-force",
    default=False,
    show_default=True,
    help="Skip cross-run dedup check and re-process all samples.",
)
@click.option(
    "--reset-fingerprints",
    is_flag=True,
    default=False,
    help="Wipe the fingerprint store before running (start dedup fresh).",
)
def run_all(
    input_path: str | None,
    mock: bool,
    resume: bool,
    workers: int,
    agent: bool,
    steering: str,
    threshold: float,
    force: bool,
    reset_fingerprints: bool,
):
    """Run the complete pipeline: ingest → generate → validate → filter → evaluate → analyze."""
    import time as _time

    cfg = _load_config(mock=mock)
    cfg.generation.max_workers = workers

    # Validate config bounds and write access
    try:
        cfg.validate()
    except ValueError as exc:
        _echo(f"[bold red]Config validation failed:[/bold red] {exc}")
        sys.exit(1)

    # Create versioned run directory and write manifest
    run_dir = cfg.run_dir()
    (run_dir / "manifest.json").write_text(
        json.dumps(cfg.run_manifest(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Cross-run fingerprint store
    fp_path = cfg.storage.data_dir / "fingerprints.json"
    if reset_fingerprints and fp_path.exists():
        fp_path.unlink()
        _echo("  [yellow]Fingerprint store wiped — dedup starts fresh.[/yellow]")
    fp_store = FingerprintStore(fp_path)

    _echo(
        f"\n[bold green]Dataset Builder[/bold green] "
        f"({'Mock LLM' if cfg.use_mock_llm else cfg.llm.model})"
        f" | workers={workers} | run_id={cfg.run_id}\n"
    )

    ckpt = _load_checkpoint(cfg) if resume else {"completed": [], "session_id": None}
    done = set(ckpt.get("completed", []))
    if resume and done:
        _echo(f"[yellow]Resuming — already completed: {sorted(done)}[/yellow]")

    def _mark(step: str) -> None:
        done.add(step)
        ckpt["completed"] = sorted(done)
        _save_checkpoint(cfg, ckpt)

    pipeline_start = _time.monotonic()

    def _step_header(n: int, total: int, title: str) -> float:
        _header(f"{n} / {total}  {title}")
        return _time.monotonic()

    def _step_footer(t0: float) -> None:
        _echo(f"  [dim]↳ {_time.monotonic() - t0:.1f}s[/dim]")

    # 1. INGEST
    t0 = _step_header(1, 6, "INGESTION")
    if "ingest" in done:
        ingestion_results = [
            IngestionResult(**r) for r in _load_jsonl(cfg.storage.data_dir / "ingested.jsonl")
        ]
        _echo(f"  [dim]Skipped (resumed) — {len(ingestion_results)} chunk(s)[/dim]")
    else:
        ingestion_results = step_ingest(cfg, input_path)
        _save_jsonl([r.to_dict() for r in ingestion_results], cfg.storage.data_dir / "ingested.jsonl")
        _mark("ingest")
    _step_footer(t0)

    # 2. GENERATE
    t0 = _step_header(2, 6, "GENERATION" + (" (Multi-Agent)" if agent else ""))
    if "generate" in done:
        raw_samples = [DatasetSample.from_dict(r) for r in _load_jsonl(cfg.storage.raw_path())]
        _echo(f"  [dim]Skipped (resumed) — {len(raw_samples)} sample(s)[/dim]")
    else:
        if agent:
            from generation.orchestrator import (
                MultiAgentOrchestrator,
                OrchestratorConfig,
                SteeringMode,
            )
            orch_cfg = OrchestratorConfig(
                steering_mode=SteeringMode(steering),
                critic_pass_threshold=threshold,
                critic_review_threshold=max(0.0, threshold - 0.25),
                show_dashboard=not os.environ.get("CI"),
            )
            orch = MultiAgentOrchestrator(config=cfg, orch_config=orch_cfg)
            orch_result = orch.run(ingestion_results)
            if orch_result.aborted:
                _echo("[yellow]Orchestration aborted — saving collected samples.[/yellow]")
            raw_samples = [DatasetSample.from_dict(d) for d in (orch_result.accepted + orch_result.fix_required)]
            if orch_result.critic_scores:
                _save_jsonl([cs.to_dict() for cs in orch_result.critic_scores], run_dir / "critic_scores.jsonl")
        else:
            raw_samples = step_generate(cfg, ingestion_results)

        # Always persist the full generated set as the raw artifact for this run
        _save_jsonl([s.to_dict() for s in raw_samples], cfg.storage.raw_path())

        # --- Cross-run dedup (in-memory only; raw file is unchanged) ---
        if not force and raw_samples:
            raw_dicts = [s.to_dict() for s in raw_samples]
            new_dicts, cross_dupes = fp_store.filter_new(raw_dicts)
            if cross_dupes:
                _echo(
                    f"  [yellow]Cross-run dedup:[/yellow] {len(cross_dupes)} already in prior runs, "
                    f"[green]{len(new_dicts)} truly new[/green]."
                )
            if not new_dicts:
                _echo(
                    f"\n[bold yellow]⚠  Nothing new to process — all {len(raw_samples)} sample(s) were "
                    f"already seen in a previous run.[/bold yellow]\n"
                    f"[dim]Tip: use --force to re-process anyway, or --reset-fingerprints to start fresh.[/dim]\n"
                )
                # Persist fingerprints (they were already saved before, nothing changes)
                _clear_checkpoint(cfg)
                sys.exit(0)
            # Only proceed with new samples downstream (don't re-overwrite raw file)
            raw_samples = [DatasetSample.from_dict(d) for d in new_dicts]
        elif force:
            _echo("  [dim]--force: skipping cross-run dedup.[/dim]")

        _mark("generate")
    _step_footer(t0)

    # Guard: if somehow 0 samples reach here, fail loudly rather than silently
    if not raw_samples:
        _echo("[bold red]✗  0 samples after generation step — cannot continue.[/bold red]")
        sys.exit(1)

    # 3. VALIDATE
    t0 = _step_header(3, 6, "VALIDATION  (Rule + HITL Simulation)")
    if "validate" in done:
        annotated_dicts = _load_jsonl(cfg.storage.annotated_path())
        annotated = []
        for r in annotated_dicts:
            ann_data = r.pop("annotation", {})
            ann = AnnotatedSample.from_sample_dict(r)
            ann.label = AnnotationLabel(ann_data.get("label", "ACCEPT"))
            annotated.append(ann)
        _echo(f"  [dim]Skipped (resumed) — {len(annotated)} annotated[/dim]")
    else:
        annotated = step_validate(cfg, raw_samples)
        _mark("validate")
    _step_footer(t0)

    # 4. FILTER
    t0 = _step_header(4, 6, "FILTERING")
    if "filter" in done:
        filtered_dicts = _load_jsonl(cfg.storage.filtered_path())
        filtered = [AnnotatedSample.from_sample_dict(r) for r in filtered_dicts]
        filter_report: dict = {}
        _echo(f"  [dim]Skipped (resumed) — {len(filtered)} samples retained[/dim]")
    else:
        filtered, filter_report = step_filter(cfg, annotated)
        _mark("filter")
    _step_footer(t0)

    # 5. EVALUATE
    t0 = _step_header(5, 6, "EVALUATION  (Metrics)")
    step_evaluate(cfg, raw_samples, filtered, filter_report)
    _step_footer(t0)

    # 6. ANALYZE
    t0 = _step_header(6, 6, "ERROR ANALYSIS")
    step_analyze(cfg, annotated)
    _step_footer(t0)

    _clear_checkpoint(cfg)

    # Persist fingerprints only after full successful run
    if not force:
        fp_store.save()

    # Copy artifacts to versioned run directory
    import shutil as _shutil
    for src in [
        cfg.storage.raw_path(),
        cfg.storage.filtered_path(),
        cfg.storage.annotated_path(),
        cfg.storage.metrics_path(),
        cfg.storage.error_path(),
    ]:
        if src.exists():
            _shutil.copy2(src, run_dir / src.name)

    # Update data/latest symlink
    latest = cfg.storage.data_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(Path("runs") / cfg.run_id)
    except OSError:
        pass

    elapsed_total = _time.monotonic() - pipeline_start

    _header("DONE")
    _echo(
        f"\n  [bold]Run ID[/bold]          {cfg.run_id}  [dim](git {cfg.git_sha})[/dim]\n"
        f"  [bold]Total time[/bold]      {elapsed_total:.1f}s\n"
    )
    _echo(f"  Raw dataset      → [cyan]{cfg.storage.raw_path()}[/cyan]")
    _echo(f"  Annotated        → [cyan]{cfg.storage.annotated_path()}[/cyan]")
    _echo(f"  Filtered dataset → [cyan]{cfg.storage.filtered_path()}[/cyan]")
    _echo(f"  Metrics report   → [cyan]{cfg.storage.metrics_path()}[/cyan]")
    _echo(f"  Error analysis   → [cyan]{cfg.storage.error_path()}[/cyan]")
    _echo(f"  Versioned run    → [cyan]{run_dir}[/cyan]")
    _echo(f"  Latest symlink   → [cyan]{latest}[/cyan]\n")


@cli.command("ingest")
@click.argument("input_path")
@click.option("--mock/--no-mock", default=False, show_default=True)
def ingest_cmd(input_path: str, mock: bool):
    """Ingest a text or JSON file and save normalised chunks."""
    cfg = _load_config(mock=mock)
    _header("INGESTION")
    results = step_ingest(cfg, input_path)
    out = cfg.storage.data_dir / "ingested.jsonl"
    _save_jsonl([r.to_dict() for r in results], out)


@cli.command("generate")
@click.option("--input", "input_path", default=None, help="Input file path.")
@click.option("--mock/--no-mock", default=False, show_default=True)
def generate_cmd(input_path: str | None, mock: bool):
    """Generate dataset samples from input data."""
    cfg = _load_config(mock=mock)
    _header("GENERATION")
    ingestion_results = step_ingest(cfg, input_path)
    samples = step_generate(cfg, ingestion_results)
    _save_jsonl([s.to_dict() for s in samples], cfg.storage.raw_path())


@cli.command("validate")
@click.option(
    "--dataset",
    "dataset_path",
    default=None,
    help="Path to a JSONL raw dataset. Defaults to data/raw_dataset.jsonl.",
)
@click.option("--mock/--no-mock", default=False, show_default=True)
def validate_cmd(dataset_path: str | None, mock: bool):
    """Validate a raw dataset JSONL file."""
    cfg = _load_config(mock=mock)
    _header("VALIDATION")
    path = Path(dataset_path) if dataset_path else cfg.storage.raw_path()
    if not path.exists():
        _echo(f"[red]Dataset not found: {path}[/red]")
        sys.exit(1)
    records = _load_jsonl(path)
    samples = [DatasetSample.from_dict(r) for r in records]
    step_validate(cfg, samples)


@cli.command("filter")
@click.option("--mock/--no-mock", default=False, show_default=True)
def filter_cmd(mock: bool):
    """Apply quality filtering to the annotated dataset."""
    cfg = _load_config(mock=mock)
    _header("FILTERING")
    path = cfg.storage.annotated_path()
    if not path.exists():
        _echo(f"[red]Annotated dataset not found: {path}. Run 'validate' first.[/red]")
        sys.exit(1)
    records = _load_jsonl(path)
    annotated = []
    for r in records:
        ann_data = r.pop("annotation", {})
        ann = AnnotatedSample.from_sample_dict(r)
        ann.label = AnnotationLabel(ann_data.get("label", "ACCEPT"))
        annotated.append(ann)
    step_filter(cfg, annotated)


@cli.command("evaluate")
@click.option("--mock/--no-mock", default=False, show_default=True)
def evaluate_cmd(mock: bool):
    """Compute and compare metrics for raw vs filtered datasets."""
    cfg = _load_config(mock=mock)
    _header("EVALUATION")
    raw_path = cfg.storage.raw_path()
    filtered_path = cfg.storage.filtered_path()
    if not raw_path.exists() or not filtered_path.exists():
        _echo("[red]Run 'run-all' or generate + validate + filter first.[/red]")
        sys.exit(1)
    raw_dicts = _load_jsonl(raw_path)
    filtered_dicts = _load_jsonl(filtered_path)
    raw_metrics = compute_metrics(raw_dicts)
    filtered_metrics = compute_metrics(filtered_dicts)
    reporter = MetricsReporter(output_path=cfg.storage.metrics_path())
    reporter.report(raw_metrics, filtered_metrics, {})


@cli.command("analyze")
@click.option("--mock/--no-mock", default=False, show_default=True)
def analyze_cmd(mock: bool):
    """Run error analysis on the annotated dataset."""
    cfg = _load_config(mock=mock)
    _header("ERROR ANALYSIS")
    path = cfg.storage.annotated_path()
    if not path.exists():
        _echo(f"[red]Annotated dataset not found: {path}. Run 'validate' first.[/red]")
        sys.exit(1)
    records = _load_jsonl(path)
    annotated = []
    for r in records:
        ann_data = r.pop("annotation", {})
        ann = AnnotatedSample.from_sample_dict(r)
        ann.label = AnnotationLabel(ann_data.get("label", "ACCEPT"))
        for reason in ann_data.get("rejection_reasons", []):

            class _R:
                def __init__(self, code, message):
                    self.code = code
                    self.message = message
                def to_dict(self):
                    return {"code": self.code, "message": self.message}

            ann.rejection_reasons.append(_R(reason["code"], reason["message"]))
        annotated.append(ann)
    step_analyze(cfg, annotated)


@cli.command("guidelines")
def guidelines_cmd():
    """Print the human-in-the-loop annotation guidelines."""
    click.echo(annotation_guidelines())


@cli.command("evolve")
@click.argument("input_path")
@click.option(
    "--rounds", default=2, show_default=True,
    help="Number of Evol-Instruct evolution rounds.",
)
@click.option(
    "--ops",
    default="add_constraints,deepen,concretise,increase_reasoning",
    show_default=True,
    help="Comma-separated list of evolution operations to apply.",
)
@click.option(
    "--mock/--no-mock", default=False, show_default=True,
    help="Use mock LLM for evolution (always uses template mode regardless).",
)
@click.option(
    "--output", "output_path", default=None,
    help="Path to write evolved prompts JSONL. Default: data/evolved_prompts.jsonl",
)
def evolve_cmd(
    input_path: str,
    rounds: int,
    ops: str,
    mock: bool,
    output_path: str | None,
):
    """
    Run Evol-Instruct prompt evolution on seed instructions from INPUT_PATH.

    Reads the file as plain text lines (one instruction per line) or as
    JSON articles (extracts the 'content' field). Writes evolved prompts to
    JSONL with full lineage metadata.

    Example::

        python main.py evolve data/sample_inputs/sample_text.txt --rounds 3
    """
    cfg = _load_config(mock=mock)
    _header("EVOL-INSTRUCT PROMPT EVOLUTION")

    # Load seeds from file
    p = Path(input_path)
    if not p.exists():
        _echo(f"[red]Input file not found: {p}[/red]")
        sys.exit(1)

    seeds: list[str] = []
    if p.suffix.lower() == ".json":
        raw_json = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw_json, list):
            for item in raw_json:
                if isinstance(item, dict) and "content" in item:
                    seeds.append(item["content"][:500])
                elif isinstance(item, str):
                    seeds.append(item)
        else:
            seeds.append(str(raw_json)[:500])
    else:
        text = p.read_text(encoding="utf-8")
        import re as _re
        lines = [ln.strip() for ln in _re.split(r"[.?!]\s+", text) if len(ln.strip()) > 20]
        seeds = lines[:50]

    if not seeds:
        _echo("[yellow]No usable seed prompts found in the input file.[/yellow]")
        sys.exit(0)

    _echo(f"  Loaded [bold]{len(seeds)}[/bold] seed prompt(s) from {p.name}")

    operations = [o.strip() for o in ops.split(",") if o.strip()]
    evolve_cfg = EvolveConfig(
        n_rounds=rounds,
        operations=operations,
        use_llm_evolution=False,  # always template mode for now
    )
    evolver = PromptEvolver(config=evolve_cfg, seed=42)
    evolved = evolver.evolve(seeds, n_rounds=rounds)

    surviving = [ep for ep in evolved if not ep.discarded]
    discarded = [ep for ep in evolved if ep.discarded]

    _echo(
        f"  Evolution complete: [bold]{len(evolved)}[/bold] total evolved, "
        f"[green]{len(surviving)}[/green] surviving, "
        f"[yellow]{len(discarded)}[/yellow] discarded"
    )

    # Stats by operation
    from collections import Counter
    op_counts = Counter(ep.operation for ep in surviving)
    for op, count in op_counts.most_common():
        _echo(f"    {op:<26} {count} prompt(s)")

    # Show top-3 most complex evolved prompts
    top3 = sorted(surviving, key=lambda e: e.complexity_score, reverse=True)[:3]
    _echo("\n  Top-3 most complex evolved prompts:")
    for i, ep in enumerate(top3, 1):
        _echo(
            f"    [{i}] [{ep.operation}] score={ep.complexity_score:.3f}\n"
            f"        {ep.prompt[:140]}{'...' if len(ep.prompt) > 140 else ''}"
        )

    # Write output
    out_path = Path(output_path) if output_path else cfg.storage.data_dir / "evolved_prompts.jsonl"
    _save_jsonl([ep.to_dict() for ep in evolved], out_path)
    _echo(f"\n  Evolved prompts saved → [cyan]{out_path}[/cyan]")


@cli.command("generate-agent")
@click.option(
    "--input", "input_path",
    default=None,
    help="Path to a text file or JSON articles file. Defaults to bundled sample data.",
)
@click.option(
    "--mock/--no-mock",
    default=False,
    show_default=True,
    help="Use mock LLM instead of Ollama (no Ollama server needed).",
)
@click.option(
    "--steering",
    type=click.Choice(["auto", "review-low", "review-all"], case_sensitive=False),
    default="auto",
    show_default=True,
    help=(
        "Human steering mode.  "
        "auto=critic decides everything; "
        "review-low=human reviews low-scoring samples; "
        "review-all=human reviews every sample."
    ),
)
@click.option(
    "--threshold",
    default=0.70,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Critic pass threshold (composite score ≥ this → ACCEPT in auto mode).",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=click.IntRange(min=1, max=16),
    help="Number of parallel LLM generation threads.",
)
@click.option(
    "--output", "output_path",
    default=None,
    help="Path to write the accepted samples JSONL. Default: data/raw_dataset.jsonl",
)
@click.option(
    "--no-dashboard",
    is_flag=True,
    default=False,
    help="Disable the live Rich dashboard (useful for CI / log-only runs).",
)
def generate_agent_cmd(
    input_path: str | None,
    mock: bool,
    steering: str,
    threshold: float,
    workers: int,
    output_path: str | None,
    no_dashboard: bool,
):
    """
    Run multi-agent generation with critic scoring + optional human steering.

    The pipeline:

    \b
      Generator Agent  →  CriticAgent (4-axis scoring)
                       →  Steering Gate (auto / review-low / review-all)
                       →  LiveMetricsTracker (Rich dashboard)
                       →  Save accepted samples to JSONL

    Examples::

        # Fully automatic (critic decides, rich dashboard visible)
        python main.py generate-agent --mock

        # Human reviews samples below the 0.70 threshold
        python main.py generate-agent --mock --steering review-low

        # Human reviews every sample
        python main.py generate-agent --mock --steering review-all --threshold 0.6

        # Use real Ollama backend with 4 workers
        python main.py generate-agent --workers 4 --steering auto
    """
    from generation.orchestrator import MultiAgentOrchestrator, OrchestratorConfig, SteeringMode

    cfg = _load_config(mock=mock)
    cfg.generation.max_workers = workers

    _echo(
        f"\n[bold green]Multi-Agent Generation[/bold green]"
        f" ({'Mock LLM' if cfg.use_mock_llm else cfg.llm.model})"
        f" | steering={steering}  threshold={threshold}  workers={workers}\n"
    )

    # Ingest
    _header("INGESTION")
    ingestion_results = step_ingest(cfg, input_path)

    # Build orchestrator config
    orch_cfg = OrchestratorConfig(
        steering_mode=SteeringMode(steering),
        critic_pass_threshold=threshold,
        critic_review_threshold=max(0.0, threshold - 0.25),
        show_dashboard=not no_dashboard,
    )

    # Run orchestration
    _header("MULTI-AGENT GENERATION + CRITIC")
    orch = MultiAgentOrchestrator(config=cfg, orch_config=orch_cfg)
    result = orch.run(ingestion_results)

    if result.aborted:
        _echo("[yellow]Run was aborted by user — saving collected samples.[/yellow]")

    # Save accepted samples
    out = Path(output_path) if output_path else cfg.storage.raw_path()
    all_output = result.accepted + result.fix_required
    _save_jsonl(all_output, out)

    _echo(f"\n  Accepted + Fix-required → [cyan]{out}[/cyan]")
    _echo(f"  Rejected               → discarded ({len(result.rejected)} samples)")

    if result.metrics_snapshot:
        _header("METRICS SNAPSHOT (accepted samples)")
        snap = result.metrics_snapshot
        _echo(f"  Schema validity       {snap.get('schema_validity_rate', 0):.1%}")
        _echo(f"  Task consistency      {snap.get('task_consistency_score', 0):.1%}")
        _echo(f"  Diversity score       {snap.get('diversity_score', 0):.3f}")
        _echo(f"  Collapse risk         {snap.get('collapse_risk_score', 0):.3f}  "
              f"({snap.get('collapse_warning') or 'OK'})")


@cli.command("list-runs")
def list_runs_cmd():
    """List all versioned pipeline runs with their manifests."""
    cfg = Config()
    runs_dir = cfg.storage.data_dir / "runs"
    if not runs_dir.exists():
        _echo("[yellow]No runs found.[/yellow]")
        return
    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        _echo("[yellow]No runs found.[/yellow]")
        return
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="Pipeline Runs", show_header=True, header_style="bold cyan")
        table.add_column("Run ID", style="cyan")
        table.add_column("Git SHA")
        table.add_column("Model")
        table.add_column("Tasks")
        table.add_column("Config Hash")
        for d in run_dirs:
            manifest_path = d / "manifest.json"
            if manifest_path.exists():
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
                table.add_row(
                    m.get("run_id", d.name),
                    m.get("git_sha", "?"),
                    m.get("model", "?"),
                    ", ".join(m.get("task_types", [])),
                    m.get("config_hash", "?"),
                )
            else:
                table.add_row(d.name, "?", "?", "?", "?")
        console.print(table)
    except ImportError:
        for d in run_dirs:
            _echo(f"  {d.name}")


@cli.command("export")
@click.option(
    "--dataset",
    "dataset_path",
    default=None,
    help="Path to the filtered JSONL to export. Default: data/filtered_dataset.jsonl",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["argilla", "labelstudio"], case_sensitive=False),
    default="argilla",
    show_default=True,
    help="Export format.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    help="Output file path. Default: data/export_<format>.jsonl",
)
def export_cmd(dataset_path: str | None, fmt: str, output_path: str | None):
    """Export the dataset to Argilla or Label Studio annotation format."""
    from evaluation.exporter import export_argilla, export_labelstudio

    cfg = Config()
    src = Path(dataset_path) if dataset_path else cfg.storage.filtered_path()
    if not src.exists():
        _echo(f"[red]Dataset not found: {src}[/red]")
        sys.exit(1)

    records = _load_jsonl(src)
    out = Path(output_path) if output_path else cfg.storage.data_dir / f"export_{fmt}.jsonl"

    if fmt == "argilla":
        exported = export_argilla(records)
    else:
        exported = export_labelstudio(records)

    _save_jsonl(exported, out)
    _echo(f"  Exported {len(exported)} records ({fmt}) → [cyan]{out}[/cyan]")


# ─────────────────────────────────────────────────────────────────────────────
# Inline demo text (fallback when no sample files exist)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to
the natural intelligence displayed by animals including humans.  AI research has been
defined as the field of study of intelligent agents, which refers to any system that
perceives its environment and takes actions that maximize its chance of achieving its goals.

Machine learning (ML) is a subset of AI that provides systems the ability to automatically
learn and improve from experience without being explicitly programmed.  Deep learning, a
further subset of ML, uses neural networks with many layers to extract higher-level
features from raw input data.  The transformer architecture, introduced in the 2017 paper
"Attention is All You Need," became the foundation for large language models such as GPT and BERT.

Climate change refers to long-term shifts in temperatures and weather patterns.  Since the
1800s, human activities have been the main driver of climate change, primarily due to the
burning of fossil fuels like coal, oil, and gas.  The Intergovernmental Panel on Climate
Change (IPCC) concluded in its 2021 report that global surface temperature has increased
faster since 1970 than in any other 50-year period over at least the last 2,000 years.

The human genome contains approximately 3 billion base pairs of DNA, encoding roughly
20,000–25,000 protein-coding genes.  The Human Genome Project, completed in 2003, was
an international scientific research project that produced a complete sequence of the
human genome.  Advances in genomics have accelerated drug discovery, personalised medicine,
and our understanding of hereditary diseases.

Quantum computing leverages quantum mechanical phenomena such as superposition and
entanglement to perform computations.  A quantum bit, or qubit, can exist in a
superposition of the 0 and 1 states simultaneously, unlike a classical bit.
Quantum computers are expected to solve certain classes of problems—such as integer
factorisation and database searching—exponentially faster than classical computers.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Health check command
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("health-check")
@click.option("--mock/--no-mock", default=False, show_default=True,
              help="Skip LLM connectivity check (useful in CI with no Ollama server).")
def health_check_cmd(mock: bool):
    """Check Ollama connectivity, disk space, config validity, and Python deps."""
    import shutil as _shutil

    from rich.table import Table as _Table

    cfg = _load_config(mock=mock, skip_preflight=True)

    checks: list[tuple[str, bool, str]] = []  # (name, passed, detail)

    # 1. Config validity
    try:
        cfg.validate()
        checks.append(("Config valid", True, f"run_id={cfg.run_id}"))
    except ValueError as exc:
        checks.append(("Config valid", False, str(exc).splitlines()[0]))

    # 2. Disk space (≥ 500 MB free)
    try:
        free_bytes = _shutil.disk_usage(cfg.storage.data_dir).free
        free_mb = free_bytes / (1024 ** 2)
        ok = free_mb >= 500
        checks.append((
            "Disk space (≥500 MB)",
            ok,
            f"{free_mb:.0f} MB free in {cfg.storage.data_dir}",
        ))
    except OSError as exc:
        checks.append(("Disk space (≥500 MB)", False, str(exc)))

    # 3. Data dir writable
    try:
        test = cfg.storage.data_dir / ".hc_write_test"
        test.touch()
        test.unlink()
        checks.append(("Data dir writable", True, str(cfg.storage.data_dir)))
    except OSError as exc:
        checks.append(("Data dir writable", False, str(exc)))

    # 4. LLM reachability (skip for --mock)
    if mock:
        checks.append(("LLM reachable", True, "Skipped (--mock mode)"))
    else:
        from generation.llm_client import OllamaClient
        client = OllamaClient(model=cfg.llm.model, base_url=cfg.llm.base_url)
        try:
            client.health_check()
            checks.append(("LLM reachable", True, f"{cfg.llm.model} @ {cfg.llm.base_url}"))
        except RuntimeError as exc:
            checks.append(("LLM reachable", False, str(exc).splitlines()[0]))

    # 5. Required Python packages
    for pkg in ("click", "rich", "ollama"):
        try:
            __import__(pkg)
            checks.append((f"Package: {pkg}", True, ""))
        except ImportError:
            checks.append((f"Package: {pkg}", False, "not installed — run: pip install " + pkg))

    # Render results table
    tbl = _Table(title="Health Check", show_header=True, header_style="bold")
    tbl.add_column("Check", style="bold")
    tbl.add_column("Status", justify="center")
    tbl.add_column("Detail")
    all_ok = True
    for name, passed, detail in checks:
        status = "[bold green]✓ PASS[/bold green]" if passed else "[bold red]✗ FAIL[/bold red]"
        if not passed:
            all_ok = False
        tbl.add_row(name, status, detail)

    from rich.console import Console as _Console
    _Console(highlight=False).print(tbl)

    if all_ok:
        _echo("\n[bold green]All checks passed — pipeline is ready.[/bold green]\n")
    else:
        _echo("\n[bold red]One or more checks failed — fix the issues above before running.[/bold red]\n")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CBSE Mathematics commands
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("math-generate")
@click.argument("inputs", nargs=-1, required=False)
@click.option(
    "--class-level",
    default=12,
    show_default=True,
    type=click.Choice(["10", "12"], case_sensitive=False),
    help="CBSE class level to generate for (supported: 10, 12).",
)
@click.option(
    "--mock/--no-mock",
    default=False,
    show_default=True,
    help="Use offline mock LLM (no Ollama required).",
)
@click.option(
    "--problems-per-subtopic",
    default=2,
    show_default=True,
    type=click.IntRange(1, 20),
    help="Number of practice problems to generate per subtopic.",
)
@click.option(
    "--gap-fills",
    default=2,
    show_default=True,
    type=click.IntRange(0, 20),
    help="Gap-fill problems to generate per uncovered subtopic.",
)
@click.option(
    "--output", "output_path",
    default=None,
    help="Output JSONL path. Default: data/math_dataset.jsonl",
)
@click.option(
    "--model",
    default="qwen3:4b",
    show_default=True,
    help="Ollama model to use for generation.",
)
def math_generate_cmd(
    inputs: tuple[str, ...],
    class_level: str,
    mock: bool,
    problems_per_subtopic: int,
    gap_fills: int,
    output_path: str | None,
    model: str,
):
    """
    Generate CBSE Mathematics problems, explanations, and gap-fill items.

    INPUTS can be one or more PDF files (NCERT chapters, past papers), .txt,
    or .json files. If no inputs are given, generation runs from the CBSE
    syllabus definitions alone.

    Examples::

        # Offline mock run for Class 12
        python main.py math-generate --mock --class-level 12

        # Real generation from NCERT PDF + past paper
        python main.py math-generate ncert_class12.pdf prev_year_2024.pdf \\
            --class-level 12 --model qwen3:4b

        # Class 10 with 3 problems per subtopic
        python main.py math-generate textbook.pdf --class-level 10 \\
            --problems-per-subtopic 3 --mock
    """
    from cbse_math.math_generator import MathGenConfig, MathGenerator

    _header("CBSE MATH GENERATION")
    cl = int(class_level)

    cfg = MathGenConfig(
        class_level=cl,
        problems_per_subtopic=problems_per_subtopic,
        gap_fills_per_gap=gap_fills,
    )

    out = Path(output_path) if output_path else _PROJECT_ROOT / "data" / "math_dataset.jsonl"

    _echo(
        f"  Class     : {cl}\n"
        f"  LLM       : {'mock' if mock else model}\n"
        f"  Inputs    : {list(inputs) or ['(syllabus descriptions only)']}\n"
        f"  Output    : {out}\n"
    )

    gen = MathGenerator(mock=mock, model=model, config=cfg)
    samples = gen.run(inputs=list(inputs), class_level=cl, output_path=out)

    valid = [s for s in samples if not s.get("_validation_errors")]
    invalid = [s for s in samples if s.get("_validation_errors")]

    _echo(
        f"\n  [bold]Generated[/bold] {len(samples)} items  "
        f"([green]{len(valid)} valid[/green]  [yellow]{len(invalid)} invalid[/yellow])"
    )
    _echo(f"  Saved → [cyan]{out}[/cyan]")


@cli.command("math-gap-analysis")
@click.argument("inputs", nargs=-1, required=True)
@click.option(
    "--class-level",
    default=12,
    show_default=True,
    type=click.Choice(["10", "12"], case_sensitive=False),
    help="CBSE class level to analyse against (supported: 10, 12).",
)
def math_gap_analysis_cmd(inputs: tuple[str, ...], class_level: str):
    """
    Analyse syllabus coverage gaps in provided NCERT / past-paper files.

    INPUTS: one or more .pdf, .txt, or .json files.

    Prints a coverage report showing which chapters and subtopics are
    well-covered, partially covered, or entirely missing from your source
    materials.

    Example::

        python main.py math-gap-analysis ncert_class12.pdf --class-level 12
    """
    from cbse_math.gap_analyzer import GapAnalyzer
    from cbse_math.pdf_ingestor import ingest_pdf

    _header("CBSE MATH GAP ANALYSIS")
    cl = int(class_level)

    all_chunks: list[str] = []
    for inp in inputs:
        path = Path(inp)
        try:
            if path.suffix.lower() == ".pdf":
                records = ingest_pdf(str(path))
                all_chunks.extend(r["content"] for r in records)
            elif path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
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
                all_chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            _echo(f"  Ingested: {path.name}")
        except Exception as exc:
            _echo(f"  [yellow]Could not ingest {path.name}: {exc}[/yellow]")

    if not all_chunks:
        _echo("[red]No text could be extracted from inputs.[/red]")
        sys.exit(1)

    analyzer = GapAnalyzer(class_level=cl)  # type: ignore[arg-type]
    report = analyzer.analyse(all_chunks)

    _echo(f"\n{report.summary()}\n")

    # Print prioritised gap list
    gaps = report.prioritised_gaps()
    if gaps:
        _echo(f"[bold]{len(gaps)} gap(s) identified — run 'math-generate' to fill them.[/bold]")
    else:
        _echo("[bold green]No significant gaps found — good coverage![/bold green]")


@cli.command("math-latex-preview")
@click.argument("dataset_path")
@click.option(
    "--limit",
    default=5,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Number of items to preview.",
)
def math_latex_preview_cmd(dataset_path: str, limit: int):
    """
    Pretty-print LaTeX content from a math dataset JSONL file.

    Renders each item's question, answer, and solution to the terminal.
    Useful for quickly inspecting generated problems.

    Example::

        python main.py math-latex-preview data/math_dataset.jsonl --limit 10
    """
    path = Path(dataset_path)
    if not path.exists():
        _echo(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    records = _load_jsonl(path)[:limit]
    _header(f"LaTeX Preview — {path.name} (first {len(records)} items)")

    for i, rec in enumerate(records, 1):
        item_type = rec.get("item_type", "?")
        chapter = rec.get("chapter_title", "?")
        subtopic = rec.get("subtopic", "?")
        difficulty = rec.get("difficulty", "?")
        marks = rec.get("marks", "?")
        content = rec.get("content", {})

        _echo(
            f"\n[bold cyan]── Item {i} ──[/bold cyan]  "
            f"[dim]{item_type}  |  {chapter}  |  {subtopic}  |  {difficulty}  |  {marks}m[/dim]"
        )

        if item_type == "problem":
            _echo(f"[bold]Q:[/bold] {content.get('question_latex', '(none)')}")
            _echo(f"[bold]A:[/bold] {content.get('answer_latex', '(none)')}")
            hints = content.get("hints", [])
            if hints:
                _echo(f"[dim]Hint 1: {hints[0]}[/dim]")

        elif item_type == "explanation":
            _echo(f"[bold]Concept:[/bold] {str(content.get('concept_latex', '(none)'))[:300]}")
            for fml in content.get("key_formulas", [])[:2]:
                _echo(f"  Formula: {fml}")

        elif item_type == "fill_gap":
            _echo(f"[bold]Gap:[/bold] {content.get('gap_description', '(none)')}")
            _echo(f"[bold]Q:[/bold] {content.get('question_latex', '(none)')}")
            _echo(f"[bold]A:[/bold] {content.get('answer_latex', '(none)')}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
