"""n
main.py — CLI entry point for the Dataset Builder pipeline.

Commands
--------
  run-all     Run the complete pipeline end-to-end (recommended for demo)
  ingest      Normalise raw inputs into intermediate representation
  generate    Synthesise dataset samples from ingested content
  validate    Apply schema + rule validation to a raw dataset file
  filter      Run quality filtering on a validated (annotated) dataset
  evaluate    Compute and compare metrics for raw vs filtered datasets
  analyze     Run error analysis on an annotated dataset
  evolve      Run Evol-Instruct prompt evolution on a seed file
  guidelines  Print human-in-the-loop annotation guidelines

Usage examples
--------------
  python main.py run-all
  python main.py run-all --input data/sample_inputs/sample_articles.json
  python main.py evolve data/sample_inputs/sample_text.txt --rounds 2
  python main.py generate --help
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

# ── Ensure the project root is on sys.path ────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config, DEFAULT_CONFIG
from ingestion.ingestor import Ingestor, IngestionResult
from generation.generator import DatasetGenerator
from generation.evolver import PromptEvolver, EvolveConfig
from schema.dataset_schema import DatasetSample
from validation.rule_validator import RuleValidator, annotation_guidelines
from validation.llm_reviewer import LLMReviewer
from validation.annotation import AnnotatedSample, AnnotationLabel
from filtering.pipeline import FilteringPipeline
from evaluation.metrics import compute_metrics
from evaluation.reporter import MetricsReporter
from analysis.error_analyzer import ErrorAnalyzer

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset_builder")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(mock: bool = False) -> Config:
    cfg = Config()
    if mock:
        cfg.llm.provider = "mock"
    cfg.ensure_dirs()
    return cfg


def _save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _echo(f"  Saved {len(records)} records → {path}")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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

def step_ingest(cfg: Config, input_path: Optional[str] = None) -> List[IngestionResult]:
    """Ingest inputs and return IngestionResult list."""
    ingestor = Ingestor()
    results: List[IngestionResult] = []

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
    cfg: Config, ingestion_results: List[IngestionResult]
) -> List[DatasetSample]:
    """Generate dataset samples from ingestion results."""
    generator = DatasetGenerator(cfg)
    samples = generator.generate_from_ingestion(ingestion_results)
    _save_jsonl(
        [s.to_dict() for s in samples],
        cfg.storage.raw_path(),
    )
    _echo(
        f"  Generated [bold]{len(samples)}[/bold] raw samples "
        f"({'mock LLM' if cfg.use_mock_llm else cfg.llm.model})"
    )
    return samples


def step_validate(
    cfg: Config, samples: List[DatasetSample]
) -> List[AnnotatedSample]:
    """Validate samples (rule-based + optional LLM review)."""
    rule_validator = RuleValidator(min_confidence=cfg.filtering.min_confidence)
    sample_dicts = [s.to_dict() for s in samples]
    annotated = rule_validator.validate_batch(sample_dicts)

    # LLM reviewer on FIX_REQUIRED samples
    llm_reviewer = LLMReviewer(llm_client=None)  # uses mock heuristics
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
    cfg: Config, annotated: List[AnnotatedSample]
) -> tuple[List[AnnotatedSample], dict]:
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
    raw_samples: List[DatasetSample],
    filtered_samples: List[AnnotatedSample],
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
    cfg: Config, annotated: List[AnnotatedSample]
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
def run_all(input_path: Optional[str], mock: bool):
    """Run the complete pipeline: ingest → generate → validate → filter → evaluate → analyze."""
    cfg = _load_config(mock=mock)
    _echo(
        f"\n[bold green]Dataset Builder[/bold green] "
        f"({'Mock LLM' if cfg.use_mock_llm else cfg.llm.model})\n"
    )

    # 1. INGEST
    _header("1 / 6  INGESTION")
    ingestion_results = step_ingest(cfg, input_path)

    # 2. GENERATE
    _header("2 / 6  GENERATION")
    raw_samples = step_generate(cfg, ingestion_results)

    # 3. VALIDATE
    _header("3 / 6  VALIDATION  (Rule + HITL Simulation)")
    annotated = step_validate(cfg, raw_samples)

    # 4. FILTER
    _header("4 / 6  FILTERING")
    filtered, filter_report = step_filter(cfg, annotated)

    # 5. EVALUATE
    _header("5 / 6  EVALUATION  (Metrics)")
    step_evaluate(cfg, raw_samples, filtered, filter_report)

    # 6. ANALYZE
    _header("6 / 6  ERROR ANALYSIS")
    step_analyze(cfg, annotated)

    _header("DONE")
    _echo(f"\n  Raw dataset      → [cyan]{cfg.storage.raw_path()}[/cyan]")
    _echo(f"  Annotated        → [cyan]{cfg.storage.annotated_path()}[/cyan]")
    _echo(f"  Filtered dataset → [cyan]{cfg.storage.filtered_path()}[/cyan]")
    _echo(f"  Metrics report   → [cyan]{cfg.storage.metrics_path()}[/cyan]")
    _echo(f"  Error analysis   → [cyan]{cfg.storage.error_path()}[/cyan]\n")


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
def generate_cmd(input_path: Optional[str], mock: bool):
    """Generate dataset samples from input data."""
    cfg = _load_config(mock=mock)
    _header("GENERATION")
    ingestion_results = step_ingest(cfg, input_path)
    step_generate(cfg, ingestion_results)


@cli.command("validate")
@click.option(
    "--dataset",
    "dataset_path",
    default=None,
    help="Path to a JSONL raw dataset. Defaults to data/raw_dataset.jsonl.",
)
@click.option("--mock/--no-mock", default=False, show_default=True)
def validate_cmd(dataset_path: Optional[str], mock: bool):
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
    output_path: Optional[str],
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

    seeds: List[str] = []
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
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
