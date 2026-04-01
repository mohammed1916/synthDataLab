# System Architecture

## Overview

SynthDataLab is structured as a linear **pipeline of composable stages**, each
implemented as an independent Python package under `dataset_builder/`. Stages
communicate via plain Python lists of well-typed dataclasses — no message
queues, no shared mutable state, no databases.

The multi-agent path adds a _vertical_ supervision layer (Critic + Steering)
that wraps the same Generator stage without modifying it.

---

## High-level data flow

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        dataset_builder/                               │
  │                                                                       │
  │  ┌──────────┐    ┌────────────────────────────────────────────────┐  │
  │  │  ingest  │    │             Standard pipeline                  │  │
  │  │ ─────── │    │                                                 │  │
  │  │  .txt   │    │  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │  │
  │  │  .json  │───►│  │ generate │─►│ validate │─►│   filter     │ │  │
  │  │  .png   │    │  └──────────┘  └──────────┘  └──────┬───────┘ │  │
  │  └──────────┘    │       ▲                             │         │  │
  │                  │       │ (generate-agent path)       ▼         │  │
  │                  │  ┌────┴───────┐       ┌──────────────────┐   │  │
  │                  │  │  Critic    │       │    evaluate       │   │  │
  │                  │  │  Agent     │       │  (10+ metrics)    │   │  │
  │                  │  └────┬───────┘       └──────────────────┘   │  │
  │                  │       │                        │               │  │
  │                  │  ┌────▼───────┐       ┌────────▼──────────┐  │  │
  │                  │  │  Human     │       │    analyze         │  │  │
  │                  │  │  Steering  │       │  (error patterns)  │  │  │
  │                  │  └────────────┘       └───────────────────┘  │  │
  │                  └────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    data/ (JSONL files, metrics JSON, logs/)
```

---

## Module map

```
dataset_builder/
│
├── main.py                  CLI entry point (Click); 10 commands
├── config.py                Central config dataclasses
│
├── ingestion/
│   ├── ingestor.py          Dispatcher: routes files to specific ingestors
│   ├── text_ingestor.py     Splits plain text into overlapping chunks
│   ├── image_ingestor.py    OCR via pytesseract (optional)
│   └── __init__.py
│
├── generation/
│   ├── generator.py         DatasetGenerator: batch + stream generation
│   ├── llm_client.py        OllamaClient (retries) + MockLLMClient (thread-safe)
│   ├── critic_agent.py      4-axis heuristic quality scorer
│   ├── orchestrator.py      MultiAgentOrchestrator: Generator + Critic + Steering
│   └── evolver.py           Evol-Instruct prompt evolution
│
├── prompts/
│   ├── templates.py         Per-task-type prompt templates
│   └── few_shot_examples.py Few-shot examples for each task
│
├── schema/
│   └── dataset_schema.py    DatasetSample dataclass + jsonschema validator
│
├── validation/
│   ├── rule_validator.py    Schema rules + per-task-type heuristics
│   ├── llm_reviewer.py      LLM-based review (optionally real Ollama call)
│   └── annotation.py        AnnotatedSample dataclass + AnnotationLabel enum
│
├── filtering/
│   ├── pipeline.py          FilteringPipeline: confidence + dedup + length
│   └── deduplicator.py      Jaccard-similarity deduplication
│
├── evaluation/
│   ├── metrics.py           DatasetMetrics + compute_metrics (10 metrics)
│   ├── live_metrics.py      LiveMetricsTracker + Rich live dashboard
│   └── reporter.py          MetricsReporter: pretty-print + JSON write
│
├── analysis/
│   └── error_analyzer.py    ErrorAnalyzer: rejection-reason breakdown
│
├── tests/                   73 pytest tests (5 modules)
│   ├── test_schema.py
│   ├── test_metrics.py
│   ├── test_validator.py
│   ├── test_evolver.py
│   └── test_integration.py
│
└── data/
    ├── sample_inputs/       Bundled demo data
    ├── logs/                Rotating log files (pipeline_<ts>.log)
    └── *.jsonl / *.json     Pipeline outputs
```

---

## Key design decisions

### Atomic file writes

`_save_jsonl()` in `main.py` never overwrites a file in-place. It always
writes to a `.tmp` sibling, calls `os.fsync`, then `os.replace` (atomic rename
on POSIX). A crash mid-write leaves the original file intact — no torn
datasets.

### Thread safety

- `MockLLMClient` uses `threading.Lock` around `random.Random` calls.
- `LiveMetricsTracker` uses `threading.Lock` for all snapshot mutations.
- `OllamaClient` is inherently thread-safe (each call is stateless HTTP).

### Retry / back-off

`OllamaClient.complete()` retries up to `max_retries` (default 3) times with
exponential back-off: sleep `2^attempt + uniform(0, 1)` seconds between
attempts. Network flaps and model restarts are handled transparently.

### Collapse prevention

The pipeline computes **vocabulary entropy**, **bigram entropy**, and a
composite **collapse risk score** per evaluation run. The multi-agent
orchestrator re-evaluates collapse risk every 10 samples and pushes the
result to the live dashboard. A `CRITICAL` reading (≥ 0.70) triggers a
log warning and turns the dashboard gauge red.

### Checkpointing

`run-all --resume` persists completed step names to `data/logs/checkpoint.json`
after each step. On re-run with `--resume`, already-completed steps are loaded
from disk and skipped. The checkpoint is cleared on successful completion.

### Cross-run deduplication

`filtering/fingerprint_store.py` maintains a SHA-256 fingerprint set across
all pipeline runs. Each fingerprint is derived from:

```
SHA-256( lower(strip(input_text)) + "|" + task_type )[:32]
```

On each `run-all` execution:

1. All generated samples are **always** written to `raw_dataset.jsonl` unchanged.
2. `FingerprintStore.filter_new()` partitions into _(new, already-seen)_ **in-memory only** — the raw file is never overwritten.
3. Only genuinely new samples proceed to validation, filtering, and evaluation.
4. If all samples are already seen, the run exits cleanly with code `0` and a descriptive message.
5. After a successful full run, `FingerprintStore.save()` atomically persists the updated set.

Escape hatches: `--force` (skip dedup this run) and `--reset-fingerprints` (wipe the store).

### Config versioning and run lineage

Every `Config` instance carries:

| Property      | Source                                     | Purpose                                 |
| ------------- | ------------------------------------------ | --------------------------------------- |
| `run_id`      | 8-char hex from `datetime` + entropy       | Unique identifier for each pipeline run |
| `git_sha`     | `git rev-parse --short HEAD`               | Code version at time of run             |
| `config_hash` | SHA-256 of JSON-serialised config snapshot | Detect config drift between runs        |

`Config.validate()` raises `ValueError` for misconfigured thresholds, a
non-writable data directory, or `< 500 MB` free disk space. It is called
automatically at the start of `run-all`.

Each `run-all` creates `data/runs/<run_id>/manifest.json` containing all
three properties plus model name, task types, and timestamp. A `data/latest`
symlink always points to the most recent `runs/<run_id>/`.

---

## Dependency graph (simplified)

```
main.py
  ├── config.py
  ├── ingestion/ingestor.py
  ├── generation/generator.py
  │     └── generation/llm_client.py
  ├── generation/orchestrator.py
  │     ├── generation/generator.py
  │     ├── generation/critic_agent.py
  │     └── evaluation/live_metrics.py
  ├── validation/rule_validator.py
  ├── validation/llm_reviewer.py
  ├── filtering/pipeline.py
  ├── evaluation/metrics.py
  ├── evaluation/reporter.py
  └── analysis/error_analyzer.py
```
