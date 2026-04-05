# CLI Reference

All commands are run from inside the `dataset_builder/` directory:

```bash
cd dataset_builder
python main.py <command> [OPTIONS]
```

---

## Global help

```bash
python main.py --help
```

---

## `run-all`

Run the complete 6-step pipeline end-to-end.

```
python main.py run-all [OPTIONS]
```

### Options

| Flag                     | Default        | Description                                                                 |
| ------------------------ | -------------- | --------------------------------------------------------------------------- |
| `--input PATH`           | bundled sample | Path to a `.txt` or `.json` articles file                                   |
| `--mock / --no-mock`     | `--no-mock`    | Use mock LLM (no Ollama server required)                                    |
| `--resume / --no-resume` | `--no-resume`  | Resume from last checkpoint                                                 |
| `--workers N`            | `1`            | Parallel LLM generation threads (1–16)                                      |
| `--agent / --no-agent`   | `--no-agent`   | Use `MultiAgentOrchestrator` (CriticAgent + Steering) for generation        |
| `--steering MODE`        | `auto`         | Steering mode when `--agent` is set: `auto` \| `review-low` \| `review-all` |
| `--threshold FLOAT`      | `0.70`         | Critic pass threshold for `--agent` mode (0.0–1.0)                          |
| `--force / --no-force`   | `--no-force`   | Skip cross-run dedup; reprocess all samples                                 |
| `--reset-fingerprints`   | off            | Wipe the fingerprint store before running (start dedup fresh)               |

### Steps executed

1. **Ingestion** — normalise inputs into chunks
2. **Generation** — synthesise samples via LLM
3. **Validation** — rule-based + HITL simulation
4. **Filtering** — quality thresholds + deduplication
5. **Evaluation** — compute all metrics
6. **Error analysis** — breakdown rejection patterns

### Examples

```bash
# Basic mock run
python main.py run-all --mock

# Full run with 4 workers using your own data
python main.py run-all --input articles.json --workers 4

# Resume a failed run from the last completed step
python main.py run-all --mock --resume

# Multi-agent run with human review of low-scoring samples
python main.py run-all --mock --agent --steering review-low --threshold 0.65

# Force reprocess all samples from this run (bypass cross-run dedup)
python main.py run-all --mock --force

# Wipe fingerprint history and start fresh
python main.py run-all --mock --reset-fingerprints
```

### Cross-run deduplication

`run-all` maintains a SHA-256 fingerprint store at `data/fingerprints.json`.
On each run it computes fingerprints for every generated sample; samples already
seen in previous runs are filtered out before validation begins. The raw
artifact file is always written in full — only what goes _downstream_ is filtered.

If all generated samples have already been processed:

```
⚠  Nothing new to process — all 30 sample(s) were already seen in a previous run.
Tip: use --force to re-process anyway, or --reset-fingerprints to start fresh.
```

The command exits with code `0` in this case (not an error).

### Versioned run artifacts

Every successful `run-all` writes to `data/runs/<run_id>/`:

```
data/runs/d50c5250/
├── manifest.json          # {run_id, git_sha, config_hash, model, task_types, ...}
├── raw_dataset.jsonl
├── annotated_dataset.jsonl
├── filtered_dataset.jsonl
├── metrics_report.json
├── error_analysis.json
└── critic_scores.jsonl    # only when --agent is set
```

A `data/latest` symlink always points to the most recent `runs/<run_id>/`.

---

## `generate-agent`

Multi-agent generation with a **CriticAgent** quality gate and an optional
**human steering** interface. Shows a live Rich dashboard during generation.

```
python main.py generate-agent [OPTIONS]
```

### Options

| Flag                 | Default                  | Description                            |
| -------------------- | ------------------------ | -------------------------------------- |
| `--input PATH`       | bundled sample           | Source data file                       |
| `--mock / --no-mock` | `--no-mock`              | Use mock LLM                           |
| `--steering MODE`    | `auto`                   | `auto` \| `review-low` \| `review-all` |
| `--threshold FLOAT`  | `0.70`                   | Critic pass threshold (0.0–1.0)        |
| `--workers N`        | `1`                      | Parallel LLM workers                   |
| `--output PATH`      | `data/raw_dataset.jsonl` | Output file path                       |
| `--no-dashboard`     | off                      | Disable Rich live dashboard            |

### Steering modes explained

| Mode         | Behaviour                                                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `auto`       | Critic score ≥ threshold → ACCEPT; < threshold − 0.25 → REJECT; between → FIX_REQUIRED. No human needed.                            |
| `review-low` | Samples below `--threshold` are surfaced for human review. All others auto-accept.                                                  |
| `review-all` | Every sample shown to human with critic scores. Human keys: `a`=approve, `r`=reject, `f`=fix, `s`=skip, `q`=quit, `?`=show content. |

### Live dashboard

When `--no-dashboard` is not set, a full-screen Rich panel shows real-time:

- **Progress bar** — N/total, percentage, throughput (samp/sec), ETA
- **Decisions** — ACCEPT / REJECT / FIX_REQUIRED counts + mini bar charts
- **Task types** — distribution across qa / extraction / reasoning / …
- **Quality / Health** — mean confidence, mean critic score, collapse risk level
- **Recent samples** — last 10 sample IDs with per-sample scores

### Examples

```bash
# Auto mode, mock LLM
python main.py generate-agent --mock

# Human reviews samples below 0.65
python main.py generate-agent --mock --steering review-low --threshold 0.65

# Human reviews every sample, no dashboard (for terminal logging)
python main.py generate-agent --mock --steering review-all --no-dashboard

# Real Ollama, 4 workers, aggressive threshold
python main.py generate-agent --workers 4 --threshold 0.80
```

---

## `math-generate`

Generate CBSE math items (`problem`, `explanation`, `fill_gap`) from PDFs/text/JSON.

```
python main.py math-generate [INPUTS...] [OPTIONS]
```

### Options

| Flag                       | Default              | Description                              |
| -------------------------- | -------------------- | ---------------------------------------- |
| `--class-level`            | `12`                 | CBSE class (`10` or `12`)               |
| `--mock / --no-mock`       | `--no-mock`          | Use offline mock LLM                     |
| `--problems-per-subtopic`  | `2`                  | Practice problems per subtopic           |
| `--gap-fills`              | `2`                  | Gap-fill problems per uncovered subtopic |
| `--model MODEL`            | `qwen3:4b`           | Ollama model (ignored in mock mode)      |
| `--output PATH`            | `data/math_dataset.jsonl` | Output JSONL path                   |

### Examples

```bash
python main.py math-generate --mock --class-level 12
python main.py math-generate ncert12.pdf pyq2024.pdf --class-level 12 --model qwen3:4b
python main.py math-generate textbook10.pdf --class-level 10 --problems-per-subtopic 3 --mock
```

---

## `math-gap-analysis`

Analyze syllabus coverage from one or more `.pdf`, `.txt`, or `.json` files.

```
python main.py math-gap-analysis INPUTS... [OPTIONS]
```

### Options

| Flag            | Default | Description                |
| --------------- | ------- | -------------------------- |
| `--class-level` | `12`    | CBSE class (`10` or `12`) |

### Example

```bash
python main.py math-gap-analysis ncert12.pdf --class-level 12
```

---

## `math-latex-preview`

Preview LaTeX fields from a generated math JSONL file.

```
python main.py math-latex-preview DATASET_PATH [--limit N]
```

### Example

```bash
python main.py math-latex-preview data/math_dataset.jsonl --limit 10
```

---

## `ingest`

Normalise a single file and write chunks to `data/ingested.jsonl`.

```
python main.py ingest INPUT_PATH [--mock]
```

### Examples

```bash
python main.py ingest data/sample_inputs/sample_text.txt
python main.py ingest data/sample_inputs/sample_articles.json
```

---

## `generate`

Generate dataset samples from the most recently ingested chunks.

```
python main.py generate [--input PATH] [--mock]
```

---

## `validate`

Apply rule-based schema validation + HITL simulation to a raw dataset JSONL.

```
python main.py validate [--dataset PATH] [--mock]
```

---

## `filter`

Run quality filtering (confidence threshold + deduplication) on the
annotated dataset.

```
python main.py filter [--mock]
```

---

## `evaluate`

Compute and compare 10+ metrics for raw vs filtered datasets, writing
`data/metrics_report.json`.

```
python main.py evaluate [--mock]
```

---

## `analyze`

Run error analysis on the annotated dataset, writing `data/error_analysis.json`.

```
python main.py analyze [--mock]
```

---

## `evolve`

Run Evol-Instruct prompt evolution on seed instructions.

```
python main.py evolve INPUT_PATH [OPTIONS]
```

### Options

| Flag            | Default                      | Description                          |
| --------------- | ---------------------------- | ------------------------------------ |
| `--rounds N`    | `2`                          | Number of evolution rounds           |
| `--ops LIST`    | all 4 ops                    | Comma-separated evolution operations |
| `--output PATH` | `data/evolved_prompts.jsonl` | Output file                          |

### Available operations

| Operation            | Effect                                             |
| -------------------- | -------------------------------------------------- |
| `add_constraints`    | Add specific constraints to make the prompt harder |
| `deepen`             | Deepen the required reasoning or detail level      |
| `concretise`         | Replace vague language with concrete specifics     |
| `increase_reasoning` | Require explicit multi-step reasoning              |

### Examples

```bash
python main.py evolve data/sample_inputs/sample_text.txt --rounds 3
python main.py evolve articles.json --rounds 2 --ops add_constraints,deepen
```

---

## `guidelines`

Print the human-in-the-loop annotation guidelines to stdout.

```
python main.py guidelines
```

---

## `health-check`

Pre-flight check for Ollama connectivity, disk space, config validity, and
required Python packages. Useful before kicking off a long run.

```
python main.py health-check [--mock]
```

### Options

| Flag                 | Default     | Description                                |
| -------------------- | ----------- | ------------------------------------------ |
| `--mock / --no-mock` | `--no-mock` | Skip LLM connectivity check (useful in CI) |

### Checks performed

| Check                | What it verifies                              |
| -------------------- | --------------------------------------------- |
| Config valid         | `Config.validate()` passes                    |
| Disk space (≥500 MB) | At least 500 MB free in the data directory    |
| Data dir writable    | Can create a test file in `data/`             |
| LLM reachable        | Ollama `/api/tags` responds + model is loaded |
| Package: click       | `import click` succeeds                       |
| Package: rich        | `import rich` succeeds                        |
| Package: ollama      | `import ollama` succeeds                      |

Exits with code `0` if all checks pass, `1` if any fail.

### Example

```bash
python main.py health-check --mock
# → prints Rich table of ✓ PASS / ✗ FAIL rows
# → "All checks passed — pipeline is ready."
```

---

## `list-runs`

List all versioned pipeline runs stored in `data/runs/`, sorted newest first.

```
python main.py list-runs
```

Prints a Rich table with columns: **Run ID**, **Git SHA**, **Model**, **Tasks**, **Config Hash**.

```bash
python main.py list-runs
#  Run ID    │ Git SHA  │ Model     │ Tasks                    │ Config Hash
#  d50c5250  │ 145f51f  │ mock      │ qa, extraction, reasoning │ 4a7b3c1d
```

---

## `export`

Export the filtered dataset to an annotation platform format.

```
python main.py export [OPTIONS]
```

### Options

| Flag              | Default                       | Description                 |
| ----------------- | ----------------------------- | --------------------------- |
| `--dataset PATH`  | `data/filtered_dataset.jsonl` | Source JSONL file to export |
| `--format FORMAT` | `argilla`                     | `argilla` or `labelstudio`  |
| `--output PATH`   | `data/export_<format>.jsonl`  | Output file path            |

### Examples

```bash
# Export to Argilla format (default)
python main.py export

# Export to Label Studio format
python main.py export --format labelstudio

# Export a specific dataset file
python main.py export --dataset data/filtered_dataset.jsonl --format labelstudio --output out.jsonl
```

---

## Exit codes

| Code | Meaning                                                                          |
| ---- | -------------------------------------------------------------------------------- |
| `0`  | Success (including clean exit when cross-run dedup finds nothing new to process) |
| `1`  | Fatal error (file not found, bad config, Ollama health check failure, etc.)      |
