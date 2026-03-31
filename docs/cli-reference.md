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

| Flag                     | Default        | Description                               |
| ------------------------ | -------------- | ----------------------------------------- |
| `--input PATH`           | bundled sample | Path to a `.txt` or `.json` articles file |
| `--mock / --no-mock`     | `--no-mock`    | Use mock LLM (no Ollama server required)  |
| `--resume / --no-resume` | `--no-resume`  | Resume from last checkpoint               |
| `--workers N`            | `1`            | Parallel LLM generation threads (1–16)    |

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
```

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

## Exit codes

| Code | Meaning                                        |
| ---- | ---------------------------------------------- |
| `0`  | Success                                        |
| `1`  | Fatal error (file not found, bad config, etc.) |
