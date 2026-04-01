# Quickstart Guide

Get SynthDataLab running locally in under 5 minutes.

---

## Prerequisites

| Requirement | Version | Purpose                                              |
| ----------- | ------- | ---------------------------------------------------- |
| Python      | ≥ 3.10  | Runtime                                              |
| pip         | any     | Package installation                                 |
| Ollama      | any     | LLM backend _(optional — mock mode needs no server)_ |

---

## 1 — Install

```bash
# Clone the repo
git clone https://github.com/mohammed1916/synthDataLab.git
cd synthDataLab

# Create a virtual environment (recommended)
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r dataset_builder/requirements.txt

# (Optional) install dev extras for testing / linting
pip install -e ".[dev]"
```

---

## 2 — Run with mock LLM (no Ollama needed)

The fastest way to see the full pipeline:

```bash
cd dataset_builder
python main.py run-all --mock
```

This runs all 6 pipeline steps using a built-in mock LLM that produces
realistic synthetic samples without any network calls.

Expected output:

```
── 1 / 6  INGESTION ──────────────────────────────────────────
  Ingested 5 chunk(s).
── 2 / 6  GENERATION ─────────────────────────────────────────
  [progress bar]
  Generated 15 raw samples (mock-llm-v1)
── 3 / 6  VALIDATION  (Rule + HITL Simulation) ───────────────
  Validation: 11 ACCEPT  2 FIX_REQUIRED  2 REJECT
── 4 / 6  FILTERING ──────────────────────────────────────────
  Filtering: 11 samples retained (100.0% retention rate)
── 5 / 6  EVALUATION  (Metrics) ──────────────────────────────
  [metrics table]
── 6 / 6  ERROR ANALYSIS ─────────────────────────────────────
  Error analysis saved → data/error_analysis.json
── DONE ──────────────────────────────────────────────────────
  Raw dataset      → data/raw_dataset.jsonl
  Annotated        → data/annotated_dataset.jsonl
  Filtered dataset → data/filtered_dataset.jsonl
  Metrics report   → data/metrics_report.json
  Error analysis   → data/error_analysis.json
```

---

## 3 — Try the multi-agent mode with live dashboard

The `generate-agent` command adds a **CriticAgent** layer and a **Real-time
Rich dashboard** showing quality decisions as they happen:

```bash
# Fully automatic (critic decides everything)
python main.py generate-agent --mock

# Human reviews every sample with critic score below 0.70
python main.py generate-agent --mock --steering review-low

# Human reviews every single sample
python main.py generate-agent --mock --steering review-all
```

The live dashboard looks like:

```
╭─ SynthDataLab · Live Generation Dashboard ─────────────────────────────────╮
│  Progress  ████████████░░░░░  9/15  60%    ⚡ 4.1 samp/sec  ETA 1s          │
│                                                                              │
│  Decisions          Task Types         Quality / Health                      │
│  ✓ ACCEPT   7 (78%) ███████   qa       5 (56%)  Confidence   0.831          │
│  ✗ REJECT   1 (11%) █         extract  3 (33%)  Critic score 0.712          │
│  ~ FIX      1 (11%) █         reason   1 (11%)  Collapse risk LOW (0.12)    │
│                                                                              │
│  Recent Samples ────────────────────────────────────────────────────────    │
│  ✓ qa             conf=0.85 critic=0.79  s_abc123456789…                    │
│  ✗ reasoning      conf=0.32 critic=0.28  s_def456789012…                    │
╰─────────────────────────────────────────────────────────────────────────────╯
```

---

## 4 — Run with a real Ollama backend

```bash
# Pull the model first (one-time)
ollama pull qwen3:4b

# Run the pipeline
python main.py run-all

# Or the multi-agent mode
python main.py generate-agent --steering review-low
```

---

## 5 — Use your own data

Pass any plain text file or a JSON articles file:

```bash
# Plain text
python main.py run-all --input /path/to/my_text.txt

# JSON array of articles: [{"title": "...", "content": "..."}]
python main.py run-all --input /path/to/articles.json
```

---

## 6 — Pre-flight check

Before running a long real-LLM run, verify the environment is healthy:

```bash
# Check Ollama, disk space, config, and Python packages
python main.py health-check

# In mock / CI mode (skips LLM connectivity check)
python main.py health-check --mock
```

---

## 7 — Run the test suite

```bash
make test
# or
cd dataset_builder && python -m pytest tests/ -v
```

All 138 tests should pass.

---

## 8 — Use Make shortcuts

```bash
make help              # show all targets
make run-mock          # run full pipeline with mock LLM
make run-mock-parallel # parallel generation (4 workers)
make test-cov          # tests with HTML coverage report
make lint              # ruff check
make typecheck         # mypy
make check             # lint + format-check + typecheck
make docker-build      # build Docker image
make docker-run        # run in Docker (mock, data in ./data-out/)
```

---

## Output files

After a successful run, these files are written to `dataset_builder/data/`:

| File                      | Format  | Contents                                    |
| ------------------------- | ------- | ------------------------------------------- |
| `raw_dataset.jsonl`       | JSONL   | All generated samples (including rejected)  |
| `annotated_dataset.jsonl` | JSONL   | Samples with validation annotations         |
| `filtered_dataset.jsonl`  | JSONL   | High-quality samples only                   |
| `metrics_report.json`     | JSON    | Full metrics report (raw vs filtered)       |
| `error_analysis.json`     | JSON    | Error patterns and rejection reasons        |
| `fingerprints.json`       | JSON    | SHA-256 fingerprint store (cross-run dedup) |
| `runs/<run_id>/`          | dir     | Versioned snapshot + `manifest.json`        |
| `latest/`                 | symlink | Points to most recent `runs/<run_id>/`      |
| `evolved_prompts.jsonl`   | JSONL   | Evol-Instruct evolved prompts (if run)      |
| `logs/pipeline_<ts>.log`  | Text    | Full run log with timestamps                |
