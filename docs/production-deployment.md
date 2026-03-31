# Production Deployment

This guide covers running SynthDataLab in environments beyond a local
developer laptop: Docker containers, CI/CD pipelines, and long-running
production jobs.

---

## Docker

### Build

```bash
# From the repository root
docker build -t synthdatalab:latest .
```

The two-stage Dockerfile produces a minimal `python:3.11-slim` image:

- **Stage 1 (builder)** — installs `gcc` and compiles all Python deps
- **Stage 2 (runtime)** — copies compiled packages; no compiler in final image
- Runs as non-root user `appuser`
- `VOLUME ["/app/dataset_builder/data"]` declared for output persistence

### Run (mock mode)

```bash
mkdir -p data-out
docker run --rm \
  -v "$(pwd)/data-out:/app/dataset_builder/data" \
  synthdatalab:latest run-all --mock
```

Output files appear in `./data-out/`.

### Run (real Ollama)

Ollama must be reachable from inside the container. On macOS/Windows Docker
Desktop, `host.docker.internal` resolves to the host machine:

```bash
docker run --rm \
  -v "$(pwd)/data-out:/app/dataset_builder/data" \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=qwen3:4b \
  synthdatalab:latest run-all
```

On Linux (where `host.docker.internal` doesn't exist by default):

```bash
docker run --rm \
  -v "$(pwd)/data-out:/app/dataset_builder/data" \
  --network=host \
  -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  synthdatalab:latest run-all
```

### Run the multi-agent pipeline in Docker

```bash
docker run --rm \
  -v "$(pwd)/data-out:/app/dataset_builder/data" \
  synthdatalab:latest generate-agent --mock --no-dashboard
```

> `--no-dashboard` is recommended in Docker since the Rich live panel
> requires a real TTY. Metrics are written to the log file instead.

### Open a debug shell

```bash
docker run --rm -it \
  -v "$(pwd)/data-out:/app/dataset_builder/data" \
  --entrypoint /bin/bash \
  synthdatalab:latest
```

---

## Environment variables

Copy `.env.example` → `.env` and set your values:

```bash
cp .env.example .env
$EDITOR .env
```

| Variable             | Default                   | Description                                  |
| -------------------- | ------------------------- | -------------------------------------------- |
| `OLLAMA_BASE_URL`    | `http://localhost:11434`  | Ollama server URL                            |
| `OLLAMA_MODEL`       | `qwen3:4b`                | Model tag (must be pulled via `ollama pull`) |
| `LOG_LEVEL`          | `INFO`                    | Python logging level                         |
| `MAX_WORKERS`        | `1`                       | Default parallel LLM threads                 |
| `DEFAULT_TASK_TYPES` | `qa,extraction,reasoning` | Comma-separated task types                   |

> **Security note:** Never commit `.env` to version control. It is in
> `.gitignore`. Use `.env.example` as a template only.

---

## CI/CD — GitHub Actions

The repository ships with `.github/workflows/ci.yml` providing 4 jobs:

### Job 1 — `lint`

Runs on every push and pull-request to `main`:

```
ruff check dataset_builder/
ruff format --check dataset_builder/
```

### Job 2 — `typecheck`

```
mypy dataset_builder/ --ignore-missing-imports
```

### Job 3 — `test`

Matrix: Python `3.10`, `3.11`, `3.12`:

```
pytest tests/ --cov=. --cov-report=xml
```

Coverage XML is uploaded to Codecov on Python 3.11.

### Job 4 — `smoke-test`

Depends on `test`. Runs the full mock pipeline and asserts all 4 output
files exist:

```bash
python main.py run-all --mock
test -f data/raw_dataset.jsonl
test -f data/annotated_dataset.jsonl
test -f data/filtered_dataset.jsonl
test -f data/metrics_report.json
```

Artifacts (output JSONL files) are retained for 7 days.

### Adding secrets

For the Codecov upload, add `CODECOV_TOKEN` to your repository secrets
under _Settings → Secrets and variables → Actions_.

---

## Logging

Every pipeline run writes a structured log file to:

```
dataset_builder/data/logs/pipeline_<YYYYMMDD_HHMMSS>.log
```

Format:

```
2026-03-31T10:25:01  INFO      dataset_builder  File logging active → data/logs/pipeline_20260331_102501.log
2026-03-31T10:25:01  INFO      dataset_builder  Ingested 5 chunk(s)
2026-03-31T10:25:02  WARNING   generation.llm_client  Attempt 1/3 — ConnectionError: …
2026-03-31T10:25:04  INFO      generation.llm_client  Attempt 2/3 succeeded
```

Log files are **not rotated automatically** (each run creates a new timestamped
file). Add a cron job or logrotate rule to clean old logs:

```bash
# Keep only the last 30 log files
ls -t dataset_builder/data/logs/*.log | tail -n +31 | xargs rm -f
```

---

## Checkpointing long-running jobs

For large input corpora where a full run might take hours:

```bash
# Start a run
python main.py run-all --workers 4

# If interrupted, resume from the last completed step
python main.py run-all --workers 4 --resume
```

The checkpoint is stored at `data/logs/checkpoint.json`. It is cleared
automatically on clean completion. To force a fresh start:

```bash
rm dataset_builder/data/logs/checkpoint.json
```

---

## Performance tuning

| Bottleneck           | Diagnosis                              | Remedy                                                                     |
| -------------------- | -------------------------------------- | -------------------------------------------------------------------------- |
| Slow generation      | `throughput < 1 samp/sec` in dashboard | Increase `--workers` (try 4); switch to a faster model; lower `max_tokens` |
| High rejection rate  | `REJECT > 30%`                         | Check input quality; lower `--threshold`; use `review-low` to inspect      |
| High collapse risk   | Dashboard gauge ≥ MEDIUM               | Add more diverse source material; enable all 5 task types; run `evolve`    |
| OOM in parallel mode | Worker crashes                         | Reduce `--workers`; reduce `max_tokens`                                    |
| Disk space           | Large JSONL files                      | Compress: `gzip data/*.jsonl`                                              |

---

## File size limits

The ingestion layer enforces a **50 MB per-file limit**. Files larger than
this will raise a `ValueError` with a human-readable message. To process
larger corpora, split them first:

```bash
# Split a large JSON articles file into 10 MB chunks
split -b 10m big_corpus.json chunk_

# Process each chunk separately
for f in chunk_*; do
  python main.py run-all --input "$f" --mock --resume
done
```

---

## Security checklist

- [ ] `.env` file is in `.gitignore` and never committed
- [ ] Docker image runs as `appuser` (non-root)
- [ ] `OLLAMA_BASE_URL` points to a trusted server (not public internet)
- [ ] Data directory is mounted as a volume, not baked into the image
- [ ] Log files do not contain secrets (the pipeline never logs credentials)
- [ ] File size guard (50 MB) prevents accidental OOM from malicious inputs
- [ ] Atomic file writes prevent torn/partial dataset files on crash
