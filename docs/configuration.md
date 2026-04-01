# Configuration Reference

All configuration is centralised in `dataset_builder/config.py`.
The `DEFAULT_CONFIG` singleton is used throughout the pipeline.
You can override any field programmatically or via the CLI flags
described in [CLI Reference](cli-reference.md).

---

## Run versioning properties

These are **read-only computed properties** on every `Config` instance.
They are set automatically at construction time and written to each
run's `manifest.json`.

| Property      | Type  | Source                                             | Purpose                                         |
| ------------- | ----- | -------------------------------------------------- | ----------------------------------------------- |
| `run_id`      | `str` | 8-char hex derived from UTC datetime + entropy     | Unique identifier for each `run-all` invocation |
| `git_sha`     | `str` | `git rev-parse --short HEAD` (falls back to `"?"`) | Code version pinned to each run                 |
| `config_hash` | `str` | SHA-256 of JSON-serialised config snapshot         | Detect config drift between runs                |

---

## `Config.validate()`

Called automatically at the start of `run-all`. Raises `ValueError` if any
of the following conditions are found:

- `filtering.min_confidence` outside `[0, 1]`
- `filtering.max_duplicate_similarity` outside `(0, 1]`
- `filtering.max_output_length` ≤ `filtering.min_output_length`
- `generation.max_workers` < 1
- `storage.data_dir` is not writable
- `< 500 MB` free disk space in the data directory

You can also call `health-check --mock` to see a formatted table of all checks.

---

## `LLMConfig`

Controls the language-model backend.

| Field             | Type    | Default                    | Description                                                 |
| ----------------- | ------- | -------------------------- | ----------------------------------------------------------- |
| `provider`        | `str`   | `"ollama"`                 | `"ollama"` or `"mock"`. Use `--mock` CLI flag to switch.    |
| `model`           | `str`   | `"qwen3:4b"`               | Ollama model tag. Run `ollama pull <model>` first.          |
| `temperature`     | `float` | `0.7`                      | Sampling temperature (0 = deterministic, 1 = creative).     |
| `max_tokens`      | `int`   | `2048`                     | Maximum output tokens per LLM call.                         |
| `request_timeout` | `int`   | `120`                      | Seconds before an Ollama request times out.                 |
| `max_retries`     | `int`   | `3`                        | Retry attempts with exponential back-off on network errors. |
| `base_url`        | `str`   | `"http://localhost:11434"` | Ollama server URL. Set via `OLLAMA_BASE_URL` env var.       |

### Environment variable overrides

You can set these in a `.env` file (copy `.env.example` → `.env`):

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
LOG_LEVEL=INFO
```

---

## `GenerationConfig`

Controls how synthetic samples are produced.

| Field               | Type        | Default                           | Description                                                                 |
| ------------------- | ----------- | --------------------------------- | --------------------------------------------------------------------------- |
| `samples_per_input` | `int`       | `3`                               | Samples generated per ingested chunk (currently advisory).                  |
| `task_types`        | `List[str]` | `["qa","extraction","reasoning"]` | Task types to generate. Add `"reasoning_trace"`, `"preference"` to broaden. |
| `batch_size`        | `int`       | `10`                              | LLM call batch size (advisory).                                             |
| `max_workers`       | `int`       | `1`                               | Parallel LLM threads. `1` = sequential. Use `--workers N` CLI flag.         |

### Supported task types

```python
["qa", "extraction", "reasoning", "reasoning_trace", "preference"]
```

---

## `EvolutionConfig`

Controls Evol-Instruct prompt evolution (`evolve` command).

| Field                 | Type        | Default   | Description                                                                           |
| --------------------- | ----------- | --------- | ------------------------------------------------------------------------------------- |
| `enabled`             | `bool`      | `False`   | Automatically run evolution in `run-all`.                                             |
| `n_rounds`            | `int`       | `2`       | Evolution rounds (each round applies one operation per seed).                         |
| `operations`          | `List[str]` | all 4 ops | Operations to apply: `add_constraints`, `deepen`, `concretise`, `increase_reasoning`. |
| `max_seeds_per_round` | `int`       | `50`      | Cap on seeds processed per evolution round.                                           |
| `use_llm_evolution`   | `bool`      | `False`   | `True` = use LLM to rewrite prompts; `False` = template substitution.                 |

---

## `FilteringConfig`

Quality thresholds for the filtering pipeline.

| Field                      | Type    | Default | Description                                     |
| -------------------------- | ------- | ------- | ----------------------------------------------- |
| `min_confidence`           | `float` | `0.60`  | Samples below this confidence are discarded.    |
| `max_duplicate_similarity` | `float` | `0.85`  | Jaccard similarity threshold for deduplication. |
| `min_output_length`        | `int`   | `20`    | Minimum output character count.                 |
| `max_output_length`        | `int`   | `8000`  | Maximum output character count.                 |

---

## `StorageConfig`

File paths for all pipeline outputs.

| Field              | Type   | Default                     | Description                 |
| ------------------ | ------ | --------------------------- | --------------------------- |
| `data_dir`         | `Path` | `dataset_builder/data/`     | Root data directory.        |
| `raw_output`       | `str`  | `"raw_dataset.jsonl"`       | Raw samples filename.       |
| `annotated_output` | `str`  | `"annotated_dataset.jsonl"` | Annotated samples filename. |
| `filtered_output`  | `str`  | `"filtered_dataset.jsonl"`  | Filtered samples filename.  |
| `metrics_output`   | `str`  | `"metrics_report.json"`     | Metrics report filename.    |
| `error_output`     | `str`  | `"error_analysis.json"`     | Error analysis filename.    |

---

## `OrchestratorConfig` (multi-agent mode)

Used by the `generate-agent` command.

| Field                     | Type           | Default | Description                                                           |
| ------------------------- | -------------- | ------- | --------------------------------------------------------------------- |
| `steering_mode`           | `SteeringMode` | `AUTO`  | `auto` / `review-low` / `review-all`.                                 |
| `critic_pass_threshold`   | `float`        | `0.70`  | Critic composite score ≥ this → ACCEPT.                               |
| `critic_review_threshold` | `float`        | `0.45`  | Score between `review_threshold` and `pass_threshold` → FIX_REQUIRED. |
| `auto_reject_below`       | `float`        | `0.30`  | Composite score below this → hard REJECT.                             |
| `collapse_check_interval` | `int`          | `10`    | Re-compute collapse risk every N samples.                             |
| `show_dashboard`          | `bool`         | `True`  | Show Rich live dashboard. Disable with `--no-dashboard`.              |
| `save_critic_metadata`    | `bool`         | `True`  | Attach `critic` key to sample metadata for debugging.                 |

---

## Tuning advice

### For higher-quality output

- Increase `min_confidence` to `0.75`
- Lower `critic_pass_threshold` slightly (e.g. `0.65`) to allow more FIX_REQUIRED samples rather than hard-rejecting
- Enable `reasoning_trace` and `preference` in `task_types` for richer diversity

### For faster generation

- Set `max_workers` to 4 (respects Ollama's parallel request capacity)
- Lower `max_tokens` to `1024` for shorter LLM calls
- Lower `max_retries` to `1` if the model is reliable

### For anti-collapse measures

- Enable all 5 task types for maximum output diversity
- Run `evolve` with 3+ rounds before generation
- Monitor `collapse_risk_score` in the dashboard — CRITICAL (≥ 0.70) means
  you should diversify your input sources immediately

### For multi-run dataset growth

- Do **not** use `--reset-fingerprints` unless intentionally restarting from scratch
- Add new input sources between runs — cross-run dedup ensures only genuinely
  new samples pass through the quality pipeline
- Use `list-runs` to audit run history and compare `config_hash` values to
  detect unintended config changes across runs
