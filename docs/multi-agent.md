# Multi-Agent System

The `generate-agent` command overlays a **supervised generation loop** on top
of the standard LLM generation stage.  Three agents collaborate:

1. **Generator Agent** — the same `DatasetGenerator` used in `run-all`,
   operated in streaming mode (one sample at a time via `generate_stream`).
2. **Critic Agent** — a heuristic quality scorer that runs immediately after
   each sample is produced, with no additional LLM calls.
3. **Steering Gate** — either automatic (threshold-based) or interactive
   (human TTY prompt), which gives the final accept / reject / fix verdict.

A fourth component — the **LiveMetricsTracker** — collects every event and
renders the live Rich dashboard throughout the run.

---

## Architecture diagram

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  MultiAgentOrchestrator.run(ingestion_results)                       │
  │                                                                      │
  │  for chunk × task_type in work_items:                                │
  │                                                                      │
  │    ┌──────────────────┐                                              │
  │    │  Generator Agent │  _generate_one(chunk, task_type)             │
  │    │  (DatasetGenerator)│ → DatasetSample (confidence 0–1)           │
  │    └────────┬─────────┘                                              │
  │             │ sample_dict                                             │
  │             ▼                                                         │
  │    ┌──────────────────┐                                              │
  │    │   Critic Agent   │  score(sample_dict)                          │
  │    │  (CriticAgent)   │  → CriticScore {relevance, coherence,        │
  │    └────────┬─────────┘              groundedness, fluency}          │
  │             │ composite 0–1                                           │
  │             ▼                                                         │
  │    ┌──────────────────────────────────────────────────────────────┐  │
  │    │                    Steering Gate                              │  │
  │    │                                                               │  │
  │    │  AUTO mode         composite ≥ pass_threshold  → ACCEPT       │  │
  │    │                    composite ≥ review_threshold → FIX_REQ     │  │
  │    │                    otherwise                   → REJECT       │  │
  │    │                                                               │  │
  │    │  REVIEW_LOW mode   composite < pass_threshold  → TTY prompt   │  │
  │    │                    otherwise                   → ACCEPT       │  │
  │    │                                                               │  │
  │    │  REVIEW_ALL mode   every sample                → TTY prompt   │  │
  │    └──────────────────────────────────────────────────────────────┘  │
  │             │ ACCEPT | REJECT | FIX_REQUIRED | QUIT                  │
  │             ▼                                                         │
  │    ┌──────────────────┐                                              │
  │    │ LiveMetricsTracker│ record(sample_id, task_type, conf,          │
  │    │  (Rich dashboard) │        critic_score, status)                │
  │    └──────────────────┘                                              │
  │             │  every 10 samples → compute_metrics → update risk      │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## CriticAgent scoring

The critic evaluates each sample on **four orthogonal axes** and returns a
`CriticScore` dataclass.

### Axis 1 — Relevance (0–1)

Measures whether the output content relates to the input passage, using
Jaccard word overlap rescaled to `[0, 1]`:

```
relevance = min(1.0, jaccard(input_words, output_words) × 2.5)
```

A pure hallucination scores near 0; a good summary scores near 0.8–1.0.

### Axis 2 — Coherence (0–1)

Structural completeness: what fraction of the task's **required output keys**
are present and non-trivial?

```
coherence = filled_required_keys / total_required_keys
          + bonus(extra_non_empty_keys, max=0.15)
```

Required keys per task type:

| Task type | Required keys |
|-----------|---------------|
| `qa` | `question`, `answer`, `evidence` |
| `extraction` | `entities`, `relations`, `key_facts` |
| `reasoning` | `reasoning_steps`, `conclusion`, `confidence_explanation` |
| `reasoning_trace` | `think`, `answer`, `verification`, `confidence` |
| `preference` | `prompt`, `chosen`, `rejected`, `preference_margin` |

### Axis 3 — Groundedness (0–1)

Are the evidence / reasoning fields anchored to the source passage?

```
groundedness = min(1.0, (evidence_words ∩ input_words) / evidence_words × 2.0)
```

Checks the "evidence" field for `qa`, "key_facts" for `extraction`,
"reasoning_steps" for `reasoning`, etc.

### Axis 4 — Fluency (0–1)

Surface-form quality:

- Hard 0 if output contains error markers (`_raw_response`, `_parse_error`, …)
- Hard 0 if output is empty
- Per-field penalty for fields shorter than 15 chars or longer than 10,000 chars

### Composite score and verdict

```
composite = (relevance + coherence + groundedness + fluency) / 4

composite ≥ 0.70  → PASS     (auto-accept in AUTO mode)
composite ≥ 0.45  → REVIEW   (FIX_REQUIRED in AUTO mode)
composite < 0.45  → FAIL     (auto-reject in AUTO mode)
```

The critic metadata is attached to `sample["metadata"]["critic"]` when
`save_critic_metadata=True` (default on), enabling post-run analysis.

---

## Steering modes

### AUTO

All decisions are made by the threshold comparison.  No user interaction.

```bash
python main.py generate-agent --mock --steering auto --threshold 0.70
```

### REVIEW_LOW

The human is shown only samples that score below `--threshold`.
High-scoring samples flow through automatically.  This is the most practical
mode: you only spend time reviewing borderline cases.

```bash
python main.py generate-agent --mock --steering review-low --threshold 0.65
```

When the human review prompt appears:

```
── Human Review (7/15) ─────────────────────────────────────
╭─ Critics Scores ──────────────────────╮
│      Task type  reasoning             │
│    Relevance    0.48                  │
│    Coherence    0.55                  │
│  Groundedness   0.32                  │
│      Fluency    0.91                  │
│    Composite    0.565 → REVIEW        │
╰───────────────────────────────────────╯
  [a] approve   [r] reject   [f] fix-required   [s] skip   [q] quit   [?] show sample
  Action:
```

### REVIEW_ALL

Every sample is shown to the human.  Best for small batches or auditing a
new model's output quality.

```bash
python main.py generate-agent --mock --steering review-all
```

---

## Human steering keys

| Key | Action |
|-----|--------|
| `a` (or Enter) | **Approve** — add to accepted list |
| `r` | **Reject** — add to rejected list (not saved) |
| `f` | **Fix required** — add to fix_required list |
| `s` | **Skip** — pass through without critic judgement (treat as ACCEPT) |
| `q` | **Quit** — stop generation; save what was collected so far |
| `?` | **Show** — pretty-print the full sample JSON |

---

## Collapse monitoring

Every `collapse_check_interval` (default 10) samples the orchestrator calls
`compute_metrics(accepted_so_far)` and pushes the result to the dashboard:

| Score | Label | Dashboard colour | Recommended action |
|-------|-------|------------------|--------------------|
| < 0.30 | LOW | 🟢 Green | None |
| 0.30–0.49 | MEDIUM | 🟡 Yellow | Monitor |
| 0.50–0.69 | HIGH | 🔴 Red | Diversify input sources |
| ≥ 0.70 | CRITICAL | 🔴 Red bold | Halt; add new sources; re-run |

A `CRITICAL` reading also emits a `WARNING` log line.

---

## OrchestrationResult

`MultiAgentOrchestrator.run()` returns an `OrchestrationResult`:

```python
result.accepted          # List[Dict] — approved samples
result.rejected          # List[Dict] — discarded samples
result.fix_required      # List[Dict] — samples needing repair
result.total_generated   # int
result.acceptance_rate   # float in [0, 1]
result.critic_scores     # List[CriticScore]
result.metrics_snapshot  # Dict with DatasetMetrics fields (on accepted)
result.aborted           # bool — True if user pressed 'q'
result.summary()         # compact dict for logging
```

Both `accepted` and `fix_required` samples are written to the output JSONL by
the `generate-agent` command.

---

## Extending the multi-agent system

### Add a new scoring axis

1. Open `generation/critic_agent.py`.
2. Add a new method `_score_<axis>(self, …) -> float`.
3. Add the field to `CriticScore` and include it in the `composite` formula.

### Add a new steering mode

1. Add a value to `SteeringMode` in `generation/orchestrator.py`.
2. Handle it in `MultiAgentOrchestrator._apply_steering()`.

### Make the critic call a real LLM

Replace `CriticAgent.score()` with an LLM call that asks: *"Rate this sample
on a scale of 0–10 for relevance, coherence, groundedness, and fluency."*
Parse the response into a `CriticScore`.  The rest of the system is unchanged.
