# Metrics Guide

SynthDataLab computes **10 quality metrics** across every dataset, split into
three groups: _structural quality_, _confidence stats_, and _collapse
early-warning_.

Metrics are computed twice — on the raw (pre-filter) dataset and the filtered
dataset — so the report shows before-vs-after improvement.

---

## Quick reference table

| #   | Metric                 | Range | Higher = ? | Collapse relevance |
| --- | ---------------------- | ----- | ---------- | ------------------ |
| 1   | Schema Validity Rate   | 0–1   | Better     | —                  |
| 2   | Task Consistency Score | 0–1   | Better     | —                  |
| 3   | Completeness Score     | 0–1   | Better     | —                  |
| 4   | Hallucination Rate     | 0–1   | **Worse**  | Indirect           |
| 5   | Diversity Score (TTR)  | 0–1   | Better     | ✓ Core             |
| 6   | Mean Confidence        | 0–1   | Better     | —                  |
| 7   | Min / Max Confidence   | 0–1   | —          | —                  |
| 8   | Vocabulary Entropy     | bits  | Better     | ✓ Core             |
| 9   | Bigram Entropy         | bits  | Better     | ✓ Core             |
| 10  | Collapse Risk Score    | 0–1   | **Worse**  | ✓ Composite        |

---

## Structural quality metrics

### 1. Schema Validity Rate

**What it measures:** What fraction of samples pass JSON schema validation?

```
schema_validity_rate = valid_samples / total_samples
```

A sample is _valid_ if it matches the JSON Schema registered for its task
type in `schema/dataset_schema.py`.

**Target:** ≥ 0.95 after filtering.

**If low:** Check model temperature (too high → malformed JSON); check
`max_tokens` (too low → truncated responses).

---

### 2. Task Consistency Score

**What it measures:** Does each sample's `output` dict contain the keys
_expected_ for its declared `task_type`?

```
task_consistency_score = consistent_samples / total_samples
```

A sample is consistent if all _required_ output keys are present and
non-empty (e.g., a `qa` sample must have `question`, `answer`, `evidence`).

**Target:** ≥ 0.90 after filtering.

**If low:** The LLM is ignoring the output schema. Lower temperature, add
more few-shot examples in `prompts/few_shot_examples.py`, or tighten the
system prompt.

---

### 3. Completeness Score

**What it measures:** Mean fraction of required output fields that are
populated (non-null, non-empty) across all samples.

```
completeness_score = mean(present_fields / required_fields  for each sample)
```

Unlike Task Consistency, this is a fractional score — a sample with 2 of 3
required fields scores 0.67, not 0.

**Target:** ≥ 0.85 after filtering.

---

### 4. Hallucination Rate

**What it measures:** For `qa` task samples — what fraction of answers share
less than 20% word overlap with the input passage?

```
hallucination_rate = hallucinated_qa_samples / total_qa_samples
```

A low overlap is a proxy for factual hallucination: the model is making up
content not supported by the source.

**Target:** ≤ 0.15 (less than 15% of QA samples hallucinated).

**Note:** This is a heuristic. High-quality paraphrasing can trigger false
positives. Pair with human spot-checks.

---

## Confidence statistics

### 5. Mean Confidence

**What it measures:** The mean of the `metadata.confidence` field across all
samples. Confidence is estimated at generation time based on output
completeness and field coverage.

**Target:** ≥ 0.75.

---

### 6. Min / Max Confidence

Useful for spotting outliers. A very low `min_confidence` (< 0.3) usually
indicates that a few samples completely failed to parse. These should be
filtered out.

---

## Collapse early-warning metrics

These metrics directly address **Model Collapse** (Shumailov et al., _Nature_ 2024) — the progressive lexical and semantic narrowing that occurs when a
model is repeatedly fine-tuned on its own outputs.

### 7. Diversity Score (Type-Token Ratio)

**What it measures:** Lexical diversity of all output text, computed as the
Type-Token Ratio (TTR):

```
diversity_score = unique_tokens / total_tokens
```

TTR = 1.0 means every token is unique; TTR = 0.0 means all tokens are
identical.

**Healthy range:** ≥ 0.35.

**Why it matters:** A collapsing dataset has decreasing TTR because the model
keeps generating the same phrases and patterns.

---

### 8. Vocabulary Entropy

**What it measures:** Shannon entropy of the unigram token frequency
distribution:

```
H_vocab = - Σ p(token) × log₂(p(token))   bits
```

Higher entropy = more even/spread distribution = more diverse vocabulary.

**Healthy range:** ≥ 6.0 bits.

**Warning:** < 6.0 bits suggests vocabulary is suspiciously narrow — the
model may be defaulting to a small set of template phrases.

---

### 9. Bigram Entropy

**What it measures:** Same Shannon entropy, but over bigrams (pairs of
consecutive tokens).

Higher bigram entropy = more varied sentence structure.

**Healthy range:** ≥ 8.0 bits.

---

### 10. Collapse Risk Score

**What it measures:** A composite 0–1 risk index that combines diversity
score, vocabulary entropy, and hallucination rate into a single collapse
signal:

```python
entropy_penalty  = max(0, (6.0 - vocab_entropy) / 6.0)
ttr_penalty      = max(0, (0.35 - diversity) / 0.35)
halluc_penalty   = hallucination_rate

collapse_risk = clamp(
    0.4 × entropy_penalty
  + 0.4 × ttr_penalty
  + 0.2 × halluc_penalty,
  min=0.0, max=1.0
)
```

**Thresholds:**

| Score     | Label    | Recommended action                      |
| --------- | -------- | --------------------------------------- |
| < 0.30    | LOW      | No action needed                        |
| 0.30–0.49 | MEDIUM   | Monitor; consider more input diversity  |
| 0.50–0.69 | HIGH     | Add new source materials immediately    |
| ≥ 0.70    | CRITICAL | Halt generation; investigation required |

---

## Live dashboard metrics (generate-agent mode)

When using `generate-agent`, additional **real-time driver metrics** are shown
in the live Rich dashboard as each sample is produced:

| Metric                        | Where shown            | Update frequency |
| ----------------------------- | ---------------------- | ---------------- |
| Progress (N/total, %)         | Dashboard header       | Every sample     |
| Throughput (samp/sec)         | Dashboard header       | Every sample     |
| ETA                           | Dashboard header       | Every sample     |
| Decisions (ACCEPT/REJECT/FIX) | Decisions panel        | Every sample     |
| Task type distribution        | Task Types panel       | Every sample     |
| Mean confidence               | Quality / Health panel | Every sample     |
| Mean critic score             | Quality / Health panel | Every sample     |
| Collapse risk label + score   | Quality / Health panel | Every 10 samples |
| Recent sample log             | Recent Samples panel   | Every sample     |

See the [Multi-Agent guide](multi-agent.md) for the full dashboard screenshot
and how to interpret the collapse gauge.

---

## Reading the metrics report

After `run-all` or `evaluate`, `data/metrics_report.json` contains a JSON
object with:

```json
{
  "raw": { ... DatasetMetrics fields ... },
  "filtered": { ... DatasetMetrics fields ... },
  "improvement": {
    "schema_validity_rate": +0.12,
    "diversity_score": +0.05,
    "hallucination_rate": -0.08,
    ...
  },
  "filter_report": { ... FilterReport fields ... },
  "generated_at": "2026-03-31T10:25:00Z"
}
```

Use `evaluate` to regenerate it any time after the pipeline has run.

---

## Acting on warnings

| Warning                         | Likely cause                 | Fix                                                      |
| ------------------------------- | ---------------------------- | -------------------------------------------------------- |
| `schema_validity_rate < 0.80`   | LLM producing malformed JSON | Lower temperature; add `json_mode` if Ollama supports it |
| `task_consistency_score < 0.80` | LLM ignoring task schema     | Improve system prompt; add few-shot examples             |
| `hallucination_rate > 0.25`     | Source passages too short    | Use longer, richer input texts                           |
| `diversity_score < 0.35`        | Collapse risk emerging       | Enable all 5 task types; run Evol-Instruct evolution     |
| `collapse_risk_score ≥ 0.70`    | Active collapse              | Immediately diversify input; retrain with external data  |
