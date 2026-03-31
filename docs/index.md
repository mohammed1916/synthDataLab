# SynthDataLab Documentation

Welcome to the SynthDataLab documentation hub. Use the pages below to go from
zero to a running pipeline, understand every configuration knob, and operate
the system in production.

---

## Contents

| Page                                              | Description                                                |
| ------------------------------------------------- | ---------------------------------------------------------- |
| [Quickstart](quickstart.md)                       | Install, run your first pipeline, see outputs in 5 minutes |
| [CLI Reference](cli-reference.md)                 | Every command, flag, and example                           |
| [Configuration](configuration.md)                 | Full config.py reference with defaults and tuning advice   |
| [Architecture](architecture.md)                   | System design, data-flow diagrams, module map              |
| [Multi-Agent System](multi-agent.md)              | Generator → Critic → Steering pipeline deep-dive           |
| [Metrics Guide](metrics-guide.md)                 | What every metric means and how to act on it               |
| [Production Deployment](production-deployment.md) | Docker, CI/CD, environment variables, monitoring           |

---

## What is SynthDataLab?

SynthDataLab is an **industrial-grade synthetic dataset pipeline** that:

1. **Ingests** source material (text files, JSON articles, images)
2. **Generates** structured training samples via an LLM (Ollama or mock)
3. **Validates** samples with schema rules + optional human review
4. **Filters** out low-quality, duplicate, or out-of-spec samples
5. **Evaluates** dataset health with 10+ quality metrics including
   Model-Collapse early-warning indicators
6. **Multi-agent mode** layers a heuristic Critic Agent and a Human
   Steering Gate on top of generation for supervised quality control

### Supported task types

| Task type         | Description                           | Key output fields                         |
| ----------------- | ------------------------------------- | ----------------------------------------- |
| `qa`              | Question/answer with evidence         | `question`, `answer`, `evidence`          |
| `extraction`      | Named-entity and relation extraction  | `entities`, `relations`, `key_facts`      |
| `reasoning`       | Multi-step chain-of-thought           | `reasoning_steps`, `conclusion`           |
| `reasoning_trace` | Visible `<think>` scratchpad + answer | `think`, `answer`, `verification`         |
| `preference`      | RLHF preference pairs                 | `chosen`, `rejected`, `preference_margin` |

### Key research motivation

SynthDataLab directly addresses **Model Collapse** (Shumailov et al., _Nature_ 2024) — the
phenomenon where models trained on synthetic data progressively lose diversity over
generations. The pipeline's collapse-risk metrics, vocabulary-entropy gauges, and
Evol-Instruct prompt evolution are all designed to keep generated data diverse.

---

## Quick navigation

```
make run-mock              # Run the full 6-step pipeline with mock LLM
make test                  # Run the 73-test pytest suite
python main.py --help      # See all CLI commands
python main.py generate-agent --mock --steering review-low
```
