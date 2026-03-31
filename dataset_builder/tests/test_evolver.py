"""tests/test_evolver.py — Unit tests for Evol-Instruct prompt evolution."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from generation.evolver import PromptEvolver, EvolveConfig, EvolvedPrompt


SEEDS = [
    "What is machine learning?",
    "Explain how neural networks work.",
    "Describe the role of attention in transformers.",
    "How does gradient descent optimise model weights?",
]


# ── Configuration ─────────────────────────────────────────────────────────────

def test_evolve_config_defaults():
    cfg = EvolveConfig()
    assert cfg.n_rounds >= 1
    assert len(cfg.operations) > 0
    assert cfg.use_llm_evolution is False


# ── Evolved prompt structure ──────────────────────────────────────────────────

def test_evolved_prompts_have_required_fields():
    evolver = PromptEvolver(EvolveConfig(n_rounds=1), seed=0)
    evolved = evolver.evolve(SEEDS[:2])
    for ep in evolved:
        assert isinstance(ep, EvolvedPrompt)
        assert ep.prompt
        assert ep.seed_prompt
        assert ep.operation
        assert ep.round_number >= 1
        assert 0.0 <= ep.complexity_score <= 1.0


def test_evolve_produces_output_for_each_seed_per_round():
    ops = ["add_constraints", "deepen"]
    cfg = EvolveConfig(n_rounds=1, operations=ops)
    evolver = PromptEvolver(cfg, seed=1)
    seeds = SEEDS[:2]
    evolved = evolver.evolve(seeds)
    # Each seed × each op × n_rounds = 2×2×1 = 4 total (before quality filter)
    assert len(evolved) <= len(seeds) * len(ops) * cfg.n_rounds
    assert len(evolved) >= 1


# ── Operations ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("op", ["add_constraints", "deepen", "concretise", "increase_reasoning"])
def test_each_operation_produces_evolved_prompt(op):
    cfg = EvolveConfig(n_rounds=1, operations=[op])
    evolver = PromptEvolver(cfg, seed=42)
    evolved = evolver.evolve(["What is machine learning?"])
    surviving = [e for e in evolved if not e.discarded]
    assert len(surviving) >= 1, f"Operation '{op}' produced no surviving prompts"
    assert all(e.operation == op for e in surviving)


def test_evolved_prompt_differs_from_seed():
    cfg = EvolveConfig(n_rounds=1, operations=["add_constraints"])
    evolver = PromptEvolver(cfg, seed=7)
    evolved = evolver.evolve(["What is machine learning?"])
    for ep in evolved:
        if not ep.discarded:
            assert ep.prompt != ep.seed_prompt


# ── Quality filter ────────────────────────────────────────────────────────────

def test_quality_filter_discards_identical_prompts():
    """Force a degenerate case where evolved == seed to confirm discard fires."""
    from generation.evolver import EvolvedPrompt
    cfg = EvolveConfig(n_rounds=1, operations=["add_constraints"])
    evolver = PromptEvolver(cfg, seed=99)
    ep = EvolvedPrompt(
        prompt="same prompt", seed_prompt="same prompt",
        operation="add_constraints", round_number=1,
        complexity_score=0.0, discarded=False, discard_reason="",
    )
    evolver._quality_filter(ep, "same prompt")
    assert ep.discarded


def test_quality_filter_discards_too_short():
    from generation.evolver import EvolvedPrompt
    cfg = EvolveConfig(n_rounds=1)
    evolver = PromptEvolver(cfg)
    seed = "A much longer seed prompt that covers a lot of ground in the domain."
    short_evolved = "Hi."
    ep = EvolvedPrompt(
        prompt=short_evolved, seed_prompt=seed,
        operation="deepen", round_number=1,
        complexity_score=0.0, discarded=False, discard_reason="",
    )
    evolver._quality_filter(ep, seed)
    assert ep.discarded


def test_quality_filter_accepts_good_evolution():
    from generation.evolver import EvolvedPrompt
    cfg = EvolveConfig(n_rounds=1)
    evolver = PromptEvolver(cfg)
    seed = "Explain neural networks."
    evolved = (
        "Explain neural networks, specifically focusing on backpropagation. "
        "Include a discussion of vanishing gradients and how residual connections mitigate them."
    )
    ep = EvolvedPrompt(
        prompt=evolved, seed_prompt=seed,
        operation="deepen", round_number=1,
        complexity_score=0.0, discarded=False, discard_reason="",
    )
    evolver._quality_filter(ep, seed)
    assert not ep.discarded


# ── Complexity scoring ────────────────────────────────────────────────────────

def test_complexity_score_increases_with_constraints():
    evolver = PromptEvolver(EvolveConfig())
    simple = "What is AI?"
    complex_ = (
        "What is AI? Focus on symbolic reasoning versus neural approaches. "
        "Include at least three concrete examples, avoid vague generalizations, "
        "and justify each claim with evidence from published research."
    )
    assert evolver._complexity_score(complex_, simple) > evolver._complexity_score(simple, simple)


# ── Multi-round evolution ─────────────────────────────────────────────────────

def test_multi_round_evolution_traces_lineage():
    cfg = EvolveConfig(n_rounds=2, operations=["deepen"])
    evolver = PromptEvolver(cfg, seed=5)
    evolved = evolver.evolve(["What is AI?"])
    surviving = [e for e in evolved if not e.discarded]
    round_nums = {e.round_number for e in evolved}
    assert 1 in round_nums
    assert 2 in round_nums


def test_to_dict_is_serialisable():
    import json
    cfg = EvolveConfig(n_rounds=1, operations=["deepen"])
    evolver = PromptEvolver(cfg, seed=0)
    evolved = evolver.evolve(["Test prompt."])
    for ep in evolved:
        d = ep.to_dict()
        json.dumps(d)   # must not raise
