"""tests/test_integration.py — End-to-end integration tests using mock LLM."""
import sys
import json
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from config import Config
from ingestion.ingestor import Ingestor
from generation.generator import DatasetGenerator
from generation.llm_client import MockLLMClient
from validation.rule_validator import RuleValidator
from filtering.pipeline import FilteringPipeline
from evaluation.metrics import compute_metrics
from schema.dataset_schema import DatasetSample, validate_sample
from validation.annotation import AnnotationLabel


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_config(tmp_path):
    cfg = Config()
    cfg.llm.provider = "mock"
    cfg.storage.data_dir = tmp_path / "data"
    cfg.storage.data_dir.mkdir(parents=True)
    cfg.generation.task_types = ["qa", "extraction", "reasoning"]
    cfg.generation.max_workers = 1
    return cfg


SAMPLE_TEXT = (
    "Machine learning is a field of artificial intelligence that uses statistical techniques "
    "to give computer systems the ability to learn from data. "
    "Neural networks are inspired by the structure of the human brain and consist of "
    "interconnected layers of nodes that process information in parallel. "
    "The transformer architecture introduced the attention mechanism that allows models "
    "to weigh the importance of different parts of the input."
)


# ── Ingestion ─────────────────────────────────────────────────────────────────

def test_ingest_text_produces_chunks():
    ingestor = Ingestor()
    results = ingestor.ingest_text(SAMPLE_TEXT, source_name="test")
    assert len(results) >= 1
    for r in results:
        assert r.content
        assert r.metadata


def test_ingest_file_size_guard(tmp_path):
    large_file = tmp_path / "big.txt"
    # Write exactly 51MB
    large_file.write_bytes(b"x" * (51 * 1024 * 1024))
    ingestor = Ingestor()
    with pytest.raises(ValueError, match="too large"):
        ingestor.ingest_file(str(large_file))


def test_ingest_json_articles(tmp_path):
    articles = [
        {"title": "AI", "content": SAMPLE_TEXT, "source": "test"},
        {"title": "ML", "content": "Deep learning uses many-layered neural networks.", "source": "test"},
    ]
    json_file = tmp_path / "articles.json"
    json_file.write_text(json.dumps(articles), encoding="utf-8")
    ingestor = Ingestor()
    results = ingestor.ingest_json(str(json_file))
    assert len(results) >= 2


def test_ingest_nonexistent_file_raises():
    ingestor = Ingestor()
    with pytest.raises(FileNotFoundError):
        ingestor.ingest_file("/nonexistent/path/file.txt")


# ── Generation ────────────────────────────────────────────────────────────────

def test_mock_generation_produces_samples(mock_config):
    ingestor = Ingestor()
    chunks = ingestor.ingest_text(SAMPLE_TEXT, source_name="test")
    generator = DatasetGenerator(config=mock_config, llm_client=MockLLMClient(seed=42))
    samples = generator.generate_from_ingestion(chunks)
    assert len(samples) == len(chunks) * len(mock_config.generation.task_types)


def test_generated_samples_have_required_fields(mock_config):
    ingestor = Ingestor()
    chunks = ingestor.ingest_text(SAMPLE_TEXT, source_name="test")
    generator = DatasetGenerator(config=mock_config, llm_client=MockLLMClient(seed=1))
    samples = generator.generate_from_ingestion(chunks)
    for s in samples:
        d = s.to_dict()
        assert "id" in d
        assert "task_type" in d
        assert "output" in d
        assert "metadata" in d


def test_parallel_generation_same_count_as_sequential(mock_config):
    ingestor = Ingestor()
    chunks = ingestor.ingest_text(SAMPLE_TEXT * 3, source_name="test")

    mock_config.generation.max_workers = 1
    gen_seq = DatasetGenerator(config=mock_config, llm_client=MockLLMClient(seed=0))
    seq_samples = gen_seq.generate_from_ingestion(chunks)

    mock_config.generation.max_workers = 3
    gen_par = DatasetGenerator(config=mock_config, llm_client=MockLLMClient(seed=0))
    par_samples = gen_par.generate_from_ingestion(chunks)

    assert len(seq_samples) == len(par_samples)


# ── Validation pipeline ───────────────────────────────────────────────────────

def test_validation_accepts_reject_sums_to_total(mock_config):
    ingestor = Ingestor()
    chunks = ingestor.ingest_text(SAMPLE_TEXT, source_name="test")
    generator = DatasetGenerator(config=mock_config, llm_client=MockLLMClient(seed=42))
    samples = generator.generate_from_ingestion(chunks)

    rv = RuleValidator(min_confidence=mock_config.filtering.min_confidence)
    annotated = rv.validate_batch([s.to_dict() for s in samples])

    total = len(annotated)
    a = sum(1 for a in annotated if a.label == AnnotationLabel.ACCEPT)
    r = sum(1 for a in annotated if a.label == AnnotationLabel.REJECT)
    f = sum(1 for a in annotated if a.label == AnnotationLabel.FIX_REQUIRED)
    assert a + r + f == total


# ── Filtering ─────────────────────────────────────────────────────────────────

def test_filtering_retains_high_quality_samples(mock_config):
    ingestor = Ingestor()
    chunks = ingestor.ingest_text(SAMPLE_TEXT * 5, source_name="test")
    generator = DatasetGenerator(config=mock_config, llm_client=MockLLMClient(seed=99))
    samples = generator.generate_from_ingestion(chunks)

    rv = RuleValidator(min_confidence=mock_config.filtering.min_confidence)
    annotated = rv.validate_batch([s.to_dict() for s in samples])

    pipeline = FilteringPipeline(mock_config.filtering)
    filtered, report = pipeline.run(annotated)

    assert len(filtered) <= len(annotated)
    assert report.overall_retention_rate <= 1.0
    assert report.overall_retention_rate >= 0.0


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_compute_metrics_on_generated_samples(mock_config):
    ingestor = Ingestor()
    chunks = ingestor.ingest_text(SAMPLE_TEXT * 3, source_name="test")
    generator = DatasetGenerator(config=mock_config, llm_client=MockLLMClient(seed=77))
    samples = generator.generate_from_ingestion(chunks)

    m = compute_metrics([s.to_dict() for s in samples])
    assert m.total_samples == len(samples)
    assert 0.0 <= m.schema_validity_rate <= 1.0
    assert m.vocabulary_entropy >= 0.0
    assert m.bigram_entropy >= 0.0
    assert 0.0 <= m.collapse_risk_score <= 1.0


# ── Atomic writes ──────────────────────────────────────────────────────────────

def test_save_jsonl_is_atomic_no_partial_on_error(tmp_path):
    """If _save_jsonl fails mid-write, the original file must not be corrupted."""
    import contextlib
    from main import _save_jsonl

    out = tmp_path / "out.jsonl"
    # Write initial good content
    _save_jsonl([{"a": 1}], out)
    assert out.exists()

    original = out.read_text()

    # Introduce a non-serialisable object to force failure
    bad = [{"a": object()}]
    with pytest.raises(TypeError):
        _save_jsonl(bad, out)

    # Original file must still be intact
    assert out.read_text() == original
    # No leftover .tmp files
    assert list(tmp_path.glob("*.tmp")) == []
