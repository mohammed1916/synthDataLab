"""evaluation — Metrics computation and report generation."""
from .metrics import DatasetMetrics, compute_metrics
from .reporter import MetricsReporter

__all__ = ["DatasetMetrics", "compute_metrics", "MetricsReporter"]
