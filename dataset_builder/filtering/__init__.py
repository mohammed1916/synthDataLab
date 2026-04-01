"""filtering — Quality filtering pipeline."""
from .deduplicator import Deduplicator
from .pipeline import FilteringPipeline, FilteringReport

__all__ = ["FilteringPipeline", "FilteringReport", "Deduplicator"]
