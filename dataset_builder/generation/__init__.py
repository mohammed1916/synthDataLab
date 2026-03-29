"""generation — Synthetic dataset generation layer."""
from .generator import DatasetGenerator
from .llm_client import OllamaClient, MockLLMClient

__all__ = ["DatasetGenerator", "OllamaClient", "MockLLMClient"]
