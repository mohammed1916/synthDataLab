"""generation — Synthetic dataset generation layer."""
from .generator import DatasetGenerator
from .llm_client import OllamaClient, MockLLMClient
from .evolver import PromptEvolver, EvolveConfig, EvolvedPrompt

__all__ = [
    "DatasetGenerator",
    "OllamaClient",
    "MockLLMClient",
    "PromptEvolver",
    "EvolveConfig",
    "EvolvedPrompt",
]
