"""prompts — Prompt templates and few-shot example library."""
from .templates import PromptTemplates, TaskType
from .few_shot_examples import FEW_SHOT_EXAMPLES

__all__ = ["PromptTemplates", "TaskType", "FEW_SHOT_EXAMPLES"]
