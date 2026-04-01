"""prompts — Prompt templates and few-shot example library."""
from .few_shot_examples import FEW_SHOT_EXAMPLES
from .templates import PromptTemplates, TaskType

__all__ = ["PromptTemplates", "TaskType", "FEW_SHOT_EXAMPLES"]
