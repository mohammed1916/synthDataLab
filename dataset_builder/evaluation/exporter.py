"""
exporter.py — Export filtered datasets to external annotation tool formats.

Supported formats
-----------------
  argilla      JSON-Lines file compatible with Argilla v2 TextClassification /
               FeedbackDataset records.  Each line is a self-contained Argilla
               record that can be uploaded with the Argilla Python client:

                   import argilla as rg
                   rg.init(api_url="...", api_key="...")
                   records = [rg.FeedbackRecord(**r) for r in jsonl_lines]

  labelstudio  JSON file in Label Studio JSON import format.  Each object is
               an annotatable task compatible with the NLP Text Classification
               and NLP Text Summarization Label Studio templates.

Usage
-----
  from evaluation.exporter import export_argilla, export_labelstudio

  records = [json.loads(l) for l in Path("data/filtered_dataset.jsonl").open()]
  argilla_lines   = export_argilla(records)
  labelstudio_tasks = export_labelstudio(records)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Argilla field/question structure per task type
_ARGILLA_TASK_LABELS: dict[str, list[str]] = {
    "qa": ["correct", "partially_correct", "incorrect"],
    "extraction": ["complete", "partial", "missing"],
    "reasoning": ["valid", "flawed", "invalid"],
    "reasoning_trace": ["valid", "flawed", "invalid"],
    "preference": ["good", "acceptable", "poor"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Argilla export
# ─────────────────────────────────────────────────────────────────────────────

def export_argilla(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert dataset records to Argilla FeedbackRecord format (JSON-Lines).

    Each record becomes an Argilla FeedbackRecord with:
    - ``fields``: the input context and serialised output
    - ``metadata``: run_id, task_type, confidence, critic scores if present
    - ``suggestions``: auto-generated label hints from confidence / critic score
    - ``responses``: empty list (to be filled by human annotators)

    Args:
        records: List of DatasetSample-compatible dicts (e.g. from JSONL file).

    Returns:
        List of Argilla-compatible dicts, one per record.
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        task_type = rec.get("task_type", "unknown")
        input_text = str(rec.get("input", ""))
        output = rec.get("output", {})
        metadata = rec.get("metadata", {})

        # Serialise output as a readable string for the annotation UI
        try:
            output_text = json.dumps(output, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            output_text = str(output)

        # Derive a simple quality label hint from confidence/critic
        confidence: float = float(metadata.get("confidence", 0.0))
        critic: dict[str, Any] = metadata.get("critic", {})
        composite: float = float(critic.get("composite", confidence))
        labels = _ARGILLA_TASK_LABELS.get(task_type, ["good", "acceptable", "poor"])
        if composite >= 0.70:
            suggested_label = labels[0]
        elif composite >= 0.45:
            suggested_label = labels[1]
        else:
            suggested_label = labels[-1]

        argilla_record: dict[str, Any] = {
            "fields": {
                "input": input_text[:2000],   # cap at 2 K chars for UI
                "output": output_text[:4000],
            },
            "metadata": {
                "run_id": metadata.get("run_id", ""),
                "task_type": task_type,
                "source": metadata.get("source", rec.get("source_file", "")),
                "confidence": round(confidence, 4),
                "critic_composite": round(composite, 4),
                "critic_verdict": critic.get("verdict", ""),
            },
            "suggestions": [
                {
                    "question_name": "quality_label",
                    "value": suggested_label,
                    "score": round(composite, 4),
                    "agent": "synthDataLab-critic",
                }
            ],
            "responses": [],
        }
        out.append(argilla_record)

    logger.info("Exported %d records to Argilla format.", len(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Label Studio export
# ─────────────────────────────────────────────────────────────────────────────

def export_labelstudio(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert dataset records to Label Studio JSON task format.

    Each task has:
    - ``data``: the text fields shown in the Label Studio UI
    - ``predictions``: pre-annotations derived from critic scores (optional)
    - ``meta``: arbitrary metadata visible in the task details

    The output list should be saved as a single ``.json`` file and uploaded
    via Label Studio's "Import" → "JSON" workflow.

    Args:
        records: List of DatasetSample-compatible dicts.

    Returns:
        List of Label Studio task dicts.
    """
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(records, start=1):
        task_type = rec.get("task_type", "unknown")
        input_text = str(rec.get("input", ""))
        output = rec.get("output", {})
        metadata = rec.get("metadata", {})

        try:
            output_text = json.dumps(output, ensure_ascii=False)
        except (TypeError, ValueError):
            output_text = str(output)

        confidence: float = float(metadata.get("confidence", 0.0))
        critic: dict[str, Any] = metadata.get("critic", {})
        composite: float = float(critic.get("composite", confidence))

        # Derive a simple quality pre-annotation
        if composite >= 0.70:
            quality = "high"
        elif composite >= 0.45:
            quality = "medium"
        else:
            quality = "low"

        task: dict[str, Any] = {
            "id": i,
            "data": {
                "text": input_text[:3000],
                "output": output_text[:5000],
                "task_type": task_type,
            },
            "meta": {
                "sample_id": rec.get("id", f"sample_{i}"),
                "run_id": metadata.get("run_id", ""),
                "source": metadata.get("source", ""),
                "confidence": round(confidence, 4),
                "critic_composite": round(composite, 4),
                "critic_verdict": critic.get("verdict", ""),
            },
            "predictions": [
                {
                    "model_version": "synthDataLab-critic-v1",
                    "score": round(composite, 4),
                    "result": [
                        {
                            "from_name": "quality",
                            "to_name": "text",
                            "type": "choices",
                            "value": {"choices": [quality]},
                        }
                    ],
                }
            ],
            "annotations": [],
        }
        out.append(task)

    logger.info("Exported %d tasks to Label Studio format.", len(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face / Benchmark export
# ─────────────────────────────────────────────────────────────────────────────

def export_huggingface(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert dataset records to a flattened format suitable for Hugging Face Datasets.

    Standardises the schema:
    - ``instruction``: The system instruction plus input context
    - ``input``: The specific input text (if any)
    - ``output``: The serialised model response
    - ``metadata_*``: Flattened metadata fields (confidence, run_id, etc)

    Args:
        records: List of DatasetSample-compatible dicts.

    Returns:
        List of flattened dicts.
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        task_type = rec.get("task_type", "unknown")
        input_text = str(rec.get("input", ""))
        output = rec.get("output", {})
        metadata = rec.get("metadata", {})

        # Standardise the "instruction" and "output" based on task type
        # to match common fine-tuning formats (Alpaca / Instruction-tune)
        instruction = rec.get("instruction", "")
        
        # Flattened sample
        sample: dict[str, Any] = {
            "instruction": instruction,
            "input": input_text,
            "task_type": task_type,
            "id": rec.get("id", ""),
            "source": rec.get("source_file", ""),
            "confidence": float(metadata.get("confidence", 0.0)),
            "run_id": metadata.get("run_id", ""),
        }

        # Add task-specific outputs as columns for easier benchmarking
        if isinstance(output, dict):
            for k, v in output.items():
                # Avoid collision with standard columns
                col_name = f"output_{k}"
                if isinstance(v, (list, dict)):
                    sample[col_name] = json.dumps(v, ensure_ascii=False)
                else:
                    sample[col_name] = v
                    
            # Also provide a single serialised 'output' string
            sample["output"] = json.dumps(output, ensure_ascii=False)
        else:
            sample["output"] = str(output)

        out.append(sample)

    logger.info("Exported %d samples to Hugging Face / Benchmark format.", len(out))
    return out


def save_parquet(records: list[dict[str, Any]], output_path: str | Path) -> bool:
    """
    Try to save records as a Parquet file using pandas/pyarrow.
    
    Returns True if successful, False if dependencies are missing.
    """
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False, engine="pyarrow")
        return True
    except ImportError:
        logger.warning("Optional dependencies 'pandas' and 'pyarrow' are required for Parquet export.")
        return False
    except Exception as exc:
        logger.error("Failed to write Parquet file: %s", exc)
        return False
