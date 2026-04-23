"""
benchmark_engine.py — Benchmark external models on the CBSE Math dataset.

This script loads the generated math benchmark dataset and runs performance
tests across multiple models to evaluate their accuracy and reasoning quality.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

# --- Ensure project root is on path ---
import sys
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from generation.llm_client import OllamaClient
except ImportError:
    OllamaClient = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("math_benchmark")

class MathBenchmarker:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self.console = Console()
        self.results = []

    def load_dataset(self) -> pd.DataFrame:
        if self.dataset_path.suffix == ".parquet":
            return pd.read_parquet(self.dataset_path)
        elif self.dataset_path.suffix == ".jsonl":
            return pd.read_json(self.dataset_path, lines=True)
        else:
            raise ValueError(f"Unsupported format: {self.dataset_path.suffix}")

    def run_benchmark(self, models: list[str], limit: int = 5):
        df = self.load_dataset()
        if limit:
            df = df.head(limit)

        self.console.print(f"[bold green]Starting benchmark on {len(df)} samples across {len(models)} models...[/bold green]\n")

        for idx, row in df.iterrows():
            content = row["content"]
            # content is already a dict in parquet if saved correctly, or a serialised JSON
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except:
                    pass
            
            # MathSample content varies by item_type (problem, explanation, fill_gap)
            item_type = row.get("item_type", "problem")
            if item_type == "explanation":
                question = f"Explain the concept of {row.get('subtopic')} in {row.get('chapter_title')}."
                ground_truth = content.get("summary", "")
            else:
                question = content.get("question_latex", "")
                ground_truth = content.get("answer_latex", "")
            
            if not question:
                continue

            self.console.print(f"[bold]Item {idx+1} ({item_type}):[/bold] {question[:100]}...")

            for model_name in models:
                start_time = time.monotonic()
                response = self._get_model_response(model_name, question)
                elapsed = time.monotonic() - start_time
                
                # Simple evaluation: exact match (cleaned) or keyword pass
                score = self._evaluate_math(response, ground_truth)
                
                self.results.append({
                    "item_id": idx,
                    "model": model_name,
                    "response": response,
                    "ground_truth": ground_truth,
                    "score": score,
                    "latency": elapsed
                })
                
                status = "[green]CORRECT[/green]" if score == 1.0 else "[red]INCORRECT[/red]"
                self.console.print(f"  ↳ [cyan]{model_name:<12}[/cyan] | {status} | {elapsed:.2f}s")

    def _get_model_response(self, model_name: str, question: str) -> str:
        if model_name == "mock":
            # Mock correct answer half the time
            return "Mock solution with $x=1$ or $2$." if time.time() % 2 > 1 else "Wrong answer $x=5$."
        
        if OllamaClient:
            client = OllamaClient(model=model_name)
            prompt = f"Solve this CBSE Math problem. Output only the final answer in LaTeX format within $...$ at the end.\n\nProblem: {question}"
            try:
                return client.complete(system_prompt="You are a brilliant Math teacher.", user_prompt=prompt)
            except Exception as e:
                return f"Error: {e}"
        return "Model client not available"

    def _evaluate_math(self, response: str, ground_truth: str) -> float:
        """Heuristic math evaluation."""
        if not response or not ground_truth:
            return 0.0
        
        # Strip LaTeX fluff
        def clean(s):
            return s.replace("$", "").replace(" ", "").replace("\\text{", "").replace("}", "").lower()
        
        c_response = clean(response)
        c_gt = clean(ground_truth)
        
        if c_gt in c_response:
            return 1.0
        
        return 0.0

    def print_report(self):
        df_res = pd.DataFrame(self.results)
        summary = df_res.groupby("model").agg({
            "score": "mean",
            "latency": "mean"
        }).reset_index()

        table = Table(title="[bold blue]Model Benchmark Results[/bold blue]")
        table.add_column("Model")
        table.add_column("Accuracy", justify="right")
        table.add_column("Avg Latency", justify="right")

        for _, row in summary.iterrows():
            table.add_row(
                row["model"],
                f"{row['score']:.1%}",
                f"{row['latency']:.2f}s"
            )
        
        self.console.print("\n")
        self.console.print(table)

if __name__ == "__main__":
    dataset = "/Users/MohammedIbrahim/Documents/a/synthDataLab/dataset_builder/kaggle_upload/math_benchmark.parquet"
    benchmarker = MathBenchmarker(dataset)
    # Testing with gemma3 and a mock comparison
    benchmarker.run_benchmark(models=["gemma3:4b", "mock"], limit=3)
    benchmarker.print_report()
