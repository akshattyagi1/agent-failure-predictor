"""Run known-answer tasks through the real local agent and label the outcomes."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .generate_dataset import DEFAULT_OUTPUT_ROOT, build_processed_data, read_jsonl, write_json
from .runner import run_agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK_BANK = PROJECT_ROOT / "data" / "tasks" / "real_evaluation_task_bank.jsonl"


def load_evaluation_tasks(path: Path | str) -> list[dict[str, Any]]:
    """Load tasks with explicit, version-controlled evaluation rules."""
    tasks = read_jsonl(Path(path))
    if not tasks:
        raise ValueError("Evaluation task bank must contain at least one task.")
    for task in tasks:
        if not {"task_id", "task_type", "task", "evaluation"} <= set(task):
            raise ValueError(f"Task is missing required fields: {task}")
    return tasks


def answer_matches(answer: str, evaluation: dict[str, Any]) -> bool:
    """Apply a deterministic task-specific success rule to a completed answer."""
    rule_type = evaluation["type"]
    normalized = answer.casefold()
    if rule_type == "contains_all":
        return all(str(expected).casefold() in normalized for expected in evaluation["expected"])
    if rule_type == "numeric":
        expected = float(evaluation["expected"])
        values = [float(value) for value in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", answer)]
        return any(math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9) for value in values)
    raise ValueError(f"Unknown evaluation type: {rule_type}")


def run_real_evaluation(
    number_of_runs: int,
    model: str,
    task_bank: Path = DEFAULT_TASK_BANK,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    experiment_name: str | None = None,
    prediction_endpoint: str | None = None,
    resume: bool = False,
) -> Path:
    """Create an immutable experiment of genuine agent behavior and judged outcomes."""
    if number_of_runs < 1:
        raise ValueError("number_of_runs must be at least 1.")
    tasks = load_evaluation_tasks(task_bank)
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = output_root / (experiment_name or f"real_ollama_{created}")
    if experiment.exists() and not resume:
        raise FileExistsError(f"Experiment already exists: {experiment}")
    raw_directory = experiment / "raw"
    processed_directory = experiment / "processed"
    evaluations_path = raw_directory / "evaluations.jsonl"
    if resume:
        if not evaluations_path.exists():
            raise ValueError(
                "Cannot resume: this experiment has no incremental evaluation records. Start a new experiment name."
            )
        evaluations = read_jsonl(evaluations_path)
        raw_runs = read_jsonl(raw_directory / "runs.jsonl")
        if len(raw_runs) != len(evaluations):
            raise ValueError("Cannot resume: raw runs and judged evaluations are out of sync.")
        manifest_runs = [
            {"run_id": row["run_id"], "run_index": index, "task_id": row["task_id"],
             "task_type": row["task_type"], "model": model}
            for index, row in enumerate(evaluations)
        ]
    else:
        raw_directory.mkdir(parents=True)
        evaluations = []
        manifest_runs = []

    def write_manifest(status: str) -> None:
        manifest = {
            "synthetic": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "model": model,
            "number_of_runs": number_of_runs,
            "completed_runs": len(evaluations),
            "task_bank": str(task_bank),
            "prediction_endpoint": prediction_endpoint,
            "runs": manifest_runs,
        }
        # Windows can occasionally reject a rapid rewrite of a file temporarily
        # held by an editor, indexer, or sync client. The telemetry is written
        # first, so retry only this derived progress snapshot.
        for attempt in range(3):
            try:
                write_json(experiment / "manifest.json", manifest)
                return
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.25 * (attempt + 1))

    write_manifest("running")
    for index in range(len(evaluations), number_of_runs):
        task = tasks[index % len(tasks)]
        before_runs = read_jsonl(raw_directory / "runs.jsonl")
        answer = ""
        execution_error: str | None = None
        try:
            answer = run_agent(
                task["task"], model, raw_directory, prediction_endpoint=prediction_endpoint,
                environment_id=task.get("environment_id", "default"), declared_task_type=task["task_type"],
            )
        except RuntimeError as error:
            execution_error = str(error)
        after_runs = read_jsonl(raw_directory / "runs.jsonl")
        if len(after_runs) != len(before_runs) + 1:
            raise RuntimeError("Expected exactly one completed run record per evaluation task.")
        run = after_runs[-1]
        task_success = execution_error is None and answer_matches(answer, task["evaluation"])
        failure_type = None if task_success else ("agent_execution_failure" if execution_error else "incorrect_answer")
        evaluation = {
            "run_id": run["run_id"],
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "task_template_id": task.get("task_template_id", task["task_id"]),
            "environment_id": task.get("environment_id", "default"),
            "agent_answer": answer,
            "evaluation_rule": task["evaluation"],
            "task_success": task_success,
            "failed": not task_success,
            "failure_type": failure_type,
            "execution_error": execution_error,
        }
        evaluations.append(evaluation)
        with evaluations_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(evaluation, sort_keys=True) + "\n")
        manifest_runs.append({
            "run_id": run["run_id"], "run_index": index, "task_id": task["task_id"],
            "task_type": task["task_type"], "task_template_id": task.get("task_template_id", task["task_id"]),
            "environment_id": task.get("environment_id", "default"), "model": model,
        })
        write_manifest("running")
    outcome_overrides = {
        row["run_id"]: {"failed": row["failed"], "failure_type": row["failure_type"], "task_success": row["task_success"]}
        for row in evaluations
    }
    build_processed_data(
        raw_directory, processed_directory, manifest_runs,
        synthetic=False, outcome_overrides=outcome_overrides,
    )
    write_manifest("completed")
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and evaluate genuine local Ollama-agent tasks.")
    parser.add_argument("--runs", type=int, default=30, help="Use a multiple of 30 to cover every evaluation task evenly.")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--task-bank", type=Path, default=DEFAULT_TASK_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-name")
    parser.add_argument("--prediction-endpoint", help="Optional local FastAPI URL for shadow-mode scores.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted checkpointing-enabled experiment.")
    args = parser.parse_args()
    experiment = run_real_evaluation(
        args.runs, args.model, args.task_bank, args.output_root, args.experiment_name, args.prediction_endpoint, args.resume,
    )
    print(f"Real-run evaluation created: {experiment}")


if __name__ == "__main__":
    main()
