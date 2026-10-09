"""Generate a small, reproducible synthetic telemetry dataset from the test agent."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .failure_injection import FailureInjectionConfig
from .runner import run_agent
from features.engineering import engineer_checkpoint_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK_BANK = PROJECT_ROOT / "data" / "tasks" / "task_bank.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "generated"


def load_task_bank(path: Path) -> list[dict[str, str]]:
    """Load the fixed, version-controlled tasks used in synthetic experiments."""
    with path.open(encoding="utf-8") as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    if not tasks:
        raise ValueError("Task bank must contain at least one task.")
    return tasks


def policy_profiles() -> dict[str, dict[str, Any]]:
    """Balanced profiles: healthy, recoverable faults, and likely terminal faults."""
    return {
        "healthy": {"max_retries": 0},
        "recoverable_timeout": {"max_retries": 1, "tool_timeout_probability": 0.35},
        "terminal_timeout": {"max_retries": 0, "tool_timeout_probability": 0.80},
        "mixed_faults": {
            "max_retries": 1,
            "tool_timeout_probability": 0.15,
            "tool_unavailable_probability": 0.10,
            "invalid_arguments_probability": 0.05,
            "no_results_probability": 0.05,
            "added_latency_probability": 0.20,
            "min_added_latency_ms": 25,
            "max_added_latency_ms": 75,
        },
        "progressive_timeout": {
            "max_retries": 2,
            "progressive_timeout_probability": 0.70,
            "progressive_timeout_attempts": 3,
            "progressive_latency_increment_ms": 25,
        },
        "degraded_results": {
            "max_retries": 1,
            "empty_result_probability": 0.25,
            "malformed_result_probability": 0.25,
            "progressive_timeout_probability": 0.20,
            "progressive_timeout_attempts": 2,
            "progressive_latency_increment_ms": 25,
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_processed_data(
    raw_directory: Path,
    processed_directory: Path,
    manifest_rows: list[dict[str, Any]],
    *,
    synthetic: bool = True,
    outcome_overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Make run rows and non-leaking, after-event checkpoint features from raw JSONL."""
    processed_directory.mkdir(parents=True, exist_ok=True)
    steps = read_jsonl(raw_directory / "steps.jsonl")
    runs = read_jsonl(raw_directory / "runs.jsonl")
    manifest_by_run = {row["run_id"]: row for row in manifest_rows if row.get("run_id")}
    run_by_id = {row["run_id"]: row for row in runs}

    run_rows: list[dict[str, Any]] = []
    for run in runs:
        manifest = manifest_by_run.get(run["run_id"], {})
        outcome = (outcome_overrides or {}).get(run["run_id"], {})
        run_rows.append({
            **run,
            **outcome,
            "synthetic": synthetic,
            "task_id": manifest.get("task_id"),
            "task_template_id": manifest.get("task_template_id"),
            "environment_id": manifest.get("environment_id"),
            "policy_profile": manifest.get("policy_profile"),
            "injection_seed": manifest.get("injection_seed"),
        })

    state_by_run: dict[str, dict[str, float]] = defaultdict(lambda: {
        "events_observed": 0, "llm_calls_so_far": 0, "tool_calls_so_far": 0,
        "tool_errors_so_far": 0, "retries_so_far": 0, "injected_failures_so_far": 0,
        "cumulative_latency_ms": 0.0,
    })
    checkpoint_rows: list[dict[str, Any]] = []
    for event in steps:
        run_id = event["run_id"]
        state = state_by_run[run_id]
        state["events_observed"] += 1
        state["cumulative_latency_ms"] += event.get("latency_ms") or 0
        if event["event_type"] == "llm_call":
            state["llm_calls_so_far"] += 1
        if event["event_type"] == "tool_call":
            state["tool_calls_so_far"] += 1
            state["tool_errors_so_far"] += int(event.get("tool_success") is False)
            state["retries_so_far"] += int((event.get("retry_count") or 0) > 0)
            state["injected_failures_so_far"] += int(event.get("failure_injected") is True)
        manifest = manifest_by_run.get(run_id, {})
        outcome = (outcome_overrides or {}).get(run_id) or run_by_id.get(run_id, {})
        checkpoint_rows.append({
            "run_id": run_id,
            "checkpoint_index": int(state["events_observed"]),
            "task_id": manifest.get("task_id"),
            "task_template_id": manifest.get("task_template_id"),
            "environment_id": manifest.get("environment_id"),
            "task_type": event.get("task_type"),
            "policy_profile": manifest.get("policy_profile"),
            "event_type": event.get("event_type"),
            **state,
            # This is the future label. Do not use it as a model feature.
            "eventual_failed": outcome.get("failed"),
            "eventual_failure_type": outcome.get("failure_type"),
        })

    write_csv(processed_directory / "runs.csv", run_rows)
    write_csv(processed_directory / "checkpoints.csv", checkpoint_rows)
    engineer_checkpoint_features(steps, runs, manifest_rows, outcome_overrides).to_csv(
        processed_directory / "checkpoints_engineered.csv", index=False
    )


def generate_dataset(
    number_of_runs: int,
    model: str,
    task_bank: Path,
    output_root: Path,
    experiment_name: str | None = None,
) -> Path:
    """Run the test agent repeatedly and return the generated experiment directory."""
    if number_of_runs < 1:
        raise ValueError("number_of_runs must be at least 1.")
    tasks = load_task_bank(task_bank)
    profiles = policy_profiles()
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_name = experiment_name or f"synthetic_{created}"
    experiment = output_root / experiment_name
    if experiment.exists():
        raise FileExistsError(f"Experiment already exists: {experiment}")
    raw_directory = experiment / "raw"
    configs_directory = experiment / "configs"
    processed_directory = experiment / "processed"
    raw_directory.mkdir(parents=True)
    configs_directory.mkdir()

    manifest_rows: list[dict[str, Any]] = []
    profile_names = list(profiles)
    for index in range(number_of_runs):
        task = tasks[index % len(tasks)]
        profile_name = profile_names[index % len(profile_names)]
        seed = 10_000 + index
        settings = {**profiles[profile_name], "seed": seed}
        config_path = configs_directory / f"run_{index:04d}_{profile_name}.json"
        write_json(config_path, settings)
        before_runs = read_jsonl(raw_directory / "runs.jsonl")
        try:
            run_agent(task["task"], model, raw_directory, failure_config=config_path)
        except RuntimeError:
            # Terminal faults are expected observations in a labelled dataset.
            pass
        after_runs = read_jsonl(raw_directory / "runs.jsonl")
        if len(after_runs) != len(before_runs) + 1:
            raise RuntimeError("Expected exactly one completed run record per generated task.")
        run = after_runs[-1]
        manifest_rows.append({
            "run_id": run["run_id"], "run_index": index, "task_id": task["task_id"],
            "task_type": task["task_type"], "policy_profile": profile_name,
            "injection_seed": seed, "model": model, "failure_config": str(config_path),
        })

    write_json(experiment / "manifest.json", {
        "synthetic": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "number_of_runs": number_of_runs,
        "task_bank": str(task_bank),
        "profiles": profiles,
        "runs": manifest_rows,
    })
    build_processed_data(raw_directory, processed_directory, manifest_rows)
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic telemetry from the local Llama test agent.")
    parser.add_argument("--runs", type=int, default=12, help="Number of runs; begin small before scaling up.")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--task-bank", type=Path, default=DEFAULT_TASK_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-name")
    args = parser.parse_args()
    experiment = generate_dataset(args.runs, args.model, args.task_bank, args.output_root, args.experiment_name)
    print(f"Dataset created: {experiment}")


if __name__ == "__main__":
    main()
