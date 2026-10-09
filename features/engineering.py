"""Convert raw step telemetry into non-leaking, predictive checkpoint features."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_ROOT = PROJECT_ROOT / "data" / "generated"
RECENT_WINDOW = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def latest_experiment() -> Path:
    experiments = sorted(path for path in GENERATED_ROOT.iterdir() if (path / "raw" / "steps.jsonl").exists())
    if not experiments:
        raise FileNotFoundError("No generated experiment with raw telemetry was found.")
    return experiments[-1]


def engineer_checkpoint_features(
    steps: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    manifest_runs: list[dict[str, Any]],
    outcome_overrides: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Create only features knowable immediately after each observed event."""
    outcomes = {run["run_id"]: run for run in runs}
    manifests = {run["run_id"]: run for run in manifest_runs}
    steps_by_run: dict[str, list[dict[str, Any]]] = {}
    for event in steps:
        steps_by_run.setdefault(event["run_id"], []).append(event)

    rows: list[dict[str, Any]] = []
    for run_id, events in steps_by_run.items():
        outcome = (outcome_overrides or {}).get(run_id) or outcomes.get(run_id)
        if outcome is None:
            raise ValueError(f"Step events reference a run without a final outcome: {run_id}")
        manifest = manifests.get(run_id, {})
        state: dict[str, float] = {
            "llm_calls_so_far": 0,
            "tool_calls_so_far": 0,
            "tool_errors_so_far": 0,
            "retries_so_far": 0,
            "cumulative_latency_ms": 0.0,
            "input_tokens_so_far": 0,
            "output_tokens_so_far": 0,
            "consecutive_tool_errors": 0,
            "repeated_tool_calls_so_far": 0,
            "empty_results_so_far": 0,
            "malformed_results_so_far": 0,
        }
        tool_counts: Counter[str] = Counter()
        recent_tool_outcomes: deque[int] = deque(maxlen=RECENT_WINDOW)
        tool_latencies: list[float] = []
        last_success_event_index: int | None = None

        for event_index, event in enumerate(events, start=1):
            state["cumulative_latency_ms"] += float(event.get("latency_ms") or 0)
            state["input_tokens_so_far"] += int(event.get("input_tokens") or 0)
            state["output_tokens_so_far"] += int(event.get("output_tokens") or 0)
            last_event_was_tool_error = 0
            if event["event_type"] == "llm_call":
                state["llm_calls_so_far"] += 1
            elif event["event_type"] == "tool_call":
                state["tool_calls_so_far"] += 1
                tool_name = event.get("tool_name") or "unknown_tool"
                if tool_counts[tool_name] > 0:
                    state["repeated_tool_calls_so_far"] += 1
                tool_counts[tool_name] += 1
                tool_latency = float(event.get("latency_ms") or 0)
                previous_tool_latency = tool_latencies[-1] if tool_latencies else tool_latency
                tool_latencies.append(tool_latency)
                success = event.get("tool_success") is True
                result_quality = event.get("tool_result_quality") or "normal"
                state["empty_results_so_far"] += int(result_quality == "empty")
                state["malformed_results_so_far"] += int(result_quality == "malformed")
                error = int(not success)
                state["tool_errors_so_far"] += error
                state["retries_so_far"] += int((event.get("retry_count") or 0) > 0)
                recent_tool_outcomes.append(error)
                last_event_was_tool_error = error
                if error:
                    state["consecutive_tool_errors"] += 1
                else:
                    state["consecutive_tool_errors"] = 0
                    last_success_event_index = event_index
            else:
                raise ValueError(f"Unknown telemetry event type: {event['event_type']}")

            tool_calls = state["tool_calls_so_far"]
            rows.append({
                "run_id": run_id,
                "checkpoint_index": event_index,
                "task_type": event.get("task_type"),
                "event_type": event["event_type"],
                "task_id": manifest.get("task_id"),
                "policy_profile": manifest.get("policy_profile"),
                "synthetic": True,
                **state,
                "tool_error_rate_so_far": state["tool_errors_so_far"] / tool_calls if tool_calls else 0.0,
                "recent_tool_error_rate_3": sum(recent_tool_outcomes) / len(recent_tool_outcomes) if recent_tool_outcomes else 0.0,
                "unique_tools_used_so_far": len(tool_counts),
                "average_tool_latency_ms_so_far": sum(tool_latencies) / len(tool_latencies) if tool_latencies else 0.0,
                "max_tool_latency_ms_so_far": max(tool_latencies, default=0.0),
                "last_tool_latency_delta_ms": (tool_latencies[-1] - previous_tool_latency) if tool_latencies else 0.0,
                "steps_since_last_successful_tool": (
                    event_index - last_success_event_index if last_success_event_index is not None else event_index
                ),
                "last_event_was_tool_error": last_event_was_tool_error,
                "eventual_failed": outcome["failed"],
                "eventual_failure_type": outcome.get("failure_type"),
            })
    return pd.DataFrame(rows)


def build_experiment_features(experiment: Path, output_name: str = "checkpoints_engineered.csv") -> Path:
    """Read one experiment's immutable raw logs and write a derived CSV beside them."""
    steps = read_jsonl(experiment / "raw" / "steps.jsonl")
    runs = read_jsonl(experiment / "raw" / "runs.jsonl")
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    frame = engineer_checkpoint_features(steps, runs, manifest["runs"])
    output = experiment / "processed" / output_name
    frame.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create engineered checkpoint features from raw telemetry.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-name", default="checkpoints_engineered.csv")
    args = parser.parse_args()
    output = build_experiment_features(args.dataset or latest_experiment(), args.output_name)
    print(f"Engineered checkpoint features created: {output}")


if __name__ == "__main__":
    main()
