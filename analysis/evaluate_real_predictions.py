"""Evaluate live failure alerts against deterministically judged real agent runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_ROOT = PROJECT_ROOT / "data" / "generated"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def latest_real_experiment(root: Path) -> Path:
    experiments = sorted(
        path for path in root.iterdir()
        if (path / "raw" / "evaluations.jsonl").exists() and (path / "raw" / "steps.jsonl").exists()
    )
    if not experiments:
        raise FileNotFoundError("No real-run experiment with evaluations and steps was found.")
    return experiments[-1]


def safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    lines = ["| Task type | Runs | Failures | Alerted runs |", "| --- | ---: | ---: | ---: |"]
    for task_type, row in frame.iterrows():
        lines.append(
            f"| {task_type} | {int(row['runs'])} | {int(row['failures'])} | {int(row['alerted_runs'])} |"
        )
    return "\n".join(lines)


def make_real_evaluation_report(experiment: Path, output_directory: Path) -> Path:
    """Write run-level alert metrics without retraining or changing the saved model."""
    evaluations = read_jsonl(experiment / "raw" / "evaluations.jsonl")
    steps = read_jsonl(experiment / "raw" / "steps.jsonl")
    if not evaluations:
        raise ValueError("No evaluation rows were found.")

    steps_by_run: dict[str, list[dict[str, Any]]] = {}
    for event in steps:
        steps_by_run.setdefault(event["run_id"], []).append(event)

    audit_rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        events = steps_by_run.get(evaluation["run_id"], [])
        prediction_events = [event for event in events if event.get("failure_prediction_probability") is not None]
        alerts = [event for event in prediction_events if event.get("failure_prediction_label") == "high_risk"]
        probabilities = [float(event["failure_prediction_probability"]) for event in prediction_events]
        first_alert_event = next((index for index, event in enumerate(events, start=1) if event in alerts), None)
        audit_rows.append({
            "run_id": evaluation["run_id"],
            "task_id": evaluation["task_id"],
            "task_type": evaluation["task_type"],
            "actual_failed": bool(evaluation["failed"]),
            "actual_failure_type": evaluation.get("failure_type"),
            "alerted": bool(alerts),
            "prediction_event_count": len(prediction_events),
            "first_alert_event": first_alert_event,
            "max_failure_probability": max(probabilities) if probabilities else None,
            "agent_answer": evaluation.get("agent_answer", ""),
        })

    audit = pd.DataFrame(audit_rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_directory / "run_level_prediction_audit.csv", index=False)

    true_positive = int((audit["actual_failed"] & audit["alerted"]).sum())
    false_positive = int((~audit["actual_failed"] & audit["alerted"]).sum())
    false_negative = int((audit["actual_failed"] & ~audit["alerted"]).sum())
    true_negative = int((~audit["actual_failed"] & ~audit["alerted"]).sum())
    total = len(audit)
    failed = int(audit["actual_failed"].sum())
    successful = total - failed
    metrics = {
        "runs": total,
        "actual_failures": failed,
        "successful_runs": successful,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "precision": safe_rate(true_positive, true_positive + false_positive),
        "recall": safe_rate(true_positive, true_positive + false_negative),
        "specificity": safe_rate(true_negative, true_negative + false_positive),
        "accuracy": safe_rate(true_positive + true_negative, total),
        "false_alert_rate_on_successful_runs": safe_rate(false_positive, successful),
    }
    (output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    by_task = audit.groupby("task_type").agg(
        runs=("run_id", "count"), failures=("actual_failed", "sum"), alerted_runs=("alerted", "sum")
    ).sort_index()

    matrix = [[true_negative, false_positive], [false_negative, true_positive]]
    plt.figure(figsize=(4.8, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["No alert", "Alert"])
    plt.yticks([0, 1], ["Succeeded", "Failed"])
    plt.xlabel("Live model alert")
    plt.ylabel("Actual task outcome")
    plt.title("Run-level live alert evaluation")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row][column]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_directory / "run_level_confusion_matrix.png", dpi=160)
    plt.close()

    report = f"""# Real-run prediction evaluation: {experiment.name}

## Scope

This report compares the live FastAPI alert emitted during each run with the deterministic task outcome in `raw/evaluations.jsonl`.

- Runs evaluated: {total}
- Actual failures: {failed}
- Successful runs: {successful}
- Runs with at least one live prediction: {int((audit['prediction_event_count'] > 0).sum())}

## Run-level confusion matrix

| Actual outcome / alert | No alert | Alert |
| --- | ---: | ---: |
| Succeeded | {true_negative} | {false_positive} |
| Failed | {false_negative} | {true_positive} |

## Metrics

| Metric | Value |
| --- | ---: |
| Precision | {percentage(metrics['precision'])} |
| Recall | {percentage(metrics['recall'])} |
| Specificity | {percentage(metrics['specificity'])} |
| Accuracy | {percentage(metrics['accuracy'])} |
| False-alert rate on successful runs | {percentage(metrics['false_alert_rate_on_successful_runs'])} |

An alert means that at least one checkpoint in a run was labelled `high_risk`. This is intentionally a run-level early-warning view; it does not claim that every alerted checkpoint itself was failing.

## Outcome and alert mix by task type

{markdown_table(by_task)}

## Artifacts

- `run_level_prediction_audit.csv`: one auditable row per run, including answer, outcome, and maximum live risk.
- `metrics.json`: machine-readable metrics.
- `run_level_confusion_matrix.png`: visual run-level confusion matrix.

## Interpretation limits

This experiment has only {total} runs. Treat these values as a pipeline-validation result, not a stable measure of generalization. In particular, a high false-alert rate indicates that the synthetic-trained threshold may not transfer to genuine Ollama behavior. Expand the task bank and collect more real runs before changing or retraining the model.
"""
    report_path = output_directory / "REAL_RUN_EVALUATION_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate live alerts against real judged agent runs.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    experiment = args.dataset or latest_real_experiment(GENERATED_ROOT)
    output = args.output or experiment / "real_evaluation"
    report = make_real_evaluation_report(experiment, output)
    print(f"Real-run evaluation report created: {report}")


if __name__ == "__main__":
    main()
