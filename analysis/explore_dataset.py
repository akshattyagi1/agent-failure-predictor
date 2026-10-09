"""Create a repeatable exploratory-data-analysis report for one synthetic experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GENERATED_ROOT = PROJECT_ROOT / "data" / "generated"

LABEL_COLUMNS = {"failed", "failure_type", "eventual_failed", "eventual_failure_type"}
EXPERIMENT_CONTROL_COLUMNS = {"policy_profile", "injection_seed", "task_id", "synthetic"}


def latest_experiment(root: Path) -> Path:
    experiments = sorted(path for path in root.iterdir() if path.is_dir() and (path / "processed" / "runs.csv").exists())
    if not experiments:
        raise FileNotFoundError(f"No completed experiments found under {root}")
    return experiments[-1]


def save_chart(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def percentage(series: pd.Series) -> pd.Series:
    return series.div(series.sum()).mul(100).round(1)


def markdown_profile_table(frame: pd.DataFrame) -> str:
    """Format the small profile summary without requiring pandas' tabulate extra."""
    lines = ["| Profile | Runs | Failure rate |", "| --- | ---: | ---: |"]
    for profile, row in frame.iterrows():
        lines.append(f"| {profile} | {int(row['count'])} | {row['failure_rate_percent']:.1f}% |")
    return "\n".join(lines)


def make_report(experiment: Path, output_directory: Path) -> Path:
    """Inspect a dataset without fitting a model or treating labels as features."""
    runs = pd.read_csv(experiment / "processed" / "runs.csv")
    checkpoints = pd.read_csv(experiment / "processed" / "checkpoints.csv")
    engineered_path = experiment / "processed" / "checkpoints_engineered.csv"
    feature_frame = pd.read_csv(engineered_path) if engineered_path.exists() else checkpoints
    feature_source = engineered_path.name if engineered_path.exists() else "checkpoints.csv"
    output_directory.mkdir(parents=True, exist_ok=True)

    if runs.empty or checkpoints.empty:
        raise ValueError("The processed CSV files must contain data.")
    if "failed" not in runs or "eventual_failed" not in checkpoints:
        raise ValueError("Required labels are missing from the processed data.")

    runs["failed"] = runs["failed"].astype(bool)
    checkpoints["eventual_failed"] = checkpoints["eventual_failed"].astype(bool)
    failure_counts = runs["failed"].value_counts().reindex([False, True], fill_value=0)
    failure_rate = runs["failed"].mean() * 100
    missing_runs = runs.isna().sum()
    missing_checkpoints = feature_frame.isna().sum()

    # 1. Overall class balance.
    plt.figure(figsize=(5.5, 3.8))
    labels = ["Completed", "Failed"]
    values = [failure_counts[False], failure_counts[True]]
    bars = plt.bar(labels, values, color=["#4C78A8", "#E45756"])
    plt.title("Run outcome balance")
    plt.ylabel("Runs")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    save_chart(output_directory / "01_run_outcome_balance.png")

    # 2. Failure rate by synthetic profile. It diagnoses the generator, not model performance.
    by_profile = runs.groupby("policy_profile", dropna=False)["failed"].agg(["count", "mean"]).sort_values("mean", ascending=False)
    by_profile["failure_rate_percent"] = (by_profile["mean"] * 100).round(1)
    plt.figure(figsize=(7.2, 4.2))
    bars = plt.bar(by_profile.index.astype(str), by_profile["failure_rate_percent"], color="#E45756")
    plt.title("Failure rate by injection profile")
    plt.ylabel("Failure rate (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, by_profile["failure_rate_percent"]):
        plt.text(bar.get_x() + bar.get_width() / 2, min(value + 3, 98), f"{value:.1f}%", ha="center")
    save_chart(output_directory / "02_failure_rate_by_profile.png")

    # 3. Run-level observable outcome differences. These are descriptive only.
    metric_columns = [column for column in ["num_tool_errors", "num_retries", "total_latency_ms"] if column in runs]
    if metric_columns:
        figure, axes = plt.subplots(1, len(metric_columns), figsize=(4.5 * len(metric_columns), 4.2))
        if len(metric_columns) == 1:
            axes = [axes]
        for axis, column in zip(axes, metric_columns):
            values = [runs.loc[~runs["failed"], column].dropna(), runs.loc[runs["failed"], column].dropna()]
            axis.boxplot(values, tick_labels=labels, showmeans=True)
            axis.set_title(column.replace("_", " "))
            axis.set_ylabel("Milliseconds" if column == "total_latency_ms" else "Count")
        save_chart(output_directory / "03_run_metrics_by_outcome.png")

    # 4. Checkpoint availability: how early a failure could be assessed.
    checkpoint_counts = checkpoints.groupby("eventual_failed")["checkpoint_index"].count().reindex([False, True], fill_value=0)
    plt.figure(figsize=(5.5, 3.8))
    bars = plt.bar(labels, [checkpoint_counts[False], checkpoint_counts[True]], color=["#4C78A8", "#E45756"])
    plt.title("Observed checkpoints by eventual outcome")
    plt.ylabel("Checkpoint rows")
    for bar, value in zip(bars, [checkpoint_counts[False], checkpoint_counts[True]]):
        plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    save_chart(output_directory / "04_checkpoint_balance.png")

    feature_candidates = [
        column for column in feature_frame.columns
        if column.endswith("_so_far") or column in {"checkpoint_index", "cumulative_latency_ms"}
    ]
    prohibited = sorted((LABEL_COLUMNS | EXPERIMENT_CONTROL_COLUMNS) & set(feature_frame.columns))
    missing_run_fields = missing_runs[missing_runs > 0].to_dict()
    missing_checkpoint_fields = missing_checkpoints[missing_checkpoints > 0].to_dict()
    report = f"""# Exploratory Data Analysis: {experiment.name}

## Dataset scope

- Source: synthetic/simulated telemetry only.
- Runs: {len(runs)}
- Step checkpoints: {len(checkpoints)}
- Unique run IDs in checkpoints: {checkpoints['run_id'].nunique()}
- Duplicate run-level IDs: {int(runs['run_id'].duplicated().sum())}
- Overall terminal failure rate: {failure_rate:.1f}% ({failure_counts[True]} failed / {len(runs)} total)

## Outcome balance

| Outcome | Runs | Share |
| --- | ---: | ---: |
| Completed | {failure_counts[False]} | {100 - failure_rate:.1f}% |
| Failed | {failure_counts[True]} | {failure_rate:.1f}% |

## Failure rate by injection profile

{markdown_profile_table(by_profile)}

## Feature safety / leakage audit

Feature audit source: `{feature_source}`.

Use checkpoint-level columns that describe what has happened **so far** as model features:

{', '.join(feature_candidates) if feature_candidates else 'No candidate features found.'}

Never use these as model features:

- Labels: {', '.join(sorted(LABEL_COLUMNS & set(checkpoints.columns)))}
- Experiment controls: {', '.join(sorted(EXPERIMENT_CONTROL_COLUMNS & set(checkpoints.columns)))}
- `run_id`: identifier only.

`policy_profile` is especially important: it tells us which synthetic fault recipe generated a run. It may diagnose the generator, but using it as a predictor feature would create artificial leakage and hurt portability to LangChain/CrewAI later.

## Data quality checks

- Missing run-level fields: {missing_run_fields or 'none'}
- Missing checkpoint-level fields: {missing_checkpoint_fields or 'none'}
- Run IDs in checkpoints absent from run table: {len(set(checkpoints['run_id']) - set(runs['run_id']))}

## Interpretation limits

This experiment is small and synthetic. Its charts validate that the logging and labels behave as expected; they do **not** establish a useful predictive model yet. Before modelling, generate enough runs with a reasonable failure balance and split by `run_id` so checkpoints from the same run never appear in both training and test data.
"""
    report_path = output_directory / "EDA_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EDA report and charts for a generated synthetic experiment.")
    parser.add_argument("--dataset", type=Path, help="Experiment folder. Defaults to the latest completed dataset.")
    parser.add_argument("--output", type=Path, help="Directory for report and charts. Defaults to <dataset>/eda.")
    args = parser.parse_args()
    experiment = args.dataset or latest_experiment(DEFAULT_GENERATED_ROOT)
    output = args.output or experiment / "eda"
    report = make_report(experiment, output)
    print(f"EDA report created: {report}")


if __name__ == "__main__":
    main()
