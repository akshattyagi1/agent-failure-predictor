"""Train and evaluate a leakage-safe Logistic Regression early-warning baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_ROOT = PROJECT_ROOT / "data" / "generated"

# Every feature must describe data visible when a checkpoint is created.
BASIC_FEATURE_COLUMNS = [
    "checkpoint_index",
    "cumulative_latency_ms",
    "llm_calls_so_far",
    "retries_so_far",
    "tool_calls_so_far",
    "tool_errors_so_far",
]
ENGINEERED_FEATURE_COLUMNS = BASIC_FEATURE_COLUMNS + [
    "input_tokens_so_far",
    "output_tokens_so_far",
    "consecutive_tool_errors",
    "repeated_tool_calls_so_far",
    "tool_error_rate_so_far",
    "recent_tool_error_rate_3",
    "unique_tools_used_so_far",
    "average_tool_latency_ms_so_far",
    "max_tool_latency_ms_so_far",
    "last_tool_latency_delta_ms",
    "steps_since_last_successful_tool",
    "last_event_was_tool_error",
    "empty_results_so_far",
    "malformed_results_so_far",
]
FEATURE_SETS = {"basic": BASIC_FEATURE_COLUMNS, "engineered": ENGINEERED_FEATURE_COLUMNS}
# Kept as the default import for the existing tree-model comparison.
FEATURE_COLUMNS = BASIC_FEATURE_COLUMNS
LABEL_COLUMN = "eventual_failed"


def latest_experiment() -> Path:
    experiments = sorted(path for path in GENERATED_ROOT.iterdir() if (path / "processed" / "checkpoints.csv").exists())
    if not experiments:
        raise FileNotFoundError("No generated experiment with checkpoints.csv was found.")
    return experiments[-1]


def load_checkpoints(
    experiment: Path, feature_columns: list[str] = FEATURE_COLUMNS, checkpoint_filename: str = "checkpoints.csv"
) -> pd.DataFrame:
    frame = pd.read_csv(experiment / "processed" / checkpoint_filename)
    missing = set(feature_columns + [LABEL_COLUMN, "run_id"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Checkpoint dataset is missing required columns: {sorted(missing)}")
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(bool)
    per_run_label_count = frame.groupby("run_id")[LABEL_COLUMN].nunique()
    if (per_run_label_count != 1).any():
        raise ValueError("Every checkpoint within a run must have the same eventual outcome label.")
    return frame


def split_by_run(frame: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split run IDs first, preventing same-run checkpoint leakage."""
    run_labels = frame.groupby("run_id", as_index=False)[LABEL_COLUMN].first()
    class_counts = run_labels[LABEL_COLUMN].value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError("Need at least two successful and two failed runs for a stratified run-level split.")
    train_ids, test_ids = train_test_split(
        run_labels["run_id"], test_size=test_size, random_state=seed,
        stratify=run_labels[LABEL_COLUMN],
    )
    return frame[frame["run_id"].isin(train_ids)].copy(), frame[frame["run_id"].isin(test_ids)].copy()


def split_train_validation_test(
    frame: pd.DataFrame, test_size: float, validation_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create three stratified partitions by run ID, never by individual events."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1.")
    train_validation, test = split_by_run(frame, test_size, seed)
    validation_share_of_remaining = validation_size / (1 - test_size)
    train, validation = split_by_run(train_validation, validation_share_of_remaining, seed + 1)
    return train, validation, test


def make_pipeline(feature_columns: list[str] = FEATURE_COLUMNS) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return Pipeline([
        ("features", ColumnTransformer([("numeric", numeric, feature_columns)], remainder="drop")),
        ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42)),
    ])


def classification_metrics(y_true: pd.Series, probabilities: pd.Series, threshold: float) -> dict[str, float]:
    predictions = probabilities.ge(threshold)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def choose_threshold(y_true: pd.Series, probabilities: pd.Series, target_recall: float) -> tuple[float, pd.DataFrame]:
    """Choose the most precise validation threshold that satisfies the recall target."""
    candidates = sorted({0.0, 0.5, 1.0, *probabilities.tolist()})
    rows = []
    for threshold in candidates:
        rows.append({"threshold": threshold, **classification_metrics(y_true, probabilities, threshold)})
    search = pd.DataFrame(rows)
    eligible = search[search["recall"] >= target_recall]
    if eligible.empty:
        selected = search.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        selected = eligible.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]
    return float(selected["threshold"]), search


def run_alert_summary(predictions: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Summarise whether each test run received an alert and how early it happened."""
    rows = []
    for run_id, group in predictions.groupby("run_id"):
        alerts = group[group["failure_probability"] >= threshold]
        total_events = int(group["checkpoint_index"].max())
        first_alert = int(alerts["checkpoint_index"].min()) if not alerts.empty else None
        rows.append({
            "run_id": run_id,
            "eventual_failed": bool(group[LABEL_COLUMN].iloc[0]),
            "alerted": first_alert is not None,
            "first_alert_checkpoint": first_alert,
            "warning_lead_events": total_events - first_alert if first_alert is not None else None,
        })
    alerts = pd.DataFrame(rows)
    failed = alerts[alerts["eventual_failed"]]
    successful = alerts[~alerts["eventual_failed"]]
    return {
        "failed_run_alert_rate": round(float(failed["alerted"].mean()), 4),
        "successful_run_false_alert_rate": round(float(successful["alerted"].mean()), 4),
        "median_warning_lead_events_for_alerted_failed_runs": (
            round(float(failed.loc[failed["alerted"], "warning_lead_events"].median()), 2)
            if failed["alerted"].any() else None
        ),
    }


def save_plots(y_true: pd.Series, probabilities: pd.Series, threshold: float, output: Path, model_name: str) -> None:
    predictions = probabilities.ge(threshold)
    matrix = confusion_matrix(y_true, predictions, labels=[False, True])
    plt.figure(figsize=(4.8, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["Completed", "Failed"])
    plt.yticks([0, 1], ["Completed", "Failed"])
    plt.xlabel("Predicted outcome")
    plt.ylabel("Actual outcome")
    plt.title(f"Checkpoint confusion matrix (threshold {threshold:.2f})")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    plt.figure(figsize=(5.2, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–recall curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output / "precision_recall_curve.png", dpi=160)
    plt.close()

    fraction_positive, mean_predicted = calibration_curve(y_true, probabilities, n_bins=min(5, len(y_true)))
    plt.figure(figsize=(5.2, 4))
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.plot(mean_predicted, fraction_positive, marker="o", label=model_name)
    plt.xlabel("Predicted failure probability")
    plt.ylabel("Observed failure frequency")
    plt.title("Calibration curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "calibration_curve.png", dpi=160)
    plt.close()


def train(
    experiment: Path,
    output: Path,
    test_size: float,
    validation_size: float,
    target_recall: float,
    seed: int,
    feature_columns: list[str] = FEATURE_COLUMNS,
    checkpoint_filename: str = "checkpoints.csv",
) -> dict[str, Any]:
    """Train a baseline and save the model, test predictions, and honest metrics."""
    frame = load_checkpoints(experiment, feature_columns, checkpoint_filename)
    train_frame, validation_frame, test_frame = split_train_validation_test(frame, test_size, validation_size, seed)
    pipeline = make_pipeline(feature_columns)
    pipeline.fit(train_frame[feature_columns], train_frame[LABEL_COLUMN])
    validation_probabilities = pd.Series(
        pipeline.predict_proba(validation_frame[feature_columns])[:, 1], index=validation_frame.index
    )
    threshold, threshold_search = choose_threshold(validation_frame[LABEL_COLUMN], validation_probabilities, target_recall)
    probabilities = pd.Series(pipeline.predict_proba(test_frame[feature_columns])[:, 1], index=test_frame.index)
    y_true = test_frame[LABEL_COLUMN]
    test_classification = classification_metrics(y_true, probabilities, threshold)

    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": str(experiment),
        "model": "logistic_regression",
        "label": LABEL_COLUMN,
        "features": feature_columns,
        "checkpoint_file": checkpoint_filename,
        "split_unit": "run_id",
        "random_seed": seed,
        "threshold_selection": "validation_precision_maximized_subject_to_target_recall",
        "target_validation_recall": target_recall,
        "selected_threshold": round(threshold, 6),
        "train_runs": int(train_frame["run_id"].nunique()),
        "validation_runs": int(validation_frame["run_id"].nunique()),
        "test_runs": int(test_frame["run_id"].nunique()),
        "train_checkpoints": int(len(train_frame)),
        "validation_checkpoints": int(len(validation_frame)),
        "test_checkpoints": int(len(test_frame)),
        "test_failure_rate": round(float(y_true.mean()), 4),
        **{name: round(value, 4) for name, value in test_classification.items()},
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
    }
    joblib.dump({"pipeline": pipeline, "features": feature_columns, "threshold": threshold, "metrics": metrics}, output / "model.joblib")
    predictions_frame = test_frame[["run_id", "checkpoint_index", LABEL_COLUMN]].copy()
    predictions_frame["failure_probability"] = probabilities
    predictions_frame["prediction_at_selected_threshold"] = probabilities.ge(threshold)
    predictions_frame.to_csv(output / "test_predictions.csv", index=False)
    threshold_search.to_csv(output / "validation_threshold_search.csv", index=False)
    alert_metrics = run_alert_summary(predictions_frame, threshold)
    metrics.update(alert_metrics)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_plots(y_true, probabilities, threshold, output, "Logistic Regression")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Logistic Regression baseline from checkpoint telemetry.")
    parser.add_argument("--dataset", type=Path, default=None, help="Generated experiment directory; defaults to latest.")
    parser.add_argument("--output", type=Path, default=None, help="Directory for model and evaluation files.")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="basic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = args.dataset or latest_experiment()
    checkpoint_file = "checkpoints_engineered.csv" if args.feature_set == "engineered" else "checkpoints.csv"
    model_directory = "logistic_regression_engineered" if args.feature_set == "engineered" else "logistic_regression"
    output = args.output or PROJECT_ROOT / "models" / model_directory / dataset.name
    metrics = train(
        dataset, output, args.test_size, args.validation_size, args.target_recall, args.seed,
        FEATURE_SETS[args.feature_set], checkpoint_file,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
