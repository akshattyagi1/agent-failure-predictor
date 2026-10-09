"""Compare synthetic-only, real-only, and source-balanced hybrid failure models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from models.train_logistic_regression import (
    ENGINEERED_FEATURE_COLUMNS,
    LABEL_COLUMN,
    choose_threshold,
    classification_metrics,
    make_pipeline,
    run_alert_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYNTHETIC = PROJECT_ROOT / "data" / "generated" / "dataset_500_gradual_failures_v1"
DEFAULT_REAL_DEVELOPMENT = PROJECT_ROOT / "data" / "generated" / "real_v2_templates_90runs_v1"
DEFAULT_EXTERNAL_TEST = PROJECT_ROOT / "data" / "generated" / "real_ollama_schema_protocol_90runs_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "real_aware_comparison"


def load_experiment(experiment: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(experiment / "processed" / "checkpoints_engineered.csv")
    required = set(ENGINEERED_FEATURE_COLUMNS + [LABEL_COLUMN, "run_id", "checkpoint_index"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{experiment.name} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(bool)
    frame["source"] = source
    # Older experiments did not record a template ID. Treat each of their runs as
    # its own template rather than failing, but new experiments should provide it.
    if "task_template_id" not in frame.columns:
        frame["task_template_id"] = frame["run_id"].astype(str)
    frame["task_template_id"] = frame["task_template_id"].fillna(frame["run_id"]).astype(str)
    # A source prefix prevents an unrelated synthetic and real template with the
    # same ID string from accidentally being treated as the same task.
    frame["template_group"] = source + "::" + frame["task_template_id"]
    frame["run_id"] = source + "::" + frame["run_id"].astype(str)
    return frame


def split_by_template(frame: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole task templates, never individual repeated attempts.

    GroupShuffleSplit is deliberately used instead of a normal row/run split:
    attempts of the same prompt template can be highly similar and would make a
    test score overly optimistic if some attempts appeared in training.
    """
    if "template_group" not in frame.columns:
        raise ValueError("Template split requires a template_group column.")
    group_count = frame["template_group"].nunique()
    if group_count < 4:
        raise ValueError("Need at least four task templates for a template-level split.")
    groups = frame["template_group"]
    # GroupShuffleSplit is not stratified. Try deterministic seeds until both
    # partitions contain failures and successes, which threshold selection needs.
    for offset in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + offset)
        train_index, test_index = next(splitter.split(frame, frame[LABEL_COLUMN], groups=groups))
        train = frame.iloc[train_index].copy()
        test = frame.iloc[test_index].copy()
        if train[LABEL_COLUMN].nunique() == 2 and test[LABEL_COLUMN].nunique() == 2:
            return train, test
    raise ValueError("Could not make a template-level split containing both outcome classes in each partition.")


def split_train_validation_test_by_template(
    frame: pd.DataFrame, test_size: float, validation_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create disjoint train/validation/test partitions at task-template level."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1.")
    train_validation, test = split_by_template(frame, test_size, seed)
    validation_share_of_remaining = validation_size / (1 - test_size)
    train, validation = split_by_template(train_validation, validation_share_of_remaining, seed + 1)
    return train, validation, test


def source_balanced_weights(frame: pd.DataFrame) -> pd.Series:
    """Give each source equal total fit weight despite different checkpoint counts."""
    counts = frame["source"].value_counts()
    return frame["source"].map(lambda source: len(frame) / (len(counts) * counts[source]))


def evaluate_external(pipeline: Any, threshold: float, external: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    probabilities = pd.Series(pipeline.predict_proba(external[ENGINEERED_FEATURE_COLUMNS])[:, 1], index=external.index)
    labels = external[LABEL_COLUMN]
    predictions = external[["run_id", "checkpoint_index", LABEL_COLUMN, "source"]].copy()
    predictions["failure_probability"] = probabilities
    predictions["prediction_at_selected_threshold"] = probabilities.ge(threshold)
    metrics: dict[str, Any] = {
        **{name: round(value, 4) for name, value in classification_metrics(labels, probabilities, threshold).items()},
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 4),
        **run_alert_summary(predictions, threshold),
    }
    return metrics, predictions


def train_candidate(name: str, development: pd.DataFrame, external: pd.DataFrame, target_recall: float, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    train, validation, _ = split_train_validation_test_by_template(development, 0.20, 0.20, seed)
    pipeline = make_pipeline(ENGINEERED_FEATURE_COLUMNS)
    fit_arguments: dict[str, Any] = {}
    if name == "hybrid_source_balanced":
        fit_arguments["model__sample_weight"] = source_balanced_weights(train)
    pipeline.fit(train[ENGINEERED_FEATURE_COLUMNS], train[LABEL_COLUMN], **fit_arguments)
    validation_probability = pd.Series(
        pipeline.predict_proba(validation[ENGINEERED_FEATURE_COLUMNS])[:, 1], index=validation.index
    )
    threshold, threshold_search = choose_threshold(validation[LABEL_COLUMN], validation_probability, target_recall)
    external_metrics, predictions = evaluate_external(pipeline, threshold, external)
    metrics = {
        "candidate": name,
        "development_sources": sorted(development["source"].unique().tolist()),
        "external_test_source": sorted(external["source"].unique().tolist()),
        "features": ENGINEERED_FEATURE_COLUMNS,
        "split_unit": "task_template_id",
        "template_overlap_between_train_validation": False,
        "threshold_selection": "validation_precision_maximized_subject_to_target_recall",
        "target_validation_recall": target_recall,
        "selected_threshold": round(threshold, 6),
        "development_runs": int(development["run_id"].nunique()),
        "train_runs": int(train["run_id"].nunique()),
        "validation_runs": int(validation["run_id"].nunique()),
        "external_test_runs": int(external["run_id"].nunique()),
        "external_test_failure_rate": round(float(external[LABEL_COLUMN].mean()), 4),
        **external_metrics,
    }
    predictions["candidate"] = name
    predictions.attrs["threshold_search"] = threshold_search
    return metrics, predictions


def make_report(rows: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Real-data-aware model comparison",
        "",
        "All candidates split development data by whole task templates, select a threshold on validation templates, then evaluate once on the separate protocol-enforced real-agent experiment. The external experiment is not used to fit models or choose thresholds.",
        "",
        "| Candidate | Threshold | Precision | Recall | Specificity | False-alert rate | ROC-AUC | PR-AUC | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['selected_threshold']:.3f} | {row['precision']:.1%} | {row['recall']:.1%} | "
            f"{1 - row['successful_run_false_alert_rate']:.1%} | {row['successful_run_false_alert_rate']:.1%} | "
            f"{row['roc_auc']:.3f} | {row['pr_auc']:.3f} | {row['brier_score']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "The real-only development set is small, and all real tasks come from a constrained local task bank. These results compare useful baselines; they do not establish a universally deployable model. The held-out protocol experiment also represents a changed agent configuration, making it a useful but difficult distribution-shift test.",
    ])
    (output / "MODEL_COMPARISON_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(synthetic: Path, real_development: Path, external_test: Path, output: Path, target_recall: float, seed: int) -> pd.DataFrame:
    synthetic_frame = load_experiment(synthetic, "synthetic")
    real_frame = load_experiment(real_development, "real_diversified_v2")
    external_frame = load_experiment(external_test, "real_schema_protocol_held_out")
    candidates = {
        "synthetic_only": synthetic_frame,
        "real_only": real_frame,
        "hybrid_source_balanced": pd.concat([synthetic_frame, real_frame], ignore_index=True),
    }
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    all_predictions = []
    for name, development in candidates.items():
        metrics, predictions = train_candidate(name, development, external_frame, target_recall, seed)
        rows.append(metrics)
        threshold_search = predictions.attrs.pop("threshold_search")
        threshold_search.to_csv(output / f"{name}_validation_thresholds.csv", index=False)
        all_predictions.append(predictions)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "comparison_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(output / "external_test_predictions.csv", index=False)
    (output / "comparison_metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    make_report(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare synthetic-only, real-only, and hybrid failure models.")
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--real-development", type=Path, default=DEFAULT_REAL_DEVELOPMENT)
    parser.add_argument("--external-test", type=Path, default=DEFAULT_EXTERNAL_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = compare(args.synthetic, args.real_development, args.external_test, args.output, args.target_recall, args.seed)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
