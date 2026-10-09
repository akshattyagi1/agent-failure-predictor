"""Train comparable Random Forest and XGBoost early-warning models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

try:  # Supports `python -m models.train_tree_models`.
    from models.train_logistic_regression import (
        FEATURE_COLUMNS, LABEL_COLUMN, PROJECT_ROOT, choose_threshold, classification_metrics,
        latest_experiment, load_checkpoints, run_alert_summary, save_plots,
        split_train_validation_test, train as train_logistic_regression,
    )
except ModuleNotFoundError:  # Supports `python models/train_tree_models.py`.
    from train_logistic_regression import (
        FEATURE_COLUMNS, LABEL_COLUMN, PROJECT_ROOT, choose_threshold, classification_metrics,
        latest_experiment, load_checkpoints, run_alert_summary, save_plots,
        split_train_validation_test, train as train_logistic_regression,
    )


def make_random_forest() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])


def make_xgboost(train_labels: pd.Series) -> Pipeline:
    positive_count = int(train_labels.sum())
    negative_count = int((~train_labels).sum())
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=negative_count / positive_count,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )),
    ])


def write_feature_importance(pipeline: Pipeline, output: Path) -> None:
    model = pipeline.named_steps["model"]
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)


def fit_and_evaluate(
    model_name: str,
    display_name: str,
    pipeline: Pipeline,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output: Path,
    target_recall: float,
    dataset: Path,
    seed: int,
) -> dict[str, Any]:
    pipeline.fit(train_frame[FEATURE_COLUMNS], train_frame[LABEL_COLUMN])
    validation_probabilities = pd.Series(
        pipeline.predict_proba(validation_frame[FEATURE_COLUMNS])[:, 1], index=validation_frame.index
    )
    threshold, threshold_search = choose_threshold(validation_frame[LABEL_COLUMN], validation_probabilities, target_recall)
    probabilities = pd.Series(pipeline.predict_proba(test_frame[FEATURE_COLUMNS])[:, 1], index=test_frame.index)
    y_true = test_frame[LABEL_COLUMN]
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": model_name,
        "dataset": str(dataset),
        "features": FEATURE_COLUMNS,
        "split_unit": "run_id",
        "random_seed": seed,
        "target_validation_recall": target_recall,
        "selected_threshold": round(threshold, 6),
        **{name: round(value, 4) for name, value in classification_metrics(y_true, probabilities, threshold).items()},
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
    }
    predictions = test_frame[["run_id", "checkpoint_index", LABEL_COLUMN]].copy()
    predictions["failure_probability"] = probabilities
    predictions["prediction_at_selected_threshold"] = probabilities.ge(threshold)
    metrics.update(run_alert_summary(predictions, threshold))
    joblib.dump({"pipeline": pipeline, "features": FEATURE_COLUMNS, "threshold": threshold, "metrics": metrics}, output / "model.joblib")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    threshold_search.to_csv(output / "validation_threshold_search.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_feature_importance(pipeline, output)
    save_plots(y_true, probabilities, threshold, output, display_name)
    return metrics


def write_comparison(metrics: list[dict[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(metrics)
    columns = [
        "model", "selected_threshold", "precision", "recall", "f1", "roc_auc", "pr_auc",
        "failed_run_alert_rate", "successful_run_false_alert_rate", "median_warning_lead_events_for_alerted_failed_runs",
    ]
    comparison = frame[columns].sort_values("pr_auc", ascending=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    lines = ["# Tree Model Comparison", "", "Models use the same run-level train/validation/test split and features.", "", "| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Failed-run alert rate | Successful-run false-alert rate | Median warning lead (events) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['model']} | {row['selected_threshold']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | {row['failed_run_alert_rate']:.3f} | {row['successful_run_false_alert_rate']:.3f} | {row['median_warning_lead_events_for_alerted_failed_runs']} |"
        )
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest and XGBoost using the same split as the logistic baseline.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = args.dataset or latest_experiment()
    frame = load_checkpoints(dataset)
    train_frame, validation_frame, test_frame = split_train_validation_test(
        frame, args.test_size, args.validation_size, args.seed
    )
    output_root = PROJECT_ROOT / "models"
    logistic_metrics = train_logistic_regression(
        dataset, output_root / "logistic_regression" / dataset.name,
        args.test_size, args.validation_size, args.target_recall, args.seed,
    )
    forest_metrics = fit_and_evaluate(
        "random_forest", "Random Forest", make_random_forest(), train_frame, validation_frame, test_frame,
        output_root / "random_forest" / dataset.name, args.target_recall, dataset, args.seed,
    )
    xgboost_metrics = fit_and_evaluate(
        "xgboost", "XGBoost", make_xgboost(train_frame[LABEL_COLUMN]), train_frame, validation_frame, test_frame,
        output_root / "xgboost" / dataset.name, args.target_recall, dataset, args.seed,
    )
    comparison_output = output_root / "model_comparison" / dataset.name
    comparison_output.mkdir(parents=True, exist_ok=True)
    write_comparison([logistic_metrics, forest_metrics, xgboost_metrics], comparison_output)
    print((comparison_output / "COMPARISON.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
