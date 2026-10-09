"""Generate official SHAP explanations for the selected Logistic Regression model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

try:  # Supports `python -m models.explain_linear_model`.
    from models.train_logistic_regression import load_checkpoints, split_train_validation_test
except ModuleNotFoundError:  # Supports `python models/explain_linear_model.py`.
    from train_logistic_regression import load_checkpoints, split_train_validation_test

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def linear_shap_values(model: Any, background: np.ndarray, values: np.ndarray) -> tuple[float, np.ndarray]:
    """Return additive log-odds attributions around the mean background vector."""
    baseline = background.mean(axis=0)
    coefficients = np.asarray(model.coef_[0])
    expected_log_odds = float(model.intercept_[0] + baseline @ coefficients)
    contributions = (values - baseline) * coefficients
    return expected_log_odds, contributions


def write_explanations(model_directory: Path, dataset: Path, top_n: int = 5) -> Path:
    bundle = joblib.load(model_directory / "model.joblib")
    pipeline = bundle["pipeline"]
    features: list[str] = bundle["features"]
    metrics = bundle["metrics"]
    frame = load_checkpoints(dataset, features, metrics.get("checkpoint_file", "checkpoints.csv"))
    train_frame, _, test_frame = split_train_validation_test(frame, 0.20, 0.20, metrics["random_seed"])
    transformer = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["model"]
    background = np.asarray(transformer.transform(train_frame[features]))
    test_values = np.asarray(transformer.transform(test_frame[features]))
    explainer = shap.LinearExplainer(classifier, background)
    shap_explanation = explainer(test_values)
    contributions = np.asarray(shap_explanation.values)
    expected_log_odds = float(np.mean(np.asarray(shap_explanation.base_values)))
    probabilities = pipeline.predict_proba(test_frame[features])[:, 1]

    output = model_directory / "explanations"
    output.mkdir(exist_ok=True)
    (output / "explanation_baseline.json").write_text(
        json.dumps({
            "features": features,
            "background_mean_transformed": background.mean(axis=0).tolist(),
            "expected_log_odds": expected_log_odds,
            "method": "official_shap_linear_explainer",
        }, indent=2) + "\n", encoding="utf-8"
    )
    global_importance = pd.DataFrame({
        "feature": features,
        "mean_absolute_contribution_log_odds": np.abs(contributions).mean(axis=0),
        "mean_signed_contribution_log_odds": contributions.mean(axis=0),
    }).sort_values("mean_absolute_contribution_log_odds", ascending=False)
    global_importance.to_csv(output / "global_feature_attribution.csv", index=False)

    plt.figure(figsize=(7, 5))
    ordered = global_importance.iloc[::-1]
    plt.barh(ordered["feature"], ordered["mean_absolute_contribution_log_odds"])
    plt.xlabel("Mean absolute contribution to failure log-odds")
    plt.title("Global linear feature attribution")
    plt.tight_layout()
    plt.savefig(output / "global_feature_attribution.png", dpi=160)
    plt.close()

    rows = test_frame[["run_id", "checkpoint_index", "eventual_failed"]].copy()
    rows["failure_probability"] = probabilities
    selected_indices = rows.nlargest(top_n, "failure_probability").index
    local_explanations = []
    for index in selected_indices:
        position = test_frame.index.get_loc(index)
        feature_rows = []
        for feature, value, contribution in zip(features, test_frame.loc[index, features], contributions[position]):
            feature_rows.append({
                "feature": feature,
                "value": float(value),
                "contribution_log_odds": round(float(contribution), 6),
                "direction": "increases_failure_risk" if contribution > 0 else "decreases_failure_risk",
            })
        feature_rows.sort(key=lambda item: abs(item["contribution_log_odds"]), reverse=True)
        local_explanations.append({
            "run_id": rows.loc[index, "run_id"],
            "checkpoint_index": int(rows.loc[index, "checkpoint_index"]),
            "failure_probability": round(float(rows.loc[index, "failure_probability"]), 6),
            "eventual_failed": bool(rows.loc[index, "eventual_failed"]),
            "top_factors": feature_rows[:5],
        })
    (output / "top_risk_explanations.json").write_text(
        json.dumps({
            "method": "official_shap_linear_explainer",
            "expected_log_odds": round(expected_log_odds, 6),
            "top_risk_checkpoints": local_explanations,
        }, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Model explanations

Method: official `shap.LinearExplainer` using a mean-background baseline.
Contributions are in failure **log-odds**, not percentage points. Positive values increase failure risk; negative values reduce it.

SHAP successfully loaded in the local Windows environment and generated these artifacts.

## Most influential features globally

| Feature | Mean absolute contribution (log-odds) |
| --- | ---: |
""" + "\n".join(
        f"| {row.feature} | {row.mean_absolute_contribution_log_odds:.4f} |"
        for row in global_importance.itertuples()
    ) + "\n"
    (output / "EXPLANATIONS.md").write_text(report, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain a trained Logistic Regression failure-risk model.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    print(f"Explanations created: {write_explanations(args.model_dir, args.dataset, args.top_n)}")


if __name__ == "__main__":
    main()
