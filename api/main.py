"""Local inference API for the engineered failure-risk model."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "logistic_regression_engineered" / "dataset_500_gradual_failures_v1"


class PredictionRequest(BaseModel):
    """Feature values must describe the agent at its current checkpoint only."""

    features: dict[str, float] = Field(description="Current, non-leaking telemetry features keyed by feature name.")


class Factor(BaseModel):
    feature: str
    value: float
    contribution_log_odds: float
    direction: str


class PredictionResponse(BaseModel):
    failure_probability: float
    prediction: str
    threshold: float
    top_factors: list[Factor]
    model_name: str
    explanation_method: str


class PredictionService:
    def __init__(self, bundle: dict[str, Any], baseline: np.ndarray) -> None:
        self.pipeline = bundle["pipeline"]
        self.features: list[str] = bundle["features"]
        self.threshold = float(bundle["threshold"])
        self.metrics = bundle["metrics"]
        self.baseline = baseline

    @classmethod
    def from_model_directory(cls, model_directory: Path) -> "PredictionService":
        bundle = joblib.load(model_directory / "model.joblib")
        baseline_path = model_directory / "explanations" / "explanation_baseline.json"
        if not baseline_path.exists():
            raise FileNotFoundError("Explanation baseline missing. Run models/explain_linear_model.py first.")
        baseline = np.asarray(json.loads(baseline_path.read_text(encoding="utf-8"))["background_mean_transformed"])
        return cls(bundle, baseline)

    def predict(self, features: dict[str, float]) -> PredictionResponse:
        missing = sorted(set(self.features) - set(features))
        unexpected = sorted(set(features) - set(self.features))
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            raise ValueError("Invalid feature set; " + "; ".join(details))
        frame = pd.DataFrame([{name: features[name] for name in self.features}])
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        transformer = self.pipeline.named_steps["features"]
        classifier = self.pipeline.named_steps["model"]
        transformed = np.asarray(transformer.transform(frame))[0]
        contributions = (transformed - self.baseline) * np.asarray(classifier.coef_[0])
        factors = [
            Factor(
                feature=name,
                value=float(features[name]),
                contribution_log_odds=round(float(contribution), 6),
                direction="increases_failure_risk" if contribution > 0 else "decreases_failure_risk",
            )
            for name, contribution in zip(self.features, contributions)
        ]
        factors.sort(key=lambda factor: abs(factor.contribution_log_odds), reverse=True)
        return PredictionResponse(
            failure_probability=round(probability, 6),
            prediction="high_risk" if probability >= self.threshold else "low_risk",
            threshold=round(self.threshold, 6),
            top_factors=factors[:5],
            model_name=self.metrics["model"],
            explanation_method="linear_shap_compatible_runtime_attribution",
        )


@lru_cache
def get_service() -> PredictionService:
    model_dir = Path(os.getenv("AGENT_FAILURE_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    return PredictionService.from_model_directory(model_dir)


app = FastAPI(title="Agent Failure Predictor", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    service = get_service()
    return {"status": "ok", "model": service.metrics["model"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        return get_service().predict(request.features)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
