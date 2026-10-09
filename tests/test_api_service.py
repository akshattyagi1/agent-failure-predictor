import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from api.main import PredictionService


def test_prediction_service_returns_probability_and_factors(tmp_path):
    feature = "tool_errors_so_far"
    pipeline = Pipeline([
        ("features", ColumnTransformer([("numeric", "passthrough", [feature])])),
        ("model", LogisticRegression(random_state=42)),
    ])
    training = pd.DataFrame({feature: [0.0, 0.0, 1.0, 2.0]})
    pipeline.fit(training, [False, False, True, True])
    model_dir = tmp_path / "model"
    explanations = model_dir / "explanations"
    explanations.mkdir(parents=True)
    joblib.dump({"pipeline": pipeline, "features": [feature], "threshold": 0.5, "metrics": {"model": "logistic_regression"}}, model_dir / "model.joblib")
    (explanations / "explanation_baseline.json").write_text(
        json.dumps({"background_mean_transformed": [0.75]}), encoding="utf-8"
    )
    result = PredictionService.from_model_directory(model_dir).predict({feature: 2.0})
    assert result.failure_probability > 0.5
    assert result.top_factors[0].feature == feature
