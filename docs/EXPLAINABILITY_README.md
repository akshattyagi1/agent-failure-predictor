# Model Explainability

Run explanations for the current best engineered Logistic Regression model:

```powershell
python models/explain_linear_model.py `
  --dataset data/generated/dataset_500_gradual_failures_v1 `
  --model-dir models/logistic_regression_engineered/dataset_500_gradual_failures_v1
```

It creates an `explanations/` folder containing:

- `global_feature_attribution.csv`: average importance across unseen test checkpoints.
- `global_feature_attribution.png`: chart of that global importance.
- `top_risk_explanations.json`: top factors for the highest-risk test checkpoints.
- `EXPLANATIONS.md`: a readable summary.

## SHAP implementation

SHAP is installed and working locally. The offline report uses official `shap.LinearExplainer` for the trained Logistic Regression model.

The API uses the equivalent lightweight linear contribution calculation at request time:

```text
contribution = (standardised feature value - mean background value) × model coefficient
```

The contributions add to the model's failure log-odds. Positive contributions raise risk; negative contributions lower it. When we later promote a tree model, we can add official `shap.TreeExplainer` for that model too.
