# Random Forest and XGBoost Comparison

Run both tree models on the same 60/20/20 run-level split as Logistic Regression:

```powershell
python models/train_tree_models.py --dataset data/generated/dataset_500_runs_v1
```

It writes each model's pipeline, probabilities, calibration/PR/confusion-matrix plots, validation threshold search, and feature importance under `models/random_forest/` and `models/xgboost/`.

The combined comparison is saved in:

```text
models/model_comparison/<experiment-name>/COMPARISON.md
```

Compare more than ROC-AUC. Prefer a model that maintains high failed-run alert coverage while reducing the successful-run false-alert rate and warning earlier than the final event.
