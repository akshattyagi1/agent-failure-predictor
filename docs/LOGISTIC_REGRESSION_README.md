# Logistic Regression Baseline

This is the first failure-risk model. It predicts `eventual_failed` from telemetry available **at each checkpoint**, while the agent run is still in progress.

## Run it

```powershell
python models/train_logistic_regression.py --dataset data/generated/synthetic_20260913T143345Z
```

It writes the trained pipeline and evaluation artifacts to:

```text
models/logistic_regression/<experiment-name>/
├── model.joblib
├── metrics.json
├── test_predictions.csv
├── validation_threshold_search.csv
├── confusion_matrix.png
├── precision_recall_curve.png
└── calibration_curve.png
```

## Features

The baseline uses only these safe, after-event checkpoint features:

- `checkpoint_index`
- `cumulative_latency_ms`
- `llm_calls_so_far`
- `retries_so_far`
- `tool_calls_so_far`
- `tool_errors_so_far`

It does not use labels, the task ID, synthetic-fault profile, synthetic-only
`injected_failures_so_far`, or run ID. `tool_errors_so_far` is used instead so
the feature set can transfer to real agent telemetry later.

## Honest evaluation and threshold rule

The split happens by `run_id` before model training. Thus, all checkpoints from one agent run stay entirely in either train, validation, or test. This avoids the common mistake of letting nearly identical events from the same run inflate test performance.

The validation set chooses the highest-precision alert threshold that still meets the configured recall target (95% by default). The untouched test set then reports the final metrics at that chosen threshold. This is more honest than tuning the threshold directly on test results.

The evaluation also reports the fraction of failed runs that received at least one alert, the false-alert rate for successful runs, and the median number of remaining events when an alerted failed run first crossed the threshold.

## Reading the metrics

- **Recall**: how many eventual failures it catches. This is especially important here.
- **Precision**: how often a warning is correct.
- **F1**: balance between precision and recall at the default 0.50 threshold.
- **PR-AUC**: ranking quality when failure cases are less common.
- **ROC-AUC**: general probability ranking quality.
- **Brier score**: probability calibration; lower is better.

With only 12 runs, metrics are a smoke test only. Generate 50–100 runs before comparing models or claiming predictive quality.
