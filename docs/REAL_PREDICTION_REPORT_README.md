# Real-run prediction report

This report evaluates the deployed live predictor against the actual task outcome of a real Ollama-agent experiment. It is an evaluation report, not a model-training step.

## What it joins

- `raw/steps.jsonl`: checkpoint-level probabilities and `high_risk`/`low_risk` alerts returned by FastAPI.
- `raw/evaluations.jsonl`: deterministic judgement of the final answer.

The report reduces each run to one decision: **was the run alerted at least once?** It then compares that decision with whether the task truly failed.

## Run it

After running a real evaluation experiment:

```powershell
python -m analysis.evaluate_real_predictions --dataset data/generated/real_ollama_baseline_v1
```

Outputs are written to:

```text
data/generated/real_ollama_baseline_v1/real_evaluation/
  REAL_RUN_EVALUATION_REPORT.md
  metrics.json
  run_level_prediction_audit.csv
  run_level_confusion_matrix.png
```

## Metrics

- **Precision:** among alerted runs, how many really failed?
- **Recall:** among failed runs, how many received an alert?
- **Specificity:** among successful runs, how many avoided an alert?
- **False-alert rate:** among successful runs, how many still received an alert?

For early-warning systems, recall matters, but a model that alerts on every run has poor precision and zero specificity. This report makes that trade-off visible.

## Important limitation

Run-level metrics are useful for the first pilot, but nine runs are not enough to claim reliable performance. Use this report after each real experiment, then compare results after expanding the task bank and collecting a larger held-out set.
