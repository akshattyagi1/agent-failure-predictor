# Experiment results and interpretation

## Reading rule

Each experiment answers a different question. Synthetic experiments test controlled trajectories; real experiments test deployed prediction on genuine local Ollama behavior. Agent-version changes alter the telemetry distribution, so the figures are not interchangeable.

## Synthetic model result

`dataset_500_gradual_failures_v1` is the main synthetic training dataset.

| Metric | Result |
| --- | ---: |
| Test precision | 68.1% |
| Test recall | 95.0% |
| Test F1 | 79.3% |
| Test ROC-AUC | 95.2% |
| Test PR-AUC | 93.4% |
| Brier score | 0.095 |
| Median lead time | 2 events |

Interpretation: the predictor recognizes deliberately generated fault patterns. It does not prove real-agent performance.

## Real-run experiments

### `real_ollama_baseline_v1` — 9-run smoke test

| Runs | Failures | Precision | Recall | Specificity | False-alert rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 4 | 44.4% | 100.0% | 0.0% | 100.0% |

The online API and judge worked, but the synthetic-trained threshold alerted on every run.

### `real_ollama_expanded_v1` — 30-task pilot

| Runs | Failures | Precision | Recall | Specificity | False-alert rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 6 | 22.2% | 100.0% | 12.5% | 87.5% |

Task diversity confirmed that the original model was over-sensitive on genuine traces.

### 90-run comparison

| Experiment | Agent version | Failures | Precision | Recall | Specificity | False-alert rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `real_ollama_90runs_v1` | `0.1.0` schema blind | 21 | 22.4% | 90.5% | 4.3% | 95.7% |
| `real_ollama_schema_aware_90runs_v2` | `0.2.0` optional schema tool | 20 | 91.7% | 55.0% | 98.6% | 1.4% |
| `real_ollama_schema_protocol_90runs_v1` | `0.3.0` schema protocol | 26 | 43.3% | 50.0% | 73.4% | 26.6% |

### Findings

- `0.1.0` produced invented SQL columns, semantic document errors, malformed tool requests, and incorrect answers.
- `0.2.0` never called `describe_database_schema`. Its high precision came with nine missed failures; it is not evidence of universally better prediction.
- `0.3.0` recorded 31 blocked blind queries and 24 successful schema discoveries. Only two database errors occurred after discovery, both SQL syntax/quoting errors. The 3B model nevertheless struggled with the longer multi-step protocol, increasing overall failures.

The changing predictor metrics are the project’s central real-world result: synthetic-only training does not yield stable alert behavior across genuine agent/tool configurations.

## Real-data-aware model comparison

The comparison pipeline used synthetic gradual-failure data plus the schema-blind 90-run experiment as development data. The protocol-enforced 90-run experiment remained fully external: it was not used to fit models or select thresholds.

| Candidate | External precision | External recall | External ROC-AUC | External PR-AUC | Run false-alert rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic-only retrained baseline | 56.4% | 21.2% | 57.4% | 42.0% | 26.6% |
| Real-only | 28.0% | 93.3% | 58.4% | 37.6% | 100.0% |
| Source-balanced hybrid | 34.5% | 56.7% | 57.3% | 41.6% | 100.0% |

Interpretation: the real-only and hybrid candidates overfit the small schema-blind development set and alert on every successful held-out protocol run. The retrained synthetic-only baseline has lower recall but the best external precision and false-alert rate among these candidates. No candidate should replace the currently served model automatically.

Artifacts: `models/real_aware_comparison/MODEL_COMPARISON_REPORT.md`, `comparison_metrics.csv`, and `external_test_predictions.csv`.

## Version 2 diversified real development data

`real_v2_templates_90runs_v1` is a clean 90-run collection with authoritative `task_template_id` and `environment_id` metadata. It contains 30 distinct templates, each run three times:

| Task type | Runs | Distinct templates | Environments | Failures |
| --- | ---: | ---: | ---: | ---: |
| Calculation | 30 | 10 | 1 | 0 |
| Document lookup | 30 | 10 | 3 | 6 |
| Database lookup | 30 | 10 | 2 | 18 |

The existing synthetic-only model scored this collection with 56.3% precision, 75.0% recall, 78.8% specificity, and a 21.2% false-alert rate. These values are a new real-data baseline, not a retrained-model result.

## Template-grouped real-data-aware comparison

The comparison was rerun using the diversified V2 data as development data. It splits by `task_template_id`: all three repeats of a task stay in training, validation, or the internal test partition together. The protocol-enforced experiment remains fully external.

| Candidate | External precision | External recall | External ROC-AUC | External PR-AUC | Run false-alert rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic-only retrained baseline | 48.0% | 23.1% | 57.8% | 43.6% | 35.9% |
| Real-only, V2 templates | 36.5% | 73.1% | 68.0% | 43.8% | 100.0% |
| Source-balanced hybrid, V2 templates | 33.2% | 62.5% | 60.6% | 46.6% | 100.0% |

Interpretation: grouping makes the development evaluation more honest, but 30 templates is still a small training bank. No retrained candidate is ready to replace the served synthetic baseline: real-only and hybrid models continue to alert on every successful external run. The next evidence-building step is a new task bank whose templates are never used during training or threshold selection.

## Final held-out V3 evaluation: 300 runs

`real_v3_heldout_300runs_v1` is the final evaluation of the currently served synthetic-trained predictor on 30 new task templates, with each template repeated 10 times. V3 was not used for fitting, threshold selection, or feature choices.

| Metric | Value |
| --- | ---: |
| Runs | 300 |
| Actual failures | 103 |
| Precision | 59.3% |
| Recall | 65.0% |
| Specificity | 76.6% |
| Accuracy | 72.7% |
| False-alert rate on successful runs | 23.4% |

The run-level counts were 67 true positives, 46 false positives, 36 false negatives, and 151 true negatives. The predictor alerted on every database task (100 runs; 63 failed), while it alerted on only 7 document-lookup runs and 6 calculation runs. This indicates that tool/workflow type is strongly associated with its current risk signal.

Interpretation: this is credible held-out evidence that the project pipeline works and that the current synthetic model has partial real-world signal. It is not yet a broadly deployable predictor: roughly one in four successful runs still caused an alert, and 35% of failing runs were missed. Preserve V3 as evaluation-only data.

## Artifact locations

Every completed real experiment includes:

```text
data/generated/<experiment>/
  raw/steps.jsonl
  raw/runs.jsonl
  raw/evaluations.jsonl
  processed/runs.csv
  processed/checkpoints_engineered.csv
  real_evaluation/REAL_RUN_EVALUATION_REPORT.md
  real_evaluation/metrics.json
  real_evaluation/run_level_prediction_audit.csv
```

Future training/evaluation must split by complete `run_id`, never by individual checkpoint row. Prefer keeping a whole real experiment unseen until final testing.
