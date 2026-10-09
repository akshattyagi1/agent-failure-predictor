# Real-data-aware model comparison

This experiment compares three Logistic Regression training sources without using the final real protocol experiment for fitting or threshold selection:

1. `synthetic_only`: synthetic gradual-failure development data.
2. `real_only`: schema-blind real-agent development data.
3. `hybrid_source_balanced`: both sources, weighted so their checkpoint totals contribute equally during fitting.

The protocol-enforced real-agent experiment is an external held-out test. This tests transfer across a changed agent/tool workflow rather than reporting an optimistic in-sample number.

## Leakage protection: split by task template

The V2 real task bank repeats each underlying task template three times. For example, three independent runs may all ask the agent to retrieve the balance for the same account using the same wording and environment. Those attempts can differ because LLM generation is variable, but they are still very similar examples.

The comparison now assigns a whole `task_template_id` to exactly one development partition: training, validation, or the unused internal test partition. It never puts attempt 1 of a template in training and attempt 2 in validation. This is stricter than the previous run-level split and gives a more honest estimate of whether the predictor generalizes to a new task rather than recognizing a familiar prompt pattern.

Older experiments without `task_template_id` remain readable: the script conservatively treats each of their runs as a separate template. New experiments should always record the real template identifier.

Run from the project root:

```powershell
python -m models.compare_real_aware_models
```

Outputs go to `models/real_aware_comparison/`:

```text
comparison_metrics.csv
comparison_metrics.json
external_test_predictions.csv
MODEL_COMPARISON_REPORT.md
*_validation_thresholds.csv
```

This does not replace the currently served API model automatically. It is an evidence-gathering comparison first. Promote a model only after reviewing its held-out behavior and documenting why it is preferable.
