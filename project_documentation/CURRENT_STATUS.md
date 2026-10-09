# Current status and next step

## What works today

- Local Ollama tool-using agent with bounded runs.
- Validated append-only JSONL telemetry.
- Controlled synthetic failure injection and reproducible datasets.
- Leakage-safe feature engineering.
- Trained Logistic Regression predictor with live FastAPI inference.
- Official SHAP explainability artifacts.
- Deterministic real-task judging and reproducible alert reports.
- Safe tool diagnostics and schema-discovery protocol experiments.
- Incremental real-experiment checkpointing and resume support.

## Current deployed model

The deployed model is an engineered Logistic Regression classifier trained on synthetic gradual-failure data. It is our locally trained model, not a general pre-trained failure-prediction model.

It is useful as a baseline and API demonstration, but should not yet be presented as dependable for arbitrary agents.

## Known limitations

1. Synthetic-only training produces unstable live thresholds on real runs.
2. Real data is still small and task-bank-specific.
3. `llama3.2:3b` is inconsistent at multi-step tool calling.
4. The deterministic judge does not assess broad open-ended factuality.
5. The project is local and single-user, with no production authentication or data controls.
6. Framework adapters for AgentHub, LangChain, and CrewAI are future work.

## Real-data-aware model comparison

Do not train and test on the same experiment.

```text
Development data
  synthetic gradual-failure data + selected real baseline data
    -> train candidate models
    -> choose/calibrate threshold on validation data

Held-out data
  separate real experiment(s)
    -> evaluate once
    -> report precision, recall, specificity, false-alert rate, calibration
```

Compare:

1. Existing synthetic-only Logistic Regression.
2. Real-only Logistic Regression, clearly labelled data-limited.
3. Hybrid synthetic-plus-real Logistic Regression.

All splits must group complete `task_template_id` values (and therefore their contained `run_id` values). Ideally one real experiment is untouched until final evaluation.

## First comparison result

The first comparison is complete. It found that the real-only and source-balanced hybrid candidates overfit the small real development experiment, producing a 100% false-alert rate on successful protocol-enforced held-out runs. The retrained synthetic-only candidate had the best held-out precision and false-alert rate of the three, but insufficient recall.

Therefore, the current next research action is **not** to promote a new API model. It is to expand and diversify real development data, then repeat the same untouched-experiment evaluation protocol.

## Diversified data collection and grouped comparison completed

The first diversified collection is complete: `real_v2_templates_90runs_v1` has 90 real runs across 30 explicitly tracked templates and multiple document/database environments. The comparison pipeline now keeps every repeated attempt of a template entirely within one partition.

The first template-grouped comparison still does **not** justify promoting a new served model. On the external protocol experiment, the real-only and hybrid candidates alerted on every successful run; the synthetic-only candidate lowered false alerts but missed many failures. The result is useful evidence, not a production claim.

## Next research step

The separate final-evaluation bank was run as `real_v3_heldout_300runs_v1`: 300 runs across 30 new template IDs, each repeated 10 times. It remains outside development work. Its live synthetic-model evaluation recorded 59.3% precision, 65.0% recall, 76.6% specificity, and a 23.4% false-alert rate. These are final evaluation evidence for the current model—not data to use for retraining.

The next project phase is GitHub polish: a root README, reproducible setup, architecture diagram, selected small example artifacts, and an honest results/limitations section. If a future model is trained using V3, it requires a fresh unseen task bank for evaluation.

## GitHub polish after the model comparison

- Root README with architecture, setup, demo, results, and limitations.
- Clean `.gitignore` and reproducible installation.
- Selected lightweight sample artifacts instead of unnecessary bulky local data.
- Screenshots of FastAPI docs and a real evaluation report.
- Future AgentHub telemetry-adapter roadmap.
