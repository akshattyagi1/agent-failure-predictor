# Held-out real task bank v3

`data/tasks/real_evaluation_task_bank_v3_held_out.jsonl` is the project's final-evaluation task bank. It contains 30 new task templates: 10 calculations, 10 document lookups, and 10 database lookups.

## Why it is separate

This bank must stay out of all model fitting, feature selection, threshold selection, and development-data exploration. It answers a single question: after we have made our modelling choices, does a predictor behave usefully on new task templates?

The tasks use the existing deterministic local environments so answers can still be judged reproducibly. “Held out” means the task wording, task IDs, and `task_template_id` values are new; it does not claim that the environments themselves are unknown to the agent.

## Run it

Run each template three times (90 agent runs) to measure variation in local LLM behavior. For a larger final measurement, increase `--runs` to a multiple of 30 and use a matching new experiment name:

```powershell
python -m agent_failure_predictor.evaluate_real_runs --runs 90 --task-bank data/tasks/real_evaluation_task_bank_v3_held_out.jsonl --experiment-name real_v3_heldout_90runs_v1 --prediction-endpoint http://127.0.0.1:8000
```

If an interrupted run needs to continue, use the same command with `--resume` after the experiment name. Do not reuse this experiment as development data after inspecting its metrics.

Then produce the predictor report (replace `<experiment-name>` with the name you used):

```powershell
python -m analysis.evaluate_real_predictions --dataset data/generated/<experiment-name>
```

Keep the outputs as final evidence. If a model is later retrained with this bank, create a fresh V4 bank for its final evaluation.
