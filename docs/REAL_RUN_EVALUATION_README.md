# Real Ollama-run evaluation

This experiment collects **genuine** behavior from the local Ollama agent. It does not inject failures. Each task has a version-controlled expected result, so the project can automatically determine whether the agent completed the task correctly.

## What it measures

There are two different outcomes:

- **Operational failure:** the agent loop crashed, hit its step limit, or a tool failed terminally.
- **Task failure:** the agent returned an incorrect answer even though the loop completed.

For this experiment, `failed` means either kind of failure. This is the outcome label used for later real-world model evaluation.

## Run an experiment

Start the API in one terminal if you want live shadow-mode predictions:

```powershell
uvicorn api.main:app --reload
```

Then run 30 real tasks in another terminal:

```powershell
python -m agent_failure_predictor.evaluate_real_runs --runs 30 --experiment-name real_ollama_expanded_v1 --prediction-endpoint http://127.0.0.1:8000
```

Use a new experiment name every time. The bank now has 30 tasks: 10 calculations, 10 document lookups, and 10 database lookups. Thirty runs covers each task once. More runs create repeated trials over the same task set; use 90 runs for three trials per task.

### Interrupted collections

The harness now saves each judged evaluation immediately. If a future collection is interrupted, rerun the same command with `--resume`:

```powershell
python -m agent_failure_predictor.evaluate_real_runs --runs 90 --experiment-name real_ollama_schema_aware_90runs_v2 --prediction-endpoint http://127.0.0.1:8000 --resume
```

Only experiments started with the current checkpointing version can resume. An older partial collection with raw runs but no `evaluations.jsonl` cannot be graded retrospectively because the final agent answers were never written to raw step telemetry.

## Outputs

```text
data/generated/real_ollama_baseline_v1/
  raw/steps.jsonl        # agent events and optional live predictions
  raw/runs.jsonl         # operational summaries
  raw/evaluations.jsonl  # answer, expected-rule, task success/failure
  processed/runs.csv     # combined labels and summaries
  processed/checkpoints_engineered.csv
  manifest.json
```

The evaluator currently uses deterministic numeric matching and required-phrase matching. It deliberately avoids an LLM judge, so labels are reproducible and easy to audit. The task bank is [real_evaluation_task_bank.jsonl](../data/tasks/real_evaluation_task_bank.jsonl).

## Important limitation

Thirty tasks are a meaningful pilot, but not enough data to retrain a reliable model. First use this harness to check whether live predictions correspond to actual outcomes. A 90-run collection provides three trials per task, but expand task diversity further before claiming broad generalization.
