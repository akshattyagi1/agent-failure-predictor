# Synthetic Dataset Generator

This generator runs the local Llama test agent repeatedly with a fixed task bank and different synthetic-failure profiles. It produces labelled data for the first failure-prediction model.

## Run a small experiment

From the project root:

```powershell
python -m agent_failure_predictor.generate_dataset --runs 12
```

Start with 12 runs. Each run makes local Llama calls, so larger experiments take real time. Once you inspect the results, try 50 or 100:

```powershell
python -m agent_failure_predictor.generate_dataset --runs 50
```

## Output

Every invocation creates a timestamped experiment in `data/generated/`:

```text
data/generated/synthetic_.../
├── configs/             # exact per-profile injection configs used
├── raw/
│   ├── steps.jsonl      # immutable event log
│   └── runs.jsonl       # one final record per run
├── processed/
│   ├── runs.csv         # one labelled row per run
│   └── checkpoints.csv  # one labelled row after each observed event
└── manifest.json        # model, task IDs, profile, and seed for every run
```

Everything is synthetic and is explicitly marked as such in the manifest and run table.

## Task bank and failure profiles

The version-controlled tasks are in `data/tasks/task_bank.jsonl`. The generator cycles through calculator, document-search, and database tasks.

It cycles through four profiles:

- `healthy`: no injected failures.
- `recoverable_timeout`: a tool may time out, with one retry allowed.
- `terminal_timeout`: a tool is likely to time out and cannot retry.
- `mixed_faults`: timeouts, unavailable tools, invalid arguments, no-results, and small added delays.

Each generated run uses a different seed. The seed, task ID, and profile are kept in `manifest.json`, so results can be reproduced and audited.

## Why two processed files?

`runs.csv` answers: “Did the whole run ultimately fail?” It is for run-level models.

`checkpoints.csv` answers: “Based on everything observed **so far**, will this run eventually fail?” It is for real-time early-warning models. The columns ending in `_so_far` are valid features. `eventual_failed` and `eventual_failure_type` are labels only—never train on them as features.
