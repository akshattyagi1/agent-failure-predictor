# Exploratory Data Analysis

Before training a model, inspect whether the synthetic dataset is correctly labelled and whether its features are safe to use.

## Run it

Analyse the newest completed generated experiment:

```powershell
python analysis/explore_dataset.py
```

Or explicitly analyse one dataset:

```powershell
python analysis/explore_dataset.py --dataset data/generated/synthetic_20260913T143345Z
```

It writes an `eda/` folder inside the selected experiment containing a Markdown report and four PNG charts.

## What it checks

- Run and checkpoint counts
- Terminal failure class balance
- Failure rate by synthetic injection profile
- Descriptive differences in retries, tool errors, and latency
- Missing fields, duplicate run IDs, and orphan checkpoint records
- Feature/label leakage risks

## The key rule

Train an early-warning model from checkpoint columns that describe what was observed **so far**, such as `tool_errors_so_far` and `cumulative_latency_ms`.

Do not train on `eventual_failed`, `eventual_failure_type`, `failed`, `failure_type`, `run_id`, `policy_profile`, or the injected-failure configuration. Those are labels, identifiers, or experimental controls—not information a production predictor can rely on.
