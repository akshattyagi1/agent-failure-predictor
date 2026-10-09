# Agent Failure Predictor: project journey and architecture

## Goal

This project is a local, ML-powered early-warning system for tool-using LLM agents. It observes an agent while it runs, converts observed history into numeric features, and estimates whether the current run will eventually fail.

The honest portfolio claim is:

> A local Ollama agent emits structured telemetry. A Logistic Regression model trained on telemetry predicts run-level failure risk at each checkpoint. FastAPI exposes the prediction, and real-agent experiments measure whether alerts correspond to actual task outcomes.

It is not yet a universal predictor for every framework. It is a reusable telemetry and prediction prototype evaluated on one local agent and several controlled agent versions.

## Current architecture

```text
User task
  -> bounded local Ollama agent (`llama3.2:3b`)
  -> LLM calls and local tool calls
  -> Pydantic-validated JSONL telemetry
  -> online feature tracker
  -> FastAPI POST /predict
  -> probability, high/low-risk alert, top factors
  -> raw telemetry + final judged task outcome
  -> experiment report
```

| Component | Location | Responsibility |
| --- | --- | --- |
| Agent loop | `agent_failure_predictor/runner.py` | Calls Ollama, dispatches tools, records events, handles bounded steps. |
| Local tools | `agent_failure_predictor/tools.py` | Calculator, document search, read-only SQLite query, schema discovery. |
| Telemetry | `agent_failure_predictor/telemetry.py` | Pydantic step/run schemas and append-only JSONL. |
| Online predictor | `agent_failure_predictor/live_prediction.py` | Builds live features and calls FastAPI in shadow mode. |
| Features | `features/engineering.py` | Creates matching non-leaking offline features. |
| API | `api/main.py` | Serves `/health` and `/predict`. |
| Model | `models/train_logistic_regression.py` | Leakage-safe splits, threshold selection, metrics, artifact. |
| Explanations | `models/explain_linear_model.py` | Official SHAP LinearExplainer artifacts. |

## Runs, checkpoints, telemetry, and labels

A **run** is one agent attempt to answer one task. Its events share a `run_id`.

A **checkpoint** is the observable state immediately after an LLM call or tool-call attempt. The predictor uses only information known at that moment; it must never use the final outcome while predicting.

Raw event data includes latency, token counts, tool success, retry state, errors, result quality, live risk score, and top factors. Tool events also save privacy-conscious diagnostics: argument keys, redacted SQL shape, sanitized error details, and whether the error was a recoverable protocol block.

The engineered model uses 20 observable features, including checkpoint index, cumulative latency, LLM/tool counts, tool errors, retries, token totals, consecutive errors, recent error rate, repeated tools, and latency aggregates. Labels, experiment controls, and IDs are never features.

## Synthetic-data phase

New local agents mostly succeed, so controlled failure injection created initial labelled data. It can simulate timeouts, unavailable tools, invalid arguments, no results, added latency, progressive retries, empty results, and malformed results.

The main synthetic dataset is `data/generated/dataset_500_gradual_failures_v1/`. Its engineered Logistic Regression model used run-level train/validation/test splits and reported these held-out **synthetic** results:

| Metric | Result |
| --- | ---: |
| Precision | 68.1% |
| Recall | 95.0% |
| ROC-AUC | 95.2% |
| PR-AUC | 93.4% |
| Brier score | 0.095 |
| Median warning lead | 2 events |

These values demonstrate synthetic-pattern learning only; they do not prove real-agent generalization.

## API and explainability

FastAPI exposes `GET /health` and `POST /predict`. The client must submit exactly the feature schema saved with the model. The response contains probability, risk label, threshold, and top attribution factors.

The service stays local at `http://127.0.0.1:8000`. Runtime attributions use the equivalent lightweight linear calculation; offline reports use official `shap.LinearExplainer`.

## Real-run evaluation

Synthetic labels are insufficient for real behavior. The version-controlled real task bank at `data/tasks/real_evaluation_task_bank.jsonl` contains 30 tasks: 10 calculations, 10 document lookups, and 10 database lookups. Each task uses a deterministic numeric or required-phrase grading rule.

The evaluation harness saves `steps.jsonl`, `runs.jsonl`, `evaluations.jsonl`, processed CSV files, a manifest, and a live-prediction report. A run can fail operationally or semantically through an incorrect final answer.

## Agent/tool versions

| Version | Design | Finding |
| --- | --- | --- |
| `0.1.0` | Schema-blind database agent | Invented SQL columns such as `customer_name`. |
| `0.2.0` | Optional schema tool in prompt | Small model never called the tool. |
| `0.3.0` | Recoverable schema-discovery protocol guard | Blocks blind SQL; agent can discover interface then write SQL. |

The `0.3.0` guard does not answer tasks or write SQL. It enforces an interface contract: `query_database` must follow schema discovery. The agent still independently creates the query and final answer.

Small `llama3.2:3b` sometimes emits tool requests as text rather than native Ollama tool calls. The runner has a strict compatibility parser for valid JSON-shaped calls plus one exact no-argument schema-tool shorthand. It executes only registered local tools and never evaluates arbitrary text.

## Experiment durability

The real evaluation harness checkpoints each judged run immediately, writes a running manifest, and supports `--resume` for current-format interrupted experiments. This was added after an automation interruption left one older experiment with raw traces but no final answer labels.

`real_ollama_schema_aware_90runs_v1` contains 72 unlabelled diagnostic raw traces and is excluded from reported metrics.

## Current scientific position

The full chain works:

```text
real agent behavior -> telemetry -> live risk score -> deterministic task label -> reproducible report
```

The central finding is that a synthetic-only model's alert policy shifts unpredictably when real agent prompts, tools, and protocols change. That finding motivates the next stage: leakage-safe real-data-aware training and calibration.

## Work deliberately not done yet

- Universal LangChain/CrewAI adapters.
- Cloud deployment, authentication, and multi-user storage.
- Claims of universal generalization.
- Retraining on collected real data.
- Final root README, license, screenshots, and GitHub publication.
