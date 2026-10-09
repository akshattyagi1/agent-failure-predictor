# Feature Engineering

Raw telemetry records individual LLM and tool events. Feature engineering turns the history visible at each event into signals a model can learn from.

## Run it

```powershell
python -m features.engineering --dataset data/generated/dataset_500_runs_v1
```

It leaves raw logs untouched and creates:

```text
data/generated/<experiment>/processed/checkpoints_engineered.csv
```

## New features

| Feature | Meaning | Why it can help |
| --- | --- | --- |
| `tool_error_rate_so_far` | Tool errors divided by tool calls | Compares runs with different numbers of tools. |
| `consecutive_tool_errors` | Current streak of failed tools | Detects an agent degrading or getting stuck. |
| `recent_tool_error_rate_3` | Error rate in the last three tool attempts | Makes recent trouble matter more than old trouble. |
| `repeated_tool_calls_so_far` | Count of calls to a previously used tool | Detects looping behaviour. |
| `unique_tools_used_so_far` | Number of distinct tools called | Captures workflow breadth. |
| `average_tool_latency_ms_so_far` | Mean latency of tools so far | Captures a generally slow dependency. |
| `max_tool_latency_ms_so_far` | Slowest observed tool call | Captures isolated severe slowness. |
| `last_tool_latency_delta_ms` | Latest tool latency minus previous tool latency | Detects rising latency. |
| `steps_since_last_successful_tool` | Events since the last successful tool | Measures lack of recent progress. |
| `last_event_was_tool_error` | Whether the newest event failed | Captures immediate risk after an error. |
| `input_tokens_so_far` / `output_tokens_so_far` | Cumulative LLM token counts | Supports future context/budget-pressure signals. |
| `empty_results_so_far` / `malformed_results_so_far` | Count degraded tool responses | Captures non-crashing dependency quality problems. |

## Leakage rule

Every feature is computed from events up to the current checkpoint only. The final label (`eventual_failed`) and experimental controls (`policy_profile`, `task_id`, `synthetic`) remain in the CSV for analysis but must never become model features.

The fields are designed to transfer to real agent frameworks: a LangChain/CrewAI adapter needs only to emit tool name, success/error, latency, retries, and token counts into the common telemetry schema.

## Test whether the features help

Train the same Logistic Regression baseline with the engineered table:

```powershell
python models/train_logistic_regression.py --dataset data/generated/dataset_500_runs_v1 --feature-set engineered
```

Compare its metrics with `models/logistic_regression/<experiment>/metrics.json`. A feature is valuable only if it improves the untouched-test precision, PR-AUC, false-alert rate, or warning lead time—not merely because it sounds sensible.
