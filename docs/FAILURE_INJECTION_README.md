# Failure Injection and Retry Telemetry

This document explains the synthetic failure system added to the local Llama agent. It is for generating **simulated telemetry** for the future failure-prediction dataset. It is not real production failure data.

## Why this exists

The first agent produced successful runs and telemetry, but an ML classifier needs examples of both success and failure. Failure injection deliberately creates controlled tool problems—such as timeouts—so the project can record what an unhealthy run looks like.

The normal agent remains unchanged unless a failure configuration is supplied.

```text
No configuration supplied  -> tools run normally
Configuration supplied     -> tools may simulate faults and retry
```

## Files added and changed

| File | Change |
| --- | --- |
| `agent_failure_predictor/failure_injection.py` | New seeded, configurable simulator for tool faults. |
| `configs/failure_injection_demo.json` | Example configuration with modest failure probabilities. |
| `agent_failure_predictor/runner.py` | Loads a failure config, injects failures before tool calls, retries, and records terminal failures. |
| `agent_failure_predictor/telemetry.py` | Adds fields that distinguish simulated failures from normal tool errors. |
| `tests/test_failure_injection.py` | Proves always-on and always-off injection behaviour. |

## Running it

From the project root:

```powershell
python -m agent_failure_predictor.runner "What is 25 * 17?" --failure-config configs\failure_injection_demo.json
```

Without `--failure-config`, the injector uses a safe default: all probabilities are zero and retries are disabled.

## The configuration file

`configs/failure_injection_demo.json` is JSON because Python can read it without another dependency.

```json
{
  "seed": 42,
  "max_retries": 1,
  "tool_timeout_probability": 0.15,
  "tool_unavailable_probability": 0.05,
  "invalid_arguments_probability": 0.05,
  "no_results_probability": 0.05,
  "added_latency_probability": 0.15,
  "min_added_latency_ms": 250,
  "max_added_latency_ms": 750
}
```

| Setting | Meaning |
| --- | --- |
| `seed` | Makes the sequence of simulated decisions repeatable. The same config produces the same sequence for a run. |
| `max_retries` | Number of additional attempts after a tool failure. `1` means up to two total attempts. |
| `tool_timeout_probability` | Chance that an attempt raises a synthetic timeout. |
| `tool_unavailable_probability` | Chance that the tool is simulated as unavailable. |
| `invalid_arguments_probability` | Chance that arguments are simulated as invalid. |
| `no_results_probability` | Chance that the tool is simulated as returning no useful result. |
| `added_latency_probability` | Chance of adding an actual delay before the tool executes. |
| `min_added_latency_ms` / `max_added_latency_ms` | Range for that extra delay. |
| `progressive_timeout_probability` | Chance that one logical tool action enters a multi-retry timeout trajectory. |
| `progressive_timeout_attempts` | Number of consecutive timeout attempts in that trajectory. |
| `progressive_latency_increment_ms` | Added delay grows by this amount on each progressive retry. |
| `empty_result_probability` | Chance that a successful tool call returns a simulated empty result. |
| `malformed_result_probability` | Chance that a successful tool call returns malformed structured content. |

Each probability must be from `0.0` to `1.0`. On one tool attempt, the injector applies at most one fault type so the resulting label is clear.

## What happens in the runner

Before calling a tool, `runner.py` calls:

```python
added_latency_ms = injector.before_tool_call(name)
```

The injector may return normally, optionally after a delay, or it may raise an `InjectedToolError` such as `InjectedToolTimeout`.

The runner then does the following for every attempt:

1. Starts a timer.
2. Allows the injector to simulate a failure/delay.
3. Runs the requested local Python tool if no injected error occurred.
4. Writes one `tool_call` step telemetry event.
5. Retries if the attempt failed and retry attempts remain.
6. Marks the whole run failed if retries are exhausted.

The runner has a separate `try/except` around the full run, so it still writes a final run summary even when the model or a tool fails.

## Retry example

Suppose the calculator is requested and the first attempt has an injected timeout, with `max_retries` set to `1`.

```text
Attempt 0 -> calculator timeout -> step event says failure_injected=true
Attempt 1 -> calculator succeeds -> step event says tool_success=true
Llama receives calculator result -> writes final answer
Run summary -> failed=false, num_injected_failures=1, num_retries=1
```

A tool error is not automatically a failed run: a successful retry can recover it. If all retry attempts fail, the final run summary is marked `failed=true` and gets the stable failure label, for example `tool_timeout`.

## Gradual failure trajectories

`progressive_timeout_*` settings simulate a fault that gets worse across retries instead of failing in one event. With `max_retries=2` and `progressive_timeout_attempts=3`, one logical tool action creates three timeout events, each with increasing added latency, before the run fails. This gives an early-warning model more than one observable signal before the terminal event.

`empty_result_probability` and `malformed_result_probability` do not crash a tool. They alter a successful tool response and are recorded as `tool_result_quality` (`empty` or `malformed`). They represent degraded dependencies that a future real agent may need to recover from through replanning or validation.

## New step-level telemetry fields

Each tool attempt is saved in `telemetry/steps.jsonl`. These fields were added to `StepEvent`:

| Field | Meaning |
| --- | --- |
| `failure_injected` | `true` only when this error was deliberately simulated. |
| `injected_failure_type` | Stable label such as `tool_timeout` or `tool_unavailable`; otherwise `null`. |
| `added_latency_ms` | Extra synthetic delay configured for this attempt. |
| `retry_count` | Zero for the first attempt, one for the first retry, and so on. This field already existed but is now populated by the retry loop. |

Example simulated timeout event:

```json
{
  "event_type": "tool_call",
  "tool_name": "calculate",
  "tool_success": false,
  "error_type": "InjectedToolTimeout",
  "failure_injected": true,
  "injected_failure_type": "tool_timeout",
  "retry_count": 0
}
```

## New run-level telemetry field

`RunEvent`, saved in `telemetry/runs.jsonl`, now has:

| Field | Meaning |
| --- | --- |
| `num_injected_failures` | Count of simulated failure attempts during the run. |

The existing fields are now populated by the retry process:

- `num_tool_calls` counts every attempt, including retries.
- `num_tool_errors` counts failed attempts.
- `num_retries` counts additional attempts after a failure.
- `failed` is `true` only when the run cannot recover or encounters another terminal runner failure.
- `failure_type` contains the terminal reason, such as `tool_timeout`.

Example recovered run summary:

```json
{
  "num_tool_calls": 2,
  "num_tool_errors": 1,
  "num_retries": 1,
  "num_injected_failures": 1,
  "failed": false,
  "failure_type": null
}
```

## Important limitations

- These records are synthetic. Keep that label when presenting the project.
- The failure probabilities are simplified and are not estimates of real-world tool reliability.
- A fixed seed is for repeatability. A future dataset generator should vary seeds and task/config combinations across runs.
- The current agent treats exhausted tool retries as a terminal failure. It does not yet judge whether a final natural-language answer is factually correct.

## Next step

Build a dataset generator that runs varied tasks using multiple configurations and seeds, then saves clean train/validation/test-ready run and checkpoint records.
