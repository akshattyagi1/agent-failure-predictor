# Live shadow-mode prediction

This feature connects the running local Ollama agent to the FastAPI predictor. After every observed LLM or tool event, the runner builds the same 20 checkpoint features used by the trained model and sends them to `POST /predict`.

It is **shadow mode**: the prediction is observed and logged, but it never changes the agent's tool choice, retry behavior, or final answer. That makes it safe to measure the predictor before allowing it to influence an agent.

## Run it

Start the API in one PowerShell window:

```powershell
uvicorn api.main:app --reload
```

In a second PowerShell window, run an agent task with live prediction enabled:

```powershell
python -m agent_failure_predictor.runner "What is 18 multiplied by 7?" --prediction-endpoint http://127.0.0.1:8000
```

`--prediction-endpoint` is optional. Without it, the runner behaves exactly as before. A short timeout prevents an unavailable API from stopping the agent; that problem is saved in the step event's `failure_prediction_error` field.

## What is recorded

Each record in `telemetry/steps.jsonl` can now contain:

- `failure_prediction_probability`: estimated probability that the current run will eventually fail.
- `failure_prediction_label`: `high_risk` or `low_risk`.
- `failure_prediction_threshold`: the saved decision threshold used by the model.
- `failure_prediction_top_factors`: the telemetry features most responsible for this score.
- `failure_prediction_error`: why the optional prediction request could not be completed.

The run summary also counts `num_live_predictions` and `num_high_risk_predictions`.

## Why this matters

The model is still trained on synthetic data, so a live score is not proof that it will generalize. Shadow mode lets us collect real Ollama-agent traces and later compare each prediction with the actual run result. That is the next source of genuine evaluation data.
