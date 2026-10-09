# Failure-Prediction API

The FastAPI service loads the current engineered Logistic Regression model and returns a live risk score plus local attribution factors.

## Start it

First generate the explanation baseline after training:

```powershell
python models/explain_linear_model.py --dataset data/generated/dataset_500_gradual_failures_v1 --model-dir models/logistic_regression_engineered/dataset_500_gradual_failures_v1
```

Then start the API from the project root:

```powershell
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive FastAPI documentation.

## Endpoints

- `GET /health`: confirms that the model loaded.
- `POST /predict`: predicts the risk using current checkpoint features.

Example request shape:

```json
{
  "features": {
    "checkpoint_index": 4,
    "cumulative_latency_ms": 800,
    "llm_calls_so_far": 1,
    "retries_so_far": 2,
    "tool_calls_so_far": 3,
    "tool_errors_so_far": 3,
    "input_tokens_so_far": 305,
    "output_tokens_so_far": 20,
    "consecutive_tool_errors": 3,
    "repeated_tool_calls_so_far": 2,
    "tool_error_rate_so_far": 1.0,
    "recent_tool_error_rate_3": 1.0,
    "unique_tools_used_so_far": 1,
    "average_tool_latency_ms_so_far": 200,
    "max_tool_latency_ms_so_far": 300,
    "last_tool_latency_delta_ms": 100,
    "steps_since_last_successful_tool": 4,
    "last_event_was_tool_error": 1,
    "empty_results_so_far": 0,
    "malformed_results_so_far": 0
  }
}
```

The client must supply exactly the features saved with the model. This strict validation prevents silently predicting from a mismatched telemetry schema.

Set `AGENT_FAILURE_MODEL_DIR` to load another compatible model directory.

## Use it from the local agent

See [LIVE_PREDICTION_README.md](LIVE_PREDICTION_README.md). The runner can send every observed checkpoint to this API in non-blocking shadow mode with `--prediction-endpoint http://127.0.0.1:8000`.
