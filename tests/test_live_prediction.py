from agent_failure_predictor.live_prediction import CheckpointFeatureTracker, PredictionClient
from agent_failure_predictor.telemetry import StepEvent
from features.engineering import engineer_checkpoint_features


def test_live_feature_tracker_matches_offline_engineering():
    events = [
        StepEvent(run_id="run-1", model="test", step_number=1, event_type="llm_call", latency_ms=5, input_tokens=10, output_tokens=2),
        StepEvent(run_id="run-1", model="test", step_number=1, event_type="tool_call", latency_ms=10, tool_name="search", tool_success=False),
        StepEvent(run_id="run-1", model="test", step_number=1, event_type="tool_call", latency_ms=20, tool_name="search", tool_success=False, retry_count=1),
        StepEvent(run_id="run-1", model="test", step_number=1, event_type="tool_call", latency_ms=15, tool_name="search", tool_success=True, retry_count=2),
    ]
    tracker = CheckpointFeatureTracker()
    live_snapshots = [tracker.observe(event) for event in events]
    offline = engineer_checkpoint_features(
        [event.model_dump(mode="json") for event in events],
        [{"run_id": "run-1", "failed": False}],
        [{"run_id": "run-1"}],
    )

    for feature, value in live_snapshots[-1].items():
        assert offline.iloc[-1][feature] == value


def test_prediction_client_accepts_base_url_or_predict_url():
    assert PredictionClient("http://127.0.0.1:8000").endpoint == "http://127.0.0.1:8000/predict"
    assert PredictionClient("http://127.0.0.1:8000/predict").endpoint == "http://127.0.0.1:8000/predict"
