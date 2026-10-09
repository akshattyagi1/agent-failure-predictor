from features.engineering import engineer_checkpoint_features


def test_engineered_features_only_use_observed_events():
    steps = [
        {"run_id": "run-1", "event_type": "tool_call", "tool_name": "search", "tool_success": False, "latency_ms": 10, "retry_count": 0, "task_type": "document_lookup"},
        {"run_id": "run-1", "event_type": "tool_call", "tool_name": "search", "tool_success": False, "latency_ms": 20, "retry_count": 1, "task_type": "document_lookup"},
        {"run_id": "run-1", "event_type": "tool_call", "tool_name": "search", "tool_success": True, "latency_ms": 15, "retry_count": 2, "task_type": "document_lookup"},
    ]
    frame = engineer_checkpoint_features(
        steps, [{"run_id": "run-1", "failed": True, "failure_type": "tool_timeout"}],
        [{"run_id": "run-1", "task_id": "doc_001", "policy_profile": "mixed_faults"}],
    )
    second = frame.iloc[1]
    third = frame.iloc[2]
    assert second["consecutive_tool_errors"] == 2
    assert second["repeated_tool_calls_so_far"] == 1
    assert second["tool_error_rate_so_far"] == 1.0
    assert third["consecutive_tool_errors"] == 0
    assert third["steps_since_last_successful_tool"] == 0
