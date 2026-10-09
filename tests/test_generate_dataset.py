from agent_failure_predictor.generate_dataset import build_processed_data


def test_processed_data_contains_run_and_checkpoint_labels(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "steps.jsonl").write_text(
        '{"run_id":"run-1","event_type":"llm_call","latency_ms":10,"task_type":"calculation"}\n'
        '{"run_id":"run-1","event_type":"tool_call","latency_ms":5,"tool_success":false,"retry_count":0,"failure_injected":true,"task_type":"calculation"}\n',
        encoding="utf-8",
    )
    (raw / "runs.jsonl").write_text(
        '{"run_id":"run-1","failed":true,"failure_type":"tool_timeout"}\n', encoding="utf-8"
    )
    processed = tmp_path / "processed"
    build_processed_data(raw, processed, [{"run_id":"run-1", "task_id":"calc_001", "policy_profile":"terminal_timeout", "injection_seed":7}])
    runs = (processed / "runs.csv").read_text(encoding="utf-8")
    checkpoints = (processed / "checkpoints.csv").read_text(encoding="utf-8")
    assert "synthetic" in runs
    assert "eventual_failed" in checkpoints
    assert "tool_errors_so_far" in checkpoints
