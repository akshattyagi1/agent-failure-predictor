import json
from datetime import datetime, timezone

from agent_failure_predictor.telemetry import RunEvent, StepEvent, TelemetryCollector


def test_collector_writes_validated_jsonl(tmp_path):
    collector = TelemetryCollector(tmp_path)
    collector.record_step(StepEvent(
        run_id="run-1", model="llama3.2:3b", step_number=1, event_type="tool_call",
        latency_ms=10.5, tool_name="calculate", tool_success=True,
    ))
    collector.record_run(RunEvent(
        run_id="run-1", model="llama3.2:3b", started_at=datetime.now(timezone.utc),
        final_answer_generated=True, failed=False,
    ))
    step = json.loads((tmp_path / "steps.jsonl").read_text(encoding="utf-8"))
    run = json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8"))
    assert step["event_type"] == "tool_call"
    assert step["failure_prediction_probability"] is None
    assert step["tool_argument_summary"] == {}
    assert step["recoverable_tool_error"] is False
    assert run["failed"] is False
