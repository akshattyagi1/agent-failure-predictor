"""Validated, append-only JSONL telemetry for agent runs."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StepEvent(BaseModel):
    schema_version: str = "0.1"
    run_id: str
    agent_id: str = "local-llama-agent"
    agent_version: str = "0.3.0"
    model: str
    task_type: str = "unknown"
    step_number: int = Field(ge=1)
    event_type: Literal["llm_call", "tool_call"]
    timestamp: datetime = Field(default_factory=utc_now)
    latency_ms: float = Field(ge=0)
    tool_name: str | None = None
    tool_success: bool | None = None
    error_type: str | None = None
    failure_injected: bool = False
    injected_failure_type: str | None = None
    tool_result_quality: Literal["normal", "empty", "malformed"] = "normal"
    tool_argument_keys: list[str] = Field(default_factory=list)
    tool_argument_summary: dict[str, str | int | float | bool] = Field(default_factory=dict)
    tool_error_detail: str | None = None
    recoverable_tool_error: bool = False
    added_latency_ms: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    failure_prediction_probability: float | None = Field(default=None, ge=0, le=1)
    failure_prediction_label: Literal["high_risk", "low_risk"] | None = None
    failure_prediction_threshold: float | None = Field(default=None, ge=0, le=1)
    failure_prediction_top_factors: list[str] = Field(default_factory=list)
    failure_prediction_error: str | None = None


class RunEvent(BaseModel):
    schema_version: str = "0.1"
    run_id: str
    agent_id: str = "local-llama-agent"
    agent_version: str = "0.3.0"
    model: str
    task_type: str = "unknown"
    started_at: datetime
    completed_at: datetime = Field(default_factory=utc_now)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    num_llm_calls: int = Field(default=0, ge=0)
    num_tool_calls: int = Field(default=0, ge=0)
    num_tool_errors: int = Field(default=0, ge=0)
    num_retries: int = Field(default=0, ge=0)
    num_injected_failures: int = Field(default=0, ge=0)
    num_live_predictions: int = Field(default=0, ge=0)
    num_high_risk_predictions: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0, ge=0)
    final_answer_generated: bool
    failure_type: str | None = None
    failed: bool


class TelemetryCollector:
    """Stores raw, schema-validated events; features will be derived later."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def record_step(self, event: StepEvent) -> None:
        self._append(self.directory / "steps.jsonl", event)

    def record_run(self, event: RunEvent) -> None:
        self._append(self.directory / "runs.jsonl", event)

    @staticmethod
    def elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    @staticmethod
    def _append(path: Path, event: BaseModel) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n")
