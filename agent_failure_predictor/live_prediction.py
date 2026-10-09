"""Shadow-mode feature tracking and HTTP calls for live failure predictions."""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

from .telemetry import StepEvent

RECENT_WINDOW = 3


@dataclass
class CheckpointFeatureTracker:
    """Build exactly the model features from events observed in one active run."""

    checkpoint_index: int = 0
    llm_calls_so_far: int = 0
    tool_calls_so_far: int = 0
    tool_errors_so_far: int = 0
    retries_so_far: int = 0
    cumulative_latency_ms: float = 0.0
    input_tokens_so_far: int = 0
    output_tokens_so_far: int = 0
    consecutive_tool_errors: int = 0
    repeated_tool_calls_so_far: int = 0
    empty_results_so_far: int = 0
    malformed_results_so_far: int = 0
    tool_counts: Counter[str] = field(default_factory=Counter)
    recent_tool_outcomes: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    tool_latencies: list[float] = field(default_factory=list)
    last_success_event_index: int | None = None
    _previous_tool_latency: float = 0.0

    def observe(self, event: StepEvent) -> dict[str, float]:
        """Update state with an event, then return its non-leaking feature snapshot."""
        self.checkpoint_index += 1
        self.cumulative_latency_ms += event.latency_ms
        self.input_tokens_so_far += event.input_tokens or 0
        self.output_tokens_so_far += event.output_tokens or 0
        last_event_was_tool_error = 0

        if event.event_type == "llm_call":
            self.llm_calls_so_far += 1
        else:
            self.tool_calls_so_far += 1
            tool_name = event.tool_name or "unknown_tool"
            if self.tool_counts[tool_name] > 0:
                self.repeated_tool_calls_so_far += 1
            self.tool_counts[tool_name] += 1
            previous = self.tool_latencies[-1] if self.tool_latencies else event.latency_ms
            self._previous_tool_latency = previous
            self.tool_latencies.append(event.latency_ms)
            error = int(event.tool_success is not True)
            self.tool_errors_so_far += error
            self.retries_so_far += int(event.retry_count > 0)
            self.recent_tool_outcomes.append(error)
            last_event_was_tool_error = error
            self.empty_results_so_far += int(event.tool_result_quality == "empty")
            self.malformed_results_so_far += int(event.tool_result_quality == "malformed")
            if error:
                self.consecutive_tool_errors += 1
            else:
                self.consecutive_tool_errors = 0
                self.last_success_event_index = self.checkpoint_index

        last_tool_delta = (
            self.tool_latencies[-1] - self._previous_tool_latency if self.tool_latencies else 0.0
        )
        return {
            "checkpoint_index": float(self.checkpoint_index),
            "cumulative_latency_ms": self.cumulative_latency_ms,
            "llm_calls_so_far": float(self.llm_calls_so_far),
            "retries_so_far": float(self.retries_so_far),
            "tool_calls_so_far": float(self.tool_calls_so_far),
            "tool_errors_so_far": float(self.tool_errors_so_far),
            "input_tokens_so_far": float(self.input_tokens_so_far),
            "output_tokens_so_far": float(self.output_tokens_so_far),
            "consecutive_tool_errors": float(self.consecutive_tool_errors),
            "repeated_tool_calls_so_far": float(self.repeated_tool_calls_so_far),
            "tool_error_rate_so_far": self.tool_errors_so_far / self.tool_calls_so_far if self.tool_calls_so_far else 0.0,
            "recent_tool_error_rate_3": sum(self.recent_tool_outcomes) / len(self.recent_tool_outcomes) if self.recent_tool_outcomes else 0.0,
            "unique_tools_used_so_far": float(len(self.tool_counts)),
            "average_tool_latency_ms_so_far": sum(self.tool_latencies) / len(self.tool_latencies) if self.tool_latencies else 0.0,
            "max_tool_latency_ms_so_far": max(self.tool_latencies, default=0.0),
            "last_tool_latency_delta_ms": last_tool_delta,
            "steps_since_last_successful_tool": float(
                self.checkpoint_index - self.last_success_event_index
                if self.last_success_event_index is not None else self.checkpoint_index
            ),
            "last_event_was_tool_error": float(last_event_was_tool_error),
            "empty_results_so_far": float(self.empty_results_so_far),
            "malformed_results_so_far": float(self.malformed_results_so_far),
        }


@dataclass(frozen=True)
class LivePrediction:
    probability: float
    label: Literal["high_risk", "low_risk"]
    threshold: float
    top_factors: list[str]


class PredictionClient:
    """Minimal dependency-free client for the local FastAPI prediction service."""

    def __init__(self, endpoint: str, timeout_seconds: float = 1.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/predict"):
            self.endpoint = f"{self.endpoint}/predict"
        self.timeout_seconds = timeout_seconds

    def predict(self, features: dict[str, float]) -> LivePrediction:
        payload = json.dumps({"features": features}).encode("utf-8")
        request = Request(self.endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            label = data["prediction"]
            if label not in {"high_risk", "low_risk"}:
                raise ValueError(f"unknown risk label: {label}")
            return LivePrediction(
                probability=float(data["failure_probability"]),
                label=label,
                threshold=float(data["threshold"]),
                top_factors=[str(factor["feature"]) for factor in data.get("top_factors", [])],
            )
        except (URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"live prediction unavailable: {error}") from error
