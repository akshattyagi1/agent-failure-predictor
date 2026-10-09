"""Configurable, seeded failure injection for synthetic agent telemetry."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class InjectedToolError(RuntimeError):
    """Base class for simulated tool failures."""

    def __init__(self, message: str, added_latency_ms: int = 0) -> None:
        super().__init__(message)
        self.added_latency_ms = added_latency_ms


class InjectedToolTimeout(InjectedToolError):
    pass


class InjectedToolUnavailable(InjectedToolError):
    pass


class InjectedInvalidArguments(InjectedToolError):
    pass


class InjectedNoResults(InjectedToolError):
    pass


class FailureInjectionConfig(BaseModel):
    """All values default to safe/no-failure behaviour."""

    seed: int = 42
    max_retries: int = Field(default=0, ge=0, le=10)
    tool_timeout_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_unavailable_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    invalid_arguments_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    no_results_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    added_latency_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    min_added_latency_ms: int = Field(default=0, ge=0, le=30_000)
    max_added_latency_ms: int = Field(default=0, ge=0, le=30_000)
    progressive_timeout_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    progressive_timeout_attempts: int = Field(default=0, ge=0, le=10)
    progressive_latency_increment_ms: int = Field(default=0, ge=0, le=30_000)
    empty_result_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    malformed_result_probability: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def latency_range_is_valid(self) -> "FailureInjectionConfig":
        if self.min_added_latency_ms > self.max_added_latency_ms:
            raise ValueError("min_added_latency_ms cannot exceed max_added_latency_ms")
        return self


class FailureInjector:
    """Deterministically chooses synthetic faults according to a config."""

    def __init__(self, config: FailureInjectionConfig) -> None:
        self.config = config
        self._random = random.Random(config.seed)
        self._operations: dict[str, ToolOperation] = {}

    @classmethod
    def from_json(cls, path: Path | None) -> "FailureInjector":
        if path is None:
            return cls(FailureInjectionConfig())
        contents = json.loads(path.read_text(encoding="utf-8"))
        return cls(FailureInjectionConfig.model_validate(contents))

    def begin_tool_operation(self, operation_id: str) -> None:
        """Choose persistent faults once, so retries form a coherent trajectory."""
        timeout_attempts = (
            self.config.progressive_timeout_attempts
            if self._random.random() < self.config.progressive_timeout_probability
            else 0
        )
        quality = "normal"
        if self._random.random() < self.config.empty_result_probability:
            quality = "empty"
        elif self._random.random() < self.config.malformed_result_probability:
            quality = "malformed"
        self._operations[operation_id] = ToolOperation(timeout_attempts, quality)

    def end_tool_operation(self, operation_id: str) -> None:
        self._operations.pop(operation_id, None)

    def before_tool_call(self, tool_name: str, operation_id: str | None = None, retry_count: int = 0) -> int:
        """Optionally delay or raise one simulated failure before a tool runs."""
        added_latency_ms = 0
        operation = self._operations.get(operation_id) if operation_id else None
        if operation and retry_count < operation.progressive_timeout_attempts:
            added_latency_ms = (retry_count + 1) * self.config.progressive_latency_increment_ms
            if added_latency_ms:
                time.sleep(added_latency_ms / 1000)
            raise InjectedToolTimeout(
                f"Simulated progressive timeout {retry_count + 1}/{operation.progressive_timeout_attempts} calling {tool_name}",
                added_latency_ms,
            )
        if self._random.random() < self.config.added_latency_probability:
            added_latency_ms = self._random.randint(
                self.config.min_added_latency_ms, self.config.max_added_latency_ms
            )
            time.sleep(added_latency_ms / 1000)

        # One attempt has at most one injected failure, making labels unambiguous.
        if self._random.random() < self.config.tool_timeout_probability:
            raise InjectedToolTimeout(f"Simulated timeout calling {tool_name}")
        if self._random.random() < self.config.tool_unavailable_probability:
            raise InjectedToolUnavailable(f"Simulated unavailable tool: {tool_name}")
        if self._random.random() < self.config.invalid_arguments_probability:
            raise InjectedInvalidArguments(f"Simulated invalid arguments for {tool_name}")
        if self._random.random() < self.config.no_results_probability:
            raise InjectedNoResults(f"Simulated no results from {tool_name}")
        return added_latency_ms

    def alter_successful_result(self, operation_id: str, result: str) -> tuple[str, str]:
        """Return a non-crashing but degraded result for the current operation."""
        operation = self._operations.get(operation_id)
        quality = operation.result_quality if operation else "normal"
        if quality == "empty":
            return "No useful results were returned by the tool.", quality
        if quality == "malformed":
            return '{"result": "incomplete",', quality
        return str(result), quality


@dataclass(frozen=True)
class ToolOperation:
    progressive_timeout_attempts: int
    result_quality: str


def injection_name(error: Exception) -> str | None:
    """Return the stable synthetic fault label, or None for a real tool error."""
    names = {
        InjectedToolTimeout: "tool_timeout",
        InjectedToolUnavailable: "tool_unavailable",
        InjectedInvalidArguments: "invalid_tool_arguments",
        InjectedNoResults: "tool_no_results",
    }
    for error_type, name in names.items():
        if isinstance(error, error_type):
            return name
    return None
