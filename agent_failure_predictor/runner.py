"""A bounded Ollama tool-calling loop with step-level telemetry."""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ollama import chat

from .failure_injection import FailureInjector, injection_name
from .live_prediction import CheckpointFeatureTracker, PredictionClient
from .telemetry import RunEvent, StepEvent, TelemetryCollector
from .tool_diagnostics import sanitize_error_detail, summarize_tool_arguments
from .tool_protocol import DatabaseSchemaProtocol, SCHEMA_DISCOVERY_REQUIRED_MESSAGE
from .tools import AVAILABLE_TOOLS, initialise_demo_data

SYSTEM_PROMPT = """You are a concise local research assistant. Use a tool when it is needed.
Use calculate for arithmetic and search_documents for the local knowledge base.
For every database question, first call describe_database_schema before calling query_database. Never invent a database schema or tool result.
After tools return, answer the user's task directly."""


def text_tool_call(content: str | None) -> tuple[str, dict] | None:
    """Accept one JSON-shaped local tool request when a small model misses native tool-call formatting."""
    if not content:
        return None
    decoder = json.JSONDecoder()
    # Some small-model responses escape only the nested parameter JSON. Try the exact
    # response first, then this narrowly normalized form; both still face the allow-list below.
    for candidate_text in (content, content.replace(r'\"', '"')):
        for match in re.finditer(r"\{", candidate_text):
            try:
                candidate, _ = decoder.raw_decode(candidate_text[match.start():])
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue
            name = candidate.get("name")
            arguments = candidate.get("parameters", candidate.get("arguments", {}))
            if isinstance(name, str) and name in AVAILABLE_TOOLS and isinstance(arguments, dict):
                return name, arguments
    # A weak local model may emit only the no-argument schema tool's name on its final line.
    # Do not apply this shorthand to parameterized or arbitrary tools.
    final_line = next((line.strip() for line in reversed(content.splitlines()) if line.strip()), "")
    if final_line == "describe_database_schema":
        return final_line, {}
    return None


def classify_task(task: str) -> str:
    lower = task.lower()
    if any(word in lower for word in ("database", "customer", "balance", "sql")):
        return "database_lookup"
    if any(word in lower for word in ("document", "knowledge", "mentioned", "search")):
        return "document_lookup"
    if any(word in lower for word in ("calculate", "what is", "multiply", "percentage")):
        return "calculation"
    return "general"


def resolve_task_type(task: str, declared_task_type: str | None = None) -> str:
    """Trust a versioned task bank's explicit label; use keywords only for ad-hoc CLI tasks."""
    return declared_task_type or classify_task(task)


def run_agent(
    task: str,
    model: str,
    telemetry_directory: Path,
    max_steps: int = 8,
    failure_config: Path | None = None,
    prediction_endpoint: str | None = None,
    prediction_timeout_seconds: float = 1.0,
    environment_id: str = "default",
    declared_task_type: str | None = None,
) -> str:
    """Run the bounded agent and write all observable events before returning."""
    initialise_demo_data(environment_id)
    collector = TelemetryCollector(telemetry_directory)
    injector = FailureInjector.from_json(failure_config)
    database_protocol = DatabaseSchemaProtocol()
    feature_tracker = CheckpointFeatureTracker()
    prediction_client = (
        PredictionClient(prediction_endpoint, prediction_timeout_seconds) if prediction_endpoint else None
    )
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    started_timer = time.perf_counter()
    task_type = resolve_task_type(task, declared_task_type)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
    llm_calls = tool_calls = tool_errors = retries = injected_failures = 0
    input_tokens = output_tokens = 0
    answer = ""
    failure_type: str | None = None
    live_predictions = high_risk_predictions = 0

    def record_checkpoint(event: StepEvent) -> None:
        """Persist a step with its optional shadow-mode prediction attached."""
        nonlocal live_predictions, high_risk_predictions
        features = feature_tracker.observe(event)
        if prediction_client is not None:
            try:
                prediction = prediction_client.predict(features)
                event.failure_prediction_probability = prediction.probability
                event.failure_prediction_label = prediction.label
                event.failure_prediction_threshold = prediction.threshold
                event.failure_prediction_top_factors = prediction.top_factors
                live_predictions += 1
                if prediction.label == "high_risk":
                    high_risk_predictions += 1
                    print(
                        f"[shadow alert] {prediction.probability:.1%} failure risk after "
                        f"checkpoint {feature_tracker.checkpoint_index}; drivers: {', '.join(prediction.top_factors[:3])}"
                    )
            except RuntimeError as error:
                event.failure_prediction_error = str(error)
        collector.record_step(event)

    try:
        for step in range(1, max_steps + 1):
            call_started = time.perf_counter()
            response = chat(model=model, messages=messages, tools=list(AVAILABLE_TOOLS.values()))
            llm_calls += 1
            step_input_tokens = getattr(response, "prompt_eval_count", None)
            step_output_tokens = getattr(response, "eval_count", None)
            input_tokens += step_input_tokens or 0
            output_tokens += step_output_tokens or 0
            record_checkpoint(StepEvent(
                run_id=run_id, model=model, task_type=task_type, step_number=step,
                event_type="llm_call", latency_ms=collector.elapsed_ms(call_started),
                input_tokens=step_input_tokens, output_tokens=step_output_tokens,
            ))
            messages.append(response.message)
            native_calls = response.message.tool_calls or []
            calls = [(call.function.name, call.function.arguments) for call in native_calls]
            if not calls:
                fallback_call = text_tool_call(response.message.content)
                if fallback_call is not None:
                    calls = [fallback_call]
            if not calls:
                answer = response.message.content or ""
                break
            for tool_index, (name, arguments) in enumerate(calls):
                function = AVAILABLE_TOOLS.get(name)
                if function is None:
                    failure_type = "unknown_tool_requested"
                    break
                argument_keys, argument_summary = summarize_tool_arguments(arguments)
                operation_id = f"{run_id}:{step}:{tool_index}:{name}"
                injector.begin_tool_operation(operation_id)
                for retry_count in range(injector.config.max_retries + 1):
                    tool_calls += 1
                    tool_started = time.perf_counter()
                    added_latency_ms = 0
                    injected_failure_type: str | None = None
                    error_detail: str | None = None
                    recoverable_protocol_error = False
                    result_quality = "normal"
                    try:
                        if database_protocol.blocks(name):
                            raise RuntimeError(SCHEMA_DISCOVERY_REQUIRED_MESSAGE)
                        added_latency_ms = injector.before_tool_call(name, operation_id, retry_count)
                        result = function(**arguments)
                        result, result_quality = injector.alter_successful_result(operation_id, result)
                        success, error_type = True, None
                    except Exception as error:  # Capture both simulated and real tool failures.
                        result = f"Tool error: {type(error).__name__}: {error}"
                        success, error_type = False, type(error).__name__
                        error_detail = sanitize_error_detail(error)
                        recoverable_protocol_error = str(error) == SCHEMA_DISCOVERY_REQUIRED_MESSAGE
                        added_latency_ms = getattr(error, "added_latency_ms", added_latency_ms)
                        injected_failure_type = injection_name(error)
                        tool_errors += 1
                        injected_failures += int(injected_failure_type is not None)
                    record_checkpoint(StepEvent(
                        run_id=run_id, model=model, task_type=task_type, step_number=step,
                        event_type="tool_call", latency_ms=collector.elapsed_ms(tool_started), tool_name=name,
                        tool_success=success, error_type=error_type, retry_count=retry_count,
                        failure_injected=injected_failure_type is not None,
                        injected_failure_type=injected_failure_type, tool_result_quality=result_quality,
                        tool_argument_keys=argument_keys, tool_argument_summary=argument_summary,
                        tool_error_detail=error_detail,
                        recoverable_tool_error=recoverable_protocol_error,
                        added_latency_ms=added_latency_ms,
                    ))
                    if success or recoverable_protocol_error:
                        messages.append({"role": "tool", "tool_name": name, "content": str(result)})
                        if success:
                            database_protocol.record_success(name)
                        break
                    if retry_count < injector.config.max_retries:
                        retries += 1
                        continue
                    failure_type = injected_failure_type or "tool_execution_error"
                    break
                injector.end_tool_operation(operation_id)
                if failure_type:
                    break
            if failure_type:
                break
        else:
            failure_type = "max_steps_exceeded"
    except Exception as error:
        failure_type = type(error).__name__
    finally:
        failed = failure_type is not None
        collector.record_run(RunEvent(
            run_id=run_id, model=model, task_type=task_type, started_at=started_at,
            input_tokens=input_tokens, output_tokens=output_tokens,
            num_llm_calls=llm_calls, num_tool_calls=tool_calls, num_tool_errors=tool_errors,
            num_retries=retries, num_injected_failures=injected_failures,
            num_live_predictions=live_predictions, num_high_risk_predictions=high_risk_predictions,
            total_latency_ms=collector.elapsed_ms(started_timer), final_answer_generated=bool(answer),
            failure_type=failure_type, failed=failed,
        ))
    if failure_type:
        raise RuntimeError(f"Agent run failed: {failure_type}")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the instrumented local Llama agent.")
    parser.add_argument("task")
    parser.add_argument("--model", default="llama3.2:3b", help="A locally pulled Ollama model name.")
    parser.add_argument("--telemetry-dir", type=Path, default=Path("telemetry"))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--failure-config", type=Path,
        help="Path to a JSON file defining simulated failure probabilities.",
    )
    parser.add_argument(
        "--prediction-endpoint",
        help="Optional local API base URL or /predict URL. Enables non-blocking shadow-mode predictions.",
    )
    parser.add_argument("--prediction-timeout-seconds", type=float, default=1.0)
    parser.add_argument("--environment", default="default", help="Versioned local document/database environment.")
    args = parser.parse_args()
    print(run_agent(
        args.task, args.model, args.telemetry_dir, args.max_steps, args.failure_config,
        args.prediction_endpoint, args.prediction_timeout_seconds, args.environment,
    ))


if __name__ == "__main__":
    main()
