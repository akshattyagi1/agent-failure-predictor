import pytest

from agent_failure_predictor.failure_injection import (
    FailureInjectionConfig,
    FailureInjector,
    InjectedToolTimeout,
    injection_name,
)


def test_timeout_is_injected_when_probability_is_one():
    injector = FailureInjector(FailureInjectionConfig(tool_timeout_probability=1.0))
    with pytest.raises(InjectedToolTimeout) as error:
        injector.before_tool_call("calculate")
    assert injection_name(error.value) == "tool_timeout"


def test_default_config_does_not_inject_failures():
    injector = FailureInjector(FailureInjectionConfig(seed=5))
    assert injector.before_tool_call("calculate") == 0


def test_progressive_timeout_persists_across_retries():
    injector = FailureInjector(FailureInjectionConfig(
        progressive_timeout_probability=1.0, progressive_timeout_attempts=2,
        progressive_latency_increment_ms=10,
    ))
    injector.begin_tool_operation("operation-1")
    with pytest.raises(InjectedToolTimeout) as first_error:
        injector.before_tool_call("calculate", "operation-1", retry_count=0)
    assert first_error.value.added_latency_ms == 10
    with pytest.raises(InjectedToolTimeout):
        injector.before_tool_call("calculate", "operation-1", retry_count=1)
    assert injector.before_tool_call("calculate", "operation-1", retry_count=2) == 0
