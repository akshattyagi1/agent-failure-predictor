from agent_failure_predictor.tool_diagnostics import sanitize_error_detail, summarize_tool_arguments


def test_sql_diagnostics_keep_structure_but_redact_values():
    keys, summary = summarize_tool_arguments({"sql": "SELECT * FROM customers WHERE balance > 1000"})

    assert keys == ["sql"]
    assert summary["sql_statement_kind"] == "select"
    assert summary["sql_shape"] == "SELECT * FROM customers WHERE balance > <number>"
    assert "1000" not in summary["sql_shape"]


def test_error_detail_redacts_quoted_values_and_numbers():
    detail = sanitize_error_detail(ValueError('near "secret_value": syntax error at 99'))

    assert "secret_value" not in detail
    assert "99" not in detail
    assert "syntax error" in detail
