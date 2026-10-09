"""Create useful tool-failure diagnostics without storing raw argument values."""

from __future__ import annotations

import re
from typing import Any, Mapping

MAX_DETAIL_LENGTH = 200


def _redact_text(value: str) -> str:
    """Remove quoted strings and numbers, normalize whitespace, and cap length."""
    collapsed = " ".join(value.split())
    without_comments = re.sub(r"--.*$", "", collapsed)
    redacted = re.sub(r"'[^']*'|\"[^\"]*\"", "<redacted>", without_comments)
    redacted = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", redacted)
    return redacted[:MAX_DETAIL_LENGTH]


def summarize_tool_arguments(arguments: Mapping[str, Any]) -> tuple[list[str], dict[str, str | int | float | bool]]:
    """Return structured metadata, never a raw value, for a tool-call attempt."""
    keys = sorted(str(key) for key in arguments)
    summary: dict[str, str | int | float | bool] = {
        "argument_count": len(arguments),
        "string_argument_count": sum(isinstance(value, str) for value in arguments.values()),
    }
    sql = arguments.get("sql")
    if isinstance(sql, str):
        first_word = sql.lstrip().split(maxsplit=1)
        summary.update({
            "sql_statement_kind": first_word[0].casefold() if first_word else "empty",
            "sql_length": len(sql),
            "sql_has_semicolon": ";" in sql,
            "sql_shape": _redact_text(sql),
        })
    return keys, summary


def sanitize_error_detail(error: Exception) -> str:
    """Keep a bounded diagnostic shape while removing quoted values and numbers."""
    return _redact_text(str(error)) or type(error).__name__
