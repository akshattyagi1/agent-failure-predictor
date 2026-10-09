# Safe tool diagnostics

Real-run evaluation found several database-tool failures recorded only as `OperationalError`. That class name alone cannot tell us whether a tool call used malformed SQL, hit a database lock, or encountered another database problem.

The runner now records privacy-conscious diagnostic metadata on every `tool_call` event in `steps.jsonl`.

## New fields

| Field | Purpose |
| --- | --- |
| `tool_argument_keys` | Argument names supplied to the tool, such as `sql`. |
| `tool_argument_summary` | Structured argument metadata, never raw argument values. |
| `tool_error_detail` | A short, sanitized error shape when the attempt fails. |

For SQL tools, the summary includes the statement kind, length, whether it contains a semicolon, and a redacted query shape. For example:

```json
{
  "tool_argument_keys": ["sql"],
  "tool_argument_summary": {
    "sql_statement_kind": "select",
    "sql_length": 48,
    "sql_has_semicolon": false,
    "sql_shape": "SELECT * FROM customers WHERE balance > <number>"
  }
}
```

Quoted values, numeric values, SQL comments, and excessive text are removed or shortened before logging. Other tools retain only argument counts and types—not raw values.

## Why this is useful

On the next real experiment, a failed database call can be separated into categories such as:

- a non-`SELECT` request,
- an accidental semicolon rejected by the safe database tool,
- a malformed SQL shape,
- a likely dependency/database issue despite a valid `SELECT` shape.

This is diagnostic telemetry only. It does not become a predictor feature by default, because logging format and argument content may differ between agents and create portability or privacy problems.

Existing experiment files remain unchanged. The new fields appear only in future runs.
