# Database tool protocol

## Problem observed

The schema-blind baseline and the soft-prompt `0.2.0` version both showed that a local LLM may invent database columns such as `customer_name` and `customer_id`. The `0.2.0` agent was given a schema-discovery tool, but chose not to call it.

## `0.3.0` protocol guard

The agent now follows this runtime protocol:

```text
Database question
  -> describe_database_schema
  -> query_database
```

If the agent requests `query_database` before discovering the schema, the runner does not execute the SQL. It writes a `tool_call` event with:

```json
{
  "tool_success": false,
  "error_type": "RuntimeError",
  "recoverable_tool_error": true
}
```

It then returns a tool message telling the LLM to call `describe_database_schema`. The run continues rather than being labelled terminally failed. Once schema discovery succeeds, the LLM still independently chooses its SQL and explains the result.

Small local models can occasionally render a requested tool call as JSON in normal text instead of using Ollama's native tool-call field. The runner accepts this fallback only when it is valid JSON (including a narrowly normalized escaped-JSON form), names one of the explicitly registered local tools, and has an object of parameters. For the one no-argument schema tool only, it also accepts a final line exactly equal to `describe_database_schema`. Unknown names and malformed text are ignored. This compatibility layer lets a model's explicit schema-discovery request proceed without granting arbitrary code execution.

## Why this is not answer leakage

The guard does not supply the answer or prewrite SQL. It enforces the environment's interface contract, comparable to requiring an application to obtain an authentication token before calling an API.

## Versioned comparison

- `0.1.0`: schema-blind baseline.
- `0.2.0`: schema tool available, but optional; the LLM did not use it.
- `0.3.0`: schema discovery required before database queries.

Keep experiment versions separate and compare both task reliability and live-predictor metrics on held-out runs.
