# Database schema discovery

## Why this exists

The schema-blind `0.1.0` agent was asked to query a private local database but was not told its column names. In the 90-run baseline, it sometimes invented names such as `customer_name` and `customer_id`, even though the actual table contains `id`, `name`, and `balance`.

That is a realistic agent failure, but it is not a useful test of whether an agent can reason once it has access to the environment's interface.

## The `0.2.0` change

The agent now has a read-only `describe_database_schema` tool. It returns:

```text
Table customers: id (INTEGER), name (TEXT), balance (REAL).
```

The system prompt tells the agent to use this tool when it needs database table or column names. It still decides whether to call the tool, writes its own SQL, and interprets the result itself.

```text
Database question
  -> agent decides schema is needed
  -> describe_database_schema
  -> agent writes SELECT query
  -> query_database
  -> agent answers
```

This is interface discovery, not answer leakage. It is analogous to software reading an API's documentation before making a request.

## Experiment comparison

- `real_ollama_90runs_v1` is the schema-blind `0.1.0` baseline.
- The next experiment using the current code is the schema-aware `0.2.0` version.

Keep their datasets separate. The meaningful comparison is a held-out experiment with the same task bank and evaluator:

- task correctness and operational-failure rate,
- database-tool error types,
- live predictor precision, recall, and false-alert rate.

Do not combine both versions to claim an improvement without reporting the version split.
