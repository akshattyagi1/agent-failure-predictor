# Real task bank v2: templates and environment variants

`data/tasks/real_evaluation_task_bank_v2.jsonl` adds 30 previously unseen task templates across three isolated environments:

| Environment | Documents | Database table |
| --- | --- | --- |
| `default` | Agent Hub / Sydney / telemetry | `customers(id, name, balance)` |
| `accounts` | Account Monitor / Perth / audit events | `accounts(account_id, holder_name, current_balance)` |
| `orders` | Order Lens / Brisbane / run records | `orders(order_id, customer, total)` |

Every task now has a `task_template_id` and `environment_id`. These fields let later ML experiments group and split by task template or environment rather than accidentally treating repeated versions of the same task as independent examples.

Run one pass through all 30 v2 templates:

```powershell
python -m agent_failure_predictor.evaluate_real_runs --runs 30 --task-bank data/tasks/real_evaluation_task_bank_v2.jsonl --experiment-name real_v2_templates_pilot --prediction-endpoint http://127.0.0.1:8000
```

The current environment selector is intentionally process-local and designed for sequential local experiments. A future multi-user AgentHub version should use isolated per-run resources rather than shared module state.
