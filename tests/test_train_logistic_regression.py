import pandas as pd

from models.train_logistic_regression import FEATURE_COLUMNS, split_by_run, split_train_validation_test


def test_split_by_run_keeps_all_checkpoints_of_a_run_together():
    rows = []
    for run_index in range(8):
        for checkpoint_index in range(2):
            rows.append({
                "run_id": f"run-{run_index}",
                "eventual_failed": run_index >= 4,
                "checkpoint_index": checkpoint_index + 1,
                "cumulative_latency_ms": 10.0,
                "llm_calls_so_far": 1,
                "retries_so_far": 0,
                "tool_calls_so_far": 0,
                "tool_errors_so_far": 0,
            })
    frame = pd.DataFrame(rows)
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    train, test = split_by_run(frame, test_size=0.25, seed=42)
    assert set(train["run_id"]).isdisjoint(set(test["run_id"]))
    assert train["eventual_failed"].nunique() == 2
    assert test["eventual_failed"].nunique() == 2


def test_three_way_split_has_no_run_id_overlap():
    frame = pd.DataFrame([
        {
            "run_id": f"run-{run_index}", "eventual_failed": run_index >= 5,
            "checkpoint_index": 1, "cumulative_latency_ms": 10.0,
            "llm_calls_so_far": 1, "retries_so_far": 0, "tool_calls_so_far": 0, "tool_errors_so_far": 0,
        }
        for run_index in range(10)
    ])
    train, validation, test = split_train_validation_test(frame, test_size=0.2, validation_size=0.2, seed=42)
    assert set(train["run_id"]).isdisjoint(validation["run_id"])
    assert set(train["run_id"]).isdisjoint(test["run_id"])
    assert set(validation["run_id"]).isdisjoint(test["run_id"])
