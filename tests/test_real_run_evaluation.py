import pytest

from agent_failure_predictor.evaluate_real_runs import answer_matches, load_evaluation_tasks, run_real_evaluation
from agent_failure_predictor.generate_dataset import build_processed_data


def test_answer_matching_is_deterministic():
    assert answer_matches("The answer is 126.", {"type": "numeric", "expected": 126})
    assert not answer_matches("The answer is 125.", {"type": "numeric", "expected": 126})
    assert answer_matches("Asha and Jordan qualify.", {"type": "contains_all", "expected": ["asha", "jordan"]})
    assert not answer_matches("Only Asha qualifies.", {"type": "contains_all", "expected": ["asha", "jordan"]})


def test_real_labels_override_operational_outcome_in_processed_data(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "steps.jsonl").write_text(
        '{"run_id":"run-1","event_type":"llm_call","latency_ms":10,"task_type":"calculation"}\n',
        encoding="utf-8",
    )
    (raw / "runs.jsonl").write_text(
        '{"run_id":"run-1","failed":false,"failure_type":null}\n', encoding="utf-8"
    )
    processed = tmp_path / "processed"
    build_processed_data(
        raw, processed, [{"run_id": "run-1", "task_id": "real_calc_001"}],
        synthetic=False,
        outcome_overrides={"run-1": {"failed": True, "failure_type": "incorrect_answer", "task_success": False}},
    )
    runs = (processed / "runs.csv").read_text(encoding="utf-8")
    checkpoints = (processed / "checkpoints.csv").read_text(encoding="utf-8")
    assert "incorrect_answer" in runs
    assert "False" in runs
    assert ",True,incorrect_answer" in checkpoints


def test_version_controlled_real_task_bank_loads():
    tasks = load_evaluation_tasks("data/tasks/real_evaluation_task_bank.jsonl")
    assert len(tasks) >= 30
    assert all("evaluation" in task for task in tasks)
    assert {task["task_type"] for task in tasks} == {"calculation", "document_lookup", "database_lookup"}


def test_held_out_v3_bank_has_new_unique_templates():
    development = load_evaluation_tasks("data/tasks/real_evaluation_task_bank_v2.jsonl")
    held_out = load_evaluation_tasks("data/tasks/real_evaluation_task_bank_v3_held_out.jsonl")

    development_templates = {task["task_template_id"] for task in development}
    held_out_templates = [task["task_template_id"] for task in held_out]
    assert len(held_out) == 30
    assert len(held_out_templates) == len(set(held_out_templates))
    assert set(held_out_templates).isdisjoint(development_templates)


def test_resume_rejects_old_unlabelled_partial_experiment(tmp_path):
    experiment = tmp_path / "partial"
    (experiment / "raw").mkdir(parents=True)
    (experiment / "raw" / "runs.jsonl").write_text('{"run_id":"unlabelled"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="no incremental evaluation records"):
        run_real_evaluation(1, "test", output_root=tmp_path, experiment_name="partial", resume=True)
