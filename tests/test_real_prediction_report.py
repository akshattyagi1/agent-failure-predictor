import json

from analysis.evaluate_real_predictions import make_real_evaluation_report


def test_real_prediction_report_creates_run_level_metrics(tmp_path):
    experiment = tmp_path / "real_experiment"
    raw = experiment / "raw"
    raw.mkdir(parents=True)
    evaluations = [
        {"run_id": "failed-alerted", "task_id": "one", "task_type": "calculation", "failed": True, "agent_answer": "wrong"},
        {"run_id": "success-unalerted", "task_id": "two", "task_type": "calculation", "failed": False, "agent_answer": "right"},
    ]
    steps = [
        {"run_id": "failed-alerted", "failure_prediction_probability": 0.9, "failure_prediction_label": "high_risk"},
        {"run_id": "success-unalerted", "failure_prediction_probability": 0.1, "failure_prediction_label": "low_risk"},
    ]
    (raw / "evaluations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in evaluations), encoding="utf-8")
    (raw / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in steps), encoding="utf-8")

    output = experiment / "real_evaluation"
    report = make_real_evaluation_report(experiment, output)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))

    assert report.exists()
    assert metrics["true_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["precision"] == 1.0
