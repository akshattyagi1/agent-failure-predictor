import pandas as pd

from models.compare_real_aware_models import split_train_validation_test_by_template


def test_template_split_keeps_repeated_attempts_together():
    rows = []
    for template_number in range(10):
        for attempt in range(3):
            rows.append({
                "template_group": f"template-{template_number}",
                "run_id": f"run-{template_number}-{attempt}",
                "eventual_failed": bool(template_number % 2),
            })
    frame = pd.DataFrame(rows)

    train, validation, test = split_train_validation_test_by_template(frame, 0.2, 0.2, seed=42)
    partitions = [set(partition["template_group"]) for partition in (train, validation, test)]

    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])
    assert set(frame["template_group"]) == set().union(*partitions)
