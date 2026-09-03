from __future__ import annotations

from motif_transfer.contracts import stable_hash
from scripts.summarize_webshop_sokoban_effect_v13 import (
    AUTHENTIC,
    COMPARATORS,
    evaluate,
    exact_binomial_two_sided,
    receipt_hash_valid,
)


def _receipt(task: str, condition: str, success: bool, reward: float, action: str):
    body = {
        "task_id": task,
        "condition": condition,
        "strict_success": success,
        "pass_success": success,
        "official_reward": reward,
        "step_count": 1,
        "steps": [{"selected_action": action}],
        "source_decision_count": int(condition == AUTHENTIC),
        "failure": None,
        "initial_state_hash": f"state-{task}",
    }
    return body | {"receipt_sha256": stable_hash(body)}


def test_exact_binomial_requires_six_unopposed_wins() -> None:
    assert exact_binomial_two_sided(5, 0) == 0.0625
    assert exact_binomial_two_sided(6, 0) == 0.03125
    assert exact_binomial_two_sided(0, 0) == 1.0


def test_receipt_hash_validation_fails_closed() -> None:
    receipt = _receipt("webshop.1", AUTHENTIC, True, 1.0, "auth")
    assert receipt_hash_valid(receipt)
    receipt["official_reward"] = 0.0
    assert not receipt_hash_valid(receipt)


def test_evaluate_requires_every_frozen_comparator_gate() -> None:
    tasks = [f"webshop.{index}" for index in range(6)]
    conditions = [AUTHENTIC, *COMPARATORS]
    config = {"task_ids": tasks, "conditions": conditions}
    rows = []
    for task in tasks:
        rows.append(_receipt(task, AUTHENTIC, True, 1.0, "auth"))
        rows.extend(
            _receipt(task, comparator, False, 0.0, comparator)
            for comparator in COMPARATORS
        )
    result = evaluate(rows, config)
    assert result["passed"]
    assert result["scientific_status"].endswith("VALIDATED")
    assert len(result["receipt_matrix_sha256"]) == 64

    broken = [dict(row) for row in rows]
    for row in broken:
        if row["condition"] == COMPARATORS[-1]:
            row["strict_success"] = True
            row["official_reward"] = 1.0
            body = dict(row)
            body.pop("receipt_sha256")
            row["receipt_sha256"] = stable_hash(body)
    failed = evaluate(broken, config)
    assert not failed["passed"]
    assert not failed["comparison_gates"][COMPARATORS[-1]][
        "positive_strict_success_delta"
    ]
