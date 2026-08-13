import json

import pytest

from motif_transfer.contracts import stable_hash
from scripts.aggregate_webshop_neural_symbolic_v11 import aggregate


CONDITIONS = [
    "target_only",
    "target_native_myopic",
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
]


def _fixture(tmp_path):
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    frozen_config = tmp_path / "frozen.json"
    config = {
        "task_ids": ["webshop.50", "webshop.51"],
        "conditions": CONDITIONS,
        "runtime_hashes": {"runner": "runner", "grounder": "grounder"},
    }
    frozen_config.write_text(json.dumps(config))
    for task_id in config["task_ids"]:
        for condition in CONDITIONS:
            row = {
                "task_id": task_id,
                "condition": condition,
                "initial_state_hash": f"initial-{task_id}",
                "runtime_hashes": {"runner": "runner", "grounder": "grounder"},
                "strict_success": condition == "authentic_source_plus_target",
                "official_reward": (
                    1.0 if condition == "authentic_source_plus_target" else 0.0
                ),
                "step_count": 2,
                "changed_from_target_rank_zero_count": 1,
                "source_decision_count": 1,
                "failure": None,
            }
            row["receipt_sha256"] = stable_hash(row)
            path = receipt_dir / f"{task_id}.{condition}.json"
            path.write_text(json.dumps(row))
    return receipt_dir, frozen_config, config


def test_aggregate_verifies_complete_receipt_matrix(tmp_path):
    receipt_dir, frozen_config, config = _fixture(tmp_path)

    report = aggregate(receipt_dir, frozen_config, config, None)

    assert report["receipt_count"] == 10
    assert report["receipt_hashes_verified"] is True
    assert report["matched_initial_state_hashes"] is True
    assert report["zero_failures"] is True
    assert report["conditions"]["authentic_source_plus_target"][
        "strict_successes"
    ] == 2


def test_aggregate_rejects_missing_condition(tmp_path):
    receipt_dir, frozen_config, config = _fixture(tmp_path)
    (receipt_dir / "webshop.50.target_only.json").unlink()

    with pytest.raises(ValueError, match="receipt matrix mismatch"):
        aggregate(receipt_dir, frozen_config, config, None)
