from copy import deepcopy

from motif_transfer.contracts import stable_hash
from motif_transfer.direct_prospective_matrix_v1 import SOURCE_GAMES
from motif_transfer.phase2_alfworld_utility_v3 import FAILED_STATUS, PASSED_STATUS, build_report
from motif_transfer.webshop_search_automaton_v16 import AUTHENTIC, CEILING, CONDITIONS, LEDGER_BLIND, PERMUTED, RAW


def _manifest():
    return {
        "manifest_sha256": "v" * 64,
        "gates": {"maximum_discordant_loss_rate": 0.25},
        "tasks": [
            {"target_identity": f"task-{i}", "source_game": SOURCE_GAMES[i % 6]}
            for i in range(75)
        ],
    }


def _receipt(index, condition):
    success = {
        RAW: index < 20,
        AUTHENTIC: index < 36,
        PERMUTED: index < 20,
        LEDGER_BLIND: index < 8,
        CEILING: index < 36,
    }[condition]
    action = ("BACKTRACK_REPLAN", "COMMIT_VERIFY", "EXPLORE_UNTRIED")[index % 3]
    selected = f"auth-{index}" if condition in {AUTHENTIC, CEILING} else f"{condition}-{index}"
    body = {
        "target_identity": f"task-{index}", "condition": condition,
        "source_game": SOURCE_GAMES[index % 6], "initial_state_hash": f"state-{index}",
        "strict_success": success, "pass_success": success,
        "official_reward": float(success), "step_count": 5,
        "steps": [{"selected_action": selected}], "failure": None, "unsafe_commits": [],
        "v16_controller": {
            "source_decisions": 1 if condition == AUTHENTIC else 0,
            "source_action_counts": {action: 1} if condition == AUTHENTIC else {},
        },
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _receipts():
    return [_receipt(index, condition) for index in range(75) for condition in CONDITIONS]


def test_selective_fresh_matrix_passes():
    report = build_report(_manifest(), _receipts())
    assert report["status"] == PASSED_STATUS
    assert all(report["gates"].values())
    assert report["paired"][RAW]["wins"] == 16
    assert report["discordant_negative_transfer_rate"] == 0.0


def test_excess_negative_transfer_fails():
    receipts = _receipts()
    for index in range(6):
        row = next(row for row in receipts if row["target_identity"] == f"task-{index}" and row["condition"] == AUTHENTIC)
        changed = deepcopy(row)
        changed["strict_success"] = False
        changed["pass_success"] = False
        changed["official_reward"] = 0.0
        changed.pop("receipt_sha256")
        changed["receipt_sha256"] = stable_hash(changed)
        receipts[receipts.index(row)] = changed
    report = build_report(_manifest(), receipts)
    assert report["status"] == FAILED_STATUS
    assert not report["gates"]["bounded_negative_transfer"]
