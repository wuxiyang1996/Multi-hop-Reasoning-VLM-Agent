from copy import deepcopy

from motif_transfer.contracts import stable_hash
from motif_transfer.direct_prospective_matrix_v1 import SOURCE_GAMES
from motif_transfer.phase2_alfworld_utility_v1 import (
    FAILED_STATUS,
    PASSED_STATUS,
    build_report,
)
from motif_transfer.webshop_search_automaton_v16 import (
    AUTHENTIC, CEILING, CONDITIONS, LEDGER_BLIND, PERMUTED, RAW,
)


def _manifest():
    return {
        "manifest_sha256": "a" * 64,
        "target_split": "eval_in_distribution",
        "tasks": [
            {"target_identity": f"fresh-alfworld-{index}", "source_game": SOURCE_GAMES[index % 6]}
            for index in range(32)
        ],
    }


def _receipt(index, condition):
    success = {
        RAW: index < 8, AUTHENTIC: index < 20, PERMUTED: index < 8,
        LEDGER_BLIND: index < 5, CEILING: index < 20,
    }[condition]
    action = ("BACKTRACK_REPLAN", "COMMIT_VERIFY", "EXPLORE_UNTRIED")[index % 3]
    selected = f"auth-{index}" if condition in {AUTHENTIC, CEILING} else f"{condition}-{index}"
    body = {
        "target_identity": f"fresh-alfworld-{index}", "condition": condition,
        "source_game": SOURCE_GAMES[index % 6], "initial_state_hash": f"state-{index}",
        "strict_success": success, "pass_success": success,
        "official_reward": float(success), "step_count": 7,
        "steps": [{"selected_action": selected}], "failure": None, "unsafe_commits": [],
        "v16_controller": {
            "source_decisions": 1 if condition == AUTHENTIC else 0,
            "source_action_counts": {action: 1} if condition == AUTHENTIC else {},
        },
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _receipts():
    return [_receipt(index, condition) for index in range(32) for condition in CONDITIONS]


def test_fresh_alfworld_powered_matrix_passes_shared_causal_gates():
    report = build_report(_manifest(), _receipts())
    assert report["status"] == PASSED_STATUS
    assert all(report["gates"].values())
    assert report["schema_version"] == "phase2-alfworld-six-source-utility-report-v1"
    assert report["paired"][RAW]["wins"] == 12


def test_alfworld_ceiling_mismatch_fails_closed():
    receipts = _receipts()
    row = next(row for row in receipts if row["target_identity"] == "fresh-alfworld-0" and row["condition"] == CEILING)
    changed = deepcopy(row)
    changed["steps"] = [{"selected_action": "different"}]
    changed.pop("receipt_sha256")
    changed["receipt_sha256"] = stable_hash(changed)
    receipts[receipts.index(row)] = changed
    assert build_report(_manifest(), receipts)["status"] == FAILED_STATUS
