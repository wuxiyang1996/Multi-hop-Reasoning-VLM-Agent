from copy import deepcopy

from motif_transfer.contracts import stable_hash
from motif_transfer.direct_prospective_matrix_v1 import SOURCE_GAMES
from motif_transfer.phase2_webshop_utility_v1 import (
    FAILED_STATUS,
    PASSED_STATUS,
    build_report,
)
from motif_transfer.webshop_search_automaton_v16 import (
    AUTHENTIC,
    CEILING,
    CONDITIONS,
    LEDGER_BLIND,
    PERMUTED,
    RAW,
)


def _manifest():
    return {
        "manifest_sha256": "m" * 64,
        "tasks": [
            {
                "target_identity": f"fresh-{index}",
                "task_id": f"webshop.{index}",
                "source_game": SOURCE_GAMES[index % len(SOURCE_GAMES)],
            }
            for index in range(32)
        ],
    }


def _receipt(index, condition):
    strict = {
        RAW: index < 7,
        AUTHENTIC: index < 18,
        PERMUTED: index < 7,
        LEDGER_BLIND: index < 4,
        CEILING: index < 18,
    }[condition]
    selected_action = (
        f"correct-{index}"
        if condition in {AUTHENTIC, CEILING}
        else f"{condition}-{index}"
    )
    action = (
        "BACKTRACK_REPLAN", "COMMIT_VERIFY", "EXPLORE_UNTRIED"
    )[index % 3]
    body = {
        "target_identity": f"fresh-{index}",
        "task_id": f"webshop.{index}",
        "condition": condition,
        "source_game": SOURCE_GAMES[index % len(SOURCE_GAMES)],
        "initial_state_hash": f"initial-{index}",
        "strict_success": strict,
        "pass_success": strict,
        "official_reward": float(strict),
        "step_count": 5,
        "steps": [{"selected_action": selected_action}],
        "failure": None,
        "unsafe_commits": [],
        "v16_controller": {
            "source_decisions": 1 if condition == AUTHENTIC else 0,
            "source_action_counts": {action: 1} if condition == AUTHENTIC else {},
        },
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _receipts():
    return [
        _receipt(index, condition)
        for index in range(32) for condition in CONDITIONS
    ]


def test_powered_matched_matrix_passes_all_causal_utility_gates():
    report = build_report(_manifest(), _receipts())
    assert report["status"] == PASSED_STATUS
    assert all(report["gates"].values())
    assert report["paired"][RAW]["wins"] == 11
    assert report["paired"][RAW]["losses"] == 0
    assert report["paired"][RAW]["exact_two_sided_p"] < 0.05
    assert report["paired"][PERMUTED]["wins"] == 11
    assert report["paired"][LEDGER_BLIND]["wins"] == 14
    assert set(report["source_lineages"]) == set(SOURCE_GAMES)
    assert all(
        row["source_decisions"] > 0
        for row in report["source_lineages"].values()
    )


def test_target_ceiling_mismatch_fails_closed():
    receipts = _receipts()
    row = next(
        row for row in receipts
        if row["target_identity"] == "fresh-0" and row["condition"] == CEILING
    )
    changed = deepcopy(row)
    changed["steps"] = [{"selected_action": "different-target-written-action"}]
    changed.pop("receipt_sha256")
    changed["receipt_sha256"] = stable_hash(changed)
    receipts[receipts.index(row)] = changed
    report = build_report(_manifest(), receipts)
    assert report["status"] == FAILED_STATUS
    assert not report["gates"]["authentic_matches_target_native_ceiling_exactly"]


def test_corrupt_receipt_hash_fails_closed():
    receipts = _receipts()
    receipts[0]["receipt_sha256"] = "0" * 64
    report = build_report(_manifest(), receipts)
    assert report["status"] == FAILED_STATUS
    assert not report["gates"]["all_receipt_hashes_valid"]
