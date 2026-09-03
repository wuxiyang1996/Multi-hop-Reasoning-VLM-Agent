from __future__ import annotations

from copy import deepcopy

from motif_transfer.discoveryworld_env import stable_hash
from scripts.summarize_discoveryworld_v23_formal import summarize


CONDITIONS = (
    "target_native_myopic",
    "authentic_sokoban_effect_plus_target",
    "commit_availability_control_plus_target",
    "inverted_effect_control_plus_target",
    "position_prior_control_plus_target",
)


def _protocol() -> dict:
    return {
        "task_ids": ["space", "protein"],
        "conditions": list(CONDITIONS),
        "claim_boundary": "test",
        "source_contract": {
            "source_program_sha256": "source",
            "source_confirmation_sha256": "confirmation",
        },
        "formal_gates": {
            "minimum_eligible_forks": 2,
            "minimum_authentic_success_gain_vs_target_native": 1,
        },
    }


def _freeze() -> dict:
    value = {
        "receipts": [
            {"task_id": "space", "eligible": True},
            {"task_id": "protein", "eligible": True},
        ],
        "outcome_fields_read_for_eligibility": False,
    }
    value["summary_sha256"] = stable_hash(value)
    return value


def _result(task: str, scenario: str, *, target: bool, authentic: bool) -> dict:
    conditions = {"target_only_recorded": {"official_success": False}}
    for name in CONDITIONS:
        success = (
            target if name == "target_native_myopic"
            else authentic if name == "authentic_sokoban_effect_plus_target"
            else False
        )
        conditions[name] = {
            "official_success": success,
            "runtime_error": None,
            "recovery": [],
        }
    value = {
        "status": "FORMAL_MECHANISM_COMPLETE",
        "task": {"scenario": scenario},
        "conditions": conditions,
        "all_selection_receipts_valid": True,
        "all_matched_forks": True,
        "policy_runtime_saw_oracle_scorecard": False,
        "_path": __file__,
    }
    body = dict(value)
    body.pop("_path")
    value["result_sha256"] = stable_hash(body)
    return value


def test_all_predeclared_gates_pass() -> None:
    report = summarize(
        protocol=_protocol(), freeze=_freeze(),
        results={
            "space": _result("space", "Space Sick", target=False, authentic=True),
            "protein": _result("protein", "Proteomics", target=False, authentic=True),
        },
        protocol_file_sha256="p", freeze_file_sha256="f",
    )
    assert report["status"] == "FRESH_FORMAL_TRANSFER_VALIDATED"
    assert report["all_predeclared_gates_passed"]


def test_negative_transfer_fails_without_weakening_other_gates() -> None:
    results = {
        "space": _result("space", "Space Sick", target=True, authentic=False),
        "protein": _result("protein", "Proteomics", target=False, authentic=True),
    }
    report = summarize(
        protocol=_protocol(), freeze=deepcopy(_freeze()), results=results,
        protocol_file_sha256="p", freeze_file_sha256="f",
    )
    assert report["status"] == "FRESH_FORMAL_TRANSFER_FAILED"
    assert not report["gates"]["zero_authentic_negative_transfer_vs_target_native"]
