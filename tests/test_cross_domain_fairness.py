from __future__ import annotations

import copy

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.cross_domain_fairness import (
    FairnessProtocolError,
    assert_paired_target_receipts,
    audit_target_bound_suite,
    require_nonpilot_embedding,
    require_formal_suite_audit,
    require_transfer_panel,
)


def _artifact(method: str):
    body = {
        "method": method,
        "source_domains": ["game"],
        "source_episode_ids": ["e0"],
        "source_receipt_ids": ["r0"],
        "source_superset_sha256": "shared",
        "source_superset_episode_ids": ["e0"],
        "expel_refinement_rounds": 1 if method == "expel" else 0,
        "online_memory_updates_allowed": False,
        "target_actions_in_memory_allowed": False,
        "items": [],
        "retrieval_strategy": {
            "episodic_rag": "semantic",
            "random_trajectory_icl": "frozen_random",
        }.get(method, "semantic"),
        "artifact_kind": "FROZEN_TARGET_BOUND_CROSS_DOMAIN_MEMORY_BASELINE",
        "target_binding": {
            "target_domain": "webshop",
            "binding_status": "ALL_ITEMS_ABSTAINED",
            "binding_mode": "SHARED_GATE_ONLY_NO_REWRITE",
            "source_artifact_sha256": "source",
            "adaptation_payload_sha256": "adaptation",
            "item_bindings": [],
            "candidate_payload_sha256": "same-trajectories",
        },
    }
    return body | {"artifact_sha256": stable_hash(body)}


def _suite():
    return {name: _artifact(name) for name in ("expel", "awm", "reasoning_bank")}


def _rehash(artifact):
    artifact["artifact_sha256"] = stable_hash({k: v for k, v in artifact.items() if k != "artifact_sha256"})


def test_suite_requires_shared_superset_and_adaptation():
    audit = audit_target_bound_suite(_suite(), target_domain="webshop", expected_source_episodes=1)
    assert audit.source_episode_count == 1
    assert audit.formal_ready
    assert not audit.exact_baseline_ready
    assert not audit.blockers

    broken = copy.deepcopy(_suite())
    broken["awm"]["source_superset_sha256"] = "other"
    _rehash(broken["awm"])
    with pytest.raises(FairnessProtocolError, match="source superset"):
        audit_target_bound_suite(broken, target_domain="webshop", expected_source_episodes=1)


def test_upstream_pinned_suite_can_pass_formal_gate():
    audit = audit_target_bound_suite(
        _suite(), target_domain="webshop", expected_source_episodes=1,
        implementation_fidelity="upstream_pinned",
    )
    assert audit.formal_ready
    assert audit.exact_baseline_ready


def test_gated_suite_accepts_matched_trajectory_controls():
    methods = ("expel", "awm", "reasoning_bank", "episodic_rag", "random_trajectory_icl")
    suite = {method: _artifact(method) for method in methods}
    audit = audit_target_bound_suite(
        suite, target_domain="webshop", expected_source_episodes=1,
        expected_methods=methods,
    )
    assert audit.formal_ready
    assert audit.transfer_panel == "gated"


def test_formal_rejects_hashing_retriever():
    with pytest.raises(FairnessProtocolError, match="pilot-only"):
        require_nonpilot_embedding({"pilot_only": True}, run_mode="formal")
    require_nonpilot_embedding({"model": "Qwen/Qwen3-Embedding-0.6B"}, run_mode="formal")


def test_raw_and_gated_panels_cannot_be_mixed():
    gated = _artifact("expel")
    require_transfer_panel(gated, transfer_panel="gated")
    with pytest.raises(FairnessProtocolError, match="raw panel"):
        require_transfer_panel(gated, transfer_panel="raw")


def test_paired_receipts_fail_closed_on_task_or_game_mismatch():
    base = {
        "target_domain": "alfworld", "task_id": "task-1", "seed": 7,
        "decision_model": "qwen", "max_steps": 20, "resolved_game_file": "/x/game",
    }
    assert_paired_target_receipts(base, dict(base))
    wrong = dict(base, resolved_game_file="/y/game")
    with pytest.raises(FairnessProtocolError, match="resolved_game_file"):
        assert_paired_target_receipts(base, wrong)


def test_formal_requires_a_passing_hash_bound_suite_audit(tmp_path):
    with pytest.raises(FairnessProtocolError, match="requires"):
        require_formal_suite_audit(None, run_mode="formal", target_domain="webshop")
    # Pilot execution deliberately does not depend on a formal audit.
    require_formal_suite_audit(None, run_mode="pilot", target_domain="webshop")
