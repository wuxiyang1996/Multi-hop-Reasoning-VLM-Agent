from types import SimpleNamespace

import motif_transfer.phase1_common_search_ir as common_ir
from motif_transfer.phase1_common_search_ir import (
    analyze_common_search_ir,
    build_discovery_option_template_artifact,
    build_discovery_primitive_template_artifact,
    canonical_policy_sha256,
    option_rows_to_ledgers,
    validate_option_template_artifact,
)
from motif_transfer.sokoban_search_automaton_v16 import (
    BACKTRACK,
    COMMIT,
    EXPLORE,
    REFUTED,
    UNBOUND,
    VERIFIED,
)


POLICY = {UNBOUND: EXPLORE, REFUTED: BACKTRACK, VERIFIED: COMMIT}


def test_discovery_template_is_frozen_from_positive_discovery_only(
    monkeypatch,
) -> None:
    def record(receipt_id: str, action: str, reward: float):
        return SimpleNamespace(
            transition=SimpleNamespace(receipt_id=receipt_id),
            action=action,
            reward=reward,
        )

    records = (
        record("low-0", "LEFT", 1.0),
        record("low-1", "A", 0.0),
        record("best-0", "RIGHT", 2.0),
        record("best-1", "B", 3.0),
        record("best-2", "A", 0.0),
        record("heldout-0", "UP", 100.0),
        record("heldout-1", "A", 100.0),
    )
    episode = SimpleNamespace(records=records)
    executions = (
        SimpleNamespace(
            split="discovery",
            skill_id="COMMIT/ATTACK",
            execution_id="low",
            transition_receipt_ids=("low-0", "low-1"),
        ),
        SimpleNamespace(
            split="discovery",
            skill_id="COMMIT/POSITION",
            execution_id="best",
            transition_receipt_ids=("best-0", "best-1", "best-2"),
        ),
        SimpleNamespace(
            split="fresh",
            skill_id="LEAKED/FRESH",
            execution_id="heldout",
            transition_receipt_ids=("heldout-0", "heldout-1"),
        ),
    )
    monkeypatch.setattr(common_ir, "import_native_source_batch", lambda _path: (episode,))
    monkeypatch.setattr(
        common_ir,
        "build_execution_sets",
        lambda _game, _episodes: (SimpleNamespace(executions=executions),),
    )

    artifact = build_discovery_option_template_artifact(
        "/does/not/need/to/exist", game="gymv_strider", horizon=8
    )
    templates = validate_option_template_artifact(
        artifact, game="gymv_strider", horizon=8
    )

    assert artifact["selected_discovery_execution"]["skill_id"] == "COMMIT/POSITION"
    assert artifact["selected_discovery_execution"]["official_cumulative_return"] == 5.0
    assert templates[0]["family"] == "AUTHENTIC_DISCOVERY_EXECUTION"
    assert templates[0]["actions"] == [
        "RIGHT", "B", "A", "RIGHT", "B", "A", "RIGHT", "B",
    ]
    assert len({row["template_id"] for row in templates}) == 4


def test_primitive_templates_use_discovery_vocabulary_without_rewards(
    monkeypatch,
) -> None:
    def record(receipt_id: str, action: str, reward: float):
        return SimpleNamespace(
            transition=SimpleNamespace(receipt_id=receipt_id),
            action=action,
            reward=reward,
        )

    records = (
        record("d-0", "RIGHT", -100.0),
        record("d-1", "B", 0.0),
        record("d-2", "RIGHT", 100.0),
        record("f-0", "LEAKED", 1000.0),
    )
    episode = SimpleNamespace(records=records)
    executions = (
        SimpleNamespace(
            split="discovery",
            skill_id="COMMIT/POSITION",
            execution_id="discovery",
            transition_receipt_ids=("d-0", "d-1", "d-2"),
        ),
        SimpleNamespace(
            split="fresh",
            skill_id="LEAKED/FRESH",
            execution_id="fresh",
            transition_receipt_ids=("f-0",),
        ),
    )
    monkeypatch.setattr(common_ir, "import_native_source_batch", lambda _path: (episode,))
    monkeypatch.setattr(
        common_ir,
        "build_execution_sets",
        lambda _game, _episodes: (SimpleNamespace(executions=executions),),
    )

    artifact = build_discovery_primitive_template_artifact(
        "/does/not/need/to/exist", game="gymv_strider", horizon=4
    )

    assert {row["actions"][0] for row in artifact["templates"]} == {"RIGHT", "B"}
    assert all(len(set(row["actions"])) == 1 for row in artifact["templates"])
    assert "LEAKED" not in str(artifact)
    assert "reward-blind" in artifact["selection"].lower()


def _rows(states_per_split: int = 4):
    rows = []
    for split_index, split in enumerate(("development", "qualification", "heldout")):
        for state in range(states_per_split):
            verified = (state + split_index + 1) % 4
            for candidate in range(4):
                value = 2.0 if candidate == verified else float(candidate == (verified + 1) % 4)
                for repeat in range(2):
                    rows.append({
                        "status": "INTERVENTION_OBSERVED",
                        "snapshot_id": f"{split}-{state}",
                        "source_split": split,
                        "candidate_id": f"candidate-{candidate}",
                        "candidate_rank": candidate,
                        "repeat_index": repeat,
                        "observed_actions": 8,
                        "cumulative_returns": {"h8": value},
                        "final_observable_sha256": f"final-{candidate}",
                        "transition_hashes": [f"edge-{candidate}"],
                    })
    return rows


def test_ledgers_strip_native_actions_and_require_stable_unique_best() -> None:
    ledgers, audit = option_rows_to_ledgers(_rows(), primary_horizon=8)

    assert len(ledgers) == 12
    assert audit["native_action_tokens_exported_to_ir"] is False
    assert not any("candidate_action" in attempt for row in ledgers for attempt in row["attempts"])
    assert set(audit["verified_candidate_rank_counts"]) == {"0", "1", "2", "3"}


def test_common_ir_induces_v16_equivalent_policy() -> None:
    expected = canonical_policy_sha256(POLICY)
    report = analyze_common_search_ir(
        _rows(states_per_split=8),
        primary_horizon=8,
        source_gate_requirements={
            "minimum_fresh_eligible_states": 8,
            "minimum_fresh_examples_per_selected_action": 8,
            "minimum_authentic_success_rate": 1.0,
            "minimum_authentic_minus_each_destructive_control": 0.40,
            "minimum_mean_matched_advantage_per_branch": 1.0,
            "require_alpha_renaming_invariance": True,
            "require_isomorphic_exhaustive_ceiling_reported_not_used_as_gate": True,
        },
        minimum_eligible_fraction_each_split=1.0,
        expected_policy_sha256=expected,
    )

    assert report["source_gate_passed"]
    assert report["canonical_policy_equivalence_passed"]
    assert report["canonical_policy_sha256"] == expected


def test_formal_infrastructure_gate_fails_closed() -> None:
    rows = _rows(states_per_split=8)
    rows[0] = rows[0] | {"status": "INTERVENTION_FAILED"}
    report = analyze_common_search_ir(
        rows,
        primary_horizon=8,
        source_gate_requirements={
            "minimum_fresh_eligible_states": 4,
            "minimum_fresh_examples_per_selected_action": 4,
            "minimum_authentic_success_rate": 1.0,
            "minimum_authentic_minus_each_destructive_control": 0.40,
            "minimum_mean_matched_advantage_per_branch": 1.0,
            "require_alpha_renaming_invariance": True,
            "require_isomorphic_exhaustive_ceiling_reported_not_used_as_gate": True,
        },
        minimum_eligible_fraction_each_split=0.5,
        expected_policy_sha256=canonical_policy_sha256(POLICY),
        maximum_intervention_failed_rows=0,
    )

    assert not report["source_gate_passed"]
    assert report["infrastructure_gate"] == {
        "intervention_failed_rows": 1,
        "maximum_intervention_failed_rows": 0,
        "passed": False,
    }
