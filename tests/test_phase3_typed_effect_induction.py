import json

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.phase3_typed_effect_induction import (
    IMMEDIATE_EFFECT,
    MEDIUM_EFFECT,
    PERSISTENCE_EFFECT,
    SHORT_EFFECT,
    induce_typed_effect_program,
    target_trial_order,
    typed_intervention_sets_from_rows,
    validate_typed_effect_program,
)


def _row(snapshot, split, rank, repeat, effects, steps, *, horizon=16):
    body = {
        "schema_version": "PHASE1_MATCHED_OPTION_FORK_V1",
        "status": "INTERVENTION_OBSERVED",
        "game": "must_not_be_a_feature",
        "snapshot_id": snapshot,
        "source_split": split,
        "candidate_id": stable_hash([snapshot, rank]),
        "candidate_rank": rank,
        "repeat_index": repeat,
        "candidate_action": f"SECRET_NATIVE_{rank}",
        "option_template_id": None,
        "horizon": horizon,
        "continuation_mode": "common",
        "observed_actions": steps,
        "actions": [f"SECRET_NATIVE_{rank}"] * steps,
        "step_rewards": [],
        "cumulative_returns": {
            "h1": effects[0], "h2": effects[0], "h4": effects[1],
            "h8": effects[2], "h16": effects[3],
        },
        "transition_hashes": [stable_hash([snapshot, rank, i]) for i in range(steps)],
        "expected_fork_observable_sha256": "before",
        "observed_fork_observable_sha256": "before",
        "observed_native_actions_sha256": "native",
        "final_observable_sha256": "after",
        "terminated": steps < horizon,
        "truncated": False,
        "error": None,
        "claim_boundary": "audit",
    }
    return body | {"row_sha256": stable_hash(body)}


def _dataset(selected_axis=SHORT_EFFECT):
    axis = {
        IMMEDIATE_EFFECT: 0,
        SHORT_EFFECT: 1,
        MEDIUM_EFFECT: 2,
        PERSISTENCE_EFFECT: 3,
    }[selected_axis]
    rows = []
    for split, count in (("development", 8), ("qualification", 8)):
        for index in range(count):
            winner = index % 4
            snapshot = f"{split}-{index}"
            for rank in range(4):
                values = [0.0, 0.0, 0.0, 2.0 if rank == winner else 0.0]
                steps = 4
                if axis < 3:
                    values[axis] = 1.0 if rank == winner else 0.0
                else:
                    steps = 16 if rank == winner else 4
                for repeat in range(2):
                    rows.append(_row(
                        snapshot, split, rank, repeat, values, steps,
                    ))
    return rows


def test_induction_selects_typed_effect_without_exporting_native_actions():
    examples, audit = typed_intervention_sets_from_rows(
        _dataset(SHORT_EFFECT), primary_horizon=16,
    )
    program = induce_typed_effect_program(
        examples, source_receipts_sha256=stable_hash("source"),
    )
    validate_typed_effect_program(program)
    assert program["selected_effect_type"] == SHORT_EFFECT
    assert program["status"] == "SOURCE_TYPED_EFFECT_PROGRAM_QUALIFIED"
    assert audit["native_action_tokens_exported"] is False
    assert "SECRET_NATIVE" not in json.dumps(program)


def test_uninformative_source_induces_abstention():
    rows = _dataset(SHORT_EFFECT)
    # Destroy every early typed effect while retaining long-horizon labels.
    for row in rows:
        body = dict(row)
        body.pop("row_sha256")
        final = body["cumulative_returns"]["h16"]
        body["cumulative_returns"] = {
            "h1": 0.0, "h2": 0.0, "h4": 0.0, "h8": 0.0,
            "h16": final,
        }
        body["observed_actions"] = 16
        body["terminated"] = False
        body["row_sha256"] = stable_hash(body)
        row.clear(); row.update(body)
    examples, _ = typed_intervention_sets_from_rows(rows, primary_horizon=16)
    program = induce_typed_effect_program(
        examples, source_receipts_sha256=stable_hash("uninformative"),
    )
    # MEDIUM is still predictive in this synthetic alteration; explicitly use
    # a high, frozen qualification requirement to exercise induced abstention.
    program = induce_typed_effect_program(
        examples, source_receipts_sha256=stable_hash("uninformative"),
        minimum_qualification_accuracy=1.1,
    )
    assert program["status"] == "SOURCE_TYPED_EFFECT_ABSTENTION_INDUCED"
    assert program["operators"] == []


def test_target_runtime_uses_content_not_source_identity_and_fails_closed():
    examples, _ = typed_intervention_sets_from_rows(
        _dataset(PERSISTENCE_EFFECT), primary_horizon=16,
    )
    program = induce_typed_effect_program(
        examples, source_receipts_sha256=stable_hash("source"),
    )
    effects = [
        {PERSISTENCE_EFFECT: 0.2},
        {PERSISTENCE_EFFECT: 0.9},
        {PERSISTENCE_EFFECT: 0.4},
    ]
    order, reason = target_trial_order(program, effects)
    assert reason is None
    assert order == (1, 2, 0)
    clone = dict(program)
    clone["lineage_display_name"] = "renamed"
    body = dict(clone); body.pop("program_sha256")
    clone["program_sha256"] = stable_hash(body)
    assert target_trial_order(clone, effects) == (order, None)
    tied, reason = target_trial_order(program, [
        {PERSISTENCE_EFFECT: 0.5}, {PERSISTENCE_EFFECT: 0.5},
    ])
    assert tied == ()
    assert reason == "TARGET_TYPED_EFFECT_ARGMAX_NOT_UNIQUE"


def test_tampered_program_is_rejected():
    examples, _ = typed_intervention_sets_from_rows(
        _dataset(IMMEDIATE_EFFECT), primary_horizon=16,
    )
    program = induce_typed_effect_program(
        examples, source_receipts_sha256=stable_hash("source"),
    )
    program["selected_effect_type"] = "SOURCE_NAME"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_typed_effect_program(program)
