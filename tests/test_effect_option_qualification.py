from __future__ import annotations

from motif_transfer.causal_effect_options import (
    CLASS_CONTEXTUAL,
    CLASS_NULL,
    CLASS_STABLE,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.effect_option_qualification import (
    summarize_source_qualification,
    treatment_action,
)
from motif_transfer.visual_intervention_receipts import FrozenVisualSnapshot


def _artifact():
    return {
        "source_grounding": {
            "action_classes": {
                "A": CLASS_STABLE, "B": CLASS_CONTEXTUAL, "N": CLASS_NULL,
            },
            "source_policy_action_counts": {"A": 1, "B": 2, "N": 3},
        },
        "shuffled_control": {
            "action_classes": {
                "A": CLASS_NULL, "B": CLASS_STABLE, "N": CLASS_CONTEXTUAL,
            },
        },
    }


def _snapshot():
    draft = FrozenVisualSnapshot(
        snapshot_id="", game="game", episode_id="episode", episode_seed=1,
        split="qualification", step=1, prefix_actions=("A",),
        source_action="B", native_actions=("A", "B", "N"),
        expected_observable_sha256=stable_hash("obs"),
        selection_rank_sha256=stable_hash("rank"),
    )
    body = {
        "game": draft.game, "episode_id": draft.episode_id,
        "episode_seed": draft.episode_seed, "split": draft.split,
        "step": draft.step, "prefix_actions": list(draft.prefix_actions),
        "source_action": draft.source_action,
        "native_actions": list(draft.native_actions),
        "expected_observable_sha256": draft.expected_observable_sha256,
        "selection_rank_sha256": draft.selection_rank_sha256,
    }
    return FrozenVisualSnapshot(**{**draft.__dict__, "snapshot_id": stable_hash(body)})


def test_authentic_and_shuffled_use_their_own_effect_grounding():
    snapshot = _snapshot()
    authentic = treatment_action(
        _artifact(), snapshot, treatment="AUTHENTIC_EFFECT_STRUCTURE",
        mode="FULL_TREATMENT_REGIME", horizon_step=0,
    )
    shuffled = treatment_action(
        _artifact(), snapshot, treatment="SHUFFLED_EFFECT_STRUCTURE",
        mode="FULL_TREATMENT_REGIME", horizon_step=0,
    )
    assert authentic == "A"
    assert shuffled == "B"
    common_authentic = treatment_action(
        _artifact(), snapshot, treatment="AUTHENTIC_EFFECT_STRUCTURE",
        mode="COMMON_HASH_CONTINUATION", horizon_step=2,
    )
    common_shuffled = treatment_action(
        _artifact(), snapshot, treatment="SHUFFLED_EFFECT_STRUCTURE",
        mode="COMMON_HASH_CONTINUATION", horizon_step=2,
    )
    assert common_authentic == common_shuffled


def _receipt(snapshot, treatment, value):
    return {
        "status": "INTERVENTION_OBSERVED",
        "snapshot_id": snapshot,
        "mode": "FULL_TREATMENT_REGIME",
        "treatment": treatment,
        "returns": {"h1": value, "h2": value, "h4": value, "h8": value},
        "positive": {"h1": value > 0, "h2": value > 0,
                     "h4": value > 0, "h8": value > 0},
        "before_lives": 4,
        "after_lives": 4,
    }


def test_source_gate_requires_paired_uplift_over_every_control():
    rows = []
    for snapshot in ("s1", "s2", "s3"):
        rows.append(_receipt(snapshot, "AUTHENTIC_EFFECT_STRUCTURE", 2.0))
        for control in (
            "SHUFFLED_EFFECT_STRUCTURE", "ALL_ACTION_HASH_RANDOM",
            "DISCOVERY_ACTION_MARGINAL",
        ):
            rows.append(_receipt(snapshot, control, 1.0))
        rows.append(_receipt(snapshot, "REPEAT_SOURCE_ACTION", 0.0))
    summary = summarize_source_qualification(rows)
    assert summary["qualification_passed"]
    assert summary["next_step"] == "RUN_HELD_OUT_SOURCE_CONFIRMATION"

    rows[-4] = _receipt("s3", "SHUFFLED_EFFECT_STRUCTURE", 100.0)
    failed = summarize_source_qualification(rows)
    assert not failed["qualification_passed"]
    assert failed["next_step"] == "STOP_BEFORE_HELD_OUT_AND_TARGET"
