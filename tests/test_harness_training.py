from __future__ import annotations

from motif_transfer.contracts import (
    Observation,
    SourcePolicyStepRecord,
    SourceTransitionReceipt,
)
from motif_transfer.harness_training import (
    build_harness_training_examples,
    summarize_harness_training_examples,
)
from motif_transfer.instrumented_import import ImportedSourceEpisode


def _record(episode, step):
    before = Observation({"step": step}, ("L", "R"))
    after = Observation(
        {"step": step + 1}, ("L", "R"), terminal=step == 2,
    )
    receipt = SourceTransitionReceipt.create(
        before, episode_id=episode, step=step,
        selected_skill_hash="skill", action_response_hash=f"r{step}",
        action="R", action_origin="AGENT", policy_adapter="action_taking",
        after=after, reward=float(step == 2),
    )
    return SourcePolicyStepRecord(
        episode, step, before, "skill", "skill", "untrusted", f"r{step}",
        "R", "AGENT", "action_taking", after, float(step == 2), receipt,
    )


def test_training_examples_are_receipt_derived_and_split_by_episode():
    episodes = tuple(
        ImportedSourceEpisode(
            f"e{i}", "game", tuple(_record(f"e{i}", step) for step in range(3)),
            (), 1.0, True, (),
        )
        for i in range(6)
    )
    examples = build_harness_training_examples(episodes)
    summary = summarize_harness_training_examples(examples)
    assert summary["all_valid"]
    assert summary["target_data_used"] is False
    assert summary["agent_verdicts_used_as_labels"] is False
    assert set(summary["split_counts"]) == {
        "train", "validation", "source_held_out",
    }
    assert summary["objective_counts"]["NEXT_TRANSITION"] == 18
    assert summary["objective_counts"]["RECORDED_ADJACENCY"] == 18
    assert summary["objective_counts"]["TRANSITION_MEMBERSHIP"] == 36
    assert summary["objective_counts"]["OPERATIONAL_EFFECT_PROBE"] == 18
    effect = next(
        row for row in examples if row.objective == "OPERATIONAL_EFFECT_PROBE"
    )
    assert effect.target_payload["verdict"] == "OBSERVED_FROM_RECEIPT"
    assert set(effect.target_payload) == {
        "observation_changed",
        "admissible_set_changed",
        "positive_native_reward",
        "terminal",
        "verdict",
    }
