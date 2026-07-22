import pytest

from motif_transfer.binding import (
    BindingVersionSpace,
    alpha_rename_target_actions,
    validate_structural_binding,
)
from motif_transfer.contracts import (
    BindingEvidence,
    BindingHypothesis,
    EvidenceVerdict,
    Lifecycle,
    MotifCandidate,
    MotifEdge,
    MotifNode,
)


def hypothesis(binding_id, verifier_id):
    return BindingHypothesis(
        binding_id,
        "motif",
        "untrusted target claim",
        "untrusted prediction",
        ("adaptation-receipt",),
        verifier_id,
    )


def test_refuted_binding_is_removed_without_ranking_remaining_candidates():
    space = BindingVersionSpace((hypothesis("a", "v1"), hypothesis("b", "v2")))
    space.record(BindingEvidence("a", "receipt", "v1", EvidenceVerdict.REFUTED))
    assert [row.binding_id for row in space.viable()] == ["b"]


def test_wrong_verifier_cannot_change_version_space():
    space = BindingVersionSpace((hypothesis("a", "v1"),))
    with pytest.raises(ValueError):
        space.record(BindingEvidence("a", "receipt", "model-judge", EvidenceVerdict.REFUTED))
    assert [row.binding_id for row in space.viable()] == ["a"]


def test_alpha_renaming_preserves_action_equality_without_semantics():
    example = {"transitions": [
        {"action": "go red", "before_native_actions": ["go red", "look"], "after_native_actions": ["look"]},
        {"action": "look", "before_native_actions": ["look"], "after_native_actions": ["go red"]},
    ]}
    renamed = alpha_rename_target_actions(example)
    assert renamed["transitions"][0]["action"] == "TARGET_ACTION_0"
    assert renamed["transitions"][1]["action"] == "TARGET_ACTION_1"
    assert renamed["transitions"][1]["after_native_actions"] == ["TARGET_ACTION_0"]
    assert "go red" not in str(renamed)


def test_structural_binding_requires_full_contiguous_partition():
    motif = MotifCandidate(
        "m", (),
        (MotifNode("n0", ("r0",)), MotifNode("n1", ("r1",))),
        (MotifEdge("n0", "n1", ("f",)),),
        Lifecycle.CANDIDATE,
    )
    signature = validate_structural_binding(
        motif,
        target_cycle_count=3,
        node_alignment=((0, (0, 1)), (1, (2,))),
        edge_alignment=((0, (1, 2)),),
    )
    assert len(signature) == 64
    with pytest.raises(ValueError):
        validate_structural_binding(
            motif,
            target_cycle_count=3,
            node_alignment=((0, (0, 2)), (1, (1,))),
            edge_alignment=((0, (0, 1)),),
        )
