import pytest
import json

from motif_transfer.binding import (
    AttributedBinding,
    BindingArtifactStatus,
    BindingAttribution,
    BindingVersionSpace,
    FrozenBindingArtifact,
    alpha_rename_target_actions,
    validate_structural_binding,
)
from motif_transfer.artifact_io import load_frozen_binding_artifact, write_frozen_binding_artifact
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


def test_frozen_binding_artifact_detects_tampering(tmp_path):
    row = hypothesis("a", "v1")
    unsigned = {
        "schema_version": 1,
        "motif_id": "motif",
        "adaptation_example_sha256": "demo",
        "induction_repetitions": 2,
        "raw_signature_sets": (("sig",), ("sig",)),
        "alpha_signature_sets": ((), ()),
        "bindings": [{
            "hypothesis": FrozenBindingArtifact._hypothesis_dict(row),
            "attribution": BindingAttribution.TARGET_GROUNDED_PROVISIONAL.value,
        }],
        "status": BindingArtifactStatus.ADMITTED.value,
        "backend_identity_sha256": "backend",
        "call_receipt_hashes": ("receipt",),
    }
    from motif_transfer.contracts import stable_hash
    artifact = FrozenBindingArtifact(
        1, "motif", "demo", 2, (("sig",), ("sig",)), ((), ()),
        (AttributedBinding(row, BindingAttribution.TARGET_GROUNDED_PROVISIONAL),),
        BindingArtifactStatus.ADMITTED, "backend", ("receipt",), stable_hash(unsigned),
    )
    path = tmp_path / "binding.json"
    write_frozen_binding_artifact(path, artifact)
    assert load_frozen_binding_artifact(path).artifact_hash == artifact.artifact_hash
    payload = json.loads(path.read_text())
    payload["motif_id"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_binding_artifact(path)
