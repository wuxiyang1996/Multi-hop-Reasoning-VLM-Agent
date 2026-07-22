import pytest

from motif_transfer.binding import BindingVersionSpace
from motif_transfer.contracts import (
    BindingEvidence,
    BindingHypothesis,
    EvidenceVerdict,
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
