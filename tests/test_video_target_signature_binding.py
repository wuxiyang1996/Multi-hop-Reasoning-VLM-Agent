from __future__ import annotations

import json
from pathlib import Path

import pytest

from motif_transfer.source_video_operator_algebra import induce_source_video_algebra
from motif_transfer.video_target_signature_binding import (
    authorize_target_signature,
    permuted_algebra,
)


ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> tuple[dict, dict]:
    catalog = json.loads((ROOT / "configs/full_video_source_catalog_v1.json").read_text())
    algebra = induce_source_video_algebra(root=ROOT, catalog=catalog)
    bindings = json.loads((ROOT / "configs/video_target_signature_bindings_v1.json").read_text())
    return algebra, bindings


def test_all_public_question_families_have_source_signature_coverage() -> None:
    algebra, bindings = _inputs()
    for domain in ("agqa", "clevrer"):
        for family in bindings[domain]:
            receipt = authorize_target_signature(
                algebra=algebra, binding_spec=bindings,
                target_domain=domain, question_family=family,
            )
            assert receipt.status == "AUTHORIZED", (domain, family, receipt)
            assert receipt.target_outcome_read is False
            assert not receipt.missing_compositions


def test_unknown_family_abstains_and_outcome_authority_is_rejected() -> None:
    algebra, bindings = _inputs()
    receipt = authorize_target_signature(
        algebra=algebra, binding_spec=bindings,
        target_domain="agqa", question_family="unknown",
    )
    assert receipt.status == "ABSTAINED"
    with pytest.raises(ValueError, match="target outcome"):
        authorize_target_signature(
            algebra=algebra, binding_spec=bindings,
            target_domain="agqa", question_family="goal",
            target_outcome_read=True,
        )


def test_permuted_control_preserves_inventory_and_capacity_but_changes_semantics() -> None:
    algebra, _ = _inputs()
    permuted = permuted_algebra(algebra)
    assert set(permuted["primitive_names"]) == set(algebra["primitive_names"])
    assert len(permuted["primitives"]) == len(algebra["primitives"])
    assert permuted["composition_edge_count"] == algebra["composition_edge_count"]
    assert permuted["composition_edges"] != algebra["composition_edges"]
    assert all(key != value for key, value in permuted["semantic_label_map"].items())
    assert permuted["artifact_sha256"] != algebra["artifact_sha256"]


def test_graph_containment_abstains_when_a_required_edge_is_missing() -> None:
    algebra, bindings = _inputs()
    broken = dict(algebra)
    broken["composition_edges"] = [
        edge for edge in algebra["composition_edges"]
        if edge != ["RELATION_PROJECT", "PRESENCE"]
    ]
    receipt = authorize_target_signature(
        algebra=broken, binding_spec=bindings,
        target_domain="agqa", question_family="membership_question",
    )
    assert receipt.status == "ABSTAINED"
    assert receipt.missing_compositions == (("RELATION_PROJECT", "PRESENCE"),)


def test_v2_every_authentic_family_passes_and_every_matched_permutation_abstains() -> None:
    algebra, _ = _inputs()
    bindings = json.loads(
        (ROOT / "configs/video_target_signature_bindings_v2.json").read_text()
    )
    permuted = permuted_algebra(algebra)
    for domain in ("agqa", "clevrer"):
        for family in bindings[domain]:
            authentic = authorize_target_signature(
                algebra=algebra, binding_spec=bindings,
                target_domain=domain, question_family=family,
            )
            control = authorize_target_signature(
                algebra=permuted, binding_spec=bindings,
                target_domain=domain, question_family=family,
            )
            assert authentic.status == "AUTHORIZED", (domain, family, authentic)
            assert control.status == "ABSTAINED", (domain, family, control)
