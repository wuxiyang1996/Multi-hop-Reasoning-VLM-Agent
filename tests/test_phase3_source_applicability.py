import json
from pathlib import Path

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.phase3_source_applicability import (
    SourceApplicabilityPrior,
    maximum_profile_contrast_derangement,
    prior_from_frozen_artifact,
    projected_rank_scores,
)


REPO = Path(__file__).resolve().parents[1]
PROGRAMS = REPO / "configs/phase3_source_induction_v1/frozen_confirmation/programs"


def _artifact(game: str):
    return json.loads((PROGRAMS / f"{game}.json").read_text())


def test_quantile_projection_preserves_source_endpoints():
    scores = projected_rank_scores((1, 2, 3, 4), 3)
    assert scores == pytest.approx((0.1, 0.25, 0.4))


def test_six_source_interventions_produce_distinct_target_orders():
    orders = {}
    for path in sorted(PROGRAMS.glob("*.json")):
        prior = prior_from_frozen_artifact(json.loads(path.read_text()))
        orders[path.stem] = prior.trial_order(3)
    assert len(orders) == 6
    assert len(set(orders.values())) >= 3
    assert orders["tetris"] != orders["gymv_streets_of_rage_2"]


def test_prior_is_profile_content_bound_not_source_identity():
    artifact = _artifact("tetris")
    prior = prior_from_frozen_artifact(artifact)
    profile = dict(artifact["source_only_profile"])
    cloned = SourceApplicabilityPrior.from_profile(profile)
    assert cloned.prior_sha256 == prior.prior_sha256
    assert cloned.trial_order(3) == prior.trial_order(3)


def test_applicability_receipt_is_outcome_blind_and_abstains_on_singleton():
    prior = prior_from_frozen_artifact(_artifact("candy_crush"))
    admitted = prior.applicability_receipt(
        target_candidate_ids=("a", "b", "c"),
        target_grounding_sha256=stable_hash("grounding"),
    )
    assert admitted["admitted"] is True
    assert admitted["target_outcome_read"] is False
    assert admitted["ordered_target_candidate_ids"] != []
    abstained = prior.applicability_receipt(
        target_candidate_ids=("a",),
        target_grounding_sha256=stable_hash("grounding"),
    )
    assert abstained["admitted"] is False


def test_profile_hash_is_fail_closed():
    artifact = _artifact("gymv_columns")
    profile = dict(artifact["source_only_profile"])
    profile["verified_rank_distribution"] = {
        **profile["verified_rank_distribution"], "0": 999,
    }
    with pytest.raises(ValueError, match="profile hash mismatch"):
        SourceApplicabilityPrior.from_profile(profile)


def test_source_only_permutation_is_deranged_and_deterministic():
    artifacts = {
        path.stem: json.loads(path.read_text())
        for path in PROGRAMS.glob("*.json")
    }
    mapping = maximum_profile_contrast_derangement(artifacts)
    assert set(mapping) == set(artifacts)
    assert set(mapping.values()) == set(artifacts)
    assert all(source != control for source, control in mapping.items())
    assert mapping == maximum_profile_contrast_derangement(artifacts)
    assert all(
        prior_from_frozen_artifact(artifacts[source]).trial_order(4)[0]
        != prior_from_frozen_artifact(artifacts[control]).trial_order(4)[0]
        for source, control in mapping.items()
    )
    assert any(
        prior_from_frozen_artifact(artifacts[source]).trial_order(count)
        != prior_from_frozen_artifact(artifacts[control]).trial_order(count)
        for source, control in mapping.items()
        for count in (2, 3, 4)
    )
