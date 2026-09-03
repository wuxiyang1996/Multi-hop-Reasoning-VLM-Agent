from __future__ import annotations

from motif_transfer.alfworld_hierarchical_grounder import grounder_features


def test_masked_features_do_not_change_with_required_option() -> None:
    kwargs = {
        "goal": "put a hot mug in coffeemachine",
        "observation": "You see a mug on countertop 1.",
        "action": "take mug 1 from countertop 1",
        "step": 3,
        "action_history": ("go to countertop 1",),
        "feature_bins": 32,
        "mask_required_option": True,
    }
    acquire = grounder_features(required_option="ACQUIRE", **kwargs)
    place = grounder_features(required_option="PLACE", **kwargs)
    assert (acquire == place).all()
