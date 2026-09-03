from __future__ import annotations

from copy import deepcopy

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.phase3_alfworld_typed_grounder import (
    ARTIFACT_VERSION,
    EFFECT_TYPES,
    validate_artifact,
)


def _artifact():
    head = {
        "hidden_activation": "tanh",
        "layers": [{"weights": [[0.0]], "bias": [0.0]}],
    }
    body = {
        "artifact_version": ARTIFACT_VERSION,
        "effect_types": list(EFFECT_TYPES),
        "required_option_masked_for_every_head": True,
        "formal_success_read_for_training_or_qualification": False,
        "typed_effect_heads": {effect: head for effect in EFFECT_TYPES},
    }
    return body | {"artifact_sha256": stable_hash(body)}


def test_validates_content_bound_outcome_blind_artifact():
    validate_artifact(_artifact())


def test_rejects_formal_success_exposure():
    artifact = deepcopy(_artifact())
    artifact["formal_success_read_for_training_or_qualification"] = True
    body = dict(artifact)
    body.pop("artifact_sha256")
    artifact["artifact_sha256"] = stable_hash(body)
    with pytest.raises(ValueError, match="formal success"):
        validate_artifact(artifact)


def test_rejects_effect_vocabulary_change():
    artifact = deepcopy(_artifact())
    artifact["effect_types"] = list(EFFECT_TYPES[:-1])
    body = dict(artifact)
    body.pop("artifact_sha256")
    artifact["artifact_sha256"] = stable_hash(body)
    with pytest.raises(ValueError, match="vocabulary"):
        validate_artifact(artifact)
