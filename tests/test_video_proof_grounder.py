from __future__ import annotations

import numpy as np
import pytest

from motif_transfer.video_proof_grounder import (
    V14_FEATURE_NAMES,
    artifact_content_hash,
    validate_v14_artifact,
)
from motif_transfer.video_recovery_cate import FEATURE_NAMES


def _model(width: int, bias: float = 0.0) -> dict:
    return {
        "feature_mean": [0.0] * width,
        "feature_scale": [1.0] * width,
        "input_weights": [[0.0] for _ in range(width)],
        "hidden_bias": [0.0],
        "output_weights": [0.0],
        "output_bias": bias,
    }


def test_v14_artifact_validates_three_ensembles() -> None:
    artifact = {
        "status": "FROZEN_CLEVRER_PROOF_PAIRED_UPLIFT_ENSEMBLE",
        "feature_names": list(V14_FEATURE_NAMES),
        "base_feature_count": len(FEATURE_NAMES),
        "decision_threshold": 0.2,
        "model_seeds": [0, 1],
        "proof_models": [_model(len(V14_FEATURE_NAMES), 0.3)] * 2,
        "base_only_control_models": [_model(len(FEATURE_NAMES))] * 2,
        "permuted_uplift_control_models": [_model(len(V14_FEATURE_NAMES))] * 2,
    }
    artifact["artifact_sha256"] = artifact_content_hash(artifact)
    proof, base, permuted, threshold = validate_v14_artifact(artifact)
    assert threshold == 0.2
    assert np.allclose(proof.predict([[0.0] * len(V14_FEATURE_NAMES)]), [0.3])
    assert np.allclose(base.predict([[0.0] * len(FEATURE_NAMES)]), [0.0])
    assert np.allclose(permuted.predict([[0.0] * len(V14_FEATURE_NAMES)]), [0.0])


def test_v14_artifact_rejects_mutation() -> None:
    artifact = {
        "status": "FROZEN_CLEVRER_PROOF_PAIRED_UPLIFT_ENSEMBLE",
        "feature_names": list(V14_FEATURE_NAMES),
        "base_feature_count": len(FEATURE_NAMES),
        "decision_threshold": 0.2,
        "model_seeds": [0],
        "proof_models": [_model(len(V14_FEATURE_NAMES))],
        "base_only_control_models": [_model(len(FEATURE_NAMES))],
        "permuted_uplift_control_models": [_model(len(V14_FEATURE_NAMES))],
    }
    artifact["artifact_sha256"] = artifact_content_hash(artifact)
    artifact["decision_threshold"] = 0.1
    with pytest.raises(ValueError, match="content hash"):
        validate_v14_artifact(artifact)
