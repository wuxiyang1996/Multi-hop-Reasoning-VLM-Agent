from __future__ import annotations

import numpy as np
import pytest

from motif_transfer.video_recovery_cate import (
    FEATURE_NAMES,
    FrozenTanhRegressor,
    artifact_content_hash,
    build_features,
    validate_frozen_artifact,
)


def test_feature_contract_is_answer_and_program_native() -> None:
    features = build_features(
        family="counterfactual",
        question_program=["all_events", "filter_counterfact", "belong_to"],
        choice_programs=[["all_events"], ["all_events"]],
        explicit_answer="10",
        trajectory_answer="01",
        explicit_error_count=1,
    )
    assert len(features) == len(FEATURE_NAMES)
    assert features[2] == 1.0
    assert features[10] == 0.5
    assert features[13] == 1.0


def test_frozen_tanh_regressor_inference() -> None:
    width = len(FEATURE_NAMES)
    model = FrozenTanhRegressor(
        (0.0,) * width,
        (1.0,) * width,
        tuple((1.0,) for _ in range(width)),
        (0.0,),
        (2.0,),
        0.5,
    )
    prediction = model.predict([[0.0] * width, [1.0] * width])
    assert np.isclose(prediction[0], 0.5)
    assert prediction[1] > prediction[0]


def test_frozen_artifact_validates_content_hash() -> None:
    model = {
        "feature_mean": [0.0] * len(FEATURE_NAMES),
        "feature_scale": [1.0] * len(FEATURE_NAMES),
        "input_weights": [[0.0] for _ in FEATURE_NAMES],
        "hidden_bias": [0.0],
        "output_weights": [0.0],
        "output_bias": 0.0,
    }
    artifact = {
        "status": "FROZEN_TARGET_NATIVE_PAIRED_UPLIFT_GROUNDER",
        "feature_names": list(FEATURE_NAMES),
        "decision_threshold": 0.2,
        "model": model,
        "permuted_control_model": model,
    }
    artifact["artifact_sha256"] = artifact_content_hash(artifact)
    _, _, threshold = validate_frozen_artifact(artifact)
    assert threshold == 0.2

    artifact["decision_threshold"] = 0.3
    with pytest.raises(ValueError, match="content hash"):
        validate_frozen_artifact(artifact)
