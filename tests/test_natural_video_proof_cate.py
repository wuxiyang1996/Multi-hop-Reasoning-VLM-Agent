from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from motif_transfer.natural_video_proof_cate import (
    SOURCE_CONTRACT,
    artifact_content_hash,
    compile_v19_features,
    proof_binding_rotation,
    validate_v34_artifact,
)
from motif_transfer.natural_video_recovery import BASE_FEATURE_NAMES, FEATURE_NAMES


REPO = Path(__file__).resolve().parents[1]


def _model(feature_count: int) -> dict:
    return {
        "feature_mean": [0.0] * feature_count,
        "feature_scale": [1.0] * feature_count,
        "input_weights": [[0.0] for _ in range(feature_count)],
        "hidden_bias": [0.0],
        "output_weights": [0.0],
        "output_bias": 0.0,
    }


def _artifact() -> dict:
    body = {
        "schema_version": 34,
        "status": "FROZEN_NATURAL_VIDEO_SOURCE_PROOF_CATE",
        "source_contract": list(SOURCE_CONTRACT),
        "feature_names": list(FEATURE_NAMES),
        "base_feature_count": len(BASE_FEATURE_NAMES),
        "decision_threshold": 0.0,
        "model_seeds": [0],
        "source_proof_models": [_model(len(FEATURE_NAMES))],
        "base_only_control_models": [_model(len(BASE_FEATURE_NAMES))],
        "permuted_uplift_control_models": [_model(len(FEATURE_NAMES))],
        "shuffled_proof_control_models": [_model(len(FEATURE_NAMES))],
    }
    return body | {"artifact_sha256": artifact_content_hash(body)}


def test_consumed_receipt_recompiles_without_outcome_fields() -> None:
    rows = json.loads((
        REPO / "runs/natural_video_v19_expanded_formal/formal_receipts.json"
    ).read_text(encoding="utf-8"))
    compiled = compile_v19_features(rows[0])
    assert len(compiled) == len(FEATURE_NAMES)


def test_feature_compiler_rejects_forbidden_runtime_supervision() -> None:
    rows = json.loads((
        REPO / "runs/natural_video_v19_expanded_formal/formal_receipts.json"
    ).read_text(encoding="utf-8"))
    row = copy.deepcopy(rows[0])
    row["runtime_saw_gold_or_official_structure"] = True
    with pytest.raises(ValueError, match="forbidden supervision"):
        compile_v19_features(row)


def test_frozen_artifact_hash_and_contract_are_enforced() -> None:
    artifact = _artifact()
    authentic, base, permuted, shuffled, threshold = validate_v34_artifact(artifact)
    assert authentic.feature_count == len(FEATURE_NAMES)
    assert base.feature_count == len(BASE_FEATURE_NAMES)
    assert permuted.feature_count == shuffled.feature_count == len(FEATURE_NAMES)
    assert threshold == 0.0
    artifact["source_contract"][0] = "LOOK_AT_TARGET_LABEL"
    with pytest.raises(ValueError, match="contract drift"):
        validate_v34_artifact(artifact)


def test_proof_binding_rotation_is_outcome_blind_and_video_disjoint() -> None:
    rows = [
        {"benchmark": "star", "family": "Sequence", "sample_id": f"q{i}",
         "video_id": f"v{i}", "gold_answer": chr(65 + i)}
        for i in range(4)
    ]
    without_gold = [
        {key: value for key, value in row.items() if key != "gold_answer"}
        for row in rows
    ]
    mapping = proof_binding_rotation(rows, 17)
    assert mapping == proof_binding_rotation(without_gold, 17)
    assert all(index != target for index, target in enumerate(mapping))
    assert all(rows[index]["video_id"] != rows[target]["video_id"]
               for index, target in enumerate(mapping))
