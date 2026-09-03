"""Frozen target-native uplift policy for Sokoban-to-natural-video transfer.

The source contribution is deliberately small: Sokoban supplies the
COMMIT/VERIFY/REFUTED/REPLAN controller topology.  STAR/NExT-QA supply neural
groundings for the typed verification steps.  This module contains no target
answers, labels, action strings, or official scene annotations.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .natural_video_recovery import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    build_features,
)
from .video_proof_grounder import FrozenTanhEnsemble


SOURCE_CONTRACT = (
    "COMMIT",
    "VERIFY_EXPECTED_EFFECT",
    "EXPECTED_EFFECT_REFUTED",
    "REPLAN_OR_ABSTAIN",
)


def compile_v19_features(row: Mapping[str, Any]) -> tuple[float, ...]:
    """Recompile a consumed receipt using runtime-safe fields only."""

    if bool(row.get("runtime_saw_gold_or_official_structure", True)):
        raise ValueError("natural-video runtime receipt used forbidden supervision")
    compiled = build_features(
        benchmark=str(row["benchmark"]),
        family=str(row["family"]),
        primary=row["primary"],
        proof=row["proof"],
    )
    stored = tuple(map(float, row.get("features", ())))
    if len(stored) != len(FEATURE_NAMES) or not np.allclose(
        compiled, stored, rtol=0.0, atol=1e-12,
    ):
        raise ValueError("natural-video feature receipt does not recompile exactly")
    return compiled


def artifact_content_hash(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    payload = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_v34_artifact(value: Mapping[str, Any]) -> tuple[
    FrozenTanhEnsemble,
    FrozenTanhEnsemble,
    FrozenTanhEnsemble,
    FrozenTanhEnsemble,
    float,
]:
    """Validate and instantiate authentic plus development-control heads."""

    if value.get("status") != "FROZEN_NATURAL_VIDEO_SOURCE_PROOF_CATE":
        raise ValueError("unexpected natural-video CATE artifact status")
    if tuple(value.get("source_contract", ())) != SOURCE_CONTRACT:
        raise ValueError("Sokoban source controller contract drift")
    if tuple(value.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("natural-video CATE feature schema mismatch")
    if int(value.get("base_feature_count", -1)) != len(BASE_FEATURE_NAMES):
        raise ValueError("natural-video CATE base boundary mismatch")
    if artifact_content_hash(value) != value.get("artifact_sha256"):
        raise ValueError("natural-video CATE artifact hash mismatch")
    threshold = float(value["decision_threshold"])
    if not math.isfinite(threshold):
        raise ValueError("natural-video CATE threshold must be finite")
    count = len(value.get("model_seeds", ()))
    if count <= 0:
        raise ValueError("natural-video CATE requires at least one head")
    specifications = (
        ("source_proof_models", len(FEATURE_NAMES)),
        ("base_only_control_models", len(BASE_FEATURE_NAMES)),
        ("permuted_uplift_control_models", len(FEATURE_NAMES)),
        ("shuffled_proof_control_models", len(FEATURE_NAMES)),
    )
    ensembles = []
    for key, feature_count in specifications:
        models = value.get(key, ())
        if len(models) != count:
            raise ValueError(f"natural-video CATE ensemble width mismatch: {key}")
        ensembles.append(FrozenTanhEnsemble.from_list(models, feature_count))
    return (*ensembles, threshold)


def recover_mask(
    ensemble: FrozenTanhEnsemble,
    features: Sequence[Sequence[float]],
    threshold: float,
) -> np.ndarray:
    """Execute COMMIT->VERIFY and trigger REPLAN only for positive uplift."""

    return ensemble.predict(features) > threshold


def proof_binding_rotation(
    rows: Sequence[Mapping[str, Any]], seed: int,
) -> tuple[int, ...]:
    """Build an outcome-blind proof-receipt derangement within task family.

    Each row retains its own direct/proof answer distributions (the base
    features) but receives another video's typed verification proof.  The
    mapping never reads question text, options, target answers, or outcomes.
    """

    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (str(row["benchmark"]), str(row["family"]))
        cells.setdefault(key, []).append(index)
    mapping = list(range(len(rows)))
    for cell in sorted(cells):
        indices = sorted(cells[cell], key=lambda i: str(rows[i]["sample_id"]))
        if len(indices) <= 1:
            continue
        candidates = []
        for _ in range(1000):
            shuffled = indices[:]
            rng.shuffle(shuffled)
            score = (
                sum(left == right for left, right in zip(indices, shuffled)),
                sum(
                    str(rows[left]["video_id"]) == str(rows[right]["video_id"])
                    for left, right in zip(indices, shuffled)
                ),
            )
            candidates.append((score, shuffled))
            if score == (0, 0):
                break
        shuffled = min(candidates, key=lambda item: item[0])[1]
        for left, right in zip(indices, shuffled):
            mapping[left] = right
    return tuple(mapping)


__all__ = [
    "SOURCE_CONTRACT",
    "artifact_content_hash",
    "compile_v19_features",
    "proof_binding_rotation",
    "recover_mask",
    "validate_v34_artifact",
]
