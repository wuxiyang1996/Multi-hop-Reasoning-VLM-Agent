"""Matched-model CATE for source-structured natural-video verification."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .natural_video_recovery import BASE_FEATURE_NAMES, FEATURE_NAMES
from .video_proof_grounder import FrozenTanhEnsemble


SOURCE_CONTRACT = (
    "COMMIT",
    "VERIFY_EXPECTED_EFFECT",
    "EXPECTED_EFFECT_REFUTED",
    "REPLAN_OR_ABSTAIN",
)


def cross_video_binding_rotation(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell_fields: Sequence[str],
) -> tuple[int, ...]:
    """Rotate proof receipts across videos while preserving declared cells.

    Sorting rows by video and rotating by the largest video-block size produces
    an exact cross-video derangement whenever no video owns more than half of a
    cell.  Impossible cells are rejected instead of silently weakening the
    binding control.
    """

    cells: dict[tuple[str, ...], list[int]] = {}
    for index, row in enumerate(rows):
        cells.setdefault(tuple(str(row[field]) for field in cell_fields), []).append(index)
    mapping = list(range(len(rows)))
    for cell, raw_indices in sorted(cells.items()):
        indices = sorted(
            raw_indices,
            key=lambda index: (str(rows[index]["video_id"]), str(rows[index]["sample_id"])),
        )
        counts: dict[str, int] = {}
        for index in indices:
            video = str(rows[index]["video_id"])
            counts[video] = counts.get(video, 0) + 1
        maximum = max(counts.values())
        if len(counts) < 2 or maximum * 2 > len(indices):
            raise ValueError(f"cross-video binding is impossible for cell {cell}: {counts}")
        rotated = indices[maximum:] + indices[:maximum]
        if any(
            str(rows[left]["video_id"]) == str(rows[right]["video_id"])
            for left, right in zip(indices, rotated)
        ):
            raise AssertionError(f"cross-video binding construction failed for cell {cell}")
        for left, right in zip(indices, rotated):
            mapping[left] = right
    return tuple(mapping)


def artifact_content_hash(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    payload = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_v36_artifact(value: Mapping[str, Any]) -> tuple[
    FrozenTanhEnsemble,
    FrozenTanhEnsemble,
    FrozenTanhEnsemble,
    FrozenTanhEnsemble,
    float,
]:
    if value.get("status") != "FROZEN_MATCHED_MODEL_NATURAL_VIDEO_SOURCE_CATE":
        raise ValueError("unexpected V36 matched CATE status")
    if tuple(value.get("source_contract", ())) != SOURCE_CONTRACT:
        raise ValueError("V36 source controller contract drift")
    if tuple(value.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("V36 feature schema mismatch")
    if int(value.get("base_feature_count", -1)) != len(BASE_FEATURE_NAMES):
        raise ValueError("V36 base feature boundary mismatch")
    if artifact_content_hash(value) != value.get("artifact_sha256"):
        raise ValueError("V36 artifact hash mismatch")
    threshold = float(value["decision_threshold"])
    if not math.isfinite(threshold):
        raise ValueError("V36 threshold must be finite")
    count = len(value.get("model_seeds", ()))
    specifications = (
        ("source_proof_models", len(FEATURE_NAMES)),
        ("base_only_control_models", len(BASE_FEATURE_NAMES)),
        ("permuted_uplift_control_models", len(FEATURE_NAMES)),
        ("cross_video_binding_control_models", len(FEATURE_NAMES)),
    )
    ensembles = []
    for key, feature_count in specifications:
        models = value.get(key, ())
        if count <= 0 or len(models) != count:
            raise ValueError(f"V36 ensemble width mismatch: {key}")
        ensembles.append(FrozenTanhEnsemble.from_list(models, feature_count))
    return (*ensembles, threshold)


__all__ = [
    "SOURCE_CONTRACT",
    "artifact_content_hash",
    "cross_video_binding_rotation",
    "validate_v36_artifact",
]
