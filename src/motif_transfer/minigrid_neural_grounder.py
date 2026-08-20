"""Development-only neural grounding for MiniGrid orientation panels."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier

from .contracts import stable_hash
from .minigrid_orientation_recovery import (
    DIRECTION_NAMES,
    TOKENS,
    parse_neural_binding,
)


ARTIFACT_VERSION = "minigrid-orientation-neural-grounder-v1"
PANEL_ORDER = ("I", "P", "C0", *TOKENS)


def pixel_agent_features(
    image: Image.Image, *, side: int = 20, crop_radius: int = 22,
    minimum_red_pixels: int = 80,
) -> np.ndarray:
    """Return a target-state-free local crop centered from red pixels only."""

    array = np.asarray(image.convert("RGB")).astype(np.int16)
    red = (
        (array[:, :, 0] >= 170)
        & (array[:, :, 0] >= array[:, :, 1] + 80)
        & (array[:, :, 0] >= array[:, :, 2] + 80)
        & (array[:, :, 1] <= 120)
    )
    ys, xs = np.where(red)
    if len(xs) < int(minimum_red_pixels):
        raise ValueError("pixel tool could not localize the MiniGrid agent")
    center_x = int(round((int(xs.min()) + int(xs.max())) / 2))
    center_y = int(round((int(ys.min()) + int(ys.max())) / 2))
    left = max(0, center_x - int(crop_radius))
    upper = max(0, center_y - int(crop_radius))
    right = min(image.width, center_x + int(crop_radius) + 1)
    lower = min(image.height, center_y + int(crop_radius) + 1)
    crop = image.crop((left, upper, right, lower)).resize(
        (int(side), int(side)), Image.Resampling.BILINEAR,
    )
    return np.asarray(crop, dtype=np.float64).reshape(-1) / 255.0


def _serialize_model(model: MLPClassifier) -> dict[str, Any]:
    return {
        "classes": [str(value) for value in model.classes_],
        "activation": str(model.activation),
        "out_activation": str(model.out_activation_),
        "coefs": [value.tolist() for value in model.coefs_],
        "intercepts": [value.tolist() for value in model.intercepts_],
        "iterations": int(model.n_iter_),
        "loss": float(model.loss_),
    }


def _probabilities(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    for index, (coef, intercept) in enumerate(
        zip(model["coefs"], model["intercepts"], strict=True)
    ):
        value = value @ np.asarray(coef, dtype=np.float64)
        value = value + np.asarray(intercept, dtype=np.float64)
        if index < len(model["coefs"]) - 1:
            value = np.maximum(value, 0.0)
    value = value - np.max(value)
    scores = np.exp(value)
    return scores / scores.sum()


def _predict(model: Mapping[str, Any], features: np.ndarray) -> tuple[str, float, list[float]]:
    probabilities = _probabilities(model, features)
    index = int(np.argmax(probabilities))
    return (
        str(model["classes"][index]), float(probabilities[index]),
        [float(value) for value in probabilities],
    )


def train_grounder_artifact(
    rows: Sequence[Mapping[str, Any]], *, namespace: str,
    feature_side: int = 20, crop_radius: int = 22,
    orientation_hidden: tuple[int, ...] = (32,),
    direct_hidden: tuple[int, ...] = (64, 32), random_state: int = 310031,
) -> dict[str, Any]:
    """Train perception and a source-free direct policy on development only."""

    if not rows:
        raise ValueError("neural grounder training requires development rows")
    orientation_x = []
    orientation_y = []
    for row in rows:
        for label in PANEL_ORDER:
            orientation_x.append(pixel_agent_features(
                row["panels"][label], side=feature_side,
                crop_radius=crop_radius,
            ))
            orientation_y.append(str(row["directions"][label]))
    orientation = MLPClassifier(
        hidden_layer_sizes=orientation_hidden, solver="lbfgs", alpha=1e-4,
        max_iter=2000, random_state=int(random_state),
    )
    orientation.fit(np.stack(orientation_x), np.asarray(orientation_y))
    orientation_model = _serialize_model(orientation)
    if set(orientation_model["classes"]) != set(DIRECTION_NAMES):
        raise ValueError("development rows omitted an orientation class")

    direct_x = []
    direct_y = []
    for row in rows:
        task_features = []
        for label in PANEL_ORDER:
            features = pixel_agent_features(
                row["panels"][label], side=feature_side,
                crop_radius=crop_radius,
            )
            probabilities = _probabilities(orientation_model, features)
            task_features.extend(float(value) for value in probabilities)
        direct_x.append(task_features)
        direct_y.append(str(row["direct_recovery"]))
    direct = MLPClassifier(
        hidden_layer_sizes=direct_hidden, solver="lbfgs", alpha=1e-4,
        max_iter=3000, random_state=int(random_state) + 1,
    )
    direct.fit(np.asarray(direct_x), np.asarray(direct_y))
    direct_model = _serialize_model(direct)
    if set(direct_model["classes"]) != set(TOKENS):
        raise ValueError("development rows omitted a recovery token class")

    body = {
        "schema_version": ARTIFACT_VERSION,
        "status": "TARGET_DEVELOPMENT_NEURAL_GROUNDER_FROZEN",
        "namespace": str(namespace),
        "feature_contract": {
            "pixels_only_red_agent_localization": True,
            "feature_side": int(feature_side),
            "crop_radius": int(crop_radius),
            "minimum_red_pixels": 80,
            "panel_order": list(PANEL_ORDER),
        },
        "orientation_model": orientation_model,
        "direct_policy_model": direct_model,
        "training": {
            "development_tasks": len(rows),
            "orientation_panels": len(orientation_y),
            "orientation_label_counts": {
                label: orientation_y.count(label) for label in DIRECTION_NAMES
            },
            "direct_label_counts": {
                label: direct_y.count(label) for label in TOKENS
            },
            "target_native_orientation_labels_read": len(orientation_y),
            "target_native_recovery_labels_read": len(direct_y),
            "target_native_success_or_reward_read": 0,
            "complete_target_trajectories_read": 0,
            "source_program_or_identity_read": False,
            "random_state": int(random_state),
        },
    }
    return body | {"artifact_sha256": stable_hash(body)}


def validate_grounder_artifact(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("invalid neural grounder artifact hash")
    if artifact.get("schema_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported neural grounder artifact")
    training = artifact.get("training") or {}
    if training.get("target_native_success_or_reward_read") != 0:
        raise ValueError("target success leaked into neural grounder training")
    if training.get("complete_target_trajectories_read") != 0:
        raise ValueError("complete target trajectory leaked into grounder")
    if training.get("source_program_or_identity_read") is not False:
        raise ValueError("source program leaked into target grounder")


def predict_neural_binding(
    artifact: Mapping[str, Any], panels: Mapping[str, Image.Image], *,
    orientation_minimum_confidence: float,
    direct_minimum_confidence: float,
) -> dict[str, Any]:
    """Infer anonymous effects and a separately learned neural-only action."""

    validate_grounder_artifact(artifact)
    contract = artifact["feature_contract"]
    directions = {}
    confidences = {}
    direct_features = []
    panel_probabilities = {}
    for label in PANEL_ORDER:
        features = pixel_agent_features(
            panels[label], side=int(contract["feature_side"]),
            crop_radius=int(contract["crop_radius"]),
            minimum_red_pixels=int(contract["minimum_red_pixels"]),
        )
        direction, confidence, probabilities = _predict(
            artifact["orientation_model"], features,
        )
        directions[label] = direction
        confidences[label] = confidence
        panel_probabilities[label] = probabilities
        direct_features.extend(probabilities)
    direct, direct_confidence, direct_probabilities = _predict(
        artifact["direct_policy_model"], np.asarray(direct_features),
    )
    if direct_confidence < float(direct_minimum_confidence):
        direct = "ABSTAIN"
    binding = parse_neural_binding(
        {
            "directions": directions,
            "confidences": confidences,
            "direct_recovery": direct,
        },
        minimum_confidence=float(orientation_minimum_confidence),
    )
    binding["panel_probabilities"] = panel_probabilities
    binding["direct_confidence"] = direct_confidence
    binding["direct_probabilities"] = direct_probabilities
    binding["grounder_artifact_sha256"] = artifact["artifact_sha256"]
    return binding


__all__ = [
    "ARTIFACT_VERSION", "PANEL_ORDER", "pixel_agent_features",
    "predict_neural_binding", "train_grounder_artifact",
    "validate_grounder_artifact",
]
