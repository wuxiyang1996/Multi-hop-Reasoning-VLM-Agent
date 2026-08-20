from __future__ import annotations

from copy import deepcopy

from PIL import Image, ImageDraw
import pytest

from motif_transfer.minigrid_neural_grounder import (
    PANEL_ORDER,
    predict_neural_binding,
    train_grounder_artifact,
    validate_grounder_artifact,
)
from motif_transfer.minigrid_orientation_recovery import DIRECTION_NAMES, TOKENS


def _panel(direction: str) -> Image.Image:
    points = {
        "right": ((54, 32), (12, 10), (12, 54)),
        "down": ((32, 54), (10, 12), (54, 12)),
        "left": ((10, 32), (54, 10), (54, 54)),
        "up": ((32, 10), (10, 54), (54, 54)),
    }[direction]
    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).polygon(points, fill=(255, 0, 0))
    return image


def _rows():
    rows = []
    for index in range(32):
        directions = {
            label: DIRECTION_NAMES[(index + offset) % 4]
            for offset, label in enumerate(PANEL_ORDER)
        }
        rows.append({
            "panels": {label: _panel(value) for label, value in directions.items()},
            "directions": directions,
            "direct_recovery": TOKENS[index % 4],
        })
    return rows


def test_development_neural_grounder_round_trip_and_fail_closed_hash():
    artifact = train_grounder_artifact(
        _rows(), namespace="unit-neural-grounder", feature_side=16,
        crop_radius=30, orientation_hidden=(12,), direct_hidden=(12,),
        random_state=91,
    )
    validate_grounder_artifact(artifact)
    row = _rows()[3]
    binding = predict_neural_binding(
        artifact, row["panels"], orientation_minimum_confidence=0.8,
        direct_minimum_confidence=0.0,
    )
    assert binding["qualified"] is True
    assert binding["directions"] == row["directions"]
    assert binding["grounder_artifact_sha256"] == artifact["artifact_sha256"]

    tampered = deepcopy(artifact)
    tampered["training"]["target_native_success_or_reward_read"] = 1
    with pytest.raises(ValueError, match="hash"):
        validate_grounder_artifact(tampered)
