from __future__ import annotations

import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from motif_transfer.tetris_rotation_transfer import (
    parse_rotation_options,
    parse_tetris_orientation,
    select_rotation_action,
)


REPO = Path(__file__).resolve().parents[1]


def _state(board: list[str], rotations: list[list[str]]) -> str:
    lines = ["Board:", *board]
    for index, candidate in enumerate(rotations):
        lines.extend([f"Rotation {index} (candidate):", *candidate])
    return "\n".join(lines)


def test_orientation_is_anonymous_group_index() -> None:
    base = [".........."] * 20
    first = list(base)
    second = list(base)
    first[0] = "...IIII..."
    second[0] = ".....I...."
    assert parse_tetris_orientation(_state(second, [first, second])) == (1, 2)


def test_source_inverse_selects_target_native_clockwise_degree() -> None:
    prompt = "A. 90°\nB. 110°\nC. 270°\nD. 180°"
    options = parse_rotation_options(prompt)
    assert select_rotation_action(
        options, 92.0, condition="authentic_tetris_inverse"
    ) == "A"
    assert select_rotation_action(
        options, 92.0, condition="no_inverse_control"
    ) == "C"
    assert select_rotation_action(
        options, 92.0, condition="alpha_renamed_authentic"
    ) == "A"


def test_shuffled_binding_uses_donor_not_current_angle() -> None:
    options = parse_rotation_options("A) 45°\nB) 90°\nC) 180°")
    assert select_rotation_action(
        options, 45.0, condition="shuffled_binding_control",
        donor_ccw_degrees=178.0,
    ) == "C"


def test_frozen_target_splits_and_source_receipt_are_bound_before_calls() -> None:
    config = json.loads((
        REPO / "configs/tir_tetris_rotation_v1_frozen.json"
    ).read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_TARGET_ROTATION_CALLS"
    assert config["selection"]["prompt_image_answer_or_outcome_read"] is False
    assert {key: len(value) for key, value in config["splits"].items()} == {
        "consumed_development": 12,
        "qualification": 12,
        "heldout": 24,
        "reserve": 5,
    }
    all_ids = [item for values in config["splits"].values() for item in values]
    assert len(all_ids) == len(set(all_ids)) == 53
    source_path = REPO / config["source"]["artifact"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == config["source"][
        "artifact_file_sha256"
    ]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    body = dict(source)
    claimed = body.pop("artifact_sha256")
    assert stable_hash(body) == claimed == config["source"]["artifact_content_sha256"]
    assert source["raw_source_action_tokens_exported"] is False
    assert all(source["gates"].values())
    for relative, expected in config["integrity"]["file_sha256"].items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == expected
