from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from motif_transfer.tir_maze_topology import (
    execute_maze_structural_program,
    execute_maze_topology,
)


REPO = Path(__file__).resolve().parents[1]
DATASET = Path("/fs/gamma-projects/vlm-robot/datasets/TIR-Bench")


def _binding() -> dict:
    return {
        "role": "TARGET_NATIVE_NEURAL_MAZE_BINDING",
        "answer_or_gold_seen": False,
        "move_deltas": {
            "R": [1, 0], "L": [-1, 0], "U": [0, -1], "D": [0, 1],
        },
        "start_color_rgb": [255, 0, 0],
        "goal_color_rgb": [0, 128, 0],
    }


def test_real_consumed_maze_executes_bound_topology() -> None:
    artifact_path = REPO / "runs/sokoban_topology_skill_v1/discovery_artifact.json"
    if not artifact_path.is_file():
        return
    artifact = json.loads(artifact_path.read_text())
    rows = {
        str(row["id"]): row
        for row in json.loads((DATASET / "TIR-Bench.json").read_text())
    }
    for sample_id in ("550", "17", "92"):
        row = rows[sample_id]
        with Image.open(DATASET / row["image_1"]) as image:
            receipt = execute_maze_topology(
                image, row["prompt"], neural_binding=_binding(),
                source_artifact=artifact,
            )
        assert receipt["selected_answer"] == row["answer"]
        assert receipt["source_option"] == "COMMIT"


def test_direction_permutation_is_destructive() -> None:
    artifact_path = REPO / "runs/sokoban_topology_skill_v1/discovery_artifact.json"
    if not artifact_path.is_file():
        return
    artifact = json.loads(artifact_path.read_text())
    rows = {
        str(row["id"]): row
        for row in json.loads((DATASET / "TIR-Bench.json").read_text())
    }
    row = rows["550"]
    with Image.open(DATASET / row["image_1"]) as image:
        receipt = execute_maze_topology(
            image, row["prompt"], neural_binding=_binding(),
            source_artifact=artifact,
            condition="direction_permuted_source_control",
        )
    assert receipt["selected_answer"] != row["answer"]


def test_induced_shared_ir_matches_target_ceiling_and_is_alpha_invariant() -> None:
    artifact_path = REPO / "runs/sokoban_relational_structural_v2/artifact.json"
    if not artifact_path.is_file():
        return
    artifact = json.loads(artifact_path.read_text())
    rows = {
        str(row["id"]): row
        for row in json.loads((DATASET / "TIR-Bench.json").read_text())
    }
    row = rows["550"]
    with Image.open(DATASET / row["image_1"]) as image:
        authentic = execute_maze_structural_program(
            image,
            row["prompt"],
            neural_binding=_binding(),
            source_artifact=artifact,
            condition="source_induced",
        )
        renamed = execute_maze_structural_program(
            image,
            row["prompt"],
            neural_binding=_binding(),
            source_artifact=artifact,
            condition="alpha_renamed_source",
        )
        ceiling = execute_maze_structural_program(
            image,
            row["prompt"],
            neural_binding=_binding(),
            source_artifact=artifact,
            condition="target_native_ceiling",
        )
        permuted = execute_maze_structural_program(
            image,
            row["prompt"],
            neural_binding=_binding(),
            source_artifact=artifact,
            condition="source_relation_permuted",
        )
    assert authentic["selected_answer"] == row["answer"]
    assert renamed["selected_answer"] == authentic["selected_answer"]
    assert ceiling["selected_answer"] == authentic["selected_answer"]
    assert permuted["selected_answer"] != authentic["selected_answer"]
    assert authentic["source_artifact_sha256"] == artifact["artifact_sha256"]
    assert ceiling["source_artifact_sha256"] is None
