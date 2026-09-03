"""CLEVRER official factual-event adapter for grounding-isolated execution.

The official validation annotation contains object properties, observed motion
and collisions.  This adapter converts only those fields to the native NS-DR
``Simulation`` input schema.  It is valid for factual/explanatory execution;
it deliberately refuses to synthesize predictive or counterfactual rollouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .contracts import stable_hash


def official_annotation_to_factual_prediction(
    annotation: Mapping[str, Any], *, question_family: str,
) -> dict[str, Any]:
    if str(question_family).casefold() != "explanatory":
        raise ValueError(
            "official factual annotation cannot supply predictive/counterfactual rollouts"
        )
    properties = {
        int(row["object_id"]): row for row in annotation["object_property"]
    }
    objects = [{
        "id": object_id,
        "color": str(row["color"]),
        "material": str(row["material"]),
        "shape": str(row["shape"]),
    } for object_id, row in sorted(properties.items())]
    trajectory = []
    for frame in annotation["motion_trajectory"]:
        visible = []
        for raw in frame["objects"]:
            if raw.get("inside_camera_view") is not True:
                continue
            prop = properties[int(raw["object_id"])]
            location = raw["location"]
            visible.append({
                # Scale world coordinates only to preserve the legacy
                # Simulation motion threshold. Event identity is attribute based.
                "x": float(location[0]) * 10.0,
                "y": float(location[1]) * 10.0,
                "color": str(prop["color"]),
                "material": str(prop["material"]),
                "shape": str(prop["shape"]),
            })
        trajectory.append({
            "frame_index": int(frame["frame_id"]), "objects": visible,
        })
    collisions = []
    for collision in annotation["collision"]:
        collisions.append({
            "frame": int(collision["frame_id"]),
            "objects": [{
                "color": str(properties[int(object_id)]["color"]),
                "material": str(properties[int(object_id)]["material"]),
                "shape": str(properties[int(object_id)]["shape"]),
            } for object_id in collision["object_ids"]],
        })
    return {
        "objects": objects,
        "predictions": [{
            "what_if": -1, "trajectory": trajectory, "collisions": collisions,
        }],
        "authority": {
            "official_factual_annotation_read": True,
            "answer_read": False,
            "functional_program_read": False,
            "counterfactual_rollout_synthesized": False,
        },
    }


@dataclass(frozen=True)
class ClevrerOracleExecutionReceipt:
    task_id: str
    scene_index: int
    family: str
    official_graph_sha256: str
    compiled_question_sha256: str
    compiled_choices_sha256: str
    prediction: str
    answer_read: bool
    functional_program_read: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls, *, task_id: str, scene_index: int, family: str,
        official_graph_sha256: str, question_program: list[str],
        choice_programs: list[list[str]], prediction: str,
    ) -> "ClevrerOracleExecutionReceipt":
        core = {
            "task_id": task_id, "scene_index": scene_index, "family": family,
            "official_graph_sha256": official_graph_sha256,
            "compiled_question_sha256": stable_hash(question_program),
            "compiled_choices_sha256": stable_hash(choice_programs),
            "prediction": prediction, "answer_read": False,
            "functional_program_read": False,
        }
        return cls(**core, receipt_sha256=stable_hash(core))

    def validate(self) -> None:
        core = asdict(self)
        claimed = core.pop("receipt_sha256")
        if stable_hash(core) != claimed:
            raise ValueError("CLEVRER oracle execution receipt hash mismatch")
        if self.answer_read or self.functional_program_read:
            raise ValueError("CLEVRER oracle executor crossed evaluator authority")


__all__ = [
    "ClevrerOracleExecutionReceipt", "official_annotation_to_factual_prediction",
]
