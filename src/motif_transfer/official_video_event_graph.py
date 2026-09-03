"""Answer-blind adapters for official CLEVRER and AGQA scene annotations.

These adapters intentionally stop at a benchmark-native event graph.  They do
not read a QA row and therefore cannot consume its answer, functional program,
or program-derived grounding.  The graph may be shared by every matched
controller arm through :class:`SharedVideoGroundingReceipt`.

AGQA's official STSG release is a pickle.  Loading arbitrary pickle data is
unsafe, so the loader rejects every GLOBAL/class reconstruction and accepts
only the builtin containers and scalars present in the official artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import pickle
from typing import Any, BinaryIO, Mapping
import zipfile

from .contracts import stable_hash
from .video_transfer_measurement import (
    GroundingMode,
    GroundingToolBudget,
    SharedVideoGroundingReceipt,
)


ZERO_TOOL_BUDGET = GroundingToolBudget(0, 0, 0)


class _BuiltinOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"AGQA STSG attempted forbidden global {module}.{name}"
        )


def load_builtin_only_pickle(source: str | Path | BinaryIO) -> Any:
    """Load an official plain-container pickle without permitting code import."""

    if hasattr(source, "read"):
        return _BuiltinOnlyUnpickler(source).load()  # type: ignore[arg-type]
    with Path(source).open("rb") as handle:
        return _BuiltinOnlyUnpickler(handle).load()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _ref_ids(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        vertices = value.get("vertices", ())
        return _ref_ids(vertices)
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        if isinstance(item, Mapping) and item.get("id") is not None:
            result.append(str(item["id"]))
        elif isinstance(item, (str, int)):
            result.append(str(item))
    return sorted(set(result))


def normalize_agqa_stsg(video_id: str, graph: Mapping[str, Any]) -> dict[str, Any]:
    """Break cyclic STSG references into a canonical ID-only event graph."""

    frames: list[dict[str, Any]] = []
    vertices: list[dict[str, Any]] = []
    for key in sorted(graph):
        raw = graph[key]
        if not isinstance(raw, Mapping):
            continue
        vertex_id = str(raw.get("id", key))
        kind = str(raw.get("type", "unknown"))
        if kind == "frame":
            frames.append({
                "id": vertex_id,
                "seconds": raw.get("secs"),
                "objects": _ref_ids(raw.get("objects")),
                "attention": _ref_ids(raw.get("attention")),
                "contact": _ref_ids(raw.get("contact")),
                "spatial": _ref_ids(raw.get("spatial")),
                "verbs": _ref_ids(raw.get("verb")),
                "actions": _ref_ids(raw.get("actions")),
            })
            continue
        row: dict[str, Any] = {"id": vertex_id, "type": kind}
        for field in ("class", "phrase", "charades", "start", "end", "secs",
                      "frame_num", "visible"):
            value = raw.get(field)
            if isinstance(value, (str, int, float, bool)):
                row[field] = value
        bbox = raw.get("bbox")
        if isinstance(bbox, (list, tuple)) and all(
            isinstance(value, (int, float)) for value in bbox
        ):
            row["bbox"] = list(bbox)
        for relation in ("objects", "attention", "contact", "spatial", "verb",
                         "while"):
            refs = _ref_ids(raw.get(relation))
            if refs:
                row[relation] = refs
        vertices.append(row)
    return {
        "schema": "AGQA_OFFICIAL_STSG_ID_GRAPH_V1",
        "video_id": str(video_id),
        "frames": frames,
        "vertices": vertices,
        "qa_answer_read": False,
        "qa_program_read": False,
        "program_derived_grounding_read": False,
    }


def load_agqa_official_event_graph(
    stsg_pickle: str | Path, video_id: str,
) -> dict[str, Any]:
    corpus = load_builtin_only_pickle(stsg_pickle)
    if not isinstance(corpus, Mapping) or video_id not in corpus:
        raise KeyError(f"AGQA video is absent from official STSG: {video_id}")
    graph = corpus[video_id]
    if not isinstance(graph, Mapping):
        raise ValueError("AGQA STSG video entry is not a mapping")
    return normalize_agqa_stsg(video_id, graph)


def normalize_clevrer_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an official simulator annotation into a target-native graph."""

    scene_index = int(annotation["scene_index"])
    objects = []
    for raw in annotation.get("object_property", ()):
        objects.append({
            field: raw[field]
            for field in ("object_id", "color", "material", "shape")
            if field in raw
        })
    trajectories = []
    for raw_frame in annotation.get("motion_trajectory", ()):
        frame_objects = []
        for raw in raw_frame.get("objects", ()):
            frame_objects.append({
                field: raw[field]
                for field in (
                    "object_id", "location", "orientation", "velocity",
                    "angular_velocity", "inside_camera_view",
                )
                if field in raw
            })
        trajectories.append({
            "frame_id": int(raw_frame["frame_id"]),
            "objects": frame_objects,
        })
    collisions = [{
        field: raw[field]
        for field in ("object_ids", "frame_id", "location")
        if field in raw
    } for raw in annotation.get("collision", ())]
    return {
        "schema": "CLEVRER_OFFICIAL_SIMULATION_EVENT_GRAPH_V1",
        "scene_index": scene_index,
        "objects": objects,
        "motion_trajectory": trajectories,
        "collisions": collisions,
        "qa_answer_read": False,
        "qa_program_read": False,
    }


def load_clevrer_official_event_graph(
    annotation_zip: str | Path, scene_index: int,
) -> dict[str, Any]:
    name = (
        f"annotation_{scene_index // 1000 * 1000:05d}-"
        f"{(scene_index // 1000 + 1) * 1000:05d}/"
        f"annotation_{scene_index:05d}.json"
    )
    with zipfile.ZipFile(annotation_zip) as archive:
        try:
            with archive.open(name) as handle:
                annotation = json.load(handle)
        except KeyError as error:
            raise KeyError(
                f"CLEVRER scene is absent from official annotation archive: {scene_index}"
            ) from error
    return normalize_clevrer_annotation(annotation)


@dataclass(frozen=True)
class OfficialEventGraphArtifact:
    benchmark: str
    task_id: str
    split: str
    graph: Mapping[str, Any]
    source_artifact_sha256: str
    graph_sha256: str
    answer_read: bool
    program_read: bool

    @classmethod
    def create(
        cls, *, benchmark: str, task_id: str, split: str,
        graph: Mapping[str, Any], source_artifact_sha256: str,
    ) -> "OfficialEventGraphArtifact":
        artifact = cls(
            benchmark=benchmark.casefold(), task_id=str(task_id), split=str(split),
            graph=graph, source_artifact_sha256=source_artifact_sha256,
            graph_sha256=stable_hash(graph), answer_read=False, program_read=False,
        )
        artifact.validate()
        return artifact

    def validate(self) -> None:
        if self.benchmark not in {"clevrer", "agqa2"}:
            raise ValueError("unsupported official video event graph")
        if self.answer_read or self.program_read:
            raise ValueError("official event graph crossed the QA authority boundary")
        if self.graph_sha256 != stable_hash(self.graph):
            raise ValueError("official event graph hash mismatch")
        if len(self.source_artifact_sha256) != 64:
            raise ValueError("official source artifact is not content-addressed")

    def shared_receipt(self) -> SharedVideoGroundingReceipt:
        self.validate()
        # The graph itself remains in the frozen target-native store.  The
        # receipt carries its content address and small structural metadata so
        # repeated questions over one video do not repeatedly serialize a
        # potentially large cyclic-source annotation.
        state = {
            "official_event_graph_sha256": self.graph_sha256,
            "schema": self.graph.get("schema"),
            "video_or_scene_id": self.graph.get(
                "video_id", self.graph.get("scene_index")
            ),
            "frame_count": len(self.graph.get(
                "frames", self.graph.get("motion_trajectory", ())
            )),
            "vertex_or_object_count": len(self.graph.get(
                "vertices", self.graph.get("objects", ())
            )),
            "qa_answer_read": False,
            "qa_program_read": False,
        }
        return SharedVideoGroundingReceipt.create(
            benchmark=self.benchmark, task_id=self.task_id, split=self.split,
            mode=GroundingMode.ORACLE_EVENT_GRAPH, state=state,
            evidence_source_sha256=self.source_artifact_sha256,
            tool_budget=ZERO_TOOL_BUDGET, official_scene_graph_read=True,
        )


__all__ = [
    "OfficialEventGraphArtifact", "ZERO_TOOL_BUDGET",
    "load_agqa_official_event_graph", "load_builtin_only_pickle",
    "load_clevrer_official_event_graph", "normalize_agqa_stsg",
    "normalize_clevrer_annotation", "sha256_file",
]
