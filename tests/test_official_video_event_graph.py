import io
import json
import pickle
import zipfile

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.official_video_event_graph import (
    OfficialEventGraphArtifact,
    load_builtin_only_pickle,
    load_clevrer_official_event_graph,
    normalize_agqa_stsg,
)
from motif_transfer.video_transfer_measurement import VideoTransferClaim


def test_agqa_normalization_breaks_cycles_and_exposes_no_qa_authority():
    frame = {"id": "000001", "secs": 0.1, "type": "frame"}
    obj = {
        "id": "o1/000001", "type": "object", "class": "o1",
        "frame_num": "000001", "bbox": (1.0, 2.0, 3.0, 4.0),
    }
    frame["objects"] = {"names": ["o1"], "vertices": [obj]}
    frame["next"] = frame
    obj["prev"] = frame
    graph = normalize_agqa_stsg("video-1", {"000001": frame, obj["id"]: obj})
    assert graph["frames"][0]["objects"] == ["o1/000001"]
    assert graph["vertices"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert graph["qa_answer_read"] is False
    assert graph["qa_program_read"] is False
    # Acyclic and canonical-JSON serializable.
    json.dumps(graph, sort_keys=True)


def test_builtin_only_pickle_rejects_global_reconstruction():
    safe = {"v": {"000001": {"id": "000001", "type": "frame"}}}
    assert load_builtin_only_pickle(io.BytesIO(pickle.dumps(safe))) == safe
    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        load_builtin_only_pickle(io.BytesIO(pickle.dumps(Exception("bad"))))


def test_clevrer_zip_loader_and_shared_oracle_receipt(tmp_path):
    annotation = {
        "scene_index": 10001,
        "video_filename": "video_10001.mp4",
        "object_property": [{
            "object_id": 0, "color": "red", "material": "metal",
            "shape": "sphere",
        }],
        "motion_trajectory": [{
            "frame_id": 0,
            "objects": [{
                "object_id": 0, "location": [0, 1, 2],
                "velocity": [1, 0, 0], "inside_camera_view": True,
            }],
        }],
        "collision": [],
    }
    archive = tmp_path / "annotations.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "annotation_10000-11000/annotation_10001.json",
            json.dumps(annotation),
        )
    graph = load_clevrer_official_event_graph(archive, 10001)
    artifact = OfficialEventGraphArtifact.create(
        benchmark="clevrer", task_id="10001", split="validation",
        graph=graph, source_artifact_sha256=stable_hash("fixture archive"),
    )
    receipt = artifact.shared_receipt()
    assert artifact.graph_sha256 == stable_hash(graph)
    assert receipt.target_state_sha256 != artifact.graph_sha256
    assert receipt.claim == VideoTransferClaim.CONDITIONAL_SKILL_TRANSFER
    assert receipt.allowed_tools == ()
    assert receipt.gold_answer_read is False


def test_official_graph_receipt_rejects_tampering():
    artifact = OfficialEventGraphArtifact.create(
        benchmark="agqa2", task_id="v", split="test",
        graph={"schema": "x", "events": []},
        source_artifact_sha256=stable_hash("official"),
    )
    object.__setattr__(artifact, "graph", {"schema": "tampered"})
    with pytest.raises(ValueError, match="hash mismatch"):
        artifact.shared_receipt()
