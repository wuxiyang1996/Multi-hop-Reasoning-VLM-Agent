from __future__ import annotations

import json
from pathlib import Path

import pytest

from motif_transfer.clevrer_nsdr_tool_grounder import (
    bind_cached_nsdr_prediction,
    load_prediction_payload,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/fs/gamma-projects/vlm-robot/datasets/CLEVRER-official")


def test_official_cached_prediction_binds_to_raw_video_without_oracle_fields() -> None:
    video = DATA / "raw_video_validation/videos/video_10000-11000/video_10000.mp4"
    prediction = DATA / "off_the_shelf_nsdr/propnet_preds/with_edge_supervision_old/sim_10000.json"
    if not video.exists() or not prediction.exists():
        pytest.skip("official CLEVRER raw-video/NS-DR artifacts are not installed")
    config = json.loads((ROOT / "configs/clevrer_nsdr_tool_grounder_v1.json").read_text())
    receipt = bind_cached_nsdr_prediction(
        video_path=video, prediction_path=prediction, config=config,
    )
    assert receipt.video_id == 10000
    assert receipt.object_count == 4
    assert receipt.prediction_world_count == 5
    assert receipt.observed_world_present
    assert receipt.counterfactual_worlds_complete
    assert receipt.provider_calls == 0
    assert not receipt.question_read
    assert not receipt.processed_proposals_read
    assert not receipt.official_annotation_read
    assert not receipt.answer_read
    assert not receipt.source_controller_read


def test_oracle_like_prediction_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"objects": [], "predictions": [], "ground_truth": {}}))
    with pytest.raises(ValueError, match="unexpected fields"):
        load_prediction_payload(path)
