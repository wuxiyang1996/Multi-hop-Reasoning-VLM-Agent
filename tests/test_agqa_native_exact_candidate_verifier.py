import hashlib

import numpy as np

from scripts.pilot_agqa_query_grounder_v4_qwen235_adjudicator import (
    _detector_scale,
    _exact_sgdet_frames,
    _sgdet_frame_sha256,
)


def test_detector_scale_matches_short_side_600_with_long_side_cap() -> None:
    # This documents detector preprocessing only.  Serialized SGDET boxes are
    # already divided back to native coordinates and must not use this value.
    assert _detector_scale(480, 360) == 600 / 360
    assert _detector_scale(240, 1920) == 1000 / 1920


def test_exact_frame_loader_documents_identity_box_divisor() -> None:
    assert "scales[int(frame_id)] = 1.0" in __import__("inspect").getsource(
        _exact_sgdet_frames
    )


def test_sgdet_frame_hash_uses_shape_marker_and_bgr_pixels() -> None:
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    expected = hashlib.sha256(
        str(frame.shape).encode("ascii")
        + b"\0BGR_UINT8\0"
        + frame.tobytes(order="C")
    ).hexdigest()
    assert _sgdet_frame_sha256(frame) == expected


def test_sgdet_frame_hash_is_sensitive_to_channel_order() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = [1, 2, 3]
    assert _sgdet_frame_sha256(frame) != _sgdet_frame_sha256(frame[:, :, ::-1])
