from pathlib import Path

from scripts.build_two_video_transfer_bundle_v2 import artifact_key


def test_portable_extraction_does_not_change_artifact_keys():
    assert artifact_key(
        Path("/tmp/audit/runs/agqa/formal.json")
    ) == "runs/agqa/formal.json"
    assert artifact_key(
        Path("/tmp/audit/configs/frozen.json")
    ) == "configs/frozen.json"
