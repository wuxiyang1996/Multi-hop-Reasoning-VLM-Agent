import json
from pathlib import Path

from scripts.quarantine_invalid_agqa_v65_cache import quarantine


def test_only_semantically_invalid_provider_cache_is_quarantined(tmp_path: Path):
    cache = tmp_path / "call_cache"
    task = cache / "task"
    task.mkdir(parents=True)
    invalid = task / "operand_A_primary_0.hash.json"
    valid = task / "operand_A_primary_1.hash.json"
    invalid.write_text(json.dumps({
        "input_sha256": "same-input",
        "payload": {"observations": [{
            "observability": "OBSERVED", "start_frame": None,
            "end_frame": None, "evidence_frames": [],
        }]},
    }))
    valid.write_text(json.dumps({
        "input_sha256": "other-input",
        "payload": {"observations": [{
            "observability": "UNOBSERVED", "start_frame": None,
            "end_frame": None, "evidence_frames": [],
        }]},
    }))
    errors = tmp_path / "worker_errors.json"
    errors.write_text(json.dumps({
        "errors": {"task": "operand schema retries exhausted: bad"},
    }))
    receipt = quarantine(cache, errors, tmp_path / "receipt.json")
    assert receipt["quarantined_count"] == 1
    assert not invalid.exists()
    assert valid.exists()
    moved = tmp_path / receipt["rows"][0]["quarantine_relative_path"]
    assert moved.exists()
    assert receipt["rows"][0]["payload_changed"] is False
    assert receipt["target_outcome_read"] is False
