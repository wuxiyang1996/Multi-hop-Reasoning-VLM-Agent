from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _read(relative: str) -> dict:
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


def _sha(relative: str) -> str:
    return hashlib.sha256((REPO / relative).read_bytes()).hexdigest()


def test_final_manifest_inherits_exact_unread_parent_heldout() -> None:
    parent = _read("configs/sokoban_alfworld_effect_transfer_split_v2.json")
    manifest = _read("configs/procedural_game_alfworld_v1_final_manifest.json")
    assert manifest["status"] == "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET"
    assert manifest["parent_manifest"]["file_sha256"] == _sha(
        "configs/sokoban_alfworld_effect_transfer_split_v2.json"
    )
    assert manifest["parent_manifest"]["manifest_sha256"] == parent["manifest_sha256"]
    selected = manifest["cells"]["alfworld_valid_unseen"]["splits"]["held_out"]
    assert selected == parent["splits"]["held_out"]
    assert len(selected) == len(set(selected)) == 24


def test_final_config_binds_every_mutable_development_input() -> None:
    config = _read("configs/procedural_game_alfworld_v1_frozen.json")
    summary = _read("docs/results/procedural_game_alfworld_v1_summary.json")
    evidence = config["development_evidence"]
    assert evidence["origin_config_sha256"] == _sha(evidence["origin_config"])
    assert evidence["qualification_report_sha256"] == (
        summary["input_file_sha256"]["development_report"]
    )
    assert evidence["frozen_artifact_sha256"] == (
        summary["input_file_sha256"]["candidate_artifact"]
    )
    assert evidence["runner_sha256"] == _sha(
        "scripts/run_multisource_alfworld_v2_qualification.py"
    )
    for relative, expected in (
        (evidence["qualification_report"], evidence["qualification_report_sha256"]),
        (config["target"]["artifact"], evidence["frozen_artifact_sha256"]),
    ):
        # Full receipts live under ignored runs/; validate them when locally present.
        if (REPO / relative).exists():
            assert _sha(relative) == expected
