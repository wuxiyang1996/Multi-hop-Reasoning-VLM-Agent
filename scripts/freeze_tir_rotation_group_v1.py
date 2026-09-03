#!/usr/bin/env python3
"""Freeze disjoint TIR rotation splits before target calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DATASET_SHA256 = "e764c780c53fdce2e0bb64846a8119b48b731c90c7168c6c202f270062684bc8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strings(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, (str, int)):
        yield str(value)


def freeze(*, dataset: Path, source_artifact: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen config: {output}")
    if _sha256(dataset) != DATASET_SHA256:
        raise ValueError("TIR dataset hash mismatch")
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    rotation_ids = {str(row["id"]) for row in rows if row["task"] == "rotation_game"}
    if len(rotation_ids) != 75:
        raise ValueError("unexpected TIR rotation_game count")
    used: set[str] = set()
    for path in (REPO / "configs").glob("*tir*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        used.update(value for value in _strings(payload) if value in rotation_ids)
    multi = json.loads((
        REPO / "configs/real_game_multitarget_v5_manifest.json"
    ).read_text(encoding="utf-8"))
    roles = multi["targets"]["tir_bench"]["partition"]["roles"]
    for role in ("adaptation", "qualification", "held_out"):
        used.update(set(map(str, roles[role])) & rotation_ids)
    available = sorted(
        rotation_ids - used,
        key=lambda sample_id: hashlib.sha256(
            f"tetris-to-tir-rotation-v1\0{sample_id}".encode()
        ).hexdigest(),
    )
    if len(available) != 53:
        raise ValueError(f"unexpected unused rotation pool: {len(available)}")
    split = {
        "consumed_development": available[:12],
        "qualification": available[12:24],
        "heldout": available[24:48],
        "reserve": available[48:],
    }
    source = json.loads(source_artifact.read_text(encoding="utf-8"))
    if source["status"] != "SOURCE_ROTATION_GROUP_CONFIRMED":
        raise ValueError("source rotation group was not confirmed")
    config = {
        "schema_version": "tetris-to-tir-rotation-v1",
        "status": "FROZEN_BEFORE_TARGET_ROTATION_CALLS",
        "claim_boundary": (
            "Tetris supplies only an anonymous cyclic-group inverse program. "
            "A target-native neural model grounds the current image's "
            "counterclockwise displacement without seeing choices; the symbolic "
            "executor chooses a native clockwise degree option. This route is "
            "specific to TIR rotation_game and includes an extensionally "
            "identical target-written control, so it does not establish source "
            "provenance necessity."
        ),
        "selection": {
            "rule": (
                "Exclude every rotation_game ID already present in a TIR config "
                "or an opened V5 target role, then sort by "
                "sha256('tetris-to-tir-rotation-v1\\0'+id)."
            ),
            "prompt_image_answer_or_outcome_read": False,
            "previously_assigned_ids": len(used),
            "available_before_selection": len(available),
        },
        "dataset": {"file_sha256": DATASET_SHA256, "family": "rotation_game"},
        "source": {
            "artifact": str(source_artifact.resolve().relative_to(REPO.resolve())),
            "artifact_file_sha256": _sha256(source_artifact),
            "artifact_content_sha256": source["artifact_sha256"],
        },
        "splits": split,
        "model": {
            "provider": "openrouter",
            "id": "openai/gpt-4.1-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "timeout_seconds": 180,
            "max_retries": 2,
            "maximum_output_tokens": 500,
            "temperature": 0,
        },
        "media": {"max_side": 1024, "jpeg_quality": 90},
        "authority": {
            "development_report": "runs/tir_tetris_rotation_v1/consumed_development_report.json",
            "qualification_report": "runs/tir_tetris_rotation_v1/qualification_report.json",
        },
        "formal_gates": {
            "authentic_strictly_above_raw": True,
            "paired_wins_above_losses": True,
            "authentic_strictly_above_no_inverse_shuffled_and_marginal": True,
            "alpha_rename_invariance": True,
            "target_written_isomorphic_equivalence": True,
            "thresholds_may_change_after_open": False,
        },
        "integrity": {"file_sha256": {
            "scripts/run_tir_rotation_group_v1.py": _sha256(
                REPO / "scripts/run_tir_rotation_group_v1.py"
            ),
            "src/motif_transfer/tetris_rotation_transfer.py": _sha256(
                REPO / "src/motif_transfer/tetris_rotation_transfer.py"
            ),
        }},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--source-artifact", type=Path,
        default=REPO / "runs/tetris_rotation_group_v1/source_artifact.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "configs/tir_tetris_rotation_v1_frozen.json",
    )
    args = parser.parse_args()
    print(json.dumps(freeze(
        dataset=args.dataset, source_artifact=args.source_artifact, output=args.output,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
