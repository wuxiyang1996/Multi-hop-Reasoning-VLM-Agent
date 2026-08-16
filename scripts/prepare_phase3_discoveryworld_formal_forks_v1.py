#!/usr/bin/env python3
"""Derive outcome-blind formal forks after the frozen acquisition run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_discoveryworld_formal import (  # noqa: E402
    select_outcome_blind_formal_fork,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value); claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--acquisition-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_manifest = args.output_dir / "fork_manifest.json"
    if output_manifest.exists():
        raise SystemExit("refusing to overwrite frozen fork manifest")
    manifest = _read(args.manifest)
    _self_hash(manifest, "manifest_sha256")
    acquisition_summary = _read(args.acquisition_dir / "summary.json")
    _self_hash(acquisition_summary, "summary_sha256")
    if acquisition_summary.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit("acquisition/manifest mismatch")
    if acquisition_summary.get("complete_tasks") != manifest["task_count"]:
        raise SystemExit("formal acquisition is incomplete")

    rows = []
    for task in manifest["tasks"]:
        reference_path = args.acquisition_dir / f"{task['task_id']}.json"
        reference = _read(reference_path)
        _self_hash(reference, "episode_sha256")
        fork = select_outcome_blind_formal_fork(reference)
        fork_step = int(fork["fork_after_episode_step"])
        config_body = {
            "schema_version": "phase3-discoveryworld-formal-fork-config-v1",
            "status": "FORMAL_RESERVE_FROZEN_STRUCTURAL_FORK",
            "claim_boundary": manifest["claim_boundary"],
            "formal_manifest_sha256": manifest["manifest_sha256"],
            "reference_episode": _relative(reference_path),
            "reference_episode_sha256": reference["episode_sha256"],
            "fork_after_episode_step": fork_step,
            "recovery_horizon": int(manifest["fork_protocol"]["recovery_horizon"]),
            "conditions": [
                condition for condition in manifest["conditions"]
                if condition != "neural_only"
            ],
            "source_contract": dict(manifest["legacy_runner_transport_contract"]),
            "selector": dict(manifest["selector"]),
            "model": dict(manifest["typed_grounding_model"]),
            "qualification_protocol_sha256": manifest[
                "grounding_qualification"
            ]["report_sha256"],
            "fork_receipt": fork,
            "fork_receipt_sha256": fork["fork_receipt_sha256"],
            "target_native_spatial_realizer": dict(
                manifest["target_native_spatial_realizer"]
            ),
            "formal_target_outcome_read_for_fork_selection": False,
        }
        config = config_body | {"config_sha256": stable_hash(config_body)}
        config_path = args.output_dir / "forks" / f"{task['task_id']}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        rows.append({
            "task_id": task["task_id"], "fork_config": _relative(config_path),
            "fork_config_file_sha256": _file_sha256(config_path),
            "config_sha256": config["config_sha256"],
            "fork_after_episode_step": fork_step,
            "fork_receipt_sha256": fork["fork_receipt_sha256"],
        })
    body = {
        "schema_version": "phase3-discoveryworld-formal-fork-manifest-v1",
        "status": "FROZEN_AFTER_ACQUISITION_BEFORE_MATCHED_ACTIONS",
        "formal_manifest_sha256": manifest["manifest_sha256"],
        "acquisition_summary_sha256": acquisition_summary["summary_sha256"],
        "tasks": rows, "task_count": len(rows),
        "all_forks_selected_without_formal_outcome": True,
        "matched_target_action_executed_before_fork_freeze": False,
    }
    payload = body | {"fork_manifest_sha256": stable_hash(body)}
    output_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "tasks": len(rows),
        "fork_manifest_sha256": payload["fork_manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
