#!/usr/bin/env python3
"""Freeze outcome-blind matched-fork configs from qualification baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.discoveryworld_env import stable_hash  # noqa: E402
from motif_transfer.discoveryworld_qualification import (  # noqa: E402
    select_first_commit_fork,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    baseline_config_path = REPO / str(protocol["target_baseline_config"])
    expected_baseline_hash = file_sha256(baseline_config_path)
    baseline_relative = args.baseline_dir.resolve().relative_to(REPO.resolve())
    receipts = []
    generated = []
    for task_id in protocol["qualification_task_ids"]:
        episode_path = args.baseline_dir / f"{task_id}.json"
        episode = json.loads(episode_path.read_text())
        if episode.get("status") != "TARGET_ONLY_EPISODE_COMPLETE":
            raise SystemExit(f"qualification baseline is incomplete: {task_id}")
        if episode.get("runtime_hashes", {}).get("config") != expected_baseline_hash:
            raise SystemExit(f"qualification baseline config hash mismatch: {task_id}")
        receipt = select_first_commit_fork(
            episode, protocol["fork_rule"]["allowed_commit_actions"],
        )
        receipts.append(receipt)
        if not receipt["eligible"]:
            continue
        config = {
            "schema_version": "discoveryworld-sokoban-qualification-fork-v1",
            "status": "QUALIFICATION_FROZEN_NO_ADAPTATION",
            "claim_boundary": protocol["claim_boundary"],
            "reference_episode": str(baseline_relative / f"{task_id}.json"),
            "reference_episode_sha256": episode["episode_sha256"],
            "fork_after_episode_step": receipt["fork_after_episode_step"],
            "recovery_horizon": protocol["recovery_horizon"],
            "conditions": list(protocol["conditions"]),
            "source_contract": dict(protocol["source_contract"]),
            "selector": dict(protocol["selector"]),
            "model": dict(protocol["model"]),
            "qualification_protocol_sha256": file_sha256(args.protocol),
            "fork_receipt_sha256": receipt["fork_receipt_sha256"],
        }
        output_path = args.output_dir / f"{task_id}.json"
        write_json(output_path, config)
        generated.append(str(output_path.resolve().relative_to(REPO.resolve())))
    summary = {
        "schema_version": "discoveryworld-qualification-fork-freeze-v1",
        "status": "QUALIFICATION_FORKS_FROZEN",
        "protocol_file_sha256": file_sha256(args.protocol),
        "generator_file_sha256": file_sha256(Path(__file__)),
        "target_baseline_config_sha256": expected_baseline_hash,
        "outcome_fields_read_for_eligibility": False,
        "receipts": receipts,
        "generated_configs": generated,
    }
    summary["summary_sha256"] = stable_hash(summary)
    write_json(args.output_dir / "fork_freeze_receipt.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
