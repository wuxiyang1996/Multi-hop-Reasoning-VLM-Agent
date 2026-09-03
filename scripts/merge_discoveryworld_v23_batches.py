#!/usr/bin/env python3
"""Merge independently scheduled V23 target episodes with integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.discoveryworld_env import stable_hash  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_id(row: dict) -> str:
    name = str(row["scenario"]).lower().replace(" ", "_").replace("'", "")
    name = "".join(value for value in name if value.isalnum() or value == "_")
    return f"{name}.{str(row['difficulty']).lower()}.seed{int(row['seed'])}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    manifest = json.loads((REPO / config["manifest"]).read_text())
    expected = {
        task_id(row): row for row in manifest["roles"][config["manifest_role"]]
    }
    config_hash = sha256(args.config)
    selected = {}
    for identifier, task in expected.items():
        complete = []
        partial = []
        for directory in args.input_dir:
            path = directory / f"{identifier}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            if payload.get("status") == "TARGET_ONLY_EPISODE_COMPLETE":
                complete.append((path, payload))
            else:
                partial.append(str(path))
        if len(complete) != 1:
            raise SystemExit(
                f"expected exactly one complete episode for {identifier}; "
                f"found {len(complete)}, partial={partial}"
            )
        path, payload = complete[0]
        body = dict(payload)
        claimed = body.pop("episode_sha256", None)
        if claimed != stable_hash(body):
            raise SystemExit(f"episode self-hash mismatch: {path}")
        if payload.get("task") != task:
            raise SystemExit(f"task identity mismatch: {path}")
        if payload.get("runtime_hashes", {}).get("config") != config_hash:
            raise SystemExit(f"target config hash mismatch: {path}")
        if payload.get("policy_runtime_saw_oracle_scorecard") is not False:
            raise SystemExit(f"policy oracle isolation failed: {path}")
        selected[identifier] = (path, payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied_hashes = {}
    for identifier, (path, _) in selected.items():
        destination = args.output_dir / f"{identifier}.json"
        shutil.copyfile(path, destination)
        copied_hashes[identifier] = sha256(destination)
    episodes = [payload for _, payload in selected.values()]
    summary = {
        "schema_version": "discoveryworld-target-only-v23-merged-summary",
        "status": "TARGET_ONLY_FORMAL_RESERVE_COMPLETE",
        "role": "formal_reserve",
        "claim_boundary": config["claim_boundary"],
        "tasks": len(episodes),
        "successes": sum(
            bool(row["evaluation"]["official_success"]) for row in episodes
        ),
        "zero_policy_oracle_scorecard_use": all(
            row["policy_runtime_saw_oracle_scorecard"] is False for row in episodes
        ),
        "config_file_sha256": config_hash,
        "episode_file_sha256": copied_hashes,
        "episode_sha256": {
            identifier: payload["episode_sha256"]
            for identifier, (_, payload) in selected.items()
        },
        "operational_resume": {
            "task_id": "space_sick.easy.seed5",
            "interrupted_after_step": 30,
            "resume_contract": "SAME_COMPLETION_CACHE_AND_DETERMINISTIC_PREFIX_REPLAY",
            "independent_restart_used": False,
        }
    }
    summary["summary_sha256"] = stable_hash(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
