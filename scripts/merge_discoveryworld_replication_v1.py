#!/usr/bin/env python3
"""Merge independently scheduled DiscoveryWorld replication baselines."""

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_id(row: dict) -> str:
    name = str(row["scenario"]).lower().replace(" ", "_").replace("'", "")
    name = "".join(value for value in name if value.isalnum() or value == "_")
    return f"{name}.{str(row['difficulty']).lower()}.seed{int(row['seed'])}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = REPO / str(config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        _task_id(row): row
        for row in manifest["roles"][config["manifest_role"]]
    }
    config_hash = _sha256(args.config)
    selected = {}
    for identifier, task in expected.items():
        matches = []
        for directory in args.input_dir:
            path = directory / f"{identifier}.json"
            if not path.is_file():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") == "TARGET_ONLY_EPISODE_COMPLETE":
                matches.append((path, value))
        if len(matches) != 1:
            raise SystemExit(f"expected one complete {identifier}, found {len(matches)}")
        path, value = matches[0]
        body = dict(value)
        claimed = body.pop("episode_sha256", None)
        if claimed != stable_hash(body):
            raise SystemExit(f"episode self-hash mismatch: {path}")
        if value.get("task") != task:
            raise SystemExit(f"task mismatch: {path}")
        if value.get("runtime_hashes", {}).get("config") != config_hash:
            raise SystemExit(f"target config mismatch: {path}")
        if value.get("policy_runtime_saw_oracle_scorecard") is not False:
            raise SystemExit(f"oracle isolation failed: {path}")
        selected[identifier] = (path, value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied_hashes = {}
    for identifier, (path, _) in selected.items():
        destination = args.output_dir / f"{identifier}.json"
        shutil.copyfile(path, destination)
        copied_hashes[identifier] = _sha256(destination)
    values = [value for _, value in selected.values()]
    body = {
        "schema_version": "discoveryworld-target-only-replication-v1-merged",
        "status": "TARGET_ONLY_FORMAL_RESERVE_COMPLETE",
        "role": config["manifest_role"],
        "claim_boundary": config["claim_boundary"],
        "tasks": len(values),
        "successes": sum(bool(value["evaluation"]["official_success"]) for value in values),
        "zero_policy_oracle_scorecard_use": True,
        "config_file_sha256": config_hash,
        "manifest_file_sha256": _sha256(manifest_path),
        "episode_file_sha256": copied_hashes,
        "episode_sha256": {
            identifier: value["episode_sha256"]
            for identifier, (_, value) in selected.items()
        },
        "operational_disclosure": {
            "dependency_recovery": "The first scheduler attempts for later indices failed before reset because the pinned official environment dependencies were absent. Those attempts made zero target decisions and were rerun after installing the pinned dependencies.",
            "scientific_configuration_changed": False
        }
    }
    body["summary_sha256"] = stable_hash(body)
    (args.output_dir / "summary.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps(body, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
