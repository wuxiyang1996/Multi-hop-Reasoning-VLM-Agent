#!/usr/bin/env python3
"""Run six fresh DiscoveryWorld confirmation cells for the direct matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    file_sha256,
    read_object,
    validate_self_hash,
)
import scripts.run_discoveryworld_commit_recovery_v1 as matched_runner  # noqa: E402
import scripts.run_phase1_direct_discoveryworld_v1 as direct_v1  # noqa: E402


SCHEMA = "phase1-direct-discoveryworld-confirmation-v2"
STATUS = "FROZEN_BEFORE_ANY_V2_TARGET_RESET_PROVIDER_CALL_OR_OUTCOME"


def validate_v2_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != STATUS:
        raise ValueError("wrong DiscoveryWorld V2 manifest status/schema")
    validate_self_hash(manifest, "manifest_sha256")
    if manifest.get("selection_read_target_outcome") is not False:
        raise ValueError("V2 reserve selection was not outcome blind")
    if manifest.get("historical_target_outcome_reuse_allowed") is not False:
        raise ValueError("V2 permits historical outcome reuse")
    cells = list(manifest.get("cells") or ())
    if len(cells) != len(SOURCE_GAMES):
        raise ValueError("V2 must contain exactly six DiscoveryWorld cells")
    if tuple(row.get("source_game") for row in cells) != SOURCE_GAMES:
        raise ValueError("V2 source coverage/order changed")
    target_ids = [str(row.get("target_task_id")) for row in cells]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("V2 target identity was assigned twice")
    for row in cells:
        if row.get("target_domain") != "discoveryworld":
            raise ValueError("V2 contains a non-DiscoveryWorld cell")
        if row.get("selected_target_previously_executed") is not False:
            raise ValueError("V2 cell does not attest freshness")
        source_path = REPO / str(row["source_artifact"])
        if file_sha256(source_path) != str(row["source_artifact_file_sha256"]):
            raise ValueError("V2 source artifact file changed")
    for relative, expected in (manifest.get("runtime_file_sha256") or {}).items():
        if file_sha256(REPO / str(relative)) != str(expected):
            raise ValueError(f"frozen V2 runtime changed: {relative}")


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    return int(result.returncode)


def prepare_forks(
    *, manifest: Mapping[str, Any], keys: Path, output_root: Path,
) -> Path:
    target_dir = output_root / "target_only"
    fork_dir = output_root / "frozen_forks"
    target_summary = target_dir / "summary.json"
    if not target_summary.is_file():
        code = _run(
            [
                sys.executable,
                str(REPO / "scripts/run_discoveryworld_target_only_v2.py"),
                "--config", str(REPO / str(manifest["target_config"])),
                "--keys", str(keys),
                "--output-dir", str(target_dir),
                "--role", "formal_reserve",
            ],
            output_root / "target_only.log",
        )
        if code != 0 or not target_summary.is_file():
            raise RuntimeError(f"DiscoveryWorld V2 target-only failed: exit={code}")
    if not (fork_dir / "fork_freeze_receipt.json").is_file():
        code = _run(
            [
                sys.executable,
                str(REPO / "scripts/freeze_discoveryworld_qualification_forks_v1.py"),
                "--protocol", str(REPO / str(manifest["protocol"])),
                "--baseline-dir", str(target_dir),
                "--output-dir", str(fork_dir),
            ],
            output_root / "fork_freeze.log",
        )
        if code != 0 or not (fork_dir / "fork_freeze_receipt.json").is_file():
            raise RuntimeError(f"DiscoveryWorld V2 fork freeze failed: exit={code}")
    return fork_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase1_direct_prospective_v2/discoveryworld_manifest.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument("--source-game", choices=SOURCE_GAMES, action="append")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "runs/phase1_direct_prospective_v2/discoveryworld",
    )
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    validate_v2_manifest(manifest)
    fork_dir = prepare_forks(
        manifest=manifest, keys=args.keys, output_root=args.output_root,
    )
    if args.prepare_only:
        print(json.dumps({"status": "V2_FORKS_READY", "fork_dir": str(fork_dir)}, indent=2))
        return 0

    # Transport compatibility only. Candidate schema, parser, selector,
    # target-native realizer, source automaton, and all gates remain unchanged.
    suffix = "\nReturn one valid json object."
    if not matched_runner.TARGET_BINDER_SYSTEM_PROMPT.endswith(suffix):
        matched_runner.TARGET_BINDER_SYSTEM_PROMPT += suffix
    if not matched_runner.TARGET_GROUNDER_SYSTEM_PROMPT.endswith(suffix):
        matched_runner.TARGET_GROUNDER_SYSTEM_PROMPT += suffix

    games = tuple(args.source_game or SOURCE_GAMES)
    reports = [
        direct_v1._run_cell(
            manifest=dict(manifest), game=game, keys=args.keys,
            output_root=args.output_root, fork_dir=fork_dir,
        )
        for game in games
    ]
    passed = sum(
        report["cell_execution_receipt"]["status"]
        == "DIRECT_PROSPECTIVE_CELL_PASSED"
        for report in reports
    )
    print(json.dumps({
        "domain": "discoveryworld", "protocol": "v2",
        "passed": passed, "attempted": len(reports),
    }, indent=2))
    return 0 if passed == len(reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
