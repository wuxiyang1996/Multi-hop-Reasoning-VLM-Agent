#!/usr/bin/env python3
"""Verify commits, cleanliness, roles and frozen results in the five-worktree workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_NAME = "Multi-hop-Reasoning-VLM-Agent-two-agent-clean"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    package = args.package.resolve()
    harness_root = workspace / REPOSITORY_NAME
    config = _read(harness_root / "configs/reproducible_workspace_v1.json")
    artifact_manifest = _read(package / "ARTIFACTS.json")

    components = []
    for spec in config["components"]:
        root = workspace / spec["directory"]
        expected = (
            artifact_manifest["git"]["head"]
            if spec.get("commit_from_active_bundle") else spec["commit"]
        )
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root, text=True,
        ).strip()
        missing_paths = [
            value for value in spec["required_paths"]
            if not (root / value).exists()
        ]
        if head != expected or dirty or missing_paths:
            raise SystemExit(json.dumps({
                "component": spec["directory"], "head": head,
                "expected": expected, "dirty": dirty,
                "missing_paths": missing_paths,
            }, indent=2))
        components.append({
            "directory": spec["directory"], "commit": head,
            "role": spec["role"],
            "required_for_v3_substitution": spec["required_for_v3_substitution"],
        })

    subprocess.run([
        sys.executable,
        str(harness_root / "scripts/verify_server_bundle_six_benchmark_v2.py"),
        "--workspace", str(workspace), "--package", str(package),
    ], check=True)
    result = {
        "status": "REPRODUCIBLE_FIVE_WORKTREE_WORKSPACE_VERIFIED",
        "components": components,
        "portable_frozen_cohort_result_verified": True,
        "full_official_six_benchmark_outcome_reproduced": False,
        "full_protocol_sizes": config["result_boundary"]["full_protocol_sizes"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
