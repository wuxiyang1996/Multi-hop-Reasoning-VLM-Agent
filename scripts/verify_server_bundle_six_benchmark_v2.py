#!/usr/bin/env python3
"""Verify the six-benchmark package from a clean extracted workspace."""

from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    package = args.package.resolve()
    repo = workspace / REPOSITORY_NAME
    config = _read(repo / "configs/server_bundle_six_benchmark_v2.json")
    manifest = _read(package / "ARTIFACTS.json")

    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip() != manifest["git"]["head"]:
        raise SystemExit("clean checkout does not match bundle commit")
    for row in manifest["dependency_files"]:
        path = repo / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise SystemExit(f"missing dependency: {row['path']}")
        if _sha256(path) != row["sha256"]:
            raise SystemExit(f"dependency hash mismatch: {row['path']}")

    _run([sys.executable, "-m", "pytest", "-q", *config["pytest_targets"]], cwd=repo)
    output = workspace / "verification-output"
    output.mkdir(exist_ok=False)
    canonical = repo / "runs/harness_controller_qwen35_9b_mixed_v3/source_only_sft_seed20260901"
    protocol = repo / "runs/harness_controller_qwen35_9b_mixed_v3_protocol/protocol.json"
    activation = output / "portable_activation.json"
    _run([
        sys.executable, "scripts/activate_harness_9b_six_benchmark_substitution_v1.py",
        "--protocol", str(protocol),
        "--source-qualification", str(canonical / "source_mixed_qualification.json"),
        "--adapter", str(canonical / "adapter"),
        "--training-receipt", str(canonical / "training_receipt.json"),
        "--output", str(activation),
    ], cwd=repo)
    portable_action = output / "portable_action_equivalence.json"
    _run([
        sys.executable, "scripts/audit_harness_9b_six_benchmark_action_equivalence_v1.py",
        "--activation", str(activation),
        "--route-report", str(canonical / "six_benchmark_route_report.json"),
        "--route-predictions", str(canonical / "six_benchmark_route_report.predictions.jsonl"),
        "--output", str(portable_action),
    ], cwd=repo)
    _run([
        sys.executable, "scripts/build_harness_9b_six_benchmark_paper_report_v1.py",
        "--protocol", str(protocol),
        "--training-receipt", str(canonical / "training_receipt.json"),
        "--source-qualification", str(canonical / "source_mixed_qualification.json"),
        "--route-report", str(canonical / "six_benchmark_route_report.json"),
        "--action-audit", str(canonical / "six_benchmark_action_equivalence.json"),
        "--markdown", str(output / "paper_result.md"),
        "--json", str(output / "paper_result.json"),
    ], cwd=repo)
    paper = _read(output / "paper_result.json")
    action = _read(portable_action)
    if paper.get("status") != config["expected_report_status"]:
        raise SystemExit("paper report status failed")
    if action.get("status") != (
        "SIX_BENCHMARK_9B_SUBSTITUTION_ACTION_EQUIVALENCE_VALIDATED"
    ) or not all((action.get("gates") or {}).values()):
        raise SystemExit("portable action-equivalence audit failed")
    result = {
        "status": "PORTABLE_SERVER_BUNDLE_V2_VERIFIED",
        "git_head": manifest["git"]["head"],
        "dependency_files": len(manifest["dependency_files"]),
        "pytest_passes": config["expected_pytest_passes"],
        "paper_status": paper["status"],
        "action_status": action["status"],
    }
    (output / "verification_receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
