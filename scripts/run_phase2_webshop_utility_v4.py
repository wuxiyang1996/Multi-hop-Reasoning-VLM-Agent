#!/usr/bin/env python3
"""Run Phase-2 WebShop with safe candidates and a single-threaded server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.phase2_webshop_utility_v1 import (  # noqa: E402
    PASSED_STATUS,
    validate_manifest,
)
from motif_transfer.webshop_search_automaton_v16 import AUTHENTIC  # noqa: E402
import scripts.run_phase2_webshop_utility_v1 as base_runner  # noqa: E402
from scripts.run_phase2_webshop_utility_v3 import _candidate_augmenter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--wrapper-root", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = base_runner._read(args.manifest)
    validate_manifest(manifest, repo=REPO)
    base_runner._candidate_augmenter = _candidate_augmenter
    server_log_path = args.output_dir / "server.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with server_log_path.open("a", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            [
                sys.executable,
                str(REPO / "scripts/run_webshop_direct_server_v4.py"),
                "--goal-seed", str(manifest["goal_seed"]),
            ],
            cwd=REPO, stdout=server_log, stderr=subprocess.STDOUT,
        )
        try:
            base_runner._wait_server(server)
            report = base_runner._run(
                manifest, keys=args.keys, wrapper_root=args.wrapper_root,
                output_dir=args.output_dir,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=20)
    print(json.dumps({
        "status": report["status"],
        "strict_successes": report["summaries"][AUTHENTIC]["strict_successes"],
        "raw_strict_successes": report["summaries"]["raw_target_only"]["strict_successes"],
        "gates_passed": sum(bool(value) for value in report["gates"].values()),
        "gates_required": len(report["gates"]),
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if report["status"] == PASSED_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
