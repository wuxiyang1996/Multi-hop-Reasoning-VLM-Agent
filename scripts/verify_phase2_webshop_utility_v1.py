#!/usr/bin/env python3
"""Verify all frozen Phase-2 WebShop goals live without actions or provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib import request


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase2_webshop_utility_v1 import (  # noqa: E402
    validate_manifest,
)


def _wait(process: subprocess.Popen, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"WebShop server exited with {process.returncode}")
        try:
            with request.urlopen("http://127.0.0.1:3000/", timeout=3):
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("WebShop server preflight timeout")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase2_webshop_utility_v1/manifest.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/phase2_webshop_utility_v1_preflight.json",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest, repo=REPO)
    server_log = args.output.with_suffix(".server.log")
    server_log.parent.mkdir(parents=True, exist_ok=True)
    mismatches = []
    rows = []
    with server_log.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                str(REPO / "scripts/run_webshop_direct_server_v1.py"),
                "--goal-seed", str(manifest["goal_seed"]),
            ], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            _wait(server)
            for index, task in enumerate(manifest["tasks"]):
                goal_index = int(task["server_goal_index"])
                url = (
                    "http://127.0.0.1:3000/__bridge/session/"
                    f"phase2_preflight_{index:02d}_fixed_{goal_index}"
                )
                with request.urlopen(url, timeout=90) as response:
                    goal = json.loads(response.read().decode("utf-8"))["goal"]
                observed = stable_hash(goal)
                if observed != task["goal_sha256"]:
                    mismatches.append({
                        "target_identity": task["target_identity"],
                        "expected": task["goal_sha256"],
                        "observed": observed,
                    })
                rows.append({
                    "target_identity": task["target_identity"],
                    "goal_sha256": observed,
                    "asin": goal["asin"],
                })
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=20)
    gates = {
        "manifest_valid_before_preflight": True,
        "all_32_live_goal_hashes_match": len(rows) == 32 and not mismatches,
        "unique_live_goal_hashes": len({row["goal_sha256"] for row in rows}) == 32,
        "unique_live_asins": len({row["asin"] for row in rows}) == 32,
        "zero_actions": True,
        "zero_provider_calls": True,
        "zero_outcomes_read": True,
    }
    body = {
        "schema_version": "phase2-webshop-utility-preflight-v1",
        "status": (
            "PHASE2_WEBSHOP_LIVE_PREFLIGHT_PASSED"
            if all(gates.values()) else "PHASE2_WEBSHOP_LIVE_PREFLIGHT_FAILED"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "live_goals_checked": len(rows),
        "mismatches": mismatches,
        "gates": gates,
    }
    result = body | {"preflight_sha256": stable_hash(body)}
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
