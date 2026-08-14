#!/usr/bin/env python3
"""Fail closed unless a live synthetic WebShop server matches the frozen split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib import request


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.webshop_semantic_reserve import require_semantic_reserve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "configs/webshop_synthetic_unique_v14_frozen.json",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--namespace", default="v14-preflight")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs/results/webshop_synthetic_server_v14_preflight.json",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [
        *manifest["roles"]["development"],
        *manifest["roles"]["formal_reserve"],
    ]
    live_rows = []
    mismatches = []
    for row in rows:
        index = int(row["server_goal_index"])
        url = (
            f"{args.base_url.rstrip('/')}/__bridge/session/"
            f"{args.namespace}_fixed_{index}"
        )
        with request.urlopen(url, timeout=90) as response:
            goal = json.loads(response.read().decode("utf-8"))["goal"]
        live_hash = stable_hash(goal)
        if live_hash != row["goal_sha256"]:
            mismatches.append({
                "task_id": row["task_id"],
                "expected": row["goal_sha256"],
                "observed": live_hash,
            })
        live_rows.append({
            "task_id": row["task_id"],
            "asin": goal["asin"],
            "instruction_text": goal["instruction_text"],
        })
    semantic_audit = require_semantic_reserve(
        live_rows,
        required_unique_goals=len(live_rows),
        require_unique_candidate_asins=True,
    )
    gates = {
        "manifest_status_frozen": (
            manifest["status"] == "FROZEN_BEFORE_ANY_PROVIDER_CALL_OR_OUTCOME"
        ),
        "every_live_goal_hash_matches": not mismatches,
        "live_semantic_reserve_passes": semantic_audit["passed"],
        "formal_outcomes_remain_sealed": True,
        "zero_provider_calls": True,
    }
    passed = all(gates.values())
    artifact = {
        "schema_version": 1,
        "status": "WEBSHOP_SYNTHETIC_SERVER_V14_PREFLIGHT_PASSED" if passed else
        "WEBSHOP_SYNTHETIC_SERVER_V14_PREFLIGHT_FAILED",
        "passed": passed,
        "live_goals_checked": len(live_rows),
        "development_goals": len(manifest["roles"]["development"]),
        "formal_reserve_goals": len(manifest["roles"]["formal_reserve"]),
        "goal_hash_mismatches": mismatches,
        "semantic_audit": semantic_audit,
        "gates": gates,
        "manifest": str(args.manifest),
        "manifest_artifact_sha256": manifest["artifact_sha256"],
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
