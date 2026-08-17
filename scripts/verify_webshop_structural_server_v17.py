#!/usr/bin/env python3
"""Hash-check the live V17 server without reading any task outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from urllib import request


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/webshop_structural_v17_frozen.json",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--namespace", default="v17-structural-preflight")
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/webshop_structural_v17_server_preflight.json",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    schema_match = re.fullmatch(
        r"webshop-structural-v(\d+)-reserve-v1",
        str(manifest.get("schema_version")),
    )
    if schema_match is None:
        raise SystemExit("unsupported structural reserve schema")
    reserve_version = f"V{schema_match.group(1)}"
    rows = [row for role in manifest["roles"].values() for row in role]
    mismatches = []
    for row in rows:
        index = int(row["server_goal_index"])
        url = (
            f"{args.base_url.rstrip('/')}/__bridge/session/"
            f"{args.namespace}_fixed_{index}"
        )
        with request.urlopen(url, timeout=90) as response:
            goal = json.loads(response.read().decode("utf-8"))["goal"]
        observed = stable_hash(goal)
        if observed != row["goal_sha256"]:
            mismatches.append({
                "task_id": row["task_id"],
                "expected": row["goal_sha256"],
                "observed": observed,
            })
    gates = {
        "manifest_frozen": manifest.get("status")
        == f"FROZEN_BEFORE_ANY_{reserve_version}_PROVIDER_CALL_OR_OUTCOME",
        "all_live_goal_hashes_match": not mismatches,
        "all_goals_have_relation_schema": bool(rows) and all(
            bool((row.get("goal") or {}).get("goal_options")) for row in rows
        ),
        "zero_provider_calls": True,
        "formal_outcomes_unread": True,
    }
    body = {
        "schema_version": "webshop-structural-v17-server-preflight-v1",
        "status": (
            f"WEBSHOP_STRUCTURAL_{reserve_version}_SERVER_PREFLIGHT_PASSED"
            if all(gates.values()) else
            f"WEBSHOP_STRUCTURAL_{reserve_version}_SERVER_PREFLIGHT_FAILED"
        ),
        "live_goal_hashes_checked": len(rows),
        "goal_hash_mismatches": mismatches,
        "manifest_artifact_sha256": manifest["artifact_sha256"],
        "gates": gates,
        "formal_outcomes_read_or_run": False,
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
