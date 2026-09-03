#!/usr/bin/env python3
"""Run the V22 cost-qualified QUERY_OBJECT consensus collector."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
import scripts.collect_agqa2_query_object_v21 as v21  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256  # noqa: E402


def collect(**kwargs):
    config = json.loads(Path(kwargs["config_path"]).read_text())
    parent_path = REPO_ROOT / config["query_object_grounder"]["parent_consensus_collector"]
    if _sha256(parent_path) != config["query_object_grounder"][
        "parent_consensus_collector_sha256"
    ]:
        raise ValueError("QUERY_OBJECT V22 parent consensus collector hash mismatch")
    result = v21.collect(**kwargs)
    body = deepcopy(result)
    body.pop("report_sha256", None)
    body.update({
        "schema_version": "agqa2-query-object-consensus-report-v22",
        "status": result["status"].replace("V21", "V22"),
    })
    final = body | {"report_sha256": stable_hash(body)}
    Path(kwargs["output_path"]).write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n"
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", type=Path,
                        default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = collect(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.output.resolve(), workers=args.workers,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "metrics", "controls", "qualification_gates",
        "reported_provider_cost_usd", "report_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
