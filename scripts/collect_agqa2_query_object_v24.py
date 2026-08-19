#!/usr/bin/env python3
"""V24 QUERY_OBJECT collector with deterministic interval syntax repair."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_operand_normalization import (  # noqa: E402
    parse_normalized_operand_receipt,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
import scripts.collect_agqa2_active_grounding_v3 as base  # noqa: E402
import scripts.collect_agqa2_query_object_v22 as v22  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256  # noqa: E402


def collect(**kwargs):
    config = json.loads(Path(kwargs["config_path"]).read_text())
    normalization = config["query_object_grounder"]["normalization_module"]
    normalization_path = REPO_ROOT / normalization
    if _sha256(normalization_path) != config["query_object_grounder"][
        "normalization_module_sha256"
    ]:
        raise ValueError("QUERY_OBJECT V24 normalization module hash mismatch")
    original = base.parse_operand_receipt
    base.parse_operand_receipt = parse_normalized_operand_receipt
    try:
        result = v22.collect(**kwargs)
    finally:
        base.parse_operand_receipt = original
    body = deepcopy(result)
    body.pop("report_sha256", None)
    body.update({
        "schema_version": "agqa2-query-object-consensus-report-v24",
        "status": result["status"].replace("V22", "V24"),
        "deterministic_interval_envelope_normalization": True,
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
