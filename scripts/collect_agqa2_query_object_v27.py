#!/usr/bin/env python3
"""Run V27 with the unchanged V26 paired evaluator and repaired token cap."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
import scripts.collect_agqa2_query_object_v26 as v26  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256  # noqa: E402


def _relabel(result: Mapping[str, Any]) -> dict[str, Any]:
    body = deepcopy(dict(result))
    body.pop("report_sha256", None)
    prior_status = str(body["status"])
    if "V26" not in prior_status:
        raise ValueError("V27 expected a V26 paired-evaluator status")
    body.update({
        "schema_version": "agqa2-query-object-source-specific-report-v27",
        "status": prior_status.replace("V26", "V27"),
        "v26_evaluator_status_before_version_relabel": prior_status,
        "v26_paired_outcome_calculation_unchanged": True,
        "v27_primary_ontology_max_tokens": 500,
    })
    return body | {"report_sha256": stable_hash(body)}


def collect(**kwargs) -> dict[str, Any]:
    config = json.loads(Path(kwargs["config_path"]).read_text())
    wrapper = config["source_specific_evaluation"]["report_wrapper"]
    if _sha256(REPO_ROOT / wrapper["path"]) != wrapper["sha256"]:
        raise ValueError("V27 report wrapper hash mismatch")
    result = _relabel(v26.collect(**kwargs))
    Path(kwargs["output_path"]).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = collect(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.output.resolve(), workers=args.workers,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "metrics", "source_specific_metrics",
        "qualification_gates", "source_specific_qualification_gates",
        "reported_provider_cost_usd", "report_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
