#!/usr/bin/env python3
"""Run QUERY_OBJECT with bounded explanatory strings and frozen decisions."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_bounded_ontology_protocol import (  # noqa: E402
    MAX_FREE_TEXT_CHARACTERS, bounded_response_format, bounded_system_prompt,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
import scripts.collect_agqa2_query_object_v20 as v20  # noqa: E402
import scripts.collect_agqa2_query_object_v24 as v24  # noqa: E402
import scripts.collect_agqa2_query_object_v26 as v26  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256  # noqa: E402


def _relabel(result: Mapping[str, Any]) -> dict[str, Any]:
    body = deepcopy(dict(result))
    body.pop("report_sha256", None)
    prior_status = str(body["status"])
    if "V24" in prior_status:
        status = prior_status.replace("V24", "V28")
    elif "V26" in prior_status:
        status = prior_status.replace("V26", "V28")
    else:
        raise ValueError("V28 expected a V24 or V26 evaluator status")
    body.update({
        "schema_version": "agqa2-query-object-bounded-ontology-report-v28",
        "status": status,
        "parent_evaluator_status_before_version_relabel": prior_status,
        "bounded_ontology_free_text_characters": MAX_FREE_TEXT_CHARACTERS,
        "decision_confidence_evidence_schema_unchanged": True,
    })
    return body | {"report_sha256": stable_hash(body)}


def collect(**kwargs) -> dict[str, Any]:
    config = json.loads(Path(kwargs["config_path"]).read_text())
    protocol = config["query_object_grounder"]["bounded_ontology_protocol"]
    for label in ("module", "collector"):
        if _sha256(REPO_ROOT / protocol[label]) != protocol[f"{label}_sha256"]:
            raise ValueError(f"V28 bounded ontology {label} hash mismatch")
    if int(protocol["maximum_free_text_characters"]) != MAX_FREE_TEXT_CHARACTERS:
        raise ValueError("V28 bounded ontology character limit changed")
    original_format, original_system = v20._ontology_response_format, v20._ontology_system
    v20._ontology_response_format = lambda: bounded_response_format(
        original_format(), max_characters=MAX_FREE_TEXT_CHARACTERS,
    )
    v20._ontology_system = lambda: bounded_system_prompt(
        original_system(), max_characters=MAX_FREE_TEXT_CHARACTERS,
    )
    try:
        parent = v26 if "source_specific_evaluation" in config else v24
        result = parent.collect(**kwargs)
    finally:
        v20._ontology_response_format = original_format
        v20._ontology_system = original_system
    final = _relabel(result)
    Path(kwargs["output_path"]).write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n"
    )
    return final


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
    keys = [
        "status", "metrics", "qualification_gates",
        "reported_provider_cost_usd", "report_sha256",
    ]
    if "source_specific_metrics" in result:
        keys[2:2] = [
            "source_specific_metrics", "source_specific_qualification_gates",
        ]
    print(json.dumps({key: result[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
