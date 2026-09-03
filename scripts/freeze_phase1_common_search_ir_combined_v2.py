#!/usr/bin/env python3
"""Freeze a combined manifest replacing failed Streets V1 with V2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
V1_MANIFEST = REPO / "configs/phase1_common_search_ir_formal_v1/manifest.json"
STREETS_CONFIG = (
    REPO / "configs/phase1_common_search_ir_streets_formal_v2/"
    "gymv_streets_of_rage_2.json"
)
STREETS_TEMPLATE = (
    REPO / "runs/phase1_common_search_ir_streets_formal_v2/"
    "gymv_streets_of_rage_2/templates.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "configs/phase1_common_search_ir_combined_v2/manifest.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    v1 = _read(V1_MANIFEST)
    v1_body = dict(v1)
    if v1_body.pop("manifest_sha256") != _stable(v1_body):
        raise SystemExit("V1 manifest self-hash mismatch")
    streets_v1_report_path = (
        REPO / "runs/phase1_common_search_ir_formal_v1/"
        "gymv_streets_of_rage_2/report.json"
    )
    streets_v1 = _read(streets_v1_report_path)
    if streets_v1.get("source_gate_passed") is not False:
        raise SystemExit("Streets V1 is not the expected failed source gate")
    if streets_v1.get("coverage_gate", {}).get("passed") is not False:
        raise SystemExit("Streets V1 did not fail the expected coverage gate")
    pilot_path = (
        REPO / "runs/phase1_common_search_ir_v7_pilot/"
        "gymv_streets_of_rage_2/report.json"
    )
    pilot = _read(pilot_path)
    if pilot.get("source_gate_passed") is not True:
        raise SystemExit("Streets V7 redesign pilot did not pass")
    config = _read(STREETS_CONFIG)
    template = _read(STREETS_TEMPLATE)
    selected = template["selected_discovery_execution"]
    if int(config["horizon"]) != len(selected["transition_receipt_ids"]):
        raise SystemExit("Streets V2 horizon lacks discovery authority")
    if template.get("status") != "FROZEN_BEFORE_FRESH_OPTION_FORKS":
        raise SystemExit("Streets V2 template is not frozen")
    prior_seeds = set()
    for path in (REPO / "configs").glob("phase1_common_search_ir_*/**/*.json"):
        if path == STREETS_CONFIG or path.name == "manifest.json":
            continue
        value = _read(path)
        prior_seeds.update(map(int, value.get("seeds") or []))
    v2_seeds = set(map(int, config["seeds"]))
    if prior_seeds & v2_seeds:
        raise SystemExit("Streets V2 seeds overlap prior source experiments")

    v1_receipts = {
        str(row["game"]): dict(row) for row in v1["config_receipts"]
    }
    for game, receipt in v1_receipts.items():
        receipt["run_dir"] = str((
            REPO / "runs/phase1_common_search_ir_formal_v1" / game
        ).resolve())
    v1_receipts["gymv_streets_of_rage_2"] = {
        "game": "gymv_streets_of_rage_2",
        "path": str(STREETS_CONFIG.resolve()),
        "file_sha256": _sha(STREETS_CONFIG),
        "seed_count": len(v2_seeds),
        "option_grounding": "discovery_execution_controls",
        "run_dir": str((
            REPO / "runs/phase1_common_search_ir_streets_formal_v2/"
            "gymv_streets_of_rage_2"
        ).resolve()),
    }
    body = {
        "schema_version": "phase1-common-search-ir-combined-formal-manifest-v2",
        "status": "FROZEN_BEFORE_STREETS_V2_FORMAL_COLLECTION",
        "games": list(v1["games"]),
        "target_data_read_for_freeze": False,
        "protocol": dict(v1["protocol"]),
        "config_receipts": [v1_receipts[game] for game in v1["games"]],
        "inherited_v1_manifest": {
            "path": str(V1_MANIFEST.resolve()),
            "file_sha256": _sha(V1_MANIFEST),
            "manifest_sha256": v1["manifest_sha256"],
        },
        "streets_v2_design_receipts": {
            "failed_v1_report_file_sha256": _sha(streets_v1_report_path),
            "failed_v1_reason": "SPLIT_COVERAGE_0.375_BELOW_FROZEN_0.40",
            "v7_redesign_pilot_report_file_sha256": _sha(pilot_path),
            "v2_template_file_sha256": _sha(STREETS_TEMPLATE),
            "v2_template_artifact_sha256": template["artifact_sha256"],
            "changed_component_only": "SOURCE_NATIVE_OPTION_GENERATOR",
            "unchanged_gate": True,
        },
        "claim_boundary": (
            "INHERITS_FIVE_V1_FORMAL_PROTOCOLS;STREETS_V2_FROZEN_AFTER_"
            "SOURCE_ONLY_V1_FAILURE_AND_V7_PILOT;TARGET_UNREAD"
        ),
    }
    artifact = body | {"manifest_sha256": _stable(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(args.output.resolve()),
        "manifest_sha256": artifact["manifest_sha256"],
        "streets_v2_seeds": len(v2_seeds),
        "prior_seeds_excluded": len(prior_seeds),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
