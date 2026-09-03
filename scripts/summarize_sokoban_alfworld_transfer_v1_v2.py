#!/usr/bin/env python3
"""Create a compact, hash-bound diagnosis of Sokoban→ALFWorld V1/V2."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    body = dict(payload)
    claimed = str(body.pop("report_sha256", ""))
    if stable_hash(body) != claimed:
        raise ValueError(f"report hash mismatch: {path}")
    if payload.get("heldout_read") is not False:
        raise ValueError(f"held-out boundary was not preserved: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confusion(
    report: dict, condition: str, *, required_key: str,
) -> dict[str, int]:
    counts = Counter()
    for episode in report["episodes"][condition]:
        for record in episode["records"]:
            decision = record["decision"]
            selected = str(decision.get("source_selected_option") or "NONE")
            counts[f"{selected}->{decision[required_key]}"] += 1
    return dict(sorted(counts.items()))


def _task_differences(report: dict) -> list[dict]:
    authentic = report["episodes"]["authentic_source_effect_harness"]
    target = report["episodes"]["target_only"]
    reference = report["episodes"]["target_native_stage_reference"]
    rows = []
    for auth, null, ref in zip(authentic, target, reference):
        difference = int(auth["official_success"]) - int(null["official_success"])
        if difference:
            rows.append({
                "task_index": auth["task_index"],
                "task_id": auth["task_id"],
                "paired_difference": difference,
                "authentic_success": auth["official_success"],
                "target_only_success": null["official_success"],
                "target_stage_reference_success": ref["official_success"],
                "authentic_steps": auth["steps"],
                "target_only_steps": null["steps"],
                "target_stage_reference_steps": ref["steps"],
            })
    return rows


def _family_metrics(report: dict) -> dict[str, dict]:
    result = {}
    for condition in (
        "target_only", "authentic_source_effect_harness",
        "target_native_stage_reference",
    ):
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for episode in report["episodes"][condition]:
            family = str(episode["task_id"]).split("-", 1)[0]
            counts[family][0] += int(episode["official_success"])
            counts[family][1] += 1
        result[condition] = {
            family: {"successes": values[0], "tasks": values[1]}
            for family, values in sorted(counts.items())
        }
    return result


def _loss_loops(report: dict) -> list[dict]:
    authentic = report["episodes"]["authentic_source_effect_harness"]
    target = report["episodes"]["target_only"]
    rows = []
    for auth, null in zip(authentic, target):
        if not null["official_success"] or auth["official_success"]:
            continue
        actions = Counter(
            str(record["decision"]["action"]) for record in auth["records"]
        )
        mismatches = Counter(
            f"{record['decision']['source_selected_option']}->"
            f"{record['decision']['required_option_diagnostic_only']}"
            for record in auth["records"]
        )
        rows.append({
            "task_index": auth["task_index"],
            "task_id": auth["task_id"],
            "top_repeated_actions": [
                {"action": action, "count": count}
                for action, count in actions.most_common(5)
            ],
            "option_stage_counts": dict(mismatches.most_common()),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--v2-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite compact result: {args.output}")
    v1 = _read(args.v1_report)
    v2 = _read(args.v2_report)
    body = {
        "schema_version": "sokoban-alfworld-transfer-v1-v2-summary",
        "status": "REAL_GAME_TRANSFER_NOT_QUALIFIED_HELDOUT_UNREAD",
        "reports": {
            "v1": {
                "path": str(args.v1_report.resolve()),
                "file_sha256": _sha256(args.v1_report),
                "report_sha256": v1["report_sha256"],
                "status": v1["status"],
                "heldout_read": v1["heldout_read"],
                "summaries": v1["summaries"],
                "gates": v1["gates"],
                "paired_official_success": v1["paired_official_success"],
                "authentic_option_to_required_group": _confusion(
                    v1, "authentic_source_plus_harness",
                    required_key="required_option",
                ),
                "phase_control_option_to_required_group": _confusion(
                    v1, "phase_permuted_source_plus_harness",
                    required_key="required_option",
                ),
            },
            "v2": {
                "path": str(args.v2_report.resolve()),
                "file_sha256": _sha256(args.v2_report),
                "report_sha256": v2["report_sha256"],
                "status": v2["status"],
                "heldout_read": v2["heldout_read"],
                "summaries": v2["summaries"],
                "gates": v2["gates"],
                "paired_official_success": v2["paired_official_success"],
                "authentic_option_to_required_group": _confusion(
                    v2, "authentic_source_effect_harness",
                    required_key="required_option_diagnostic_only",
                ),
                "task_differences_vs_target_only": _task_differences(v2),
                "task_family_metrics": _family_metrics(v2),
                "target_only_win_loop_diagnostics": _loss_loops(v2),
            },
        },
        "diagnosis": {
            "v1": (
                "SOURCE_OPTION_OCCUPANCY_AND_RAW_FEATURE_SCALE_TRANSFERRED;_"
                "PHASE_CONTROL_OUTPERFORMED_AUTHENTIC"
            ),
            "v2": (
                "EFFECT_DIRECTION_WAS_NONRANDOM_BUT_BINARY_POSITION_COMMIT_"
                "COLLAPSED_ACQUIRE_TRANSFORM_AND_PLACE"
            ),
            "next_required_source_evidence": (
                "MATCHED_INTERVENTION_FORKS_WITH_TYPED_POSSESSION_PROPERTY_"
                "MUTATION_AND_RELATION_EFFECTS"
            ),
        },
    }
    payload = body | {"summary_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "summary_sha256": payload["summary_sha256"],
        "status": payload["status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
