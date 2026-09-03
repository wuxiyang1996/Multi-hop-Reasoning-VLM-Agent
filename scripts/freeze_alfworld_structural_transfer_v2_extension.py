#!/usr/bin/env python3
"""Freeze a powered independent extension under the unchanged V2 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


BASE = REPO / "configs/alfworld_structural_transfer_v2/manifest.json"
BASE_REPORT = REPO / "runs/alfworld_structural_transfer_v2_matched/report.json"
OUTPUT = REPO / "configs/alfworld_structural_transfer_v2_extension/manifest.json"
TASK_PATTERN = re.compile(r"pick_two_obj_and_place-[^\"\s]+/game\.tw-pddl")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _executed() -> set[str]:
    output: set[str] = set()
    for path in sorted((REPO / "runs").rglob("*.json")):
        lower = path.name.lower()
        if any(token in lower for token in ("enumeration", "manifest", "plan")):
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                output.update(TASK_PATTERN.findall(line))
    return output


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite extension manifest: {OUTPUT}")
    base = json.loads(BASE.read_text(encoding="utf-8"))
    report = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "ALFWORLD_STRUCTURAL_TRANSFER_FAILED":
        raise SystemExit("extension is only defined for the underpowered V2 result")
    if not all(
        value for key, value in report["gates"].items()
        if key != "paired_significance_vs_neural"
    ):
        raise SystemExit("V2 failed a mechanism or safety gate, not only power")
    if report["paired_comparisons"]["neural_only"] != {
        "losses": 0,
        "negative_transfer_rate": 0.0,
        "ties": 11,
        "two_sided_exact_sign_p": 1.0,
        "wins": 1,
    }:
        raise SystemExit("V2 result is not the frozen 1W/0L underpowered outcome")
    for relative, expected in base["integrity"]["file_sha256"].items():
        if _sha((REPO / relative).resolve()) != expected:
            raise SystemExit(f"V2 protocol dependency changed: {relative}")
    data = Path(base["target"]["alfworld_data"])
    root = data / "json_2.1.1" / "train"
    executed = _executed()
    candidates = sorted(
        path.relative_to(root).as_posix()
        for path in root.glob("pick_two_obj_and_place-*/trial_*/game.tw-pddl")
        if path.relative_to(root).as_posix() not in executed
    )
    ranked = sorted(candidates, key=lambda task_id: stable_hash({
        "salt": "ALFWORLD_STRUCTURAL_TRANSFER_V2_INDEPENDENT_POWERED_EXTENSION_20260817",
        "task_id": task_id,
    }))
    task_ids = ranked[:72]
    if len(task_ids) != 72:
        raise SystemExit(f"powered extension requires 72 untouched tasks; found {len(candidates)}")
    body = {key: value for key, value in base.items() if key != "config_sha256"}
    body.update({
        "schema_version": "alfworld-structural-transfer-frozen-manifest-v2-extension",
        "role": "INDEPENDENT_POWERED_FIXED_PROTOCOL_SECOND_TARGET_REPLICATION",
        "status": "FROZEN_BEFORE_INDEPENDENT_EXTENSION_EXECUTION",
        "base_v2_report": {
            "path": str(BASE_REPORT.relative_to(REPO)),
            "file_sha256": _sha(BASE_REPORT),
            "report_sha256": report["report_sha256"],
            "used_for_algorithm_or_threshold_change": False,
            "extension_trigger": "ONLY_PREDECLARED_SIGNIFICANCE_GATE_FAILED",
        },
    })
    body["target"] = dict(base["target"]) | {
        "seed": 2026081729,
        "task_ids": task_ids,
        "task_selection": "STABLE_HASH_RANK_FROM_REMAINING_EXECUTION_UNTOUCHED_POOL",
        "execution_untouched_candidate_pool_size": len(candidates),
        "selected_task_prior_execution_occurrences": {
            task_id: int(task_id in executed) for task_id in task_ids
        },
    }
    body["preregistered_gates"] = dict(base["preregistered_gates"]) | {
        "minimum_tasks": 72,
        "source_operator_admissions_min": 144,
        "independent_extension_must_pass_without_pooling_base_v2": True,
    }
    manifest = body | {"config_sha256": stable_hash(body)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT), "config_sha256": manifest["config_sha256"],
        "candidate_pool_size": len(candidates), "selected_tasks": len(task_ids),
        "prior_occurrences": sum(task_id in executed for task_id in task_ids),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
