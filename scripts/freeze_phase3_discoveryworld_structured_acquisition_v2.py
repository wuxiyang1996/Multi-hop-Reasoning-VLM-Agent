#!/usr/bin/env python3
"""Freeze component-only qualification for structured acquisition V2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402


SEEDS = tuple(seed for seed in range(51, 81) if seed != 70)
DEVELOPMENT = (
    REPO / "runs/phase3_discoveryworld_structured_acquisition_v2_development/summary.json"
)
MODEL_CONFIG = REPO / "configs/phase3_discoveryworld_structured_acquisition_v2/development.json"
RUNTIME_PATHS = (
    "src/motif_transfer/contracts.py",
    "src/motif_transfer/discoveryworld_env.py",
    "src/motif_transfer/discoveryworld_policy.py",
    "src/motif_transfer/phase3_discoveryworld_grounding.py",
    "src/motif_transfer/phase3_discoveryworld_transfer.py",
    "src/motif_transfer/phase3_discoveryworld_formal.py",
    "src/motif_transfer/discoveryworld_structured_acquisition_v2.py",
    "scripts/run_phase3_discoveryworld_structured_acquisition_v2.py",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value); claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite qualification manifest")
    development = _read(DEVELOPMENT); _self_hash(development, "summary_sha256")
    model_config = _read(MODEL_CONFIG)
    if (
        development.get("complete_tasks") != 6
        or development.get("acquisition_ready_tasks") != 6
        or development.get("acquisition_schema_fallback_rate") > 0.10
        or development.get("acquisition_repair_rate") > 0.10
        or development.get("invalid_native_actions") != 0
    ):
        raise SystemExit("structured acquisition development gate failed")
    for seed in range(121, 145):
        if (
            REPO / "runs/phase3_discoveryworld_formal_v2_acquisition" /
            f"proteomics.easy.seed{seed}.json"
        ).exists():
            raise SystemExit("V2 formal reserve was opened before qualification freeze")
    body = {
        "schema_version": "phase3-discoveryworld-structured-acquisition-freeze-v2",
        "status": "FROZEN_BEFORE_STRUCTURED_ACQUISITION_QUALIFICATION",
        "role": "qualification",
        "claim_boundary": (
            "Component-only structured acquisition qualification on previously "
            "consumed seeds51-80 except70; stops at ready fork, does not finalize "
            "the evaluator, and does not read target outcomes or source programs."
        ),
        "model": dict(model_config["model"]),
        "runtime": {
            "maximum_acquisition_steps": 24, "continuation_horizon": 0,
            "task_workers": 4, "thread_id_base": 160000,
        },
        "tasks": [{
            "task_id": f"proteomics.easy.seed{seed}",
            "scenario": "Proteomics", "difficulty": "Easy", "seed": seed,
            "state_previously_consumed": True,
        } for seed in SEEDS],
        "task_count": len(SEEDS),
        "frozen_qualification_gates": {
            "required_ready_states": len(SEEDS),
            "maximum_acquisition_schema_fallback_rate": 0.10,
            "maximum_acquisition_repair_rate": 0.10,
            "maximum_invalid_native_actions": 0,
            "evaluator_finalized": False,
            "formal_target_outcome_read": False,
        },
        "development_evidence": {
            "path": str(DEVELOPMENT.relative_to(REPO)),
            "file_sha256": _file_sha(DEVELOPMENT),
            "summary_sha256": development["summary_sha256"],
            "excluded_from_prospective_estimates": True,
        },
        "runtime_file_sha256": {
            path: _file_sha(REPO / path) for path in RUNTIME_PATHS
        },
        "formal_reserve_seed_range": [121, 144],
        "formal_reserve_opened": False,
        "formal_target_outcome_read_for_freeze": False,
    }
    payload = body | {"manifest_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "tasks": len(SEEDS),
        "manifest_sha256": payload["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
