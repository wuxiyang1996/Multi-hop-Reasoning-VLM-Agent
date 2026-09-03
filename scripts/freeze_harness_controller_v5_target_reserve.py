#!/usr/bin/env python3
"""Freeze a target-IR reserve disjoint from all consumed V3/V4 diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/harness_controller_v5_fresh_target_reserve.json"
DEFAULT_OUTPUT = REPO / "runs/harness_controller_v5_fresh_target_reserve"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _verified(spec: dict[str, Any]) -> Path:
    path = _resolve(str(spec["path"]))
    if not path.is_file() or _sha(path) != spec["sha256"]:
        raise ValueError(f"missing or hash-mismatched input: {path}")
    return path


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]


def _group(row: dict[str, Any]) -> str:
    domain = str(row["target_domain_audit_only"])
    return (
        f"video/{row['target_benchmark_audit_only']}"
        if domain == "video" else domain
    )


def freeze(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_V5_MIXED_WEIGHT_UPDATES":
        raise ValueError("V5 reserve config is not frozen before training")
    pool_path = _verified(config["candidate_pool"])
    mixed = config["source_only_mixed_dataset"]
    mixed_manifest_path = _verified(mixed["manifest"])
    _verified(mixed["train"])
    mixed_manifest = json.loads(mixed_manifest_path.read_text(encoding="utf-8"))
    if not (
        mixed_manifest.get("status")
        == "FROZEN_SOURCE_ONLY_MIXED_HARNESS_SUPERVISION"
        and mixed_manifest.get("target_data_used") is False
        and mixed_manifest.get("target_outcome_used_for_controller_labels") is False
        and all((mixed_manifest.get("gates") or {}).values())
    ):
        raise ValueError("mixed source dataset is not source-only and gate-clean")

    consumed_files = []
    consumed = []
    for diagnostic in config["consumed_target_diagnostics"]:
        dataset_path = _verified(diagnostic["dataset"])
        report_path = _verified(diagnostic["report"])
        consumed.extend(_rows(dataset_path))
        consumed_files.append({
            "dataset": {
                "path": str(dataset_path.resolve()), "sha256": _sha(dataset_path),
            },
            "report": {
                "path": str(report_path.resolve()), "sha256": _sha(report_path),
            },
        })
    consumed_ids = {str(row["example_id"]) for row in consumed}
    consumed_pairs = {str(row["pair_id"]) for row in consumed}
    consumed_prompts = {str(row["prompt"]) for row in consumed}

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(pool_path):
        pairs[str(row["pair_id"])].append(row)
    eligible: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    malformed = []
    for pair_id, rows in pairs.items():
        controls = {str(row["control_variant_audit_only"]) for row in rows}
        groups = {_group(row) for row in rows}
        if (
            len(rows) != 2 or len(groups) != 1
            or "AUTHENTIC_TARGET_NEURAL_GROUNDING" not in controls
            or len(controls) != 2
        ):
            malformed.append(pair_id)
            continue
        if (
            pair_id in consumed_pairs
            or any(str(row["example_id"]) in consumed_ids for row in rows)
            or any(str(row["prompt"]) in consumed_prompts for row in rows)
        ):
            continue
        eligible[next(iter(groups))].append((pair_id, rows))

    seed = str(config["selection_seed"])
    quotas = {
        str(group): int(count)
        for group, count in config["pair_quota_by_group"].items()
    }
    selected = []
    selected_pairs = []
    for group, quota in sorted(quotas.items()):
        ranked = sorted(
            eligible.get(group, ()),
            key=lambda item: hashlib.sha256(
                f"{seed}:{item[0]}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < quota:
            raise ValueError(f"{group} has only {len(ranked)} fresh pairs")
        for pair_id, rows in ranked[:quota]:
            selected_pairs.append(pair_id)
            for row in rows:
                selected.append(dict(row) | {
                    "target_eval_group_audit_only": group,
                })
    selected.sort(key=lambda row: str(row["example_id"]))

    selected_ids = {str(row["example_id"]) for row in selected}
    selected_prompts = {str(row["prompt"]) for row in selected}
    pair_set = set(selected_pairs)
    group_rows = Counter(row["target_eval_group_audit_only"] for row in selected)
    group_pairs = Counter(
        _group(rows[0]) for pair_id, rows in pairs.items() if pair_id in pair_set
    )
    decisions = Counter(json.loads(row["completion"])["decision"] for row in selected)
    cardinalities = Counter()
    for row in selected:
        payload = json.loads(row["prompt"].split(
            "CONTROLLER_INPUT=", 1,
        )[1].split("\nOUTPUT_JSON=", 1)[0])
        cardinalities[len(payload["candidate_effects"])] += 1
    policy = config["selection_policy"]
    gates = {
        "mixed_source_dataset_gate_clean": all(mixed_manifest["gates"].values()),
        "no_malformed_pair_selected": not (set(malformed) & pair_set),
        "pair_quotas_exact": dict(group_pairs) == quotas,
        "two_rows_per_pair": len(selected) == 2 * len(selected_pairs),
        "example_ids_unique": len(selected_ids) == len(selected),
        "prompts_unique": len(selected_prompts) == len(selected),
        "consumed_example_overlap_zero": not (selected_ids & consumed_ids),
        "consumed_pair_overlap_zero": not (pair_set & consumed_pairs),
        "consumed_prompt_overlap_zero": not (selected_prompts & consumed_prompts),
        "both_decisions_present": set(decisions) == {
            "ABSTAIN", "EXECUTE_OPERATOR",
        },
        "arity_twelve_present": cardinalities[12] > 0,
        "selection_is_prediction_and_outcome_blind": (
            policy["uses_completion_or_decision"] is False
            and policy["uses_v3_or_v4_prediction_or_correctness"] is False
            and policy["uses_target_reward_or_success"] is False
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"V5 fresh target reserve gates failed: {gates}")
    output_dir.mkdir(parents=True)
    reserve_path = output_dir / "reserve.jsonl"
    with reserve_path.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(
                row, sort_keys=True, ensure_ascii=False,
            ) + "\n")
    manifest = {
        "schema_version": "harness-controller-v5-target-reserve-preregistration-v1",
        "status": "FROZEN_PROSPECTIVE_TO_V5_BEFORE_WEIGHT_UPDATES",
        "authority": (
            "PAIR_IDENTITY_HASH_SELECTION_ONLY;NO_COMPLETION_OR_MODEL_PREDICTION_SELECTION;"
            "NO_V5_WEIGHT_UPDATE_OCCURRED;NO_TARGET_ROWS_FOR_WEIGHT_UPDATES"
        ),
        "config": {
            "path": str(config_path.resolve()), "sha256": _sha(config_path),
        },
        "candidate_pool": {
            "path": str(pool_path.resolve()), "sha256": _sha(pool_path),
        },
        "consumed_target_diagnostics": consumed_files,
        "mixed_source_manifest": {
            "path": str(mixed_manifest_path.resolve()),
            "sha256": _sha(mixed_manifest_path),
        },
        "reserve": {
            "path": str(reserve_path.resolve()), "sha256": _sha(reserve_path),
            "rows": len(selected), "pairs": len(selected_pairs),
        },
        "summary": {
            "group_row_counts": dict(sorted(group_rows.items())),
            "group_pair_counts": dict(sorted(group_pairs.items())),
            "decision_counts_audit_after_selection": dict(sorted(decisions.items())),
            "candidate_count_counts_audit_after_selection": dict(
                sorted(cardinalities.items())
            ),
        },
        "gates": gates,
        "claim_boundary": config["claim_boundary"],
    }
    (output_dir / "preregistration.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = freeze(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "status": manifest["status"], "reserve": manifest["reserve"],
        "summary": manifest["summary"], "gates": manifest["gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
