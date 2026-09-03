#!/usr/bin/env python3
"""Freeze a prediction-blind matched-pair target-IR reserve before V4 training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/harness_controller_v4_fresh_target_reserve.json"
DEFAULT_OUTPUT = REPO_ROOT / "runs/harness_controller_v4_fresh_target_reserve"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _verify(spec: dict[str, Any]) -> Path:
    path = _resolve(str(spec["path"]))
    if not path.is_file() or _sha256(path) != spec["sha256"]:
        raise ValueError(f"missing or hash-mismatched frozen input: {path}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _group(row: dict[str, Any]) -> str:
    domain = str(row["target_domain_audit_only"])
    if domain == "video":
        return f"video/{row['target_benchmark_audit_only']}"
    return domain


def _rank(seed: str, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).hexdigest()


def freeze(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_V4_CARDINALITY_WEIGHT_UPDATES":
        raise ValueError("target reserve config is not frozen before V4 training")

    pool_path = _verify(config["candidate_pool"])
    consumed_path = _verify(config["consumed_target_diagnostic"]["dataset"])
    _verify(config["consumed_target_diagnostic"]["report"])
    source_manifest_path = _verify(config["source_only_v4_dataset"]["manifest"])
    _verify(config["source_only_v4_dataset"]["train"])
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("target_data_used") is not False
        or source_manifest.get("summary", {}).get("target_data_used") is not False
        or not all(source_manifest.get("gates", {}).values())
    ):
        raise ValueError("V4 source dataset is not source-only and gate-clean")

    consumed = _read_jsonl(consumed_path)
    consumed_ids = {str(row["example_id"]) for row in consumed}
    consumed_pairs = {str(row["pair_id"]) for row in consumed}
    consumed_prompts = {str(row["prompt"]) for row in consumed}

    pool = _read_jsonl(pool_path)
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        pairs[str(row["pair_id"])].append(row)
    eligible_by_group: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    malformed_pairs = []
    for pair_id, rows in pairs.items():
        controls = {str(row["control_variant_audit_only"]) for row in rows}
        groups = {_group(row) for row in rows}
        if (
            len(rows) != 2
            or len(groups) != 1
            or "AUTHENTIC_TARGET_NEURAL_GROUNDING" not in controls
            or len(controls) != 2
        ):
            malformed_pairs.append(pair_id)
            continue
        if (
            pair_id in consumed_pairs
            or any(str(row["example_id"]) in consumed_ids for row in rows)
            or any(str(row["prompt"]) in consumed_prompts for row in rows)
        ):
            continue
        eligible_by_group[next(iter(groups))].append((pair_id, rows))

    seed = str(config["selection_seed"])
    quotas = {str(key): int(value) for key, value in config["pair_quota_by_group"].items()}
    selected: list[dict[str, Any]] = []
    selected_pair_ids = []
    for group, quota in sorted(quotas.items()):
        candidates = sorted(
            eligible_by_group.get(group, []), key=lambda item: _rank(seed, item[0]),
        )
        if len(candidates) < quota:
            raise ValueError(f"{group} has only {len(candidates)} eligible pairs")
        for pair_id, rows in candidates[:quota]:
            selected_pair_ids.append(pair_id)
            for row in rows:
                output_row = dict(row)
                output_row["target_eval_group_audit_only"] = group
                selected.append(output_row)
    selected.sort(key=lambda row: str(row["example_id"]))

    selected_ids = {str(row["example_id"]) for row in selected}
    selected_prompts = {str(row["prompt"]) for row in selected}
    group_counts = Counter(row["target_eval_group_audit_only"] for row in selected)
    group_pair_counts = Counter(
        _group(rows[0])
        for pair_id, rows in pairs.items()
        if pair_id in set(selected_pair_ids)
    )
    decisions = Counter(json.loads(row["completion"])["decision"] for row in selected)
    candidate_counts = Counter()
    for row in selected:
        controller_input = json.loads(
            row["prompt"].split("CONTROLLER_INPUT=", 1)[1].rsplit(
                "\nOUTPUT_JSON=", 1,
            )[0]
        )
        candidate_counts[len(controller_input["candidate_effects"])] += 1

    gates = {
        "source_only_v4_dataset_gate_clean": all(source_manifest["gates"].values()),
        "no_malformed_pair_selected": not malformed_pairs or not (
            set(malformed_pairs) & set(selected_pair_ids)
        ),
        "pair_quotas_exact": dict(group_pair_counts) == quotas,
        "two_rows_per_selected_pair": len(selected) == 2 * len(selected_pair_ids),
        "example_ids_unique": len(selected_ids) == len(selected),
        "prompts_unique": len(selected_prompts) == len(selected),
        "consumed_example_overlap_zero": not (selected_ids & consumed_ids),
        "consumed_pair_overlap_zero": not (set(selected_pair_ids) & consumed_pairs),
        "consumed_prompt_overlap_zero": not (selected_prompts & consumed_prompts),
        "both_decisions_present": set(decisions) == {"ABSTAIN", "EXECUTE_OPERATOR"},
        "two_candidate_states_present": candidate_counts[2] > 0,
        "selection_does_not_use_completion_or_prediction": (
            config["selection_policy"]["uses_completion_or_decision"] is False
            and config["selection_policy"]["uses_v3_prediction_or_correctness"] is False
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"fresh target reserve gates failed: {gates}")

    output_dir.mkdir(parents=True)
    reserve_path = output_dir / "reserve.jsonl"
    with reserve_path.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "harness-controller-v4-target-reserve-preregistration-v1",
        "status": "FROZEN_PROSPECTIVE_TO_V4_BEFORE_WEIGHT_UPDATES",
        "authority": (
            "PAIR_IDENTITY_HASH_SELECTION_ONLY;NO_V3_PREDICTION_SELECTION;"
            "NO_V4_WEIGHT_UPDATE_OCCURRED;NO_TARGET_ROWS_FOR_WEIGHT_UPDATES"
        ),
        "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        "candidate_pool": {"path": str(pool_path.resolve()), "sha256": _sha256(pool_path)},
        "consumed_dataset": {
            "path": str(consumed_path.resolve()), "sha256": _sha256(consumed_path),
        },
        "source_v4_manifest": {
            "path": str(source_manifest_path.resolve()),
            "sha256": _sha256(source_manifest_path),
        },
        "reserve": {
            "path": str(reserve_path.resolve()),
            "sha256": _sha256(reserve_path),
            "rows": len(selected),
            "pairs": len(selected_pair_ids),
        },
        "summary": {
            "group_row_counts": dict(sorted(group_counts.items())),
            "group_pair_counts": dict(sorted(group_pair_counts.items())),
            "decision_counts_audit_after_selection": dict(sorted(decisions.items())),
            "candidate_count_counts_audit_after_selection": dict(
                sorted(candidate_counts.items())
            ),
        },
        "gates": gates,
        "claim_boundary": config["claim_boundary"],
    }
    (output_dir / "preregistration.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = freeze(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "status": manifest["status"],
        "reserve": manifest["reserve"],
        "summary": manifest["summary"],
        "gates": manifest["gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
