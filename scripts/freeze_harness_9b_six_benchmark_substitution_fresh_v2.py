#!/usr/bin/env python3
"""Freeze a fresh anonymous presentation reserve after a consumed diagnostic.

The semantic tasks, source routes, and native receipts remain the original
pre-outcome-frozen authorities.  Only the model-facing catalog order, opaque
aliases, prompts, and example IDs are regenerated.  This script never opens a
model prediction, target outcome, or formal report body.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.portable_paths import resolve_repo_artifact  # noqa: E402
from motif_transfer.multi_ir_selector_training import (  # noqa: E402
    SELECT_SKILL,
    execute_anonymous_selection,
    format_multi_ir_selector_prompt,
)


DEFAULT_CONFIG = REPO / "configs/harness_9b_six_benchmark_substitution_fresh_v2.json"
DEFAULT_OUTPUT = REPO / "runs/harness_9b_six_benchmark_substitution_fresh_v2"
PARENT_STATUS = "FROZEN_BEFORE_FIVE_SCHEMA_9B_WEIGHT_UPDATE_OR_SUBSTITUTION_INFERENCE"
FRESH_STATUS = "FROZEN_FRESH_PRESENTATION_BEFORE_SOURCE_ONLY_PERMUTATION_UPDATE"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve(value: str) -> Path:
    return resolve_repo_artifact(value, REPO)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def _selector_payload(prompt: str) -> dict[str, Any]:
    prefix = "\nSELECTOR_INPUT="
    suffix = "\nOUTPUT_JSON="
    if prompt.count(prefix) != 1 or not prompt.endswith(suffix):
        raise ValueError("parent selector prompt has an unexpected format")
    raw = prompt.split(prefix, 1)[1][:-len(suffix)]
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("selector payload is not an object")
    return value


def freeze(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    config = _read(config_path)
    if config.get("status") != (
        "FREEZE_REQUESTED_AFTER_CONSUMED_V1_DIAGNOSTIC_BEFORE_ANY_V3_WEIGHT_UPDATE"
    ):
        raise ValueError("fresh-reserve request is not in the pre-update state")
    forbidden_adapter = _resolve(str(config["forbidden_adapter_root"]))
    if forbidden_adapter.exists():
        raise ValueError("V3 adapter already exists; fresh reserve is too late")

    parent_path = _resolve(str(config["parent_preregistration"]))
    parent = _read(parent_path)
    if not (
        parent.get("status") == PARENT_STATUS
        and all((parent.get("gates") or {}).values())
    ):
        raise ValueError("parent substitution preregistration is not gate-clean")
    parent_dataset = _resolve(parent["route_selector_replay"]["path"])
    parent_index = _resolve(parent["native_replay_index"]["path"])
    if not (
        _sha(parent_dataset) == parent["route_selector_replay"]["sha256"]
        and _sha(parent_index) == parent["native_replay_index"]["sha256"]
    ):
        raise ValueError("parent substitution artifact hash drifted")

    reserve_salt = str(config["reserve_salt"])
    old_rows = _rows(parent_dataset)
    new_rows: list[dict[str, Any]] = []
    old_to_new: dict[str, str] = {}
    old_prompts = {str(row["prompt"]) for row in old_rows}
    for old in old_rows:
        old_id = str(old["example_id"])
        payload = _selector_payload(str(old["prompt"]))
        catalog = [dict(row) for row in payload["program_catalog"]]
        requirement = dict(payload["target_native_structural_requirement"])
        old_target = json.loads(str(old["completion"]))
        if old_target.get("decision") != SELECT_SKILL:
            raise ValueError("parent reserve contains a non-selection row")
        selected_old_id = str(old_target["selected_catalog_id"])
        seed = stable_hash({
            "schema_version": config["schema_version"],
            "reserve_salt": reserve_salt,
            "parent_example_id": old_id,
        })
        order = sorted(
            range(len(catalog)),
            key=lambda index: stable_hash({"seed": seed, "index": index}),
        )
        aliases: dict[str, str] = {}
        fresh_catalog: list[dict[str, Any]] = []
        for position, old_index in enumerate(order):
            candidate = dict(catalog[old_index])
            old_alias = str(candidate["catalog_id"])
            new_alias = f"P{position}_{seed[:10]}"
            candidate["catalog_id"] = new_alias
            aliases[old_alias] = new_alias
            fresh_catalog.append(candidate)
        fresh_payload = {
            "program_catalog": fresh_catalog,
            "target_native_structural_requirement": requirement,
        }
        fresh_target = execute_anonymous_selection(
            program_catalog=fresh_catalog, target_requirement=requirement,
        )
        if fresh_target != {
            "decision": SELECT_SKILL,
            "selected_catalog_id": aliases[selected_old_id],
            "reason": "UNIQUE_ANONYMOUS_STRUCTURAL_CONTRACT_MATCH",
        }:
            raise ValueError("fresh symbolic executor changed the source route")
        fresh_prompt = format_multi_ir_selector_prompt(fresh_payload)
        fresh_completion = json.dumps(
            fresh_target, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        )
        new_id = stable_hash({
            "reserve_salt": reserve_salt,
            "parent_example_id": old_id,
            "prompt_sha256": hashlib.sha256(fresh_prompt.encode()).hexdigest(),
            "completion": fresh_target,
        })
        row = dict(old)
        row.update({
            "example_id": new_id,
            "prompt": fresh_prompt,
            "completion": fresh_completion,
            "expected_catalog_id_audit_only": aliases[selected_old_id],
            "parent_example_id_audit_only": old_id,
            "fresh_presentation_reserve_audit_only": "V2",
        })
        new_rows.append(row)
        old_to_new[old_id] = new_id

    old_index_rows = _rows(parent_index)
    new_index_rows = []
    for old in old_index_rows:
        selector_ids = [str(value) for value in old["selector_example_ids"]]
        if any(value not in old_to_new for value in selector_ids):
            raise ValueError("native replay index references an unknown parent row")
        row = dict(old)
        row["selector_example_ids"] = [old_to_new[value] for value in selector_ids]
        row["parent_selector_example_ids_audit_only"] = selector_ids
        row["fresh_presentation_reserve_audit_only"] = "V2"
        new_index_rows.append(row)

    new_rows.sort(key=lambda row: str(row["example_id"]))
    new_index_rows.sort(key=lambda row: (
        str(row["benchmark"]), str(row["formal_task_id"]),
    ))
    new_prompts = {str(row["prompt"]) for row in new_rows}
    group_counts = Counter(
        str(row["target_eval_group_audit_only"]) for row in new_rows
    )
    expected_groups = {
        str(key): int(value)
        for key, value in config["expected_group_counts"].items()
    }
    forbidden_tokens = {
        "webshop", "discoveryworld", "tirbench", "alfworld", "clevrer",
        "agqa2", "native_action", "formal_task_id", "program_sha256",
    }
    model_text = "\n".join(
        str(row["prompt"]) + str(row["completion"]) for row in new_rows
    ).lower()
    gates = {
        "parent_preregistration_gate_clean": all(parent["gates"].values()),
        "fresh_reserve_frozen_before_v3_adapter": not forbidden_adapter.exists(),
        "route_rows_exact": len(new_rows) == int(config["expected_route_rows"]),
        "native_tasks_exact": len(new_index_rows) == int(config["expected_native_tasks"]),
        "exact_six_benchmark_groups": dict(sorted(group_counts.items())) == expected_groups,
        "all_prompts_and_ids_unique": (
            len(new_prompts) == len(new_rows)
            and len(old_to_new) == len(new_rows)
        ),
        "all_presentations_fresh": new_prompts.isdisjoint(old_prompts),
        "all_expected_completions_select_one_program": all(
            json.loads(str(row["completion"]))["decision"] == SELECT_SKILL
            for row in new_rows
        ),
        "agqa_has_two_source_primitives_per_task": group_counts["agqa2"] == 1800,
        "no_target_identity_or_program_hash_in_model_text": not any(
            token in model_text for token in forbidden_tokens
        ),
        "no_target_data_for_weight_updates": all(
            row["target_data_used_for_weight_updates"] is False for row in new_rows
        ),
        "formal_outcomes_not_parsed_or_used_by_freezer": True,
    }
    if not all(gates.values()):
        raise ValueError(f"fresh substitution freeze failed: {gates}")

    output_dir.mkdir(parents=True)
    dataset_path = output_dir / "route_selector_replay.jsonl"
    with dataset_path.open("w", encoding="utf-8") as stream:
        for row in new_rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    index_path = output_dir / "native_replay_index.jsonl"
    with index_path.open("w", encoding="utf-8") as stream:
        for row in new_index_rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    body = {
        "schema_version": "harness-9b-six-benchmark-fresh-presentation-preregistration-v2",
        "status": FRESH_STATUS,
        "authority": (
            "PARENT_PREOUTCOME_TASK_AND_NATIVE_AUTHORITIES;FRESH_MODEL_PRESENTATION_ONLY;"
            "NO_TARGET_OUTCOME_READ;FROZEN_BEFORE_V3_SOURCE_ONLY_WEIGHT_UPDATE"
        ),
        "config": {"path": str(config_path.resolve()), "sha256": _sha(config_path)},
        "parent_preregistration": {
            "path": str(parent_path.resolve()), "sha256": _sha(parent_path),
            "status": parent["status"],
        },
        "task_authorities": parent["task_authorities"],
        "route_authorities": parent["route_authorities"],
        "formal_evidence": parent["formal_evidence"],
        "source_contracts": parent["source_contracts"],
        "route_selector_replay": {
            "path": str(dataset_path.resolve()), "sha256": _sha(dataset_path),
            "rows": len(new_rows), "group_counts": dict(sorted(group_counts.items())),
        },
        "native_replay_index": {
            "path": str(index_path.resolve()), "sha256": _sha(index_path),
            "tasks": len(new_index_rows),
            "group_counts": dict(sorted(Counter(
                str(row["benchmark"]) for row in new_index_rows
            ).items())),
        },
        "preregistered_gates": parent["preregistered_gates"],
        "gates": gates,
        "claim_boundary": config["claim_boundary"],
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write_json(output_dir / "preregistration.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = freeze(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "status": manifest["status"],
        "route_selector_replay": manifest["route_selector_replay"],
        "native_replay_index": manifest["native_replay_index"],
        "gates": manifest["gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
