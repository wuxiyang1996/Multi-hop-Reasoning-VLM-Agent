#!/usr/bin/env python3
"""Freeze six-benchmark neural-controller substitution before model inference.

The task identities come only from manifests frozen before their original
target outcomes.  Existing formal reports are content-addressed but deliberately
not parsed by this freezer.  AGQA2 is represented as two independently selected
source primitives followed by its already-frozen target-native composition.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.multi_ir_selector_training import (  # noqa: E402
    SELECT_SKILL,
    anonymous_contract_payload,
    execute_anonymous_selection,
    format_multi_ir_selector_prompt,
    requirement_from_contract,
)
from motif_transfer.structural_ir_applicability import (  # noqa: E402
    SourceIRContract,
)
from scripts.build_harness_multi_ir_selector_sft_v1 import (  # noqa: E402
    _load_contracts,
)


DEFAULT_CONFIG = REPO / "configs/harness_9b_six_benchmark_substitution_v1.json"
DEFAULT_OUTPUT = REPO / "runs/harness_9b_six_benchmark_substitution_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _task_ids(
    benchmark: str, authority: Mapping[str, Any],
) -> tuple[list[str], str]:
    if benchmark == "webshop":
        if authority.get("status") != "FROZEN_BEFORE_ANY_V21_PROVIDER_CALL_OR_OUTCOME":
            raise ValueError("WebShop task authority is not pre-outcome frozen")
        rows = (authority.get("roles") or {}).get("formal_reserve") or ()
        return [str(row["task_id"]) for row in rows], "roles.formal_reserve"
    if benchmark == "discoveryworld":
        if authority.get("status") != "FROZEN_BEFORE_ANY_FRESH_TARGET_RESET_OR_OUTCOME":
            raise ValueError("DiscoveryWorld task authority is not pre-outcome frozen")
        return [str(row["task_id"]) for row in authority.get("tasks") or ()], "tasks"
    if benchmark == "tirbench":
        if authority.get("status") != "FROZEN_BEFORE_FRESH_QUALIFICATION":
            raise ValueError("TIRBench task authority is not pre-outcome frozen")
        return [str(value) for value in (authority.get("splits") or {}).get("heldout") or ()], "splits.heldout"
    if benchmark == "alfworld":
        if authority.get("status") != "FROZEN_BEFORE_ANY_ALFWORLD_V13_RESERVE_RESET_OR_OUTCOME":
            raise ValueError("ALFWorld task authority is not pre-outcome frozen")
        return [str(value) for value in authority.get("task_ids") or ()], "task_ids"
    if benchmark == "clevrer":
        if authority.get("status") != "FROZEN_BEFORE_CAUSAL_QUERY_DEVELOPMENT_OUTCOMES":
            raise ValueError("CLEVRER split authority is not pre-outcome frozen")
        family_roles = ((authority.get("benchmarks") or {}).get("clevrer") or {}).get("family_roles") or {}
        output = []
        for family in sorted(family_roles):
            output.extend(map(str, family_roles[family].get("reserve") or ()))
        return output, "benchmarks.clevrer.family_roles.*.reserve"
    if benchmark == "agqa2":
        if authority.get("status") != "FROZEN_V62_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V62_CALLS":
            raise ValueError("AGQA2 selection is not pre-outcome frozen")
        return [str(row["task_id"]) for row in authority.get("samples") or ()], "samples"
    raise ValueError(f"unsupported benchmark: {benchmark}")


def _seeded_catalog(
    contracts: Sequence[SourceIRContract], *, seed: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    order = sorted(
        contracts,
        key=lambda row: stable_hash({"seed": seed, "program": row.program_sha256}),
    )
    catalog = []
    aliases = {}
    seed_hash = stable_hash({"catalog_seed": seed})
    for index, contract in enumerate(order):
        alias = f"P{index}_{seed_hash[:10]}"
        catalog.append(anonymous_contract_payload(contract, catalog_id=alias))
        aliases[contract.program_sha256] = alias
    return catalog, aliases


def _route_components(spec: Mapping[str, Any]) -> list[str]:
    value = spec["source_program_sha256"]
    return [str(row) for row in value] if isinstance(value, list) else [str(value)]


def freeze(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    config = _read(config_path)
    if config.get("status") != (
        "FREEZE_REQUESTED_BEFORE_ANY_FIVE_SCHEMA_9B_SUBSTITUTION_INFERENCE_OR_WEIGHT_UPDATE"
    ):
        raise ValueError("substitution config is not in the pre-inference role")
    catalog_config_path = _resolve(str(config["source_catalog_config"]))
    catalog_config = _read(catalog_config_path)
    contracts, confirmations, source_inputs = _load_contracts(catalog_config)
    by_program = {row.program_sha256: row for row in contracts}
    if len(contracts) != 7 or len(by_program) != 7:
        raise ValueError("five-schema substitution requires exactly seven source programs")
    if {row.ir_kind for row in contracts} != set(catalog_config["required_ir_kinds"]):
        raise ValueError("source catalog does not contain the frozen five IR schemas")

    task_authorities: dict[str, Any] = {}
    formal_evidence: dict[str, Any] = {}
    task_ids_by_benchmark: dict[str, list[str]] = {}
    task_paths: dict[str, str] = {}
    route_authorities: dict[str, Any] = {}
    for benchmark, spec in config["benchmarks"].items():
        authority_path = _resolve(str(spec["task_authority"]))
        authority = _read(authority_path)
        task_ids, task_path = _task_ids(benchmark, authority)
        if len(task_ids) != int(spec["expected_tasks"]):
            raise ValueError(f"{benchmark} task count drifted: {len(task_ids)}")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"{benchmark} contains duplicate task identities")
        task_ids_by_benchmark[benchmark] = task_ids
        task_paths[benchmark] = task_path
        task_authorities[benchmark] = {
            "path": str(authority_path.resolve()), "sha256": _sha(authority_path),
            "status": authority.get("status"), "task_identity_path": task_path,
        }
        if spec.get("route_authority"):
            route_path = _resolve(str(spec["route_authority"]))
            route = _read(route_path)
            route_authorities[benchmark] = {
                "path": str(route_path.resolve()), "sha256": _sha(route_path),
                "status": route.get("status"),
            }
        # Hash existing formal artifacts without parsing their outcomes.  Their
        # contents are opened only by the later action-equivalence bridge.
        formal_evidence[benchmark] = [
            {"path": str(_resolve(str(value)).resolve()),
             "sha256": _sha(_resolve(str(value)))}
            for value in spec["formal_evidence"]
        ]

    expected_route_statuses = {
        "clevrer": "FROZEN_BEFORE_CLEVRER_V15_RESERVE_OUTCOMES",
        "agqa2": "FROZEN_BEFORE_ANY_V62_PROVIDER_OR_OUTCOME_CALL",
    }
    for benchmark, expected in expected_route_statuses.items():
        if route_authorities[benchmark]["status"] != expected:
            raise ValueError(f"{benchmark} route authority is not pre-outcome frozen")

    rows = []
    replay_index = []
    prompts = set()
    group_counts = Counter()
    component_counts = Counter()
    for benchmark, spec in sorted(config["benchmarks"].items()):
        components = _route_components(spec)
        missing = set(components) - set(by_program)
        if missing:
            raise ValueError(f"{benchmark} source program missing from catalog: {missing}")
        for task_id in task_ids_by_benchmark[benchmark]:
            task_examples = []
            for component_index, program_sha256 in enumerate(components):
                contract = by_program[program_sha256]
                component = f"C{component_index}:{contract.ir_kind}"
                seed = stable_hash({
                    "protocol": config["schema_version"],
                    "benchmark": benchmark,
                    "task_id": task_id,
                    "component": component,
                })
                catalog, aliases = _seeded_catalog(contracts, seed=seed)
                requirement = requirement_from_contract(contract)
                input_payload = {
                    "program_catalog": catalog,
                    "target_native_structural_requirement": requirement,
                }
                completion = execute_anonymous_selection(
                    program_catalog=catalog, target_requirement=requirement,
                )
                if completion != {
                    "decision": SELECT_SKILL,
                    "selected_catalog_id": aliases[program_sha256],
                    "reason": "UNIQUE_ANONYMOUS_STRUCTURAL_CONTRACT_MATCH",
                }:
                    raise ValueError("frozen selector executor did not choose expected source program")
                prompt = format_multi_ir_selector_prompt(input_payload)
                if prompt in prompts:
                    raise ValueError("duplicate substitution prompt")
                prompts.add(prompt)
                example_id = stable_hash({
                    "benchmark": benchmark, "task_id": task_id,
                    "route_id": spec["route_id"], "component": component,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "completion": completion,
                })
                row = {
                    "example_id": example_id,
                    "objective": "SELECT_TRANSFER_PROGRAM",
                    "prompt": prompt,
                    "completion": json.dumps(
                        completion, sort_keys=True, ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "target_eval_group_audit_only": benchmark,
                    "formal_task_id_audit_only": task_id,
                    "route_id_audit_only": str(spec["route_id"]),
                    "route_component_audit_only": component,
                    "source_ir_kind_audit_only": contract.ir_kind,
                    "expected_program_sha256_audit_only": program_sha256,
                    "expected_catalog_id_audit_only": aliases[program_sha256],
                    "target_data_used_for_weight_updates": False,
                    "target_outcome_used_for_selection": False,
                }
                rows.append(row)
                task_examples.append(example_id)
                group_counts[benchmark] += 1
                component_counts[f"{benchmark}/{contract.ir_kind}"] += 1
            replay_index.append({
                "benchmark": benchmark,
                "formal_task_id": task_id,
                "route_id": str(spec["route_id"]),
                "source_program_sha256": components,
                "target_native_composition": spec.get("target_native_composition"),
                "selector_example_ids": task_examples,
                "replay_granularity": str(spec["replay_granularity"]),
                "task_authority_sha256": task_authorities[benchmark]["sha256"],
                "formal_evidence_sha256": [
                    row["sha256"] for row in formal_evidence[benchmark]
                ],
                "formal_outcome_used_for_task_selection": False,
            })

    rows.sort(key=lambda row: str(row["example_id"]))
    replay_index.sort(key=lambda row: (row["benchmark"], row["formal_task_id"]))
    expected_group_counts = {
        name: int(spec["expected_tasks"]) * len(_route_components(spec))
        for name, spec in config["benchmarks"].items()
    }
    serialized_model_text = "\n".join(
        row["prompt"] + row["completion"] for row in rows
    )
    forbidden_model_tokens = {
        "webshop", "discoveryworld", "tirbench", "alfworld", "clevrer",
        "agqa2", "native_action", "formal_task_id", "program_sha256",
    }
    gates = {
        "exact_six_benchmark_groups": set(group_counts) == set(config["benchmarks"]),
        "task_counts_exact": all(
            len(task_ids_by_benchmark[name]) == int(spec["expected_tasks"])
            for name, spec in config["benchmarks"].items()
        ),
        "route_component_counts_exact": dict(group_counts) == expected_group_counts,
        "all_task_ids_unique_within_group": all(
            len(values) == len(set(values))
            for values in task_ids_by_benchmark.values()
        ),
        "all_prompts_unique": len(prompts) == len(rows),
        "all_expected_completions_select_one_program": all(
            json.loads(row["completion"])["decision"] == SELECT_SKILL
            for row in rows
        ),
        "agqa_has_two_source_primitives_per_task": group_counts["agqa2"] == 1800,
        "five_ir_schemas_present": len({
            row["source_ir_kind_audit_only"] for row in rows
        }) == 5,
        "no_target_identity_or_program_hash_in_model_text": not any(
            token in serialized_model_text.lower() for token in forbidden_model_tokens
        ),
        "no_target_data_for_weight_updates": all(
            row["target_data_used_for_weight_updates"] is False for row in rows
        ),
        "formal_outcomes_not_parsed_or_used_by_freezer": True,
        "five_schema_adapter_not_yet_created": not (
            REPO / "runs/harness_controller_qwen35_9b_mixed_v2"
        ).exists(),
    }
    if not all(gates.values()):
        raise ValueError(f"six-benchmark substitution freeze failed: {gates}")

    output_dir.mkdir(parents=True)
    dataset_path = output_dir / "route_selector_replay.jsonl"
    with dataset_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    index_path = output_dir / "native_replay_index.jsonl"
    with index_path.open("w", encoding="utf-8") as stream:
        for row in replay_index:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    manifest_body = {
        "schema_version": "harness-9b-six-benchmark-substitution-preregistration-v1",
        "status": "FROZEN_BEFORE_FIVE_SCHEMA_9B_WEIGHT_UPDATE_OR_SUBSTITUTION_INFERENCE",
        "authority": (
            "ALL_TASK_IDENTITIES_FROM_PREOUTCOME_MANIFESTS;NO_TARGET_OUTCOME_FILTER;"
            "SOURCE_ONLY_SEVEN_PROGRAM_FIVE_SCHEMA_CATALOG;EXISTING_FORMAL_REPORTS_HASHED_NOT_PARSED"
        ),
        "config": {"path": str(config_path.resolve()), "sha256": _sha(config_path)},
        "source_catalog_config": {
            "path": str(catalog_config_path.resolve()), "sha256": _sha(catalog_config_path),
        },
        "source_inputs": source_inputs,
        "source_contracts": [
            {
                "program_sha256": row.program_sha256,
                "contract_sha256": row.contract_sha256,
                "ir_kind": row.ir_kind,
                "confirmation_sha256": confirmations[row.program_sha256],
            }
            for row in sorted(contracts, key=lambda value: value.program_sha256)
        ],
        "task_authorities": task_authorities,
        "route_authorities": route_authorities,
        "formal_evidence": formal_evidence,
        "route_selector_replay": {
            "path": str(dataset_path.resolve()), "sha256": _sha(dataset_path),
            "rows": len(rows), "group_counts": dict(sorted(group_counts.items())),
            "component_counts": dict(sorted(component_counts.items())),
        },
        "native_replay_index": {
            "path": str(index_path.resolve()), "sha256": _sha(index_path),
            "tasks": len(replay_index),
            "group_counts": dict(sorted(Counter(
                row["benchmark"] for row in replay_index
            ).items())),
        },
        "preregistered_gates": config["preregistered_gates"],
        "gates": gates,
        "claim_boundary": config["claim_boundary"],
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    _write(output_dir / "preregistration.json", manifest)
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
