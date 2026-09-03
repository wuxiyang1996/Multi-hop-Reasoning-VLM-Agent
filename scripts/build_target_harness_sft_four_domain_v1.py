#!/usr/bin/env python3
"""Extend the frozen Harness SFT pilot to DiscoveryWorld and TIRBench."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from build_target_harness_sft_pilot_v1 import (  # noqa: E402
    SYSTEM,
    _candidate_pairs,
    _format_example,
    _json_text,
    _load_programs,
    _read,
    _resolve,
    _select_pairs,
    _serialized_probability,
    _sha256,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_source_function_induction import QUALIFIED  # noqa: E402
from motif_transfer.phase3_tir_nonmaze import (  # noqa: E402
    predict_candidate_effects,
    validate_grounder_artifact as validate_tir_grounder,
)
from motif_transfer.phase3_typed_effect_induction import TYPED_EFFECTS  # noqa: E402
from motif_transfer.target_harness_sft import GroundedTargetState  # noqa: E402


DW_FEATURE_NAMES = (
    "step_fraction",
    "inventory_fraction",
    "accessible_fraction",
    "movable_fraction",
    "salient_fraction",
    "teleport_fraction",
    "candidate_feasible",
    "option_pickup",
    "option_teleport",
    "option_use",
    "history_pickup_fraction",
    "history_teleport_fraction",
    "history_use_fraction",
    "previous_pickup",
    "previous_teleport",
    "previous_use",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _validate_stable_hash(row: Mapping[str, Any], field: str) -> None:
    body = dict(row)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"stable-hash mismatch for {field}")


def _dw_facts(step: Mapping[str, Any]) -> Mapping[str, Any]:
    facts = step.get("before_target_native_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("DiscoveryWorld step omitted target-native facts")
    return facts


def _dw_option_feasible(option: str, facts: Mapping[str, Any]) -> bool:
    inventory = list(facts.get("inventory") or ())
    accessible = list(facts.get("accessible_objects") or ())
    salient = list(facts.get("salient_relative_objects") or ())
    if option == "TELEPORT_TO_OBJECT":
        return bool(salient)
    if option == "PICKUP":
        return bool(accessible)
    if option == "USE":
        return bool(inventory) and bool(accessible)
    raise ValueError(f"unsupported DiscoveryWorld option: {option}")


def _dw_features(
    *, option: str, facts: Mapping[str, Any], step_index: int,
    maximum_steps: int, history: Sequence[str],
) -> list[float]:
    inventory = list(facts.get("inventory") or ())
    accessible = list(facts.get("accessible_objects") or ())
    salient = list(facts.get("salient_relative_objects") or ())
    teleport = list(facts.get("teleport_location_names") or ())
    location = facts.get("agent_location") or {}
    movable = list(location.get("directions_you_can_move") or ())
    denominator = max(1, len(history))
    previous = history[-1] if history else ""
    values = {
        "step_fraction": step_index / max(1, maximum_steps),
        "inventory_fraction": min(len(inventory), 4) / 4.0,
        "accessible_fraction": min(len(accessible), 12) / 12.0,
        "movable_fraction": min(len(movable), 4) / 4.0,
        "salient_fraction": min(len(salient), 32) / 32.0,
        "teleport_fraction": min(len(teleport), 12) / 12.0,
        "candidate_feasible": float(_dw_option_feasible(option, facts)),
        "option_pickup": float(option == "PICKUP"),
        "option_teleport": float(option == "TELEPORT_TO_OBJECT"),
        "option_use": float(option == "USE"),
        "history_pickup_fraction": history.count("PICKUP") / denominator,
        "history_teleport_fraction": history.count("TELEPORT_TO_OBJECT") / denominator,
        "history_use_fraction": history.count("USE") / denominator,
        "previous_pickup": float(previous == "PICKUP"),
        "previous_teleport": float(previous == "TELEPORT_TO_OBJECT"),
        "previous_use": float(previous == "USE"),
    }
    return [float(values[name]) for name in DW_FEATURE_NAMES]


def _discoveryworld_supervision(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    authority_path = _resolve(str(config["authority_config"]))
    if _sha256(authority_path) != str(config["authority_config_sha256"]):
        raise SystemExit("DiscoveryWorld development authority hash mismatch")
    authority = _read(authority_path)
    if (
        authority.get("status") != "DEVELOPMENT_ON_CONSUMED_SEEDS_ONLY"
        or authority.get("role") != "development"
    ):
        raise SystemExit("DiscoveryWorld authority is not development-only")
    authorized = {str(row["task_id"]) for row in authority["tasks"]}
    train = set(map(str, config["train_tasks"]))
    validation = set(map(str, config["validation_tasks"]))
    if train & validation or train | validation != authorized:
        raise SystemExit("DiscoveryWorld SFT split does not match authority tasks")
    options = tuple(map(str, config["candidate_options"]))
    if set(options) != {"PICKUP", "TELEPORT_TO_OBJECT", "USE"}:
        raise SystemExit("DiscoveryWorld option vocabulary drift")

    states: list[dict[str, Any]] = []
    inputs = []
    root = _resolve(str(config["episode_root"]))
    for task_id, expected_sha in sorted(config["episode_file_sha256"].items()):
        path = root / f"{task_id}.json"
        if not path.is_file() or _sha256(path) != str(expected_sha):
            raise SystemExit(f"DiscoveryWorld episode hash mismatch: {path}")
        episode = _read(path)
        if str(episode.get("task_id")) != task_id:
            raise SystemExit("DiscoveryWorld episode task identity mismatch")
        if (
            episode.get("status") != "TARGET_ONLY_EPISODE_COMPLETE"
            or episode.get("evaluator_finalized") is not False
            or episode.get("formal_outcome_read_by_acquisition_or_fork_selection")
            is not False
            or episode.get("policy_runtime_saw_oracle_scorecard") is not False
            or int(episode.get("schema_fallback_steps", -1)) != 0
            or int(episode.get("invalid_native_actions", -1)) != 0
        ):
            raise SystemExit("DiscoveryWorld episode crossed the SFT authority boundary")
        steps = list(episode.get("steps") or ())
        if not steps:
            raise SystemExit("empty DiscoveryWorld development episode")
        observed_options = [str(step["action"]["action"]) for step in steps]
        if not set(observed_options) <= set(options):
            raise SystemExit("DiscoveryWorld episode contains unknown option")
        split = "train" if task_id in train else "validation"
        inputs.append({"task_id": task_id, "path": str(path), "sha256": _sha256(path)})
        history: list[str] = []
        for index, step in enumerate(steps):
            transition = step.get("transition")
            if not isinstance(transition, Mapping):
                raise SystemExit("DiscoveryWorld step omitted transition receipt")
            _validate_stable_hash(transition, "receipt_sha256")
            future = observed_options[index:index + 8]
            successor_facts = [_dw_facts(row) for row in steps[index:index + 4]]
            rows = []
            for option in options:
                rows.append({
                    "option": option,
                    "option_sha256": stable_hash({"target_native_option": option}),
                    "features": _dw_features(
                        option=option, facts=_dw_facts(step), step_index=index,
                        maximum_steps=int(authority["runtime"]["maximum_acquisition_steps"]),
                        history=history,
                    ),
                    "labels": {
                        TYPED_EFFECTS[0]: int(option in future[:1]),
                        TYPED_EFFECTS[1]: int(option in future[:4]),
                        TYPED_EFFECTS[2]: int(option in future[:8]),
                        TYPED_EFFECTS[3]: int(
                            bool(successor_facts)
                            and all(_dw_option_feasible(option, facts) for facts in successor_facts)
                        ),
                    },
                })
            states.append({
                "task_id": task_id,
                "task_sha256": stable_hash({"target_task": task_id}),
                "split": split,
                "state_receipt_sha256": str(transition["receipt_sha256"]),
                "rows": rows,
            })
            history.append(observed_options[index])
    return states, inputs


def _fit_discoveryworld_grounder(
    states: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, LogisticRegression]], dict[str, Any]]:
    train_rows = [row for state in states if state["split"] == "train" for row in state["rows"]]
    validation_rows = [
        row for state in states if state["split"] == "validation" for row in state["rows"]
    ]
    models = {}
    heads = {}
    metrics = {}
    for offset, effect in enumerate(TYPED_EFFECTS):
        train_x = np.asarray([row["features"] for row in train_rows], dtype=float)
        train_y = np.asarray([row["labels"][effect] for row in train_rows], dtype=int)
        validation_x = np.asarray([row["features"] for row in validation_rows], dtype=float)
        validation_y = np.asarray([row["labels"][effect] for row in validation_rows], dtype=int)
        if set(train_y) != {0, 1} or set(validation_y) != {0, 1}:
            raise SystemExit(f"DiscoveryWorld typed head has one class: {effect}")
        means = train_x.mean(axis=0)
        scales = train_x.std(axis=0)
        scales[scales < 1e-8] = 1.0
        model = LogisticRegression(
            C=float(config["regularization_c"]), class_weight="balanced",
            solver="liblinear", max_iter=int(config["maximum_iterations"]),
            random_state=int(config["seed"]) + offset,
        ).fit((train_x - means) / scales, train_y)
        scores = model.predict_proba((validation_x - means) / scales)[:, 1]
        auc = float(roc_auc_score(validation_y, scores))
        metrics[effect] = {
            "training_examples": len(train_y),
            "training_positive_rate": float(np.mean(train_y)),
            "validation_examples": len(validation_y),
            "validation_positive_rate": float(np.mean(validation_y)),
            "validation_auc": auc,
        }
        heads[effect] = {
            "kind": "standardized-balanced-logistic-regression-v1",
            "feature_names": list(DW_FEATURE_NAMES),
            "means": means.tolist(),
            "scales": scales.tolist(),
            "weights": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
        }
        models[effect] = (means, scales, model)
    aucs = [metrics[name]["validation_auc"] for name in TYPED_EFFECTS]
    gates = {
        "every_head_validation_auc_at_least_threshold": min(aucs)
        >= float(config["minimum_per_head_validation_auc"]),
        "macro_validation_auc_at_least_threshold": float(np.mean(aucs))
        >= float(config["minimum_macro_validation_auc"]),
    }
    artifact = {
        "schema_version": "discoveryworld-target-typed-effect-grounder-v1",
        "status": "QUALIFIED" if all(gates.values()) else "BLOCKED",
        "authority": "CONSUMED_DEVELOPMENT_STRUCTURED_ACQUISITION_ONLY;NO_SUCCESS_LABELS",
        "effect_types": list(TYPED_EFFECTS),
        "label_granularity": "target_native_option",
        "label_contract": {
            TYPED_EFFECTS[0]: "OPTION_APPEARS_IN_CONTINUATION_BY_1",
            TYPED_EFFECTS[1]: "OPTION_APPEARS_IN_CONTINUATION_BY_4",
            TYPED_EFFECTS[2]: "OPTION_APPEARS_IN_CONTINUATION_BY_8",
            TYPED_EFFECTS[3]: "OPTION_REMAINS_FEASIBLE_ACROSS_UP_TO_4_SUCCESSORS",
        },
        "heads": heads,
        "head_validation": metrics,
        "thresholds": {
            "minimum_per_head_validation_auc": float(config["minimum_per_head_validation_auc"]),
            "minimum_macro_validation_auc": float(config["minimum_macro_validation_auc"]),
        },
        "macro_validation_auc": float(np.mean(aucs)),
        "gates": gates,
        "formal_or_qualification_data_used": False,
        "reward_or_success_used": False,
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    if not all(gates.values()):
        raise SystemExit(f"DiscoveryWorld typed grounder gate failed: {artifact}")
    return models, artifact


def _candidate_subsets(rows: Sequence[Mapping[str, Any]], sizes: Sequence[int]):
    ordered = sorted(rows, key=lambda row: str(row["candidate_key"]))
    for size in sizes:
        for subset in combinations(ordered, int(size)):
            yield subset


def _ground_discoveryworld(
    *, states: Sequence[Mapping[str, Any]], models: Mapping[str, Any],
    artifact: Mapping[str, Any], subset_sizes: Sequence[int], precision: int,
    candidate_order_variants: int,
) -> list[GroundedTargetState]:
    if candidate_order_variants not in {1, 2}:
        raise ValueError("DiscoveryWorld candidate_order_variants must be 1 or 2")
    output = []
    for state in states:
        predicted = []
        for row in state["rows"]:
            vector = np.asarray(row["features"], dtype=float).reshape(1, -1)
            effects = {}
            for effect in TYPED_EFFECTS:
                means, scales, model = models[effect]
                probability = model.predict_proba((vector - means) / scales)[0, 1]
                effects[effect] = _serialized_probability(probability, precision)
            predicted.append({
                "candidate_key": row["option_sha256"],
                "effects": effects,
            })
        for subset in _candidate_subsets(predicted, subset_sizes):
            orders = [tuple(subset)]
            if candidate_order_variants == 2:
                reversed_order = tuple(reversed(subset))
                if reversed_order != orders[0]:
                    orders.append(reversed_order)
            for ordered_subset in orders:
                candidates = tuple({
                    "candidate_id": f"C{index}", "effects": dict(row["effects"]),
                } for index, row in enumerate(ordered_subset))
                grounded = GroundedTargetState(
                    target_domain="discoveryworld",
                    target_task_sha256=str(state["task_sha256"]),
                    split=str(state["split"]),
                    state_receipt_sha256=stable_hash({
                        "transition_receipt": state["state_receipt_sha256"],
                        "candidate_order": [
                            row["candidate_key"] for row in ordered_subset
                        ],
                        "grounder": artifact["artifact_sha256"],
                    }),
                    grounder_artifact_sha256=str(artifact["artifact_sha256"]),
                    candidates=candidates,
                )
                if not grounded.validate():
                    raise SystemExit("invalid DiscoveryWorld grounded state")
                output.append(grounded)
    return output


def _load_build_config(path: Path) -> dict[str, Any]:
    config = _read(path)
    if config.get("schema_version") != "target-harness-sft-four-domain-overlay-v2":
        return config
    base_path = _resolve(str(config["base_config"]))
    if _sha256(base_path) != str(config["base_config_sha256"]):
        raise SystemExit("four-domain V2 base config hash mismatch")
    base = deepcopy(_read(base_path))
    base["schema_version"] = str(config["resolved_schema_version"])
    base["status"] = str(config["status"])
    base["supersedes"] = str(config["base_config"])
    base["supersession_reason"] = str(config["supersession_reason"])
    base["discoveryworld"]["candidate_order_variants"] = int(
        config["overrides"]["discoveryworld"]["candidate_order_variants"]
    )
    return base


def _ground_tirbench(
    config: Mapping[str, Any], *, precision: int,
) -> tuple[list[GroundedTargetState], dict[str, Any]]:
    split_path = _resolve(str(config["split_manifest"]))
    if _sha256(split_path) != str(config["split_manifest_sha256"]):
        raise SystemExit("TIR split manifest hash mismatch")
    split_manifest = _read(split_path)
    split_body = dict(split_manifest)
    claimed = str(split_body.pop("config_sha256", ""))
    if stable_hash(split_body) != claimed:
        raise SystemExit("TIR split manifest self-hash mismatch")
    artifact_path = _resolve(str(config["grounder_artifact"]))
    if _sha256(artifact_path) != str(config["grounder_artifact_file_sha256"]):
        raise SystemExit("TIR grounder file hash mismatch")
    artifact = _read(artifact_path)
    validate_tir_grounder(artifact)
    if (
        artifact.get("status") != "DEVELOPMENT_GROUNDER_FROZEN_BEFORE_NEW_HOLDOUT"
        or artifact.get("artifact_sha256") != config["grounder_artifact_sha256"]
        or artifact.get("formal_outcome_read_for_training_or_calibration") is not False
        or artifact.get("source_program_updated") is not False
        or int(artifact["training_audit"]["qualification_tasks_read"]) != 0
        or int(artifact["training_audit"]["formal_tasks_read"]) != 0
    ):
        raise SystemExit("TIR grounder crossed the target authority boundary")

    states = []
    input_receipts = []
    paths = {
        "train": _resolve(str(config["development_train_receipts"])),
        "validation": _resolve(str(config["development_validation_receipts"])),
    }
    expected_hashes = {
        "train": str(config["development_train_receipts_sha256"]),
        "validation": str(config["development_validation_receipts_sha256"]),
    }
    manifest_keys = {"train": "development_train", "validation": "development_validation"}
    for split, path in paths.items():
        if _sha256(path) != expected_hashes[split]:
            raise SystemExit(f"TIR {split} receipt hash mismatch")
        receipts = json.loads(path.read_text(encoding="utf-8"))
        expected_ids = list(map(str, split_manifest["splits"][manifest_keys[split]]))
        if [str(row["sample_id"]) for row in receipts] != expected_ids:
            raise SystemExit(f"TIR {split} receipt IDs escaped the frozen split")
        input_receipts.append({"split": split, "path": str(path), "sha256": _sha256(path)})
        for receipt in receipts:
            _validate_stable_hash(receipt, "receipt_sha256")
            if (
                receipt.get("formal_outcome_exposed_to_neural_calls") is not False
                or receipt.get("source_program_or_identity_exposed_to_neural_calls") is not False
            ):
                raise SystemExit("TIR development neural calls saw forbidden authority")
            scored = []
            for candidate in receipt.get("candidates") or ():
                effects = predict_candidate_effects(
                    candidate, artifact=artifact,
                    image_size=receipt["image_size"],
                    routing=receipt["wrapper_routing"],
                )
                scored.append({
                    "candidate_key": str(candidate["candidate_id"]),
                    "effects": {
                        name: _serialized_probability(effects[name], precision)
                        for name in TYPED_EFFECTS
                    },
                })
            for subset in _candidate_subsets(scored, config["candidate_subset_sizes"]):
                candidates = tuple({
                    "candidate_id": f"C{index}", "effects": dict(row["effects"]),
                } for index, row in enumerate(subset))
                grounded = GroundedTargetState(
                    target_domain="tirbench",
                    target_task_sha256=stable_hash({
                        "target_task": str(receipt["sample_id"]),
                    }),
                    split=split,
                    state_receipt_sha256=stable_hash({
                        "receipt": str(receipt["receipt_sha256"]),
                        "candidate_subset": [row["candidate_key"] for row in subset],
                        "grounder": artifact["artifact_sha256"],
                    }),
                    grounder_artifact_sha256=str(artifact["artifact_sha256"]),
                    candidates=candidates,
                )
                if not grounded.validate():
                    raise SystemExit("invalid TIR grounded state")
                states.append(grounded)
    return states, {
        "grounder_path": str(artifact_path),
        "grounder_file_sha256": _sha256(artifact_path),
        "grounder_artifact_sha256": str(artifact["artifact_sha256"]),
        "input_receipts": input_receipts,
        "raw_target_tasks": sum(len(split_manifest["splits"][name]) for name in (
            "development_train", "development_validation",
        )),
        "grounded_candidate_set_variants": len(states),
        "qualification_tasks_read": 0,
        "formal_tasks_read": 0,
        "target_outcomes_used_for_controller_labels": False,
    }


def _load_parent(config: Mapping[str, Any]):
    root = _resolve(str(config["directory"]))
    paths = {
        "manifest": root / "manifest.json",
        "structured": root / "structured.jsonl",
        "train": root / "train.jsonl",
        "validation": root / "validation.jsonl",
    }
    for name, path in paths.items():
        expected = str(config[f"{name}_sha256"])
        if not path.is_file() or _sha256(path) != expected:
            raise SystemExit(f"parent two-domain dataset {name} hash mismatch")
    manifest = _read(paths["manifest"])
    if (
        manifest.get("status") != "FROZEN_TARGET_DEVELOPMENT_HARNESS_SUPERVISION"
        or not all(manifest.get("gates", {}).values())
        or manifest.get("target_outcome_used_for_controller_labels") is not False
        or manifest.get("formal_or_qualification_targets_used") is not False
    ):
        raise SystemExit("parent two-domain dataset is not qualified")
    return (
        manifest, _read_jsonl(paths["structured"]),
        {split: _read_jsonl(paths[split]) for split in ("train", "validation")},
        paths,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/target_harness_sft_four_domain_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/target_harness_sft_four_domain_v1",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    config = _load_build_config(args.config)
    if config.get("status") != "FROZEN_BEFORE_FOUR_DOMAIN_DATASET_BUILD":
        raise SystemExit("four-domain target Harness config is not frozen")
    precision = int(config["controller_training"]["probability_decimal_places"])

    parent_manifest, parent_structured, parent_model_rows, parent_paths = _load_parent(
        config["parent_two_domain_dataset"]
    )
    programs, program_receipts = _load_programs(
        _resolve(str(config["source_program_root"])),
        tuple(map(str, config["source_families"])),
    )
    qualified_families = sorted(
        family for family, program in programs.items()
        if program.get("status") == QUALIFIED
    )

    dw_states_raw, dw_inputs = _discoveryworld_supervision(config["discoveryworld"])
    dw_models, dw_grounder = _fit_discoveryworld_grounder(
        dw_states_raw, config["discoveryworld"]["grounder"],
    )
    dw_states = _ground_discoveryworld(
        states=dw_states_raw, models=dw_models, artifact=dw_grounder,
        subset_sizes=config["discoveryworld"]["candidate_subset_sizes"],
        precision=precision,
        candidate_order_variants=int(
            config["discoveryworld"].get("candidate_order_variants", 1)
        ),
    )
    tir_states, tir_audit = _ground_tirbench(
        config["tirbench"], precision=precision,
    )
    all_pairs = _candidate_pairs([*dw_states, *tir_states], programs)
    forbidden_inputs = {
        _json_text(row["input_payload"]) for row in parent_structured
    }
    selected_pairs = []
    quota = config["quota_per_added_domain"]
    for split, quota_key in (
        ("train", "train_authentic"),
        ("validation", "validation_authentic"),
    ):
        for domain in ("discoveryworld", "tirbench"):
            selected_pairs.extend(_select_pairs(
                all_pairs, target_domain=domain, split=split,
                quota=int(quota[quota_key]), forbidden_prompts=forbidden_inputs,
            ))
    added_examples = [row for pair in selected_pairs for row in pair]
    if not added_examples or not all(row.validate() for row in added_examples):
        raise SystemExit("added target Harness examples failed validation")

    added_structured = [asdict(row) for row in added_examples]
    all_structured = [*parent_structured, *added_structured]
    all_model_rows = {
        split: [*parent_model_rows[split], *[
            _format_example(row) for row in added_examples if row.split == split
        ]]
        for split in ("train", "validation")
    }
    domains = {str(row["target_domain"]) for row in all_structured}
    pairs_by_id = Counter(str(row["pair_id"]) for row in all_structured)
    domain_split_kind_counts = Counter(
        (
            str(row["target_domain"]), str(row["split"]),
            "authentic" if row["control_variant"]
            == "AUTHENTIC_TARGET_NEURAL_GROUNDING" else "control",
        )
        for row in all_structured
    )
    prompts = [row["prompt"] for rows in all_model_rows.values() for row in rows]
    prompt_text = "\n".join(prompts).lower()
    completion_text = "\n".join(
        row["completion"] for rows in all_model_rows.values() for row in rows
    ).lower()
    task_sets = {
        (domain, split): {
            str(row["target_task_sha256"]) for row in all_structured
            if row["target_domain"] == domain and row["split"] == split
        }
        for domain in domains for split in ("train", "validation")
    }
    expected = {"train": 450, "validation": 150}
    source_families = {str(row["source_family"]) for row in all_structured}
    controls = {str(row["control_variant"]) for row in all_structured}
    forbidden_tokens = (
        "webshop", "alfworld", "discoveryworld", "tirbench",
        "selected_action", "expert_action", "official_success",
        "official_reward", "native_actions", "target_domain",
        "teleport_to_object", "pickup", '"use"', "zoom_region",
        "extract_colors", "go to ", "click('", "fill('", "move ", "take ",
    )
    gates = {
        "exact_four_target_domains": domains
        == {"webshop", "alfworld", "discoveryworld", "tirbench"},
        "all_added_examples_valid": all(row.validate() for row in added_examples),
        "exact_one_control_per_authentic": set(pairs_by_id.values()) == {2},
        "exact_domain_split_kind_quotas": all(
            domain_split_kind_counts[(domain, split, kind)] == count
            for domain in domains for split, count in expected.items()
            for kind in ("authentic", "control")
        ),
        "example_ids_unique": len({row["example_id"] for row in all_structured})
        == len(all_structured),
        "prompts_unique": len(set(prompts)) == len(prompts),
        "train_validation_prompts_disjoint": {
            row["prompt"] for row in all_model_rows["train"]
        }.isdisjoint({row["prompt"] for row in all_model_rows["validation"]}),
        "target_tasks_disjoint_across_splits": all(
            task_sets[(domain, "train")].isdisjoint(task_sets[(domain, "validation")])
            for domain in domains
        ),
        "target_and_native_identity_absent_from_prompts": not any(
            token in prompt_text for token in forbidden_tokens
        ),
        "source_identity_absent_from_prompts": not any(
            family.lower() in prompt_text for family in source_families
        ),
        "control_identity_absent_from_prompts": not any(
            control.lower() in prompt_text for control in controls
        ),
        "reward_success_native_action_absent_from_completions": not any(
            token in completion_text for token in ("reward", "success", "native_action")
        ),
        "discoveryworld_grounder_qualified_before_export": dw_grounder["status"]
        == "QUALIFIED",
        "discoveryworld_development_only": all(
            row["task_id"] in set(config["discoveryworld"]["train_tasks"])
            | set(config["discoveryworld"]["validation_tasks"])
            for row in dw_inputs
        ),
        "tir_development_only": tir_audit["qualification_tasks_read"] == 0
        and tir_audit["formal_tasks_read"] == 0,
        "target_outcomes_not_used_for_controller_labels": all(
            row.get("target_outcome_used") is False for row in all_structured
        ),
        "parent_two_domain_gates_preserved": all(parent_manifest["gates"].values()),
    }
    if not all(gates.values()):
        raise SystemExit(f"four-domain target Harness gates failed: {gates}")

    args.output_dir.mkdir(parents=True)
    structured_path = args.output_dir / "structured.jsonl"
    with structured_path.open("w", encoding="utf-8") as stream:
        for row in sorted(all_structured, key=lambda value: value["example_id"]):
            stream.write(_json_text(row) + "\n")
    split_paths = {}
    for split, rows in all_model_rows.items():
        path = args.output_dir / f"{split}.jsonl"
        split_paths[split] = path
        with path.open("w", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda value: value["example_id"]):
                stream.write(_json_text(row) + "\n")
    grounder_path = args.output_dir / "discoveryworld_typed_grounder.json"
    grounder_path.write_text(
        json.dumps(dw_grounder, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    supervision_path = args.output_dir / "discoveryworld_grounder_supervision.jsonl"
    with supervision_path.open("w", encoding="utf-8") as stream:
        for state in dw_states_raw:
            for row in state["rows"]:
                stream.write(_json_text({
                    "target_task_sha256": state["task_sha256"],
                    "split": state["split"],
                    "state_receipt_sha256": state["state_receipt_sha256"],
                    "option_sha256": row["option_sha256"],
                    "features": row["features"],
                    "labels": row["labels"],
                }) + "\n")

    summary = {
        "examples": len(all_structured),
        "pairs": len(all_structured) // 2,
        "target_domains": sorted(domains),
        "qualified_source_families": qualified_families,
        "domain_split_kind_counts": {
            "/".join(key): value for key, value in sorted(domain_split_kind_counts.items())
        },
        "decision_counts": dict(sorted(Counter(
            str(row["target_payload"]["decision"]) for row in all_structured
        ).items())),
        "control_variant_counts": dict(sorted(Counter(
            str(row["control_variant"]) for row in all_structured
        ).items())),
        "source_family_counts": dict(sorted(Counter(
            str(row["source_family"]) for row in all_structured
        ).items())),
        "candidate_state_counts": {
            "webshop": parent_manifest["summary"]["candidate_state_counts"]["webshop"],
            "alfworld": parent_manifest["summary"]["candidate_state_counts"]["alfworld"],
            "discoveryworld_raw_transitions": len(dw_states_raw),
            "discoveryworld_candidate_set_variants": len(dw_states),
            "tirbench_raw_tasks": tir_audit["raw_target_tasks"],
            "tirbench_candidate_set_variants": len(tir_states),
        },
    }
    manifest = {
        "schema_version": "target-harness-sft-four-domain-dataset-v1",
        "status": "FROZEN_FOUR_TARGET_DEVELOPMENT_HARNESS_SUPERVISION",
        "authority": (
            "FOUR_TARGET_NATIVE_NEURAL_GROUNDERS;"
            "LABELS_FROM_FROZEN_SOURCE_PROGRAM_EXECUTOR;NO_FORMAL_TARGET_OUTCOME"
        ),
        "build_config": str(args.config.resolve()),
        "build_config_sha256": _sha256(args.config),
        "parent_two_domain_dataset": {
            "manifest": str(parent_paths["manifest"]),
            "manifest_sha256": _sha256(parent_paths["manifest"]),
            "structured_sha256": _sha256(parent_paths["structured"]),
            "train_sha256": _sha256(parent_paths["train"]),
            "validation_sha256": _sha256(parent_paths["validation"]),
        },
        "source_program_receipts": program_receipts,
        "discoveryworld": {
            "authority_config": str(_resolve(config["discoveryworld"]["authority_config"])),
            "input_episode_receipts": dw_inputs,
            "raw_transitions": len(dw_states_raw),
            "candidate_set_variants": len(dw_states),
            "grounder_artifact": str(grounder_path.resolve()),
            "grounder_artifact_sha256": str(dw_grounder["artifact_sha256"]),
            "grounder_file_sha256": _sha256(grounder_path),
            "grounder_validation": dw_grounder["head_validation"],
            "grounder_macro_validation_auc": dw_grounder["macro_validation_auc"],
            "formal_or_qualification_data_used": False,
        },
        "tirbench": tir_audit,
        "files": {
            "structured": {"path": str(structured_path.resolve()), "sha256": _sha256(structured_path)},
            "train": {"path": str(split_paths["train"].resolve()), "sha256": _sha256(split_paths["train"])},
            "validation": {"path": str(split_paths["validation"].resolve()), "sha256": _sha256(split_paths["validation"])},
            "discoveryworld_grounder_supervision": {
                "path": str(supervision_path.resolve()), "sha256": _sha256(supervision_path),
            },
        },
        "summary": summary,
        "gates": gates,
        "prompt_policy": config["prompt_policy"],
        "controller_training_contract": {
            **config["controller_training"], "exact_tokenizer_audit_required": True,
        },
        "target_data_used": True,
        "target_outcome_used_for_controller_labels": False,
        "formal_or_qualification_targets_used": False,
        "video_target_data_used": False,
        "claim_boundary": config["claim_boundary"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"], "summary": summary,
        "discoveryworld_grounder_macro_auc": dw_grounder["macro_validation_auc"],
        "gates": gates, "output_dir": str(args.output_dir),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
