#!/usr/bin/env python3
"""Build WebShop + ALFWorld target-development SFT for the 9B Harness.

The target domains provide only neural typed-effect grounding.  Exact Harness
labels come from the unchanged source-induced symbolic executor.  Formal,
qualification, held-out, reward, and official-success fields are not read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_hierarchical_grounder import action_option  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.harness_controller_training import (  # noqa: E402
    CONTROLLER_SYSTEM_PROMPT,
    format_controller_prompt,
)
from motif_transfer.phase3_alfworld_typed_grounder import (  # noqa: E402
    score_actions as score_alfworld_actions,
    validate_artifact as validate_alfworld_grounder,
)
from motif_transfer.phase3_source_function_induction import QUALIFIED  # noqa: E402
from motif_transfer.phase3_typed_effect_induction import TYPED_EFFECTS  # noqa: E402
from motif_transfer.target_harness_sft import (  # noqa: E402
    GroundedTargetState,
    TargetHarnessSFTExample,
    build_matched_target_pair,
)
from motif_transfer.webshop_neural_grounder_v5 import (  # noqa: E402
    action_verb,
    target_action_features,
)


SYSTEM = CONTROLLER_SYSTEM_PROMPT


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )


def _serialized_probability(value: float, decimal_places: int) -> float:
    """Bound prompt precision without changing the grounder's stored scores."""

    if decimal_places < 1 or decimal_places > 12:
        raise ValueError("probability_decimal_places must be in [1, 12]")
    return round(float(value), decimal_places)


def _episode_candidates(experience: Mapping[str, Any]) -> list[str]:
    metadata = experience.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    values = [
        *map(str, metadata.get("candidate_actions") or ()),
        str(experience.get("action") or ""),
    ]
    return sorted({
        value for value in values
        if value and "\n" not in value and action_verb(value) != "other"
    }, key=lambda value: (stable_hash({"target_native_action": value}), value))


def _webshop_observation(experience: Mapping[str, Any]) -> str:
    metadata = experience.get("metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("schema_canonical") or metadata.get("schema")
        if value:
            return str(value)
    return str(experience.get("state") or experience.get("raw_state") or "")


def _webshop_url(experience: Mapping[str, Any]) -> str:
    metadata = experience.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("url"):
        return str(metadata["url"])
    interface = experience.get("interface")
    if isinstance(interface, Mapping) and interface.get("url"):
        return str(interface["url"])
    return ""


def _webshop_goal(experience: Mapping[str, Any]) -> str:
    return str(experience.get("goal") or experience.get("tasks") or "")


def _webshop_supervision_states(
    *, episode_index: Mapping[str, Any], validation_tasks: set[str],
    maximum_candidates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    states = []
    input_receipts = []
    indexed_tasks = set(map(str, episode_index["adaptation_task_ids"]))
    for receipt in episode_index["adaptation_episode_receipts"]:
        path = Path(str(receipt["path"]))
        if not path.is_file() or _sha256(path) != str(receipt["sha256"]):
            raise SystemExit(f"WebShop adaptation episode receipt mismatch: {path}")
        payload = _read(path)
        task_id = str(receipt["task_id"])
        if task_id not in indexed_tasks:
            raise SystemExit("WebShop episode escaped the adaptation task set")
        experiences = list(payload.get("experiences") or ())
        if not experiences:
            raise SystemExit(f"empty WebShop adaptation episode: {path}")
        input_receipts.append({
            "task_id_sha256": stable_hash({"target_task": task_id}),
            "path": str(path),
            "sha256": _sha256(path),
        })
        for index, experience in enumerate(experiences):
            candidates = _episode_candidates(experience)[:maximum_candidates]
            if len(candidates) < 2:
                continue
            future_options = tuple(
                action_verb(str(row.get("action") or ""))
                for row in experiences[index:index + 8]
            )
            successor_option_sets = tuple(
                set(map(action_verb, _episode_candidates(row)))
                for row in experiences[index:index + 4]
            )
            history = tuple(
                str(row.get("action") or "") for row in experiences[:index]
            )
            rows = []
            for action in candidates:
                option = action_verb(action)
                labels = {
                    TYPED_EFFECTS[0]: int(option in future_options[:1]),
                    TYPED_EFFECTS[1]: int(option in future_options[:4]),
                    TYPED_EFFECTS[2]: int(option in future_options[:8]),
                    TYPED_EFFECTS[3]: int(
                        bool(successor_option_sets)
                        and all(option in values for values in successor_option_sets)
                    ),
                }
                rows.append({
                    "action": action,
                    "action_sha256": stable_hash({"target_native_action": action}),
                    "features": target_action_features(
                        observation_text=_webshop_observation(experience),
                        url=_webshop_url(experience),
                        goal=_webshop_goal(experience),
                        action=action,
                        step_index=index,
                        maximum_steps=len(experiences),
                        previous_action=history[-1] if history else None,
                    ),
                    "labels": labels,
                })
            states.append({
                "task_id": task_id,
                "task_sha256": stable_hash({"target_task": task_id}),
                "split": "validation" if task_id in validation_tasks else "train",
                "state_receipt_sha256": stable_hash({
                    "episode_sha256": str(receipt["sha256"]),
                    "step_id": str(experience.get("step_id") or index),
                }),
                "rows": rows,
            })
    return states, input_receipts


def _fit_webshop_grounder(
    *, states: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> tuple[dict[str, MLPClassifier], dict[str, Any]]:
    train_rows = [
        row for state in states if state["split"] == "train"
        for row in state["rows"]
    ]
    validation_rows = [
        row for state in states if state["split"] == "validation"
        for row in state["rows"]
    ]
    if not train_rows or not validation_rows:
        raise SystemExit("WebShop grounder needs train and validation target tasks")
    models = {}
    heads = {}
    metrics = {}
    seed = int(config["seed"])
    for offset, effect in enumerate(TYPED_EFFECTS):
        train_x = np.asarray([row["features"] for row in train_rows])
        train_y = np.asarray([row["labels"][effect] for row in train_rows])
        validation_x = np.asarray([row["features"] for row in validation_rows])
        validation_y = np.asarray([row["labels"][effect] for row in validation_rows])
        if set(map(int, train_y)) != {0, 1} or set(map(int, validation_y)) != {0, 1}:
            raise SystemExit(f"WebShop typed head has one class: {effect}")
        rng = np.random.default_rng(seed + offset)
        target = max(int(np.sum(train_y == label)) for label in (0, 1))
        chosen = np.concatenate([
            rng.choice(np.flatnonzero(train_y == label), target, replace=True)
            for label in (0, 1)
        ])
        rng.shuffle(chosen)
        model = MLPClassifier(
            hidden_layer_sizes=(int(config["hidden_units"]),),
            activation="tanh", solver="lbfgs", alpha=float(config["alpha"]),
            max_iter=int(config["maximum_iterations"]),
            random_state=seed + offset,
        ).fit(train_x[chosen], train_y[chosen])
        scores = model.predict_proba(validation_x)[:, 1]
        auc = float(roc_auc_score(validation_y, scores))
        metrics[effect] = {
            "raw_training_examples": len(train_y),
            "raw_training_positive_rate": float(np.mean(train_y)),
            "balanced_training_examples": len(chosen),
            "validation_examples": len(validation_y),
            "validation_positive_rate": float(np.mean(validation_y)),
            "validation_auc": auc,
        }
        heads[effect] = {
            "kind": "target-native-binary-mlp-v1",
            "hidden_activation": "tanh",
            "layers": [
                {"weights": weights.tolist(), "bias": bias.tolist()}
                for weights, bias in zip(model.coefs_, model.intercepts_)
            ],
        }
        models[effect] = model
    aucs = [metrics[name]["validation_auc"] for name in TYPED_EFFECTS]
    minimum_auc = float(config["minimum_per_head_validation_auc"])
    minimum_macro = float(config["minimum_macro_validation_auc"])
    gates = {
        "every_head_validation_auc_at_least_threshold": min(aucs) >= minimum_auc,
        "macro_validation_auc_at_least_threshold": float(np.mean(aucs)) >= minimum_macro,
    }
    artifact = {
        "schema_version": "webshop-target-typed-effect-grounder-v1",
        "status": "QUALIFIED" if all(gates.values()) else "BLOCKED",
        "authority": "WEBSHOP_ADAPTATION_TASKS_ONLY;NO_REWARD_OR_SUCCESS_LABELS",
        "effect_types": list(TYPED_EFFECTS),
        "label_granularity": "target_native_action_verb",
        "label_contract": {
            TYPED_EFFECTS[0]: "ACTION_VERB_APPEARS_IN_CONTINUATION_BY_1",
            TYPED_EFFECTS[1]: "ACTION_VERB_APPEARS_IN_CONTINUATION_BY_4",
            TYPED_EFFECTS[2]: "ACTION_VERB_APPEARS_IN_CONTINUATION_BY_8",
            TYPED_EFFECTS[3]: "ACTION_VERB_REMAINS_EXECUTABLE_ACROSS_UP_TO_4_SUCCESSORS",
        },
        "heads": heads,
        "head_validation": metrics,
        "thresholds": {
            "minimum_per_head_validation_auc": minimum_auc,
            "minimum_macro_validation_auc": minimum_macro,
        },
        "macro_validation_auc": float(np.mean(aucs)),
        "gates": gates,
        "formal_or_qualification_data_used": False,
        "reward_or_success_used": False,
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    if not all(gates.values()):
        raise SystemExit(f"WebShop typed grounder gate failed: {artifact}")
    return models, artifact


def _ground_webshop_states(
    *, states: Sequence[Mapping[str, Any]], models: Mapping[str, MLPClassifier],
    artifact_sha256: str, probability_decimal_places: int,
) -> list[GroundedTargetState]:
    output = []
    for state in states:
        matrix = np.asarray([row["features"] for row in state["rows"]])
        predictions = {
            effect: models[effect].predict_proba(matrix)[:, 1]
            for effect in TYPED_EFFECTS
        }
        ordered = sorted(state["rows"], key=lambda row: row["action_sha256"])
        by_sha = {row["action_sha256"]: index for index, row in enumerate(state["rows"])}
        candidates = tuple({
            "candidate_id": f"C{rank}",
            "effects": {
                effect: _serialized_probability(
                    predictions[effect][by_sha[row["action_sha256"]]],
                    probability_decimal_places,
                )
                for effect in TYPED_EFFECTS
            },
        } for rank, row in enumerate(ordered))
        grounded = GroundedTargetState(
            target_domain="webshop",
            target_task_sha256=str(state["task_sha256"]),
            split=str(state["split"]),
            state_receipt_sha256=str(state["state_receipt_sha256"]),
            grounder_artifact_sha256=artifact_sha256,
            candidates=candidates,
        )
        if grounded.validate():
            output.append(grounded)
    return output


def _ground_alfworld_states(
    *, grounder_path: Path, maximum_candidates: int,
    probability_decimal_places: int,
) -> tuple[list[GroundedTargetState], dict[str, Any]]:
    artifact = _read(grounder_path)
    validate_alfworld_grounder(artifact)
    if not str(artifact.get("status", "")).endswith("QUALIFIED"):
        raise SystemExit("ALFWorld typed grounder is not qualified")
    if artifact.get("formal_success_read_for_training_or_qualification") is not False:
        raise SystemExit("ALFWorld grounder read formal success")
    receipts_info = artifact["target_adaptation_receipts"]
    receipts_path = Path(str(receipts_info["path"]))
    if _sha256(receipts_path) != str(receipts_info["file_sha256"]):
        raise SystemExit("ALFWorld target adaptation receipt hash mismatch")
    receipts = _read(receipts_path)
    if receipts.get("qualification_or_heldout_read") is not False:
        raise SystemExit("ALFWorld receipts crossed the target evaluation boundary")
    if receipts.get("selection_used_target_outcomes") is not False:
        raise SystemExit("ALFWorld receipt selection used target outcomes")
    output = []
    for episode in receipts["episodes"]:
        partition = str(episode["partition"])
        if partition not in {"adaptation_train", "adaptation_validation"}:
            raise SystemExit(f"forbidden ALFWorld partition: {partition}")
        split = "train" if partition == "adaptation_train" else "validation"
        history: list[str] = []
        for transition in episode["transitions"]:
            scores = score_alfworld_actions(
                goal=str(transition["goal"]),
                observation=str(transition["before_observation"]),
                native_actions=tuple(map(str, transition["native_actions"])),
                step=int(transition["step"]),
                action_history=history,
                artifact=artifact,
            )
            top = sorted(scores.items(), key=lambda item: (
                -float(item[1]["target_policy_probability"]),
                str(item[1]["action_sha256"]),
            ))[:maximum_candidates]
            top = sorted(top, key=lambda item: str(item[1]["action_sha256"]))
            candidates = tuple({
                "candidate_id": f"C{rank}",
                "effects": {
                    name: _serialized_probability(
                        row["typed_effect_probabilities"][name],
                        probability_decimal_places,
                    )
                    for name in TYPED_EFFECTS
                },
            } for rank, (_, row) in enumerate(top))
            grounded = GroundedTargetState(
                target_domain="alfworld",
                target_task_sha256=stable_hash({
                    "target_task": str(episode["task_id"]),
                }),
                split=split,
                state_receipt_sha256=str(transition["receipt_sha256"]),
                grounder_artifact_sha256=str(artifact["artifact_sha256"]),
                candidates=candidates,
            )
            if grounded.validate():
                output.append(grounded)
            history.append(str(transition["expert_action"]))
    audit = {
        "grounder_path": str(grounder_path),
        "grounder_sha256": _sha256(grounder_path),
        "grounder_artifact_sha256": str(artifact["artifact_sha256"]),
        "adaptation_receipts_path": str(receipts_path),
        "adaptation_receipts_sha256": _sha256(receipts_path),
        "episodes": len(receipts["episodes"]),
        "grounded_states": len(output),
        "qualification_or_heldout_read": False,
        "target_outcomes_used_for_controller_labels": False,
    }
    return output, audit


def _load_programs(
    root: Path, families: Sequence[str],
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    programs = {}
    receipts = []
    for family in families:
        path = root / f"{family}.json"
        artifact = _read(path)
        program = artifact.get("source_function_program")
        if not isinstance(program, Mapping):
            raise SystemExit(f"malformed source program: {path}")
        programs[str(family)] = program
        receipts.append({
            "source_family": str(family),
            "path": str(path),
            "file_sha256": _sha256(path),
            "program_sha256": str(program["program_sha256"]),
            "status": str(program["status"]),
        })
    return programs, receipts


def _candidate_pairs(
    states: Sequence[GroundedTargetState],
    programs: Mapping[str, Mapping[str, Any]],
) -> list[tuple[TargetHarnessSFTExample, TargetHarnessSFTExample]]:
    output = []
    for state in states:
        for family, program in sorted(programs.items()):
            if program.get("status") != QUALIFIED:
                continue
            pair = build_matched_target_pair(
                state=state, source_family=family, program=program,
                program_receipt=str(program["program_sha256"]),
            )
            if pair is not None:
                output.append(pair)
    return output


def _select_pairs(
    pairs: Sequence[tuple[TargetHarnessSFTExample, TargetHarnessSFTExample]],
    *, target_domain: str, split: str, quota: int,
    forbidden_prompts: set[str],
) -> list[tuple[TargetHarnessSFTExample, TargetHarnessSFTExample]]:
    by_family = defaultdict(list)
    for pair in pairs:
        if pair[0].target_domain == target_domain and pair[0].split == split:
            by_family[pair[0].source_family].append(pair)
    for family in by_family:
        by_family[family].sort(key=lambda pair: (pair[0].pair_id, pair[0].example_id))
    families = sorted(by_family)
    if not families:
        raise SystemExit(f"no candidate pairs for {target_domain}/{split}")
    selected = []
    offsets = {family: 0 for family in families}
    while len(selected) < quota:
        progress = False
        for family in families:
            rows = by_family[family]
            while offsets[family] < len(rows):
                pair = rows[offsets[family]]
                offsets[family] += 1
                prompts = {_json_text(row.input_payload) for row in pair}
                if prompts & forbidden_prompts or len(prompts) != 2:
                    continue
                selected.append(pair)
                forbidden_prompts.update(prompts)
                progress = True
                break
            if len(selected) >= quota:
                break
        if not progress:
            raise SystemExit(
                f"insufficient unique pairs for {target_domain}/{split}: "
                f"needed {quota}, found {len(selected)}"
            )
    return selected


def _format_example(row: TargetHarnessSFTExample) -> dict[str, Any]:
    prompt = format_controller_prompt(
        objective=row.objective, input_payload=row.input_payload,
    )
    return {
        "example_id": row.example_id,
        "pair_id": row.pair_id,
        "target_domain_audit_only": row.target_domain,
        "source_family_audit_only": row.source_family,
        "control_variant_audit_only": row.control_variant,
        "objective": row.objective,
        "prompt": prompt,
        "completion": _json_text(row.target_payload),
        "evidence_receipt_ids": list(row.evidence_receipt_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/target_harness_sft_pilot_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/target_harness_sft_pilot_v1",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    config = _read(args.config)
    if config.get("status") != "FROZEN_BEFORE_TARGET_DATASET_BUILD":
        raise SystemExit("target Harness data config is not frozen")
    controller_training = config.get("controller_training") or {}
    probability_decimal_places = int(
        controller_training.get("probability_decimal_places", 12)
    )
    maximum_sequence_length = int(
        controller_training.get("maximum_sequence_length", 2048)
    )
    if maximum_sequence_length < 1:
        raise SystemExit("maximum_sequence_length must be positive")

    program_root = _resolve(str(config["source_program_root"]))
    programs, program_receipts = _load_programs(
        program_root, tuple(map(str, config["source_families"])),
    )
    qualified_families = sorted(
        family for family, program in programs.items()
        if program.get("status") == QUALIFIED
    )

    webshop_manifest_path = _resolve(str(config["webshop"]["partition_manifest"]))
    webshop_manifest = _read(webshop_manifest_path)
    roles = webshop_manifest["targets"]["webshop"]["partition"]["roles"]
    adaptation_tasks = set(map(str, roles["adaptation"]))
    forbidden_webshop_tasks = set(map(str, (
        *roles["qualification"], *roles["held_out"], *roles["reserve"],
    )))
    if adaptation_tasks & forbidden_webshop_tasks:
        raise SystemExit("WebShop adaptation tasks overlap forbidden target roles")
    episode_index_path = _resolve(str(config["webshop"]["episode_index_artifact"]))
    episode_index = _read(episode_index_path)
    if set(map(str, episode_index["adaptation_task_ids"])) != adaptation_tasks:
        raise SystemExit("WebShop episode index is not exactly the adaptation task set")
    validation_tasks = set(map(str, episode_index["validation"]["tasks"]))
    if not validation_tasks < adaptation_tasks:
        raise SystemExit("WebShop validation tasks must be a proper adaptation subset")
    webshop_grounder_config = config["webshop"]["grounder"]
    webshop_states_raw, webshop_inputs = _webshop_supervision_states(
        episode_index=episode_index, validation_tasks=validation_tasks,
        maximum_candidates=int(webshop_grounder_config["maximum_candidates"]),
    )
    webshop_models, webshop_grounder = _fit_webshop_grounder(
        states=webshop_states_raw, config=webshop_grounder_config,
    )
    webshop_states = _ground_webshop_states(
        states=webshop_states_raw, models=webshop_models,
        artifact_sha256=str(webshop_grounder["artifact_sha256"]),
        probability_decimal_places=probability_decimal_places,
    )

    alfworld_grounder_path = _resolve(str(config["alfworld"]["grounder_artifact"]))
    alfworld_states, alfworld_audit = _ground_alfworld_states(
        grounder_path=alfworld_grounder_path,
        maximum_candidates=int(config["alfworld"]["maximum_candidates"]),
        probability_decimal_places=probability_decimal_places,
    )
    all_pairs = _candidate_pairs(
        [*webshop_states, *alfworld_states], programs,
    )
    quota = config["quota_per_domain"]
    forbidden_prompts: set[str] = set()
    selected_pairs = []
    for split, quota_key in (
        ("train", "train_authentic"),
        ("validation", "validation_authentic"),
    ):
        for domain in ("webshop", "alfworld"):
            selected_pairs.extend(_select_pairs(
                all_pairs, target_domain=domain, split=split,
                quota=int(quota[quota_key]),
                forbidden_prompts=forbidden_prompts,
            ))
    examples = [row for pair in selected_pairs for row in pair]
    if not examples or not all(row.validate() for row in examples):
        raise SystemExit("target Harness examples failed validation")

    model_rows = [_format_example(row) for row in examples]
    prompt_sets = {
        split: {row["prompt"] for row, raw in zip(model_rows, examples) if raw.split == split}
        for split in ("train", "validation")
    }
    prompt_text = "\n".join(row["prompt"] for row in model_rows).lower()
    completions = "\n".join(row["completion"] for row in model_rows).lower()
    native_tokens = (
        "webshop", "alfworld", "selected_action", "expert_action",
        "official_success", "official_reward", "go to ", "click('",
        "fill('", "move ", "take ", "target_domain_audit_only",
    )
    pairs_by_id = Counter(row.pair_id for row in examples)
    domain_split_kind_counts = Counter(
        (row.target_domain, row.split,
         "authentic" if row.control_variant == "AUTHENTIC_TARGET_NEURAL_GROUNDING"
         else "control")
        for row in examples
    )
    expected_train = int(quota["train_authentic"])
    expected_validation = int(quota["validation_authentic"])
    gates = {
        "exact_two_target_domains": {row.target_domain for row in examples}
        == {"webshop", "alfworld"},
        "all_examples_valid": all(row.validate() for row in examples),
        "exact_one_control_per_authentic": set(pairs_by_id.values()) == {2},
        "requested_domain_split_quotas_met": all(
            domain_split_kind_counts[(domain, split, kind)] == expected
            for domain in ("webshop", "alfworld")
            for split, expected in (
                ("train", expected_train),
                ("validation", expected_validation),
            )
            for kind in ("authentic", "control")
        ),
        "prompts_unique": len({row["prompt"] for row in model_rows}) == len(model_rows),
        "train_validation_prompts_disjoint": prompt_sets["train"].isdisjoint(
            prompt_sets["validation"]
        ),
        "target_domain_and_native_actions_absent_from_prompts": not any(
            token in prompt_text for token in native_tokens
        ),
        "reward_and_success_absent_from_completions": not any(
            token in completions for token in ("reward", "success", "native_action")
        ),
        "webshop_grounder_qualified_before_export": webshop_grounder["status"]
        == "QUALIFIED",
        "webshop_only_adaptation_tasks_used": not (
            {receipt["task_id_sha256"] for receipt in webshop_inputs}
            & {stable_hash({"target_task": task}) for task in forbidden_webshop_tasks}
        ),
        "alfworld_qualification_or_heldout_not_read":
        alfworld_audit["qualification_or_heldout_read"] is False,
        "target_outcomes_not_used_for_controller_labels": all(
            row.target_outcome_used is False for row in examples
        ),
        "source_identity_hidden_from_prompts": not any(
            family.lower() in prompt_text for family in programs
        ),
        "control_variant_hidden_from_prompts": not any(
            value.lower() in prompt_text
            for value in {row.control_variant for row in examples}
        ),
    }
    if not all(gates.values()):
        raise SystemExit(f"target Harness dataset gates failed: {gates}")

    args.output_dir.mkdir(parents=True)
    structured_path = args.output_dir / "structured.jsonl"
    with structured_path.open("w", encoding="utf-8") as stream:
        for row in sorted(examples, key=lambda value: value.example_id):
            stream.write(_json_text(asdict(row)) + "\n")
    split_paths = {}
    for split in ("train", "validation"):
        path = args.output_dir / f"{split}.jsonl"
        split_paths[split] = path
        with path.open("w", encoding="utf-8") as stream:
            rows = [
                formatted for formatted, raw in zip(model_rows, examples)
                if raw.split == split
            ]
            for row in sorted(rows, key=lambda value: value["example_id"]):
                stream.write(_json_text(row) + "\n")
    webshop_grounder_path = args.output_dir / "webshop_typed_grounder.json"
    webshop_grounder_path.write_text(
        json.dumps(webshop_grounder, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    grounder_supervision_path = args.output_dir / "webshop_grounder_supervision.jsonl"
    with grounder_supervision_path.open("w", encoding="utf-8") as stream:
        for state in webshop_states_raw:
            for row in state["rows"]:
                stream.write(_json_text({
                    "target_task_sha256": state["task_sha256"],
                    "split": state["split"],
                    "state_receipt_sha256": state["state_receipt_sha256"],
                    "action_sha256": row["action_sha256"],
                    "features": list(row["features"]),
                    "labels": row["labels"],
                }) + "\n")

    summary = {
        "examples": len(examples),
        "pairs": len(selected_pairs),
        "qualified_source_families": qualified_families,
        "domain_split_kind_counts": {
            "/".join(key): value
            for key, value in sorted(domain_split_kind_counts.items())
        },
        "decision_counts": dict(sorted(Counter(
            str(row.target_payload["decision"]) for row in examples
        ).items())),
        "control_variant_counts": dict(sorted(Counter(
            row.control_variant for row in examples
        ).items())),
        "source_family_counts": dict(sorted(Counter(
            row.source_family for row in examples
        ).items())),
        "candidate_state_counts": {
            "webshop": len(webshop_states),
            "alfworld": len(alfworld_states),
        },
    }
    manifest = {
        "schema_version": "target-harness-sft-pilot-dataset-v1",
        "status": "FROZEN_TARGET_DEVELOPMENT_HARNESS_SUPERVISION",
        "authority": (
            "TARGET_NATIVE_NEURAL_GROUNDING;"
            "LABELS_FROM_FROZEN_SOURCE_PROGRAM_EXECUTOR;"
            "NO_FORMAL_TARGET_OUTCOME"
        ),
        "build_config": str(args.config.resolve()),
        "build_config_sha256": _sha256(args.config),
        "source_program_receipts": program_receipts,
        "webshop": {
            "partition_manifest": str(webshop_manifest_path),
            "partition_manifest_sha256": _sha256(webshop_manifest_path),
            "episode_index": str(episode_index_path),
            "episode_index_sha256": _sha256(episode_index_path),
            "adaptation_task_count": len(adaptation_tasks),
            "forbidden_task_count": len(forbidden_webshop_tasks),
            "input_episode_receipts": webshop_inputs,
            "raw_states": len(webshop_states_raw),
            "grounded_states": len(webshop_states),
            "grounder_artifact": str(webshop_grounder_path.resolve()),
            "grounder_artifact_sha256": _sha256(webshop_grounder_path),
            "grounder_validation": webshop_grounder["head_validation"],
            "grounder_macro_validation_auc": webshop_grounder[
                "macro_validation_auc"
            ],
        },
        "alfworld": alfworld_audit,
        "files": {
            "structured": {"path": str(structured_path.resolve()), "sha256": _sha256(structured_path)},
            "train": {"path": str(split_paths["train"].resolve()), "sha256": _sha256(split_paths["train"])},
            "validation": {"path": str(split_paths["validation"].resolve()), "sha256": _sha256(split_paths["validation"])},
            "webshop_grounder_supervision": {
                "path": str(grounder_supervision_path.resolve()),
                "sha256": _sha256(grounder_supervision_path),
            },
        },
        "summary": summary,
        "gates": gates,
        "prompt_policy": config["prompt_policy"],
        "controller_training_contract": {
            "model": str(controller_training.get("model", "Qwen/Qwen3.5-9B")),
            "maximum_sequence_length": maximum_sequence_length,
            "probability_decimal_places": probability_decimal_places,
            "exact_tokenizer_audit_required": True,
        },
        "target_data_used": True,
        "target_outcome_used_for_controller_labels": False,
        "formal_or_qualification_targets_used": False,
        "claim_boundary": str(config["claim_boundary"]),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"], "summary": summary,
        "webshop_grounder_macro_auc": webshop_grounder["macro_validation_auc"],
        "gates": gates, "output_dir": str(args.output_dir),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
