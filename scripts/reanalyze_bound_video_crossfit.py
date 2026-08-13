#!/usr/bin/env python3
"""LOVO target grounding for source-gated BIND-handle->RELATE sequences."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_structured_video_transfer as runner  # noqa: E402
from motif_transfer.bound_video_ir import _rehydrate_receipt  # noqa: E402
from motif_transfer.structured_video_transfer import _target_exact_plan_value  # noqa: E402
from motif_transfer.video_dynamics_mdp import (  # noqa: E402
    answer_belief, apply_probe_receipt, initial_state,
    source_compatible_action_features,
)


def _entropy(values: Sequence[float]) -> float:
    vector = np.clip(np.asarray(values, dtype=float), 1e-12, 1)
    vector /= vector.sum()
    return float(-np.sum(vector * np.log(vector)) / math.log(len(vector)))


def _fit_ridge(features, labels, *, alpha: float):
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(labels, dtype=float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1
    design = np.column_stack(((matrix - mean) / scale, np.ones(len(matrix))))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[-1, -1] = 0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return mean, scale, weights


def _predict(model, features) -> np.ndarray:
    mean, scale, weights = model
    matrix = np.asarray(features, dtype=float)
    design = np.column_stack(((matrix - mean) / scale, np.ones(len(matrix))))
    return design @ weights


def _one_hot_kind(kind: str, role: str) -> tuple[float, ...]:
    kinds = (
        "OBJECT_PRESENT", "OBJECT_ATTRIBUTE", "OBJECT_TRACK",
        "OBJECT_MOTION", "COLLISION", "ENTRY", "EXIT", "EVENT_ORDER",
        "CAUSAL_ANCESTOR",
    )
    return tuple(float(kind == value) for value in kinds) + (
        float(role == "BIND"), float(role == "RELATE"),
    )


def _sample_objects(source, fork):
    model, global_receipts = runner._rehydrate(source)
    bind = [i for i, probe in enumerate(model.probes) if probe.target_event_role == "BIND"]
    relate = [i for i, probe in enumerate(model.probes) if probe.target_event_role == "RELATE"]
    tracks = fork["tracks"]
    matrix = fork["bound_relation_receipts"]
    bind_receipts = {
        key: _rehydrate_receipt(value["bind_receipt"])
        for key, value in tracks.items()
    }
    bound_receipts = {
        (bind_id, relate_id): _rehydrate_receipt(value["receipt"])
        for bind_id, rows in matrix.items() for relate_id, value in rows.items()
    }
    return model, global_receipts, bind, relate, tracks, matrix, bind_receipts, bound_receipts


def _candidate_rows(source, fork):
    model, _, bind, relate, tracks, matrix, bind_receipts, bound_receipts = _sample_objects(source, fork)
    particles, probes = model.particles, model.probes
    initial = initial_state(particles, probes, max_tests=2)
    tests, _, space = source_compatible_action_features(
        particles, initial, probes, max_tests=2,
    )
    gold = str(source["gold_answer"])
    gold_index = space.index(gold)
    initial_belief = answer_belief(particles, initial, answer_space=space)
    bind_rows, relation_rows = [], []
    for bind_index in bind:
        probe = probes[bind_index]
        bind_id = probe.probe_id
        bind_feature = (
            *tests[bind_index],
            probe.expected_sensor_reliability,
            max(probe.latent_true_probability_by_particle) - min(probe.latent_true_probability_by_particle),
            *_one_hot_kind(probe.predicate_kind, "BIND"),
        )
        after_bind = apply_probe_receipt(
            particles, initial, probes, bind_receipts[bind_id],
        )
        next_tests, _, _ = source_compatible_action_features(
            particles, after_bind, probes, max_tests=2, answer_space=space,
        )
        terminal_values = []
        for relate_index in relate:
            relation = probes[relate_index]
            relate_id = relation.probe_id
            cell = matrix[bind_id][relate_id]
            track = tracks[bind_id]
            visible_fraction = sum(
                value is not None for value in track["track"]["tracks"]
            ) / len(track["track"]["tracks"])
            relation_feature = (
                *next_tests[relate_index],
                visible_fraction,
                float(track["bind_receipt"]["sensor_reliability"]),
                float(track["bind_receipt"]["observed_true"]),
                float(cell["crop_fallback_count"]) / max(1, len(cell["relation_proxy_indices"])),
                float(cell["shared_primary_entity"]),
                relation.expected_sensor_reliability,
                max(relation.latent_true_probability_by_particle) - min(relation.latent_true_probability_by_particle),
                _entropy(initial_belief),
                *_one_hot_kind(relation.predicate_kind, "RELATE"),
            )
            terminal = apply_probe_receipt(
                particles, after_bind, probes, bound_receipts[(bind_id, relate_id)],
            )
            belief = answer_belief(particles, terminal, answer_space=space)
            value = float(belief[gold_index])
            terminal_values.append((value, bool(cell["shared_primary_entity"])))
            relation_rows.append({
                "sample_id": source["sample_id"], "bind_id": bind_id,
                "relate_id": relate_id, "features": relation_feature,
                "value": value, "shared": bool(cell["shared_primary_entity"]),
            })
        shared_values = [value for value, shared in terminal_values if shared]
        bind_rows.append({
            "sample_id": source["sample_id"], "bind_id": bind_id,
            "features": bind_feature,
            "value": max(shared_values or [value for value, _ in terminal_values]),
        })
    return bind_rows, relation_rows


def _evaluate_fold(source, fork, bind_model, relation_model):
    model, global_receipts, bind, relate, tracks, matrix, bind_receipts, bound_receipts = _sample_objects(source, fork)
    particles, probes = model.particles, model.probes
    initial = initial_state(particles, probes, max_tests=2)
    tests, _, space = source_compatible_action_features(particles, initial, probes, max_tests=2)
    initial_belief = answer_belief(particles, initial, answer_space=space)
    bind_features = [
        (
            *tests[index], probes[index].expected_sensor_reliability,
            max(probes[index].latent_true_probability_by_particle) - min(probes[index].latent_true_probability_by_particle),
            *_one_hot_kind(probes[index].predicate_kind, "BIND"),
        )
        for index in bind
    ]
    bind_index = bind[int(np.argmax(_predict(bind_model, bind_features)))]
    bind_id = probes[bind_index].probe_id
    after_bind = apply_probe_receipt(particles, initial, probes, bind_receipts[bind_id])
    next_tests, _, _ = source_compatible_action_features(
        particles, after_bind, probes, max_tests=2, answer_space=space,
    )
    relation_features, shared = [], []
    for index in relate:
        relation = probes[index]
        relate_id = relation.probe_id
        cell = matrix[bind_id][relate_id]
        track = tracks[bind_id]
        visible_fraction = sum(value is not None for value in track["track"]["tracks"]) / len(track["track"]["tracks"])
        relation_features.append((
            *next_tests[index], visible_fraction,
            float(track["bind_receipt"]["sensor_reliability"]),
            float(track["bind_receipt"]["observed_true"]),
            float(cell["crop_fallback_count"]) / max(1, len(cell["relation_proxy_indices"])),
            float(cell["shared_primary_entity"]), relation.expected_sensor_reliability,
            max(relation.latent_true_probability_by_particle) - min(relation.latent_true_probability_by_particle),
            _entropy(initial_belief), *_one_hot_kind(relation.predicate_kind, "RELATE"),
        ))
        shared.append(bool(cell["shared_primary_entity"]))
    scores = _predict(relation_model, relation_features)
    eligible = [slot for slot, value in enumerate(shared) if value]
    selected_slot = max(eligible or list(range(len(relate))), key=lambda slot: scores[slot])
    relation_index = relate[selected_slot]
    relate_id = probes[relation_index].probe_id

    def commit(receipt):
        state = apply_probe_receipt(particles, after_bind, probes, receipt)
        belief = answer_belief(particles, state, answer_space=space)
        answer = space[int(np.argmax(belief))]
        return answer, answer == source["gold_answer"], float(belief[space.index(source["gold_answer"])])

    authentic = commit(bound_receipts[(bind_id, relate_id)])
    unbound = commit(global_receipts[relate_id])
    wrong_slots = [slot for slot, value in enumerate(shared) if not value]
    wrong_slot = max(wrong_slots or list(range(len(relate))), key=lambda slot: scores[slot])
    wrong_id = probes[relate[wrong_slot]].probe_id
    wrong = commit(bound_receipts[(bind_id, wrong_id)])

    # Strong global exact-DP control under the target's native observation model.
    state = initial
    selected = set()
    global_actions = []
    for step in range(2):
        _, choice = _target_exact_plan_value(
            particles, probes, state, answer_space=space,
            selected=frozenset(selected), test_cost=0,
        )
        if choice[0] == "COMMIT":
            break
        index = choice[1]
        selected.add(index)
        global_actions.append(probes[index].probe_id)
        state = apply_probe_receipt(
            particles, state, probes, global_receipts[probes[index].probe_id],
        )
    global_belief = answer_belief(particles, state, answer_space=space)
    global_answer = space[int(np.argmax(global_belief))]
    target = (
        global_answer, global_answer == source["gold_answer"],
        float(global_belief[space.index(source["gold_answer"])]),
    )
    baseline_answer = space[int(np.argmax(initial_belief))]
    return {
        "sample_id": source["sample_id"], "gold_answer": source["gold_answer"],
        "baseline_correct": baseline_answer == source["gold_answer"],
        "selected_bind_id": bind_id, "selected_relate_id": relate_id,
        "selected_relation_shared": shared[selected_slot],
        "target_exact_dp": {"actions": global_actions, "answer": target[0], "correct": target[1], "gold_probability": target[2]},
        "authentic": {"answer": authentic[0], "correct": authentic[1], "gold_probability": authentic[2]},
        "unbound": {"answer": unbound[0], "correct": unbound[1], "gold_probability": unbound[2]},
        "wrong_guard": {"relate_id": wrong_id, "answer": wrong[0], "correct": wrong[1], "gold_probability": wrong[2]},
        "node_only_correct": baseline_answer == source["gold_answer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="0.1,1,10,100")
    args = parser.parse_args()
    alphas = [float(value) for value in args.alphas.split(",")]
    sources = {
        str(row["sample_id"]): row
        for row in json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    }
    forks = json.loads((args.run_dir / "bound_relation_forks.json").read_text(encoding="utf-8"))
    fork_by_id = {str(row["sample_id"]): row for row in forks}
    sample_ids = sorted(fork_by_id)
    candidates = {
        sample_id: _candidate_rows(sources[sample_id], fork_by_id[sample_id])
        for sample_id in sample_ids
    }
    reports = {}
    for alpha in alphas:
        evaluated = []
        for held in sample_ids:
            bind_rows = [row for sample_id in sample_ids if sample_id != held for row in candidates[sample_id][0]]
            relation_rows = [row for sample_id in sample_ids if sample_id != held for row in candidates[sample_id][1]]
            bind_model = _fit_ridge(
                [row["features"] for row in bind_rows], [row["value"] for row in bind_rows], alpha=alpha,
            )
            relation_model = _fit_ridge(
                [row["features"] for row in relation_rows], [row["value"] for row in relation_rows], alpha=alpha,
            )
            evaluated.append(_evaluate_fold(
                sources[held], fork_by_id[held], bind_model, relation_model,
            ))
        count = len(evaluated)
        condition_keys = ("target_exact_dp", "authentic", "unbound", "wrong_guard")
        conditions = {
            key: {
                "correct": sum(bool(row[key]["correct"]) for row in evaluated),
                "accuracy": sum(bool(row[key]["correct"]) for row in evaluated) / count,
            }
            for key in condition_keys
        }
        conditions["node_only"] = {
            "correct": sum(bool(row["node_only_correct"]) for row in evaluated),
            "accuracy": sum(bool(row["node_only_correct"]) for row in evaluated) / count,
        }
        authentic = conditions["authentic"]["correct"]
        gates = {
            "leave_one_video_out_complete": count == len(sample_ids),
            "authentic_shared_guard": all(row["selected_relation_shared"] for row in evaluated),
            "authentic_above_target_exact_dp": authentic > conditions["target_exact_dp"]["correct"],
            "authentic_above_unbound": authentic > conditions["unbound"]["correct"],
            "authentic_above_wrong_guard": authentic > conditions["wrong_guard"]["correct"],
            "authentic_above_node_only": authentic > conditions["node_only"]["correct"],
        }
        reports[str(alpha)] = {
            "status": "CROSSFIT_EDGE_PASS" if all(gates.values()) else "CROSSFIT_EDGE_FAIL",
            "conditions": conditions, "gates": gates, "rows": evaluated,
        }
    output = {
        "schema_version": 1,
        "benchmark": str(forks[0]["benchmark"]),
        "protocol": "LEAVE_ONE_VIDEO_OUT_TARGET_VALUE_GROUNDING_ON_SOURCE_EDGE_TOPOLOGY",
        "alphas": reports,
        "any_alpha_passed": any(row["status"].endswith("PASS") for row in reports.values()),
        "claim_boundary": "Adaptation-only model-selection diagnostic; qualification remains sealed.",
    }
    path = args.run_dir / "bound_relation_crossfit_report.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "benchmark": output["benchmark"], "any_alpha_passed": output["any_alpha_passed"],
        "alphas": {key: {"status": value["status"], "conditions": value["conditions"], "gates": value["gates"]} for key, value in reports.items()},
        "report": str(path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
