#!/usr/bin/env python3
"""Evaluate a frozen source-induced structural program on fresh source seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_delta_induction import (  # noqa: E402
    StructuralPath,
    sequence_contains,
    structural_atom_descriptors,
    validate_structural_program,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _validate_collection(collection: Mapping[str, Any]) -> None:
    _self_hash(collection, "collection_sha256")
    if collection.get("split") != "qualification":
        raise ValueError("fresh collection is not qualification-only")
    if collection.get("selection", {}).get("named_effect_list_used") is not False:
        raise ValueError("fresh collection used named effects")
    for path in collection.get("paths") or ():
        _self_hash(path, "path_sha256")
        for step in path.get("steps") or ():
            _self_hash(step, "transition_sha256")
            required = {
                "before_replay_state_sha256", "source_action_ordinal",
                "delta", "after_replay_state_sha256",
            }
            if not required.issubset(step):
                raise ValueError("source intervention tuple is incomplete")
    for group in collection.get("delta_contrast_groups") or ():
        _self_hash(group, "group_sha256")


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    row = list(range(len(right) + 1))
    for index, a in enumerate(left, start=1):
        next_row = [index]
        for offset, b in enumerate(right, start=1):
            next_row.append(min(
                next_row[-1] + 1,
                row[offset] + 1,
                row[offset - 1] + int(a != b),
            ))
        row = next_row
    return row[-1]


def _blind_program_selection(
    effects: Sequence[str], programs: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    distances = {
        task: _edit_distance(effects, program["induced_sequence"])
        for task, program in programs.items()
    }
    best = min(distances.values())
    return tuple(sorted(task for task, value in distances.items() if value == best))


def _grounding_metrics(
    collection: Mapping[str, Any], sequence: Sequence[str],
) -> dict[str, int]:
    groups = {
        str(row["prefix_sha256"]): row
        for row in collection.get("delta_contrast_groups") or ()
    }
    authentic = shuffled = occurrences = unique = 0
    for path in collection.get("paths") or ():
        if not bool(path.get("success")):
            continue
        position = 0
        for step in path.get("steps") or ():
            if position >= len(sequence):
                break
            observed_atoms = [
                row["operator_type_id"]
                for row in structural_atom_descriptors(step["delta"])
            ]
            for observed in observed_atoms:
                if position >= len(sequence) or observed != sequence[position]:
                    continue
                expected = str(sequence[position])
                group = groups.get(str(step["prefix_sha256"]))
                if group is None:
                    position += 1
                    continue
                candidates = list(group["candidates"])
                winners = [
                    row for row in candidates
                    if expected in row["operator_type_ids"]
                ]
                occurrences += 1
                unique += int(len(winners) == 1)
                authentic += int(
                    len(winners) == 1
                    and int(winners[0]["source_action_ordinal"])
                    == int(step["source_action_ordinal"])
                )
                count = len(candidates)
                if count < 2:
                    raise ValueError("contrast group does not contain a control")
                offset = 1 + int(stable_hash({
                    "group_sha256": group["group_sha256"],
                    "operator_type_id": expected,
                    "control": "FULL_OPERATOR_BINDING_ROTATION_V1",
                })[:8], 16) % (count - 1)
                shuffled_winners = [
                    candidates[index]
                    for index in range(count)
                    if expected in candidates[(index + offset) % count][
                        "operator_type_ids"
                    ]
                ]
                shuffled += int(
                    len(shuffled_winners) == 1
                    and int(shuffled_winners[0]["source_action_ordinal"])
                    == int(step["source_action_ordinal"])
                )
                position += 1
    return {
        "operator_occurrences": occurrences,
        "unique_authentic_bindings": unique,
        "authentic_correct_bindings": authentic,
        "shuffled_correct_bindings": shuffled,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")

    manifest = _read(args.manifest)
    _self_hash(manifest, "manifest_sha256")
    if manifest.get("reserve_opened_at_freeze") is not False:
        raise SystemExit("reserve was not unopened at freeze")
    for relative, expected in manifest["frozen_code_sha256"].items():
        path = REPO / relative
        if _sha(path) != expected:
            raise SystemExit(f"frozen code changed: {relative}")
    if _sha(args.summary) != manifest["expected_summary_file_sha256"]:
        raise SystemExit("fresh summary does not match its post-collection receipt")
    summary = _read(args.summary)
    _self_hash(summary, "summary_sha256")
    if summary["config_file_sha256"] != manifest["reserve_config_file_sha256"]:
        raise SystemExit("fresh collection used the wrong reserve config")

    programs: dict[str, dict[str, Any]] = {}
    for task, receipt in manifest["source_programs"].items():
        path = REPO / receipt["path"]
        if _sha(path) != receipt["file_sha256"]:
            raise SystemExit(f"frozen source program changed: {task}")
        program = _read(path)
        validate_structural_program(program)
        if program["program_sha256"] != receipt["program_sha256"]:
            raise SystemExit(f"source program receipt mismatch: {task}")
        programs[task] = program

    expected = {
        (str(row["task_id"]), int(seed["seed"]))
        for row in manifest["reserve_config"]["tasks"]
        for seed in row["seeds"]
    }
    observed: set[tuple[str, int]] = set()
    collections: dict[str, list[dict[str, Any]]] = {}
    for receipt in summary.get("receipts") or ():
        path = Path(str(receipt["path"]))
        if _sha(path) != receipt["file_sha256"]:
            raise SystemExit(f"fresh collection file changed: {path}")
        collection = _read(path)
        _validate_collection(collection)
        key = (str(collection["task_id"]), int(collection["seed"]))
        if key in observed:
            raise SystemExit(f"duplicate fresh collection: {key}")
        observed.add(key)
        collections.setdefault(key[0], []).append(collection)
    if observed != expected:
        raise SystemExit("fresh source seed set is incomplete or unexpected")

    permutation = dict(manifest["frozen_source_program_permutation"])
    totals = {
        "success_paths": 0, "authentic_supported": 0,
        "permuted_supported": 0, "unique_program_selections": 0,
        "correct_program_selections": 0, "operator_occurrences": 0,
        "unique_authentic_bindings": 0, "authentic_bindings": 0,
        "shuffled_bindings": 0, "replay_mismatches": 0,
    }
    lineages = []
    for task in sorted(programs):
        program = programs[task]
        success_paths: list[StructuralPath] = []
        grounding_parts = []
        task_replay_mismatches = 0
        for collection in collections.get(task, ()):
            task_replay_mismatches += int(collection["audit"]["replay_mismatches"])
            grounding_parts.append(
                _grounding_metrics(collection, program["induced_sequence"])
            )
            success_paths.extend(
                StructuralPath(
                    split="qualification", success=True,
                    steps=tuple(path["steps"]),
                )
                for path in collection.get("paths") or ()
                if path.get("success")
            )
        authentic = sum(
            sequence_contains(path.effects, program["induced_sequence"])
            for path in success_paths
        )
        control_task = permutation[task]
        permuted = sum(
            sequence_contains(
                path.effects, programs[control_task]["induced_sequence"],
            )
            for path in success_paths
        )
        selections = [
            _blind_program_selection(path.effects, programs)
            for path in success_paths
        ]
        unique_selections = sum(len(row) == 1 for row in selections)
        correct_selections = sum(row == (task,) for row in selections)
        grounding = {
            key: sum(row[key] for row in grounding_parts)
            for key in (
                "operator_occurrences", "unique_authentic_bindings",
                "authentic_correct_bindings", "shuffled_correct_bindings",
            )
        }
        lineages.append({
            "task_id": task,
            "fresh_seeds": sorted(int(row["seed"]) for row in collections[task]),
            "success_paths": len(success_paths),
            "authentic_sequence_supported": authentic,
            "permuted_task_id": control_task,
            "permuted_sequence_supported": permuted,
            "blind_program_selections": [list(row) for row in selections],
            "unique_program_selections": unique_selections,
            "correct_program_selections": correct_selections,
            "grounding": grounding,
            "replay_mismatches": task_replay_mismatches,
        })
        totals["success_paths"] += len(success_paths)
        totals["authentic_supported"] += authentic
        totals["permuted_supported"] += permuted
        totals["unique_program_selections"] += unique_selections
        totals["correct_program_selections"] += correct_selections
        totals["operator_occurrences"] += grounding["operator_occurrences"]
        totals["unique_authentic_bindings"] += grounding[
            "unique_authentic_bindings"
        ]
        totals["authentic_bindings"] += grounding[
            "authentic_correct_bindings"
        ]
        totals["shuffled_bindings"] += grounding[
            "shuffled_correct_bindings"
        ]
        totals["replay_mismatches"] += task_replay_mismatches

    paths = totals["success_paths"]
    occurrences = totals["operator_occurrences"]
    rates = {
        "authentic_sequence_support": totals["authentic_supported"] / paths if paths else 0.0,
        "permuted_sequence_support": totals["permuted_supported"] / paths if paths else 0.0,
        "unique_program_selection_rate": totals["unique_program_selections"] / paths if paths else 0.0,
        "correct_program_selection_rate": totals["correct_program_selections"] / paths if paths else 0.0,
        "authentic_binding_accuracy": totals["authentic_bindings"] / occurrences if occurrences else 0.0,
        "shuffled_binding_accuracy": totals["shuffled_bindings"] / occurrences if occurrences else 0.0,
    }
    thresholds = manifest["preregistered_thresholds"]
    gates = {
        "all_fresh_seeds_present": observed == expected,
        "every_task_has_success_evidence": all(row["success_paths"] > 0 for row in lineages),
        "all_replays_exact": totals["replay_mismatches"] == 0,
        "authentic_sequence_support": rates["authentic_sequence_support"] >= thresholds["minimum_authentic_sequence_support"],
        "authentic_beats_source_permuted": rates["authentic_sequence_support"] - rates["permuted_sequence_support"] >= thresholds["minimum_source_permutation_gap"],
        "correct_program_is_blindly_selected": rates["correct_program_selection_rate"] >= thresholds["minimum_correct_program_selection_rate"],
        "operator_bindings_are_unique": totals["unique_authentic_bindings"] == occurrences and occurrences > 0,
        "authentic_binding_accuracy": rates["authentic_binding_accuracy"] >= thresholds["minimum_authentic_binding_accuracy"],
        "authentic_beats_shuffled_binding": rates["authentic_binding_accuracy"] - rates["shuffled_binding_accuracy"] >= thresholds["minimum_shuffled_binding_gap"],
        "program_bodies_remain_distinct": len({stable_hash(row["induced_sequence"]) for row in programs.values()}) >= 2,
    }
    body = {
        "schema_version": "source-structural-fresh-report-v5c",
        "status": "SOURCE_STRUCTURAL_FRESH_VALIDATED" if all(gates.values()) else "SOURCE_STRUCTURAL_FRESH_FAILED",
        "manifest_sha256": manifest["manifest_sha256"],
        "summary_sha256": summary["summary_sha256"],
        "lineages": lineages,
        "aggregate": totals | rates,
        "gates": gates,
        "claim_boundary": "FRESH_SOURCE_ONLY;DOMAIN_SPECIFIC_STRUCTURAL_PROGRAM_INDUCTION_VALIDATION;NO_CROSS_DOMAIN_TARGET_CLAIM",
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "aggregate": report["aggregate"],
        "gates": report["gates"], "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
