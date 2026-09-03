#!/usr/bin/env python3
"""Analyze consumed V5b structural source acquisition before fresh freezing."""

from __future__ import annotations

import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_delta_induction import (  # noqa: E402
    StructuralPath,
    induce_structural_program,
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
    for path in collection.get("paths") or ():
        _self_hash(path, "path_sha256")
        for step in path.get("steps") or ():
            _self_hash(step, "transition_sha256")
    for group in collection.get("delta_contrast_groups") or ():
        _self_hash(group, "group_sha256")


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        next_row = [i]
        for j, b in enumerate(right, start=1):
            next_row.append(min(
                next_row[-1] + 1,
                row[j] + 1,
                row[j - 1] + int(a != b),
            ))
        row = next_row
    return row[-1]


def _maximum_contrast_derangement(
    programs: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    names = tuple(sorted(programs))
    candidates = []
    for order in permutations(names):
        if any(a == b for a, b in zip(names, order)):
            continue
        mapping = dict(zip(names, order))
        distance = sum(
            _edit_distance(
                programs[left]["induced_sequence"],
                programs[right]["induced_sequence"],
            )
            for left, right in mapping.items()
        )
        candidates.append((distance, stable_hash(mapping), mapping))
    if not candidates:
        raise ValueError("three source tasks are required for derangement")
    return max(candidates, key=lambda row: row[:2])[-1]


def _grounding_metrics(
    collection: Mapping[str, Any], sequence: Sequence[str],
) -> dict[str, Any]:
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
            atoms = [
                row["operator_type_id"]
                for row in structural_atom_descriptors(step["delta"])
            ]
            for observed in atoms:
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
        "authentic_accuracy": authentic / occurrences if occurrences else 0.0,
        "shuffled_accuracy": shuffled / occurrences if occurrences else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    summary = _read(args.summary)
    _self_hash(summary, "summary_sha256")
    collections: dict[str, list[dict[str, Any]]] = {}
    for receipt in summary.get("receipts") or ():
        path = Path(str(receipt["path"]))
        if _sha(path) != receipt["file_sha256"]:
            raise SystemExit(f"collection file changed: {path}")
        collection = _read(path)
        _validate_collection(collection)
        collections.setdefault(str(collection["task_id"]), []).append(collection)

    programs = {}
    for task, rows in collections.items():
        paths = tuple(
            StructuralPath(
                split=str(path["split"]), success=bool(path["success"]),
                steps=tuple(path["steps"]),
            )
            for collection in rows for path in collection.get("paths") or ()
        )
        program = induce_structural_program(
            paths,
            source_receipts_sha256=stable_hash([
                row["collection_sha256"] for row in rows
            ]),
        )
        validate_structural_program(program)
        programs[task] = program

    permutation = _maximum_contrast_derangement(programs)
    lineages = []
    totals = {
        "success_paths": 0, "authentic_supported": 0,
        "permuted_supported": 0, "operator_occurrences": 0,
        "authentic_bindings": 0, "shuffled_bindings": 0,
    }
    for task in sorted(programs):
        program = programs[task]
        qualification = [
            row for row in collections[task] if row["split"] == "qualification"
        ]
        success_paths = [
            StructuralPath(
                split=str(path["split"]), success=True,
                steps=tuple(path["steps"]),
            )
            for collection in qualification
            for path in collection.get("paths") or ()
            if path.get("success")
        ]
        authentic = sum(
            sequence_contains(row.effects, program["induced_sequence"])
            for row in success_paths
        )
        control_task = permutation[task]
        permuted = sum(
            sequence_contains(
                row.effects, programs[control_task]["induced_sequence"],
            )
            for row in success_paths
        )
        grounding_parts = [
            _grounding_metrics(row, program["induced_sequence"])
            for row in qualification
        ]
        grounding = {
            key: sum(int(row[key]) for row in grounding_parts)
            for key in (
                "operator_occurrences", "unique_authentic_bindings",
                "authentic_correct_bindings", "shuffled_correct_bindings",
            )
        }
        occurrences = grounding["operator_occurrences"]
        grounding["authentic_accuracy"] = (
            grounding["authentic_correct_bindings"] / occurrences
            if occurrences else 0.0
        )
        grounding["shuffled_accuracy"] = (
            grounding["shuffled_correct_bindings"] / occurrences
            if occurrences else 0.0
        )
        totals["success_paths"] += len(success_paths)
        totals["authentic_supported"] += authentic
        totals["permuted_supported"] += permuted
        totals["operator_occurrences"] += occurrences
        totals["authentic_bindings"] += grounding["authentic_correct_bindings"]
        totals["shuffled_bindings"] += grounding["shuffled_correct_bindings"]
        lineages.append({
            "task_id": task,
            "program": program,
            "permuted_task_id": control_task,
            "qualification_success_paths": len(success_paths),
            "authentic_sequence_supported": authentic,
            "permuted_sequence_supported": permuted,
            "grounding": grounding,
        })

    total_paths = totals["success_paths"]
    total_occurrences = totals["operator_occurrences"]
    rates = {
        "authentic_sequence_support": (
            totals["authentic_supported"] / total_paths if total_paths else 0.0
        ),
        "permuted_sequence_support": (
            totals["permuted_supported"] / total_paths if total_paths else 0.0
        ),
        "authentic_binding_accuracy": (
            totals["authentic_bindings"] / total_occurrences
            if total_occurrences else 0.0
        ),
        "shuffled_binding_accuracy": (
            totals["shuffled_bindings"] / total_occurrences
            if total_occurrences else 0.0
        ),
    }
    program_bodies = {
        stable_hash(row["program"]["induced_sequence"]) for row in lineages
    }
    gates = {
        "exact_three_source_tasks": len(lineages) == 3,
        "all_programs_qualified": all(
            row["program"]["status"] == "SOURCE_STRUCTURAL_PROGRAM_QUALIFIED"
            for row in lineages
        ),
        "at_least_two_distinct_program_bodies": len(program_bodies) >= 2,
        "qualification_success_sequences_supported": (
            rates["authentic_sequence_support"] >= 0.90
        ),
        "authentic_program_beats_source_permuted": (
            rates["authentic_sequence_support"]
            - rates["permuted_sequence_support"] >= 0.30
        ),
        "operators_have_unique_source_bindings": (
            rates["authentic_binding_accuracy"] >= 0.90
        ),
        "authentic_bindings_beat_shuffled": (
            rates["authentic_binding_accuracy"]
            - rates["shuffled_binding_accuracy"] >= 0.50
        ),
        "no_named_policy_or_source_action_export": all(
            row["program"]["source_action_identity_exported"] is False
            and row["program"]["source_task_identity_used_as_feature"] is False
            for row in lineages
        ),
    }
    body = {
        "schema_version": "source-structural-development-report-v5b",
        "status": (
            "SOURCE_STRUCTURAL_DEVELOPMENT_PASSED"
            if all(gates.values()) else "SOURCE_STRUCTURAL_DEVELOPMENT_FAILED"
        ),
        "summary_sha256": summary["summary_sha256"],
        "source_program_permutation": permutation,
        "lineages": lineages,
        "aggregate": totals | rates,
        "gates": gates,
        "claim_boundary": (
            "CONSUMED_SOURCE_DEVELOPMENT_AND_QUALIFICATION_ONLY;FRESH_SOURCE_"
            "RESERVE_NOT_YET_FROZEN;NO_TARGET_CLAIM"
        ),
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
