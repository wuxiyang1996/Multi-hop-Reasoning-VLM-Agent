#!/usr/bin/env python3
"""Freeze outcome-blind CLEVRER development/formal causal-query splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FAMILIES = ("explanatory", "predictive", "counterfactual")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(salt: str, sample_id: str) -> str:
    return hashlib.sha256(f"{salt}|{sample_id}".encode("utf-8")).hexdigest()


def _collect_ids(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            output.update(_collect_ids(child))
    elif isinstance(value, list):
        for child in value:
            output.update(_collect_ids(child))
    elif isinstance(value, str) and ".mp4.Q" in value:
        output.add(value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", default="clevrer-causal-query-v9-20260813")
    parser.add_argument("--development-per-family", type=int, default=48)
    parser.add_argument("--formal-per-family", type=int, default=48)
    parser.add_argument("--reserve-per-family", type=int, default=24)
    args = parser.parse_args()
    for value in (
        args.development_per_family,
        args.formal_per_family,
        args.reserve_per_family,
    ):
        if value <= 0:
            raise ValueError("all split sizes must be positive")

    excluded: set[str] = set()
    exclusion_hashes = {}
    for path in args.exclude_manifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(_collect_ids(payload))
        exclusion_hashes[str(path.resolve())] = _sha256(path)

    # The selector deliberately reads only scene/question identifiers and the
    # public question_type metadata. Choice answers and programs are not
    # inspected, ranked, or serialized.
    candidates = {family: [] for family in FAMILIES}
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    for scene in annotations:
        scene_id = int(scene["scene_index"])
        for question in scene["questions"]:
            family = str(question.get("question_type") or "")
            if family not in candidates:
                continue
            sample_id = f"video_{scene_id:05d}.mp4.Q{int(question['question_id'])}"
            if sample_id not in excluded:
                candidates[family].append(sample_id)

    needed = (
        args.development_per_family
        + args.formal_per_family
        + args.reserve_per_family
    )
    roles = {"development": [], "formal": [], "reserve": []}
    family_roles = {}
    for family in FAMILIES:
        ordered = sorted(candidates[family], key=lambda value: _stable_key(args.salt, value))
        if len(ordered) < needed:
            raise ValueError(f"insufficient unconsumed {family} questions")
        development = ordered[: args.development_per_family]
        formal_start = args.development_per_family
        formal = ordered[formal_start : formal_start + args.formal_per_family]
        reserve = ordered[formal_start + args.formal_per_family : needed]
        family_roles[family] = {
            "development": development,
            "formal": formal,
            "reserve": reserve,
        }
        roles["development"].extend(development)
        roles["formal"].extend(formal)
        roles["reserve"].extend(reserve)

    if len(set().union(*(set(values) for values in roles.values()))) != sum(
        len(values) for values in roles.values()
    ):
        raise AssertionError("frozen CLEVRER roles overlap")
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_CAUSAL_QUERY_DEVELOPMENT_OUTCOMES",
        "selection_rule": "Within each public question_type, exclude every prior manifest ID, sort by sha256(salt|sample_id), then take fixed contiguous development/formal/reserve blocks.",
        "selection_fields": ["scene_index", "question_id", "question_type"],
        "forbidden_selection_fields": ["answer", "choice.answer", "program", "question", "choice"],
        "outcomes_or_answers_read_by_selector": False,
        "salt_sha256": hashlib.sha256(args.salt.encode("utf-8")).hexdigest(),
        "annotations_sha256": _sha256(args.annotations),
        "excluded_manifest_sha256": exclusion_hashes,
        "excluded_id_count": len(excluded),
        "benchmarks": {
            "clevrer": {
                "families": list(FAMILIES),
                "family_roles": family_roles,
                "splits": roles,
            }
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "excluded_id_count": len(excluded),
        "split_counts": {name: len(values) for name, values in roles.items()},
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
