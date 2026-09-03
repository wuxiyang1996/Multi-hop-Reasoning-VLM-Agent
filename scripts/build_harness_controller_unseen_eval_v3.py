#!/usr/bin/env python3
"""Freeze a deduplicated model-unseen controller evaluation set."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include", nargs="+", required=True, type=Path)
    parser.add_argument("--exclude", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite frozen unseen evaluation artifacts")

    excluded_prompts = {
        row["prompt"] for path in args.exclude for row in _rows(path)
    }
    included: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in args.include:
        for row in _rows(path):
            if row["prompt"] in excluded_prompts:
                continue
            included.setdefault(row["prompt"], []).append((path, row))
    ambiguous = {
        prompt: support for prompt, support in included.items()
        if len({row["completion"] for _, row in support}) != 1
    }
    if ambiguous:
        raise SystemExit(f"unseen prompt has inconsistent labels: {len(ambiguous)}")
    output_rows = []
    for prompt, support in sorted(included.items()):
        representative = min(
            (row for _, row in support), key=lambda row: str(row["example_id"]),
        )
        row = dict(representative)
        row["unseen_support_count"] = len(support)
        row["unseen_source_files"] = sorted({str(path.resolve()) for path, _ in support})
        row["supporting_example_ids"] = sorted({
            str(item["example_id"]) for _, item in support
        })
        output_rows.append(row)
    if not output_rows:
        raise SystemExit("no model-unseen prompts remain")
    if len({row["prompt"] for row in output_rows}) != len(output_rows):
        raise SystemExit("unseen evaluation output is not prompt-deduplicated")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    decision_counts = Counter(
        json.loads(row["completion"])["decision"] for row in output_rows
    )
    manifest = {
        "schema_version": "harness-controller-model-unseen-eval-v3",
        "status": "FROZEN_BEFORE_V3_ADAPTER_TRAINING",
        "include_files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in args.include
        ],
        "exclude_files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in args.exclude
        ],
        "output_file": str(args.output.resolve()),
        "output_file_sha256": _sha256(args.output),
        "examples": len(output_rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "prompt_overlap_with_exclusions": sum(
            row["prompt"] in excluded_prompts for row in output_rows
        ),
        "ambiguous_prompt_count": 0,
        "target_data_used": False,
        "claim_boundary": (
            "FRESH_TO_MODEL_SOURCE_FAMILY_REPLICATION;RETROSPECTIVE_SOURCE_"
            "RESERVES;NOT_A_NEW_PROSPECTIVE_TARGET_TRANSFER_CLAIM"
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
