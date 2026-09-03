#!/usr/bin/env python3
"""Freeze question-to-program supervision without reading formal programs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

from motif_transfer.agqa_typed_program import parse_program, serialize_program
from motif_transfer.contracts import stable_hash
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("compiler dataset is immutable")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    partitions = {
        "train": set(split["partitions"]["router_train"]),
        "validation": set(split["partitions"]["router_validation"]),
    }
    formal = set(split["partitions"]["formal_holdout"])
    args.output_dir.mkdir(parents=True)
    paths = {name: args.output_dir / f"{name}.jsonl" for name in partitions}
    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    counts: Counter[str] = Counter()
    roots: dict[str, Counter[str]] = {name: Counter() for name in partitions}
    formal_skipped = 0
    try:
        with zipfile.ZipFile(split["archive_path"]) as bundle, bundle.open(split["entry"]) as raw:
            for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
                video = str(row["video_id"])
                if video in formal:
                    formal_skipped += 1
                    continue
                bucket = next((name for name, videos in partitions.items() if video in videos), None)
                if bucket is None:
                    continue
                canonical = serialize_program(parse_program(str(row["program"])))
                record = {
                    "task_id": str(task_id), "question": str(row["question"]),
                    "program": canonical, "video_id": video,
                }
                handles[bucket].write(json.dumps(record, sort_keys=True) + "\n")
                counts[bucket] += 1
                roots[bucket][canonical.split("(", 1)[0]] += 1
    finally:
        for handle in handles.values():
            handle.close()
    body = {
        "schema_version": "agqa-full-program-compiler-data-v1",
        "status": "FROZEN_TARGET_NATIVE_COMPILER_SUPERVISION",
        "authority": "QUESTION_AND_FUNCTIONAL_PROGRAM_FROM_TRAIN_DEVELOPMENT_VIDEOS_ONLY",
        "formal_programs_read": False,
        "formal_answers_read": False,
        "formal_rows_skipped_before_program_access": formal_skipped,
        "split_sha256": split["split_sha256"],
        "rows": dict(counts),
        "root_counts": {name: dict(sorted(value.items())) for name, value in roots.items()},
        "files": {name: {"path": path.name, "sha256": file_hash(path)} for name, path in paths.items()},
        "runtime_visible_fields": ["question"],
        "runtime_forbidden_fields": ["answer", "program", "sg_grounding", "structural", "semantic"],
    }
    body["manifest_sha256"] = stable_hash(body)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(body, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
