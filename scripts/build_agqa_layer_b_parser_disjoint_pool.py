#!/usr/bin/env python3
"""Build an operator-free AGQA pool disjoint from semantic-parser supervision.

Input compiler rows contain question/program metadata but no answers or scene
graphs.  Programs are lowered to target-native operator-free semantics solely
to stratify and later audit a qualification cohort.  Runtime still receives
question text only and uses the frozen parser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_semantic_slots import serialize_compact_semantic_target
from motif_transfer.contracts import stable_hash


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_ids(paths: list[Path]) -> set[str]:
    output: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                output.add(str(json.loads(line)["task_id"]))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-data", type=Path, nargs="+", required=True)
    parser.add_argument("--parser-supervision", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("parser-disjoint pool artifacts are immutable")

    excluded = _task_ids(args.parser_supervision)
    seen: set[str] = set()
    source_rows = excluded_rows = duplicate_rows = kept_rows = 0
    roots: dict[str, int] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as writer:
        for source in args.compiler_data:
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    source_rows += 1
                    row = json.loads(line); task_id = str(row["task_id"])
                    if task_id in excluded:
                        excluded_rows += 1
                        continue
                    if task_id in seen:
                        duplicate_rows += 1
                        continue
                    seen.add(task_id)
                    target = serialize_compact_semantic_target(str(row["program"]))
                    semantic_root = target.split("(", 1)[0]
                    roots[semantic_root] = roots.get(semantic_root, 0) + 1
                    writer.write(json.dumps({
                        "task_id": task_id, "video_id": str(row["video_id"]),
                        "input": "parse AGQA semantics: " + str(row["question"]),
                        "target": target, "answer_read": False,
                        "scene_graph_read": False,
                        "functional_program_in_model_input": False,
                        "parser_supervision_disjoint": True,
                    }, sort_keys=True) + "\n")
                    kept_rows += 1

    body = {
        "schema_version": "agqa-layer-b-parser-disjoint-semantic-pool-v1",
        "status": "PARSER_DISJOINT_POOL_FROZEN",
        "compiler_data": [str(path) for path in args.compiler_data],
        "compiler_data_sha256s": [_file_sha256(path) for path in args.compiler_data],
        "parser_supervision": [str(path) for path in args.parser_supervision],
        "parser_supervision_sha256s": [_file_sha256(path) for path in args.parser_supervision],
        "parser_supervision_task_count": len(excluded),
        "source_rows": source_rows, "excluded_rows": excluded_rows,
        "duplicate_rows": duplicate_rows, "kept_rows": kept_rows,
        "semantic_roots": dict(sorted(roots.items())),
        "output_sha256": _file_sha256(args.output),
        "answers_read": False, "scene_graphs_read": False,
        "runtime_functional_program_allowed": False,
    }
    body["manifest_sha256"] = stable_hash(body)
    args.manifest.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
