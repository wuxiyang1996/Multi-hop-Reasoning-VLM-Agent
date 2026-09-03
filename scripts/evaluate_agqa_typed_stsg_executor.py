#!/usr/bin/env python3
"""Development-only semantic qualification of the typed STSG executor."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

from motif_transfer.agqa_stsg_typed_executor import AGQATypedSTSGExecutor
from motif_transfer.agqa_oracle_query_mdp import load_agqa_id_to_text
from motif_transfer.contracts import stable_hash
from motif_transfer.official_video_event_graph import load_builtin_only_pickle
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def clean(value) -> str:
    return " ".join(str(value).replace("_", " ").casefold().split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-data", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--stsg", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("executor evaluation is immutable")
    compiler_rows = []
    with (args.compiler_data / "validation.jsonl").open(encoding="utf-8") as f:
        for line in f:
            compiler_rows.append(json.loads(line))
            if len(compiler_rows) >= args.max_rows: break
    wanted = {row["task_id"] for row in compiler_rows}
    outcomes = {}
    with zipfile.ZipFile(args.archive) as z, z.open(args.entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            if task_id in wanted:
                outcomes[task_id] = clean(row["answer"])
                if len(outcomes) == len(wanted): break
    if set(outcomes) != wanted: raise ValueError("development outcome join incomplete")
    corpus = load_builtin_only_pickle(args.stsg)
    ontology = load_agqa_id_to_text(args.ontology)
    capabilities = json.loads(args.capabilities.read_text(encoding="utf-8"))
    stsg_sha = sha(args.stsg)
    counts: Counter[str] = Counter(); roots: dict[str, Counter[str]] = {}
    failures: Counter[str] = Counter(); failure_reasons: Counter[str] = Counter(); samples = []
    for row in compiler_rows:
        video = row["video_id"]
        executor = AGQATypedSTSGExecutor(
            graph=corpus[video], id_to_text=ontology,
            graph_sha256=stable_hash({"stsg_sha256": stsg_sha, "video_id": video}),
            authorized_operators=capabilities["authorized_operators"],
            authorized_compositions=capabilities.get("authorized_compositions"),
        )
        receipt = executor.execute(row["program"], functional_program_source="DEVELOPMENT_ORACLE")
        root = row["program"].split("(", 1)[0]
        bucket = roots.setdefault(root, Counter()); counts["rows"] += 1; bucket["rows"] += 1
        if receipt.status == "COMMITTED":
            counts["committed"] += 1; bucket["committed"] += 1
            if clean(receipt.prediction) == outcomes[row["task_id"]]:
                counts["correct"] += 1; bucket["correct"] += 1
            elif len(samples) < 100:
                samples.append({"task_id": row["task_id"], "question": row["question"],
                                "answer": outcomes[row["task_id"]], "prediction": receipt.prediction,
                                "program": row["program"], "reason": receipt.reason})
        else:
            failures[receipt.reason.split(":", 2)[0]] += 1
            failure_reasons[receipt.reason] += 1
            if len(samples) < 100:
                samples.append({"task_id": row["task_id"], "question": row["question"],
                                "answer": outcomes[row["task_id"]], "prediction": None,
                                "program": row["program"], "reason": receipt.reason})
    total = counts["rows"]
    body = {
        "schema_version": "agqa-typed-stsg-executor-development-eval-v1",
        "status": "EXECUTOR_DEVELOPMENT_EVALUATED",
        "authority": "TARGET_DEVELOPMENT_OUTCOMES_AND_ORACLE_PROGRAMS_ONLY",
        "formal_outcomes_read": False, "formal_programs_read": False,
        "metrics": {"rows": total, "committed": counts["committed"],
                    "correct": counts["correct"],
                    "coverage": counts["committed"] / total,
                    "overall_accuracy": counts["correct"] / total,
                    "conditional_accuracy": counts["correct"] / counts["committed"] if counts["committed"] else 0.0},
        "by_root": {k: dict(v) for k, v in sorted(roots.items())},
        "failure_classes": dict(failures),
        "failure_reasons": dict(failure_reasons.most_common()),
        "failure_sample": samples,
        "source_capability_artifact_sha256": capabilities["artifact_sha256"],
        "stsg_sha256": stsg_sha,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "metrics": body["metrics"],
                      "failure_classes": body["failure_classes"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
