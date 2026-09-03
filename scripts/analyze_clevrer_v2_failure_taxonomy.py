#!/usr/bin/env python3
"""Build a paper-facing, observable CLEVRER V2 failure taxonomy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from motif_transfer.contracts import stable_hash


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gold(question: dict[str, Any], family: str) -> str:
    if family == "descriptive":
        return str(question["answer"])
    return "".join("1" if row["answer"] == "correct" else "0" for row in question["choices"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen, actor = _load(args.predictions), _load(args.actor)
    annotations = {
        int(scene["scene_index"]): {int(q["question_id"]): q for q in scene["questions"]}
        for scene in _load(args.annotations)
    }
    taxonomy: Counter[str] = Counter()
    per_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in frozen["rows"]:
        family = str(row["question_family"])
        gold = _gold(annotations[int(row["video_id"])][int(row["question_id"])], family)
        neural_ok = row["predictions"]["neural_only"] == gold
        source_ok = row["predictions"]["source_induced"] == gold
        generic_ok = row["predictions"]["generic_symbolic"] == gold
        if row["source_commit"]:
            if source_ok and not neural_ok:
                label = "symbolic_recovery"
            elif not source_ok and neural_ok:
                label = "negative_transfer"
            elif source_ok:
                label = "committed_both_correct"
            else:
                label = "committed_residual_error"
        else:
            if neural_ok:
                label = "abstained_fallback_correct"
            elif generic_ok:
                label = "abstained_generic_headroom"
            else:
                label = "abstained_shared_failure"
        taxonomy[label] += 1
        per_family[family][label] += 1

    total = len(frozen["rows"])
    gates = {
        "taxonomy_partitions_all_tasks": sum(taxonomy.values()) == total,
        "paired_wins_match_formal": taxonomy["symbolic_recovery"] == 464,
        "paired_losses_match_formal": taxonomy["negative_transfer"] == 23,
        "actor_usage_complete": int(actor["provider_calls"]) == total,
        "no_causal_error_attribution_from_final_answers": True,
    }
    body = {
        "schema_version": "clevrer-v2-observable-failure-taxonomy-v1",
        "status": "CLEVRER_V2_FAILURE_TAXONOMY_VERIFIED" if all(gates.values()) else "CLEVRER_V2_FAILURE_TAXONOMY_FAILED",
        "tasks": total,
        "taxonomy": dict(sorted(taxonomy.items())),
        "per_family": {key: dict(sorted(value.items())) for key, value in sorted(per_family.items())},
        "neural_actor_cost": {
            "provider_calls": actor["provider_calls"],
            **actor["usage"],
            "receipt_backed_usd": None,
            "historical_estimate_usd": 0.24,
            "estimate_disclosure": "Approximation recorded in the experiment notes; provider receipts contain tokens but no immutable USD charge.",
        },
        "input_file_sha256s": {
            "predictions": _sha(args.predictions),
            "annotations": _sha(args.annotations),
            "actor": _sha(args.actor),
        },
        "gates": gates,
        "claim_boundary": "Categories are observable commit/correctness outcomes. committed_residual_error cannot distinguish grounder, parser, executor, or dataset ambiguity without additional intervention.",
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": body["status"], "taxonomy": body["taxonomy"],
        "neural_actor_cost": body["neural_actor_cost"],
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
