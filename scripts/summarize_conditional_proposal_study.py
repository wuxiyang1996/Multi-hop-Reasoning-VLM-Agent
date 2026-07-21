#!/usr/bin/env python3
"""Freeze a paired four-condition conditional-proposal capability summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.conditional_node_program import proposal_from_dict


CONDITIONS = ("correct", "renamed", "randomized", "target_only")


def _hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_hashed(path, hash_key):
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    claimed = unsigned.pop(hash_key, None)
    if claimed != _hash(unsigned):
        raise ValueError(f"artifact hash mismatch: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for condition in CONDITIONS:
        parser.add_argument(f"--{condition}-proposals", type=Path, required=True)
        parser.add_argument(f"--{condition}-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    condition_rows = {}
    proposal_hashes = {}
    admission_hashes = {}
    seeds_by_condition = {}
    for condition in CONDITIONS:
        proposals = _load_hashed(
            getattr(args, f"{condition}_proposals"), "artifact_sha256",
        )
        admission = _load_hashed(
            getattr(args, f"{condition}_admission"), "artifact_hash",
        )
        if (
            proposals["condition"] != condition
            or len(proposals["rows"]) != 13
            or proposals["n_invalid"] != 0
            or len(proposals["candidates"]) != 13
        ):
            raise SystemExit(f"incomplete successful enumeration: {condition}")
        admitted_hashes = {
            candidate["proposal_hash"] for candidate in admission["candidates"]
        }
        candidate_by_index = {
            int(candidate["proposal_source"].rsplit("graph", 1)[1]): candidate
            for candidate in proposals["candidates"]
        }
        rows = []
        seeds = []
        for graph_index in range(13):
            row = proposals["rows"][graph_index]
            receipt = row["receipt_payload"]
            if int(receipt["graph_index"]) != graph_index:
                raise SystemExit("merged enumeration row order mismatch")
            seeds.append(int(receipt["proposal_seed"]))
            candidate = candidate_by_index[graph_index]
            rows.append({
                "graph_index": graph_index,
                "source_hypothesis_hash": receipt["source_hypothesis_hash"],
                "proposal_seed": int(receipt["proposal_seed"]),
                "proposal_hash": proposal_from_dict(candidate).content_hash(),
                "admitted": (
                    proposal_from_dict(candidate).content_hash() in admitted_hashes
                ),
                "prompt_tokens": int(receipt["usage"].get("prompt_tokens") or 0),
                "completion_tokens": int(
                    receipt["usage"].get("completion_tokens") or 0
                ),
                "reported_cost": float(receipt["usage"].get("cost") or 0.0),
            })
        condition_rows[condition] = rows
        seeds_by_condition[condition] = seeds
        proposal_hashes[condition] = proposals["artifact_sha256"]
        admission_hashes[condition] = admission["artifact_hash"]
    if len({tuple(value) for value in seeds_by_condition.values()}) != 1:
        raise SystemExit("proposal seeds are not paired across conditions")
    metrics = {}
    for condition, rows in condition_rows.items():
        metrics[condition] = {
            "n_registered": len(rows),
            "n_exact_format_after_endpoint_retry": len(rows),
            "n_admitted": sum(row["admitted"] for row in rows),
            "admission_rate": sum(row["admitted"] for row in rows) / len(rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "completion_tokens": sum(row["completion_tokens"] for row in rows),
            "reported_cost": sum(row["reported_cost"] for row in rows),
        }
    paired = {}
    correct = condition_rows["correct"]
    for control in CONDITIONS[1:]:
        values = condition_rows[control]
        paired[control] = {
            "both_admitted": sum(
                left["admitted"] and right["admitted"]
                for left, right in zip(correct, values)
            ),
            "correct_only": sum(
                left["admitted"] and not right["admitted"]
                for left, right in zip(correct, values)
            ),
            "control_only": sum(
                not left["admitted"] and right["admitted"]
                for left, right in zip(correct, values)
            ),
            "neither": sum(
                not left["admitted"] and not right["admitted"]
                for left, right in zip(correct, values)
            ),
        }
    correct_exceeds_every_control = all(
        metrics["correct"]["n_admitted"] > metrics[control]["n_admitted"]
        for control in CONDITIONS[1:]
    )
    output = {
        "schema_version": 1,
        "study": "qwen3max_seeded_complete_graph_conditional_proposal_capability",
        "conditions": list(CONDITIONS), "n_graph_slots": 13,
        "proposal_seeds": seeds_by_condition["correct"],
        "proposal_artifact_hashes": proposal_hashes,
        "admission_artifact_hashes": admission_hashes,
        "metrics": metrics, "paired_admission": paired,
        "gate": {
            "correct_exceeds_every_control": correct_exceeds_every_control,
            "authorizes_online_source_pilot": correct_exceeds_every_control,
            "authorizes_large_scale_2x4": False,
            "reason": (
                "CORRECT_SOURCE_DOES_NOT_EXCEED_RENAMED_AND_RANDOMIZED_CONTROLS"
                if not correct_exceeds_every_control else
                "ONLINE_PILOT_REQUIRED_BEFORE_ANY_SCALE_UP"
            ),
        },
        "per_condition_rows": condition_rows,
        "claim_limit": (
            "Development proposal-capability study only; admission is not online "
            "task success or evidence of positive skill transfer."
        ),
    }
    output["artifact_sha256"] = _hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "metrics": metrics, "paired_admission": paired, "gate": output["gate"],
        "artifact_sha256": output["artifact_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
