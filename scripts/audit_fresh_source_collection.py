#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from motif_transfer.instrumented_import import import_native_source_batch
from motif_transfer.phase1_assets import read_jsonl
from motif_transfer.source_execution_motifs import (
    build_execution_traces,
    execution_affordance_report,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_one(root: Path, expected_episodes: int) -> dict:
    required = (
        "manifest.json", "events.jsonl", "episodes.jsonl",
        "replay_receipts.jsonl", "matched_policy_records.jsonl",
        "matched_policy_replays.jsonl",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        return {
            "evidence_dir": str(root.resolve()),
            "accepted": False,
            "failure_codes": ["MISSING_" + name.upper() for name in missing],
        }
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest.get("metadata") or {}
    episodes = import_native_source_batch(root)
    traces = build_execution_traces(episodes)
    matched = read_jsonl(root / "matched_policy_records.jsonl")
    replays = read_jsonl(root / "matched_policy_replays.jsonl")
    failures = []
    if len(episodes) != expected_episodes:
        failures.append("EPISODE_COUNT_MISMATCH")
    if any(episode.gaps for episode in episodes):
        failures.append("IMPORT_GAPS_PRESENT")
    if not all(record.validate() for episode in episodes for record in episode.records):
        failures.append("SOURCE_RECEIPT_VALIDATION_FAILED")
    if metadata.get("human_policy_hints") is not False:
        failures.append("NO_HUMAN_HINTS_RECEIPT_MISSING")
    if metadata.get("policy_hint_profile") != "NO_HUMAN_GAME_HINTS_V1":
        failures.append("POLICY_HINT_PROFILE_MISMATCH")
    if metadata.get("lora_checkpoint_loaded") is not True:
        failures.append("BEST_CHECKPOINT_NOT_RECEIPTED")
    if not metadata.get("runtime_code_sha256"):
        failures.append("RUNTIME_CODE_RECEIPT_MISSING")
    treatment_counts = Counter(str(row.get("treatment")) for row in matched)
    if set(treatment_counts) != {
        "B", "G_MINUS_S", "G_PLUS_S", "G_PLUS_RANDOM",
    } or len(set(treatment_counts.values())) != 1:
        failures.append("MATCHED_TREATMENTS_UNBALANCED")
    replay_status = Counter(str(row.get("status")) for row in replays)
    if set(replay_status) != {"INTERVENTION_OBSERVED"}:
        failures.append("MATCHED_REPLAY_NOT_FULLY_OBSERVED")
    matched_meta = manifest.get("matched_policy_treatments") or {}
    for name, field in (
        ("matched_policy_records.jsonl", "records_sha256"),
        ("matched_policy_replays.jsonl", "replays_sha256"),
    ):
        if matched_meta.get(field) != _sha256(root / name):
            failures.append(f"{field.upper()}_MISMATCH")
    split_counts = Counter(trace.split for trace in traces)
    if split_counts != {
        "discovery": expected_episodes // 3,
        "qualification": expected_episodes // 3,
        "held_out": expected_episodes // 3,
    }:
        failures.append("FROZEN_SPLIT_BALANCE_FAILED")
    return {
        "evidence_dir": str(root.resolve()),
        "game": sorted({episode.game for episode in episodes}),
        "accepted": not failures,
        "failure_codes": sorted(set(failures)),
        "episodes": len(episodes),
        "transitions": sum(len(episode.records) for episode in episodes),
        "treatment_counts": dict(sorted(treatment_counts.items())),
        "matched_replay_status_counts": dict(sorted(replay_status.items())),
        "execution_affordance": execution_affordance_report(traces),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of completed fresh source collections"
    )
    parser.add_argument("evidence_dirs", nargs="+", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [
        _audit_one(root, args.expected_episodes)
        for root in args.evidence_dirs
    ]
    payload = {
        "schema_version": 1,
        "authority": "MECHANICAL_FRESH_SOURCE_INTEGRITY_AUDIT",
        "rows": rows,
        "all_accepted": all(row["accepted"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "all_accepted": payload["all_accepted"],
        "games": len(rows),
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    if not payload["all_accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
