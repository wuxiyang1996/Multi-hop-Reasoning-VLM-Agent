#!/usr/bin/env python3
"""Freeze outcome-blind fresh ALFWorld valid_unseen transfer task IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


TASK_PATTERN = re.compile(
    r"(?:pick|look)_[A-Za-z0-9_\-]+/trial_[A-Za-z0-9_\-]+/game\.tw-pddl"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(seed: int, task_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{task_id}".encode()).hexdigest()


def _consumed_ids(repo: Path, valid_ids: set[str]) -> tuple[set[str], dict[str, str]]:
    consumed = set()
    receipts = {}
    pool_only_manifests = {"configs/target_manifests_v1.json"}
    for root_name in ("configs", "docs", "runs"):
        root = repo / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            relative = path.relative_to(repo).as_posix()
            if relative in pool_only_manifests:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            matches = set(TASK_PATTERN.findall(text)) & valid_ids
            if matches:
                consumed.update(matches)
                receipts[relative] = _sha256(path)
    return consumed, dict(sorted(receipts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=94701)
    parser.add_argument("--qualification-size", type=int, default=24)
    parser.add_argument("--heldout-size", type=int, default=24)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {output}")
    valid_root = (args.data_root / "json_2.1.1" / "valid_unseen").resolve()
    valid_ids = {
        path.relative_to(valid_root).as_posix()
        for path in valid_root.glob("*/trial_*/game.tw-pddl")
    }
    consumed, receipts = _consumed_ids(args.repo.resolve(), valid_ids)
    available = sorted(
        valid_ids - consumed,
        key=lambda task_id: (_rank(args.seed, task_id), task_id),
    )
    required = args.qualification_size + args.heldout_size
    if len(available) < required:
        raise SystemExit(f"only {len(available)} unconsumed tasks for {required} cells")
    qualification = available[: args.qualification_size]
    heldout = available[args.qualification_size:required]
    body = {
        "schema_version": "fresh-alfworld-sokoban-transfer-split-v1",
        "status": "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET",
        "selection_authority": "VALID_UNSEEN_PATH_NAMES_ONLY",
        "selection_used_task_contents": False,
        "selection_used_target_outcomes": False,
        "seed": args.seed,
        "valid_unseen_root": str(valid_root),
        "valid_unseen_task_count": len(valid_ids),
        "previously_consumed_task_count": len(consumed),
        "consumed_task_ids_sha256": hashlib.sha256(
            json.dumps(sorted(consumed), separators=(",", ":")).encode()
        ).hexdigest(),
        "consumed_receipt_files_sha256": receipts,
        "splits": {
            "qualification": qualification,
            "held_out": heldout,
        },
        "split_contract": (
            "SHA256_SEED_AND_RELATIVE_TASK_ID; QUALIFICATION_FIRST_THEN_HELDOUT"
        ),
    }
    payload = body | {
        "manifest_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "manifest_sha256": payload["manifest_sha256"],
        "valid_unseen_task_count": len(valid_ids),
        "previously_consumed_task_count": len(consumed),
        "qualification_size": len(qualification),
        "heldout_size": len(heldout),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
