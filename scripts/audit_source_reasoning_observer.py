#!/usr/bin/env python3
"""Audit a shadow reasoning-observer batch without interpreting game semantics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.instrumented_import import import_instrumented_batch  # noqa: E402
from motif_transfer.phase1_assets import read_jsonl  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_rows(root: Path) -> dict[tuple[int, str], dict]:
    events = read_jsonl(root / "events.jsonl")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("episode_id"))].append(event)
    result = {}
    for episode_id, rows in grouped.items():
        reset = next((row for row in rows if row.get("kind") == "RESET"), None)
        first_obs = next(
            (row for row in rows if row.get("kind") == "OBSERVATION"
             and (row.get("payload") or {}).get("step") == 0),
            None,
        )
        if reset is None or first_obs is None:
            continue
        seed = (reset.get("payload") or {}).get("requested_seed")
        if not isinstance(seed, int):
            continue
        initial_hash = stable_hash((first_obs.get("payload") or {}).get("observable_state", ""))
        env_steps = sorted(
            (row for row in rows if row.get("kind") == "ENVIRONMENT_STEP"),
            key=lambda row: int((row.get("payload") or {}).get("step", -1)),
        )
        result[(seed, initial_hash)] = {
            "episode_id": episode_id,
            "actions": [str((row.get("payload") or {}).get("executed_action", "")) for row in env_steps],
            "rewards": [float((row.get("payload") or {}).get("reward", 0.0)) for row in env_steps],
        }
    return result


def _policy_identity(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest.get("metadata") or {}
    checkpoint = metadata.get("checkpoint_receipt") or {}
    bank = metadata.get("skill_bank_receipt") or {}
    return {
        "game": metadata.get("game"),
        "model": metadata.get("model"),
        "checkpoint_files_sha256": checkpoint.get("files_sha256"),
        "skill_bank_sha256": bank.get("sha256"),
        "episode_seed_base": metadata.get("episode_seed_base"),
        "max_steps": metadata.get("max_steps"),
        "lora_checkpoint_loaded": metadata.get("lora_checkpoint_loaded"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-evidence", type=Path, required=True)
    parser.add_argument("--reference-overlay-receipt", type=Path, required=True)
    parser.add_argument("--observer-evidence", type=Path, required=True)
    parser.add_argument("--observer-overlay-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    patch_path = REPO / "patches/source_reasoning_shadow_observer.patch"
    patch_sha256 = _sha256(patch_path)
    no_hints_sha256 = _sha256(REPO / "patches/source_no_human_policy_hints.patch")
    seed_control_sha256 = _sha256(REPO / "patches/source_request_seed_control.patch")
    overlay_receipt = json.loads(args.observer_overlay_receipt.read_text(encoding="utf-8"))
    reference_overlay = json.loads(args.reference_overlay_receipt.read_text(encoding="utf-8"))
    if overlay_receipt.get("reasoning_observer") is not True:
        raise ValueError("overlay receipt is not a reasoning-observer collection")
    applied_patch_hashes = set(overlay_receipt.get("applied_patch_sha256") or [])
    if patch_sha256 not in applied_patch_hashes:
        raise ValueError("overlay receipt does not contain the shadow-observer patch")
    if overlay_receipt.get("human_policy_hints_excluded") is not True:
        raise ValueError("observer collection retained human-authored policy hints")
    if no_hints_sha256 not in applied_patch_hashes:
        raise ValueError("overlay receipt does not contain the no-human-hints patch")
    if overlay_receipt.get("request_seed_control") is not True:
        raise ValueError("observer collection lacks exact-request sampling control")
    if seed_control_sha256 not in applied_patch_hashes:
        raise ValueError("observer overlay lacks the request-seed-control patch")
    reference_patch_hashes = set(reference_overlay.get("applied_patch_sha256") or [])
    if reference_overlay.get("reasoning_observer") is not False:
        raise ValueError("reference collection must not run the reasoning observer")
    if reference_overlay.get("human_policy_hints_excluded") is not True:
        raise ValueError("reference collection retained human-authored policy hints")
    if no_hints_sha256 not in reference_patch_hashes:
        raise ValueError("reference overlay lacks the no-human-hints patch")
    if reference_overlay.get("request_seed_control") is not True:
        raise ValueError("reference collection lacks exact-request sampling control")
    if seed_control_sha256 not in reference_patch_hashes:
        raise ValueError("reference overlay lacks the request-seed-control patch")

    reference = _episode_rows(args.reference_evidence)
    observer = _episode_rows(args.observer_evidence)
    reference_identity = _policy_identity(args.reference_evidence)
    observer_identity = _policy_identity(args.observer_evidence)
    imported = import_instrumented_batch(args.observer_evidence)
    pairs = []
    for identity in sorted(set(reference) & set(observer)):
        left, right = reference[identity], observer[identity]
        pairs.append({
            "requested_seed": identity[0],
            "initial_observation_sha256": identity[1],
            "reference_episode_id": left["episode_id"],
            "observer_episode_id": right["episode_id"],
            "action_sequences_exact": left["actions"] == right["actions"],
            "reward_sequences_exact": left["rewards"] == right["rewards"],
            "reference_action_count": len(left["actions"]),
            "observer_action_count": len(right["actions"]),
        })
    cycle_gap_count = sum(
        sum(str(gap).startswith("STEP_") for gap in row.gaps) for row in imported
    )
    valid_cycles = sum(len(row.records) for row in imported)
    total_steps = valid_cycles + cycle_gap_count
    report = {
        "schema_version": 1,
        "authority": "MECHANICAL_OBSERVER_AUDIT_NO_SEMANTIC_INTERPRETATION",
        "pairing_rule": "EXACT_REQUESTED_SEED_AND_INITIAL_OBSERVATION_HASH",
        "observer_patch_sha256": patch_sha256,
        "no_human_policy_hints_patch_sha256": no_hints_sha256,
        "request_seed_control_patch_sha256": seed_control_sha256,
        "observer_overlay_receipt_sha256": _sha256(args.observer_overlay_receipt),
        "reference_overlay_receipt_sha256": _sha256(args.reference_overlay_receipt),
        "matched_pairs": pairs,
        "reference_policy_identity": reference_identity,
        "observer_policy_identity": observer_identity,
        "unmatched_reference_identities": [list(row) for row in sorted(set(reference) - set(observer))],
        "unmatched_observer_identities": [list(row) for row in sorted(set(observer) - set(reference))],
        "observer_cycle_counts": {
            "valid": valid_cycles,
            "total_seen": total_steps,
            "excluded": total_steps - valid_cycles,
        },
        "observer_import_gaps": {
            row.episode_id: list(row.gaps) for row in imported if row.gaps
        },
        "descriptive_checks": {
            "all_pairs_action_exact": bool(pairs) and all(row["action_sequences_exact"] for row in pairs),
            "all_pairs_reward_exact": bool(pairs) and all(row["reward_sequences_exact"] for row in pairs),
            "all_identities_matched": set(reference) == set(observer),
            "at_least_one_valid_closed_loop_cycle": valid_cycles > 0,
            "policy_identity_exact": reference_identity == observer_identity,
        },
        "claim_limit": (
            "No tolerance threshold is embedded. The report exposes exact observations; "
            "any non-inferiority criterion must be preregistered outside the Harness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["descriptive_checks"], sort_keys=True))


if __name__ == "__main__":
    main()
