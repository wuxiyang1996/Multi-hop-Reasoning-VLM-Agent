#!/usr/bin/env python3
"""Collect exhaustive replay forks at mechanically recorded skill-ID changes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_step(events: list[dict[str, Any]], kind: str) -> dict[int, dict[str, Any]]:
    result = {}
    for event in events:
        if event.get("kind") != kind:
            continue
        step = (event.get("payload") or {}).get("step")
        if isinstance(step, int):
            result[step] = event
    return result


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir")
    parser.add_argument("--game", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    evidence = Path(args.evidence_dir).resolve()
    output = Path(args.output).resolve()
    source_repo = Path(args.source_repo).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for name in ("manifest.json", "events.jsonl", "episodes.jsonl"):
        if not (evidence / name).is_file():
            raise FileNotFoundError(evidence / name)

    # Import the exact source replay adapter and verifier. The source checkout
    # remains read-only; this script only creates a supplemental receipt set.
    sys.path.insert(0, str(source_repo))
    from harness.replay_fork import ReplayForkVerifier  # type: ignore
    from scripts.run_instrumented_source_smoke import _SourceReplayAdapter  # type: ignore

    source_manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    metadata = source_manifest.get("metadata") or {}
    max_steps = int(metadata.get("max_steps", 0))
    if max_steps <= 0:
        raise ValueError("source manifest has no positive max_steps")

    events = _read_jsonl(evidence / "events.jsonl")
    episodes = _read_jsonl(evidence / "episodes.jsonl")
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_episode[str(event.get("episode_id", ""))].append(event)

    executor = ThreadPoolExecutor(max_workers=args.workers)
    receipts: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    gaps: list[str] = []
    replayed_prefix_action_count = 0
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        episode_events = by_episode[episode_id]
        reset_events = [row for row in episode_events if row.get("kind") == "RESET"]
        if len(reset_events) != 1:
            gaps.append(f"{episode_id}:RESET_COUNT_{len(reset_events)}")
            continue
        seed = (reset_events[0].get("payload") or {}).get("requested_seed")
        if not isinstance(seed, int):
            gaps.append(f"{episode_id}:MISSING_REQUESTED_SEED")
            continue

        steps = _by_step(episode_events, "ENVIRONMENT_STEP")
        observations = _by_step(episode_events, "OBSERVATION")
        admissibility = _by_step(episode_events, "NATIVE_ADMISSIBILITY")
        selections = _by_step(episode_events, "AGENT_PROPOSAL_SET")
        ordered_selection_steps = sorted(selections)
        switch_steps = [
            step
            for previous, step in zip(ordered_selection_steps, ordered_selection_steps[1:])
            if (selections[previous].get("payload") or {}).get("selected_skill_id")
            != (selections[step].get("payload") or {}).get("selected_skill_id")
        ]

        for switch_step in switch_steps:
            required_steps = tuple(range(switch_step + 1))
            if (
                any(step not in steps for step in required_steps)
                or switch_step not in observations
                or switch_step not in admissibility
            ):
                gaps.append(f"{episode_id}:INCOMPLETE_SWITCH_{switch_step}")
                continue
            prefix = [str(steps[step]["payload"]["executed_action"]) for step in range(switch_step)]
            original_action = str(steps[switch_step]["payload"]["executed_action"])
            native_actions = [
                str(item) for item in admissibility[switch_step]["payload"].get("native_actions", [])
            ]
            alternatives = [item for item in native_actions if item != original_action]
            before_skill = (selections[switch_step - 1].get("payload") or {}).get("selected_skill_id")
            after_skill = (selections[switch_step].get("payload") or {}).get("selected_skill_id")
            boundaries.append({
                "episode_id": episode_id,
                "step": switch_step,
                "before_skill_id": before_skill,
                "after_skill_id": after_skill,
                "alternative_count": len(alternatives),
            })
            replayed_prefix_action_count += len(prefix) * len(alternatives)

            def run_alternative(item):
                alternative_index, alternative = item
                adapter = _SourceReplayAdapter(args.game, max_steps)
                try:
                    receipt = ReplayForkVerifier().run(
                        adapter,
                        intervention_id=(
                            f"{episode_id}.fork_step_{switch_step}.switch_alt_{alternative_index}"
                        ),
                        seed=seed,
                        prefix_actions=prefix,
                        expected_fork_state_sha256=str(
                            observations[switch_step]["payload"]["observable_state_sha256"]
                        ),
                        alternative_action=alternative,
                    )
                    return asdict(receipt) | {"receipt_sha256": receipt.content_hash()}
                finally:
                    adapter.close()

            receipts.extend(executor.map(run_alternative, enumerate(alternatives)))
            print(
                json.dumps({
                    "episode_id": episode_id,
                    "switch_step": switch_step,
                    "boundaries_complete": len(boundaries),
                    "receipts_complete": len(receipts),
                }, sort_keys=True),
                flush=True,
            )

    status_counts = Counter(str(row.get("status")) for row in receipts)
    executor.shutdown(wait=True)
    if gaps:
        raise RuntimeError(f"switch replay input gaps: {gaps[:8]}")
    if not receipts or set(status_counts) != {"INTERVENTION_OBSERVED"}:
        raise RuntimeError(f"switch replay did not verify completely: {dict(status_counts)}")

    output.mkdir(parents=True)
    receipt_path = output / "replay_receipts.jsonl"
    receipt_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in receipts),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "authority": "SUPPLEMENTAL_SWITCH_BOUNDARY_REPLAY_ONLY",
        "boundary_rule": "EXACT_RECORDED_SELECTED_SKILL_ID_CHANGE_V1",
        "game": args.game,
        "source_evidence": str(evidence),
        "source_files_sha256": {
            name: _sha256(evidence / name)
            for name in ("manifest.json", "events.jsonl", "episodes.jsonl")
        },
        "episodes": len(episodes),
        "boundaries": boundaries,
        "boundary_count": len(boundaries),
        "receipt_count": len(receipts),
        "status_counts": dict(sorted(status_counts.items())),
        "replayed_prefix_action_count": replayed_prefix_action_count,
        "elapsed_seconds": time.monotonic() - started,
        "workers": args.workers,
        "receipt_file_sha256": _sha256(receipt_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "boundaries": len(boundaries),
        "receipts": len(receipts),
        "status_counts": dict(status_counts),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
