#!/usr/bin/env python3
"""Build an immutable, source-only evidence index for skill transfer.

This command never mutates the downloaded artifacts.  It hashes every source
episode and transition, quarantines legacy shared-bank verdicts, and compiles
only exact observed skill identities into the conservative v1 program IR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skill_bank.legacy_source_import import iter_legacy_bindings  # noqa: E402
from skill_bank.source_program_store import SourceProgramStore  # noqa: E402
from skill_bank.source_skill_compiler import compile_source_programs  # noqa: E402
from skill_bank.source_effects import extract_source_effects  # noqa: E402


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return _sha256_bytes(value.encode("utf-8"))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return count


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def index_episodes(source_root: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    episodes: List[Dict[str, Any]] = []
    invocations: List[Dict[str, Any]] = []
    for path in sorted(source_root.glob("*/episode_*.json")):
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes)
        game = str(payload.get("game_name") or payload.get("metadata", {}).get("game") or path.parent.name)
        episode_id = str(payload.get("episode_id") or path.stem)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        provider = str(
            metadata.get("agent_type")
            or metadata.get("model")
            or payload.get("env_name")
            or "unknown"
        )
        file_hash = _sha256_bytes(raw_bytes)
        experiences = payload.get("experiences")
        if not isinstance(experiences, list):
            raise ValueError(f"{path}: experiences must be a list")
        episodes.append(
            {
                "episode_id": episode_id,
                "game": game,
                "provider_or_run": provider,
                "n_steps": len(experiences),
                "outcome": payload.get("outcome"),
                "total_reward": metadata.get("total_reward"),
                "source_path": _display_path(path),
                "source_file_sha256": file_hash,
                "source_only": True,
            }
        )
        for ordinal, step in enumerate(experiences):
            if not isinstance(step, dict):
                raise ValueError(f"{path}: experiences[{ordinal}] is not an object")
            step_index = int(step.get("idx", ordinal))
            skills = step.get("skills") if isinstance(step.get("skills"), dict) else {}
            chosen = str(skills.get("skill_id") or "").strip()
            action = str(step.get("action") or "").strip()
            state_hash = _sha256_text(step.get("raw_state", step.get("state", "")))
            next_hash = _sha256_text(step.get("raw_next_state", step.get("next_state", "")))
            state_text = str(step.get("state") or "")
            next_state_text = str(step.get("next_state") or "")
            candidates = step.get("skill_candidates")
            if not isinstance(candidates, list):
                candidates = []
            invocations.append(
                {
                    "game": game,
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "provider_or_run": provider,
                    "chosen_skill_id": chosen,
                    "chosen_skill_name": str(skills.get("skill_name") or ""),
                    "skill_candidates": [str(item) for item in candidates],
                    "action": action,
                    "available_actions": [str(item) for item in (step.get("available_actions") or [])],
                    "reward": float(step.get("reward") or 0.0),
                    "done": bool(step.get("done")),
                    "state_sha256": state_hash,
                    "next_state_sha256": next_hash,
                    "source_effects": list(extract_source_effects(
                        game=game,
                        state=state_text,
                        next_state=next_state_text,
                        action=action,
                        reward=float(step.get("reward") or 0.0),
                        done=bool(step.get("done")),
                    )),
                    "source_path": _display_path(path),
                    "source_file_sha256": file_hash,
                    "source_only": True,
                }
            )
    return episodes, invocations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "labeling" / "gpt54_skill_labeled",
    )
    parser.add_argument(
        "--legacy-bank-root",
        type=Path,
        default=REPO_ROOT / "frontier_data" / "output" / "shared_skill_bank_grpo",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "source_evidence_index",
    )
    parser.add_argument("--min-invocations", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.source_root.is_dir():
        raise SystemExit(f"source root not found: {args.source_root}")
    if args.min_invocations < 1:
        raise SystemExit("--min-invocations must be positive")

    episodes, invocations = index_episodes(args.source_root)
    legacy = list(iter_legacy_bindings(args.legacy_bank_root)) if args.legacy_bank_root.is_dir() else []
    programs = compile_source_programs(invocations, min_invocations=args.min_invocations)

    args.output_root.mkdir(parents=True, exist_ok=True)
    n_episodes = _atomic_jsonl(args.output_root / "episodes.jsonl", episodes)
    n_invocations = _atomic_jsonl(args.output_root / "skill_invocations.jsonl", invocations)
    n_legacy = _atomic_jsonl(args.output_root / "legacy_proposals.jsonl", legacy)
    n_programs = SourceProgramStore(args.output_root / "source_programs.jsonl").replace(programs)

    by_game = Counter(row["game"] for row in episodes)
    by_skill = Counter(row["chosen_skill_id"] for row in invocations if row["chosen_skill_id"])
    missing_skill = sum(not row["chosen_skill_id"] for row in invocations)
    summary = {
        "schema_version": 1,
        "source_root": str(args.source_root),
        "legacy_bank_root": str(args.legacy_bank_root),
        "source_only": True,
        "n_episodes": n_episodes,
        "n_steps": n_invocations,
        "n_skill_invocations": n_invocations - missing_skill,
        "n_steps_without_chosen_skill": missing_skill,
        "n_legacy_proposals_quarantined": n_legacy,
        "n_source_verified_programs": n_programs,
        "episodes_by_game": dict(sorted(by_game.items())),
        "n_unique_chosen_skills": len(by_skill),
        "top_chosen_skills": by_skill.most_common(20),
        "compiler_policy": "exact (game, chosen_skill_id); no semantic clustering",
    }
    _atomic_json(args.output_root / "summary.json", summary)
    manifest = {}
    for path in sorted(args.output_root.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest[path.name] = _sha256_bytes(path.read_bytes())
    _atomic_json(args.output_root / "manifest.json", {"sha256": manifest})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
