#!/usr/bin/env python3
"""Build exact-number Game SFT solely from immutable executed game steps.

No target-domain text, semantic clustering, confidence threshold, reward
threshold, or model-generated rationale is consumed.  The resulting LoRAs are
a source behavior-cloning control; they are never evidence for admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.frozen_transfer_policy import action_prompt  # noqa: E402


TARGET_TOKENS = ("alfworld", "miniwob", "webshop", "video_holmes", "tir_bench")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_rank(value: Mapping[str, Any], seed: int) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{seed}\0{encoded}".encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return _sha256_bytes(path.read_bytes())


def _skill_prompt(
    *, game: str, task: str, state: str, candidates: Sequence[str], actions: Sequence[str]
) -> str:
    numbered = "\n".join(f"{index}. {value}" for index, value in enumerate(candidates, 1))
    recent = " -> ".join(actions[-6:]) or "none"
    return (
        "Choose one source-game skill from the exact listed candidates.\n"
        f"Domain: source_game/{game}\nTask: {task[:1000]}\n"
        f"Current state:\n{state[:3000]}\nRecent exact actions: {recent}\n"
        "Available source skills (pick ONE by number):\n"
        f"{numbered}\nReturn exactly `SKILL: N`. No other text."
    )


def _eligible_rows(source_root: Path) -> tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    by_game: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    inputs: List[Dict[str, Any]] = []
    for path in sorted(source_root.glob("*/episode_*.json")):
        raw = path.read_bytes()
        payload = json.loads(raw)
        game = str(payload.get("game_name") or path.parent.name)
        episode_id = str(payload.get("episode_id") or path.stem)
        task = str(payload.get("task") or "")
        file_hash = _sha256_bytes(raw)
        inputs.append({
            "path": _display_path(path),
            "sha256": file_hash,
            "game": game,
            "episode_id": episode_id,
        })
        history: List[str] = []
        for ordinal, step in enumerate(payload.get("experiences") or []):
            if not isinstance(step, dict):
                continue
            action = str(step.get("action") or "").strip()
            available = [str(item) for item in (step.get("available_actions") or [])]
            skills = step.get("skills") if isinstance(step.get("skills"), dict) else {}
            chosen = str(skills.get("skill_id") or "").strip()
            candidates = [str(item) for item in (step.get("skill_candidates") or [])]
            if action and action in available and chosen and chosen in candidates:
                state = str(step.get("summary_state") or step.get("summary") or step.get("state") or "")
                by_game[game].append({
                    "game": game,
                    "episode_id": episode_id,
                    "step_index": int(step.get("idx", ordinal)),
                    "task": task,
                    "state": state,
                    "history": list(history[-6:]),
                    "available_actions": available,
                    "action": action,
                    "action_index": available.index(action),
                    "skill_candidates": candidates,
                    "chosen_skill": chosen,
                    "skill_index": candidates.index(chosen),
                    "source_path": _display_path(path),
                    "source_file_sha256": file_hash,
                })
            if action:
                history.append(action)
    return dict(by_game), inputs


def _select(rows: Sequence[Dict[str, Any]], *, limit: int, seed: int) -> List[Dict[str, Any]]:
    # Exact duplicate transition labels add no information and amplify loops.
    # De-duplication is identity-based, not a semantic/quality heuristic.
    unique: Dict[tuple[str, int, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["episode_id"], row["step_index"], row["chosen_skill"], row["action"])
        unique.setdefault(key, row)
    ranked = sorted(unique.values(), key=lambda item: _stable_rank(item, seed))
    return ranked[:limit] if limit > 0 else ranked


def build(source_root: Path, output_root: Path, *, per_game: int, seed: int) -> Dict[str, Any]:
    by_game, inputs = _eligible_rows(source_root)
    hashes: Dict[str, str] = {}
    counts: Dict[str, Dict[str, Any]] = {}
    all_prompt_text: List[str] = []
    for game, rows in sorted(by_game.items()):
        selected = _select(rows, limit=per_game, seed=seed)
        action_rows = []
        skill_rows = []
        for row in selected:
            action_text = action_prompt(
                domain=f"source_game/{game}",
                goal=row["task"],
                observation=row["state"],
                actions=row["available_actions"],
                active_skill=row["chosen_skill"],
                recent_actions=row["history"],
            )
            skill_text = _skill_prompt(
                game=game,
                task=row["task"],
                state=row["state"],
                candidates=row["skill_candidates"],
                actions=row["history"],
            )
            provenance = {
                "source_only": True,
                "source_game": game,
                "source_episode_id": row["episode_id"],
                "source_step_index": row["step_index"],
                "source_file_sha256": row["source_file_sha256"],
                "label_kind": "executed_not_quality_verdict",
            }
            action_rows.append({
                "prompt": action_text,
                "completion": f"ACTION: {row['action_index'] + 1}",
                **provenance,
            })
            skill_rows.append({
                "prompt": skill_text,
                "completion": f"SKILL: {row['skill_index'] + 1}",
                **provenance,
            })
            all_prompt_text.extend([action_text, skill_text])
        game_root = output_root / game
        hashes[f"{game}/action_taking.jsonl"] = _atomic_jsonl(
            game_root / "action_taking.jsonl", action_rows,
        )
        hashes[f"{game}/skill_selection.jsonl"] = _atomic_jsonl(
            game_root / "skill_selection.jsonl", skill_rows,
        )
        counts[game] = {
            "eligible": len(rows),
            "selected": len(selected),
            "skills": dict(Counter(item["chosen_skill"] for item in selected)),
        }

    target_leaks = sorted({
        token for token in TARGET_TOKENS
        if any(token in prompt.lower() for prompt in all_prompt_text)
    })
    if target_leaks:
        raise RuntimeError(f"target-domain token leak in source-only prompts: {target_leaks}")
    manifest = {
        "schema_version": 1,
        "dataset_kind": "source_only_executed_game_behavior_cloning",
        "source_only": True,
        "target_examples": 0,
        "target_gradient_updates": 0,
        "semantic_clustering": False,
        "reward_or_confidence_filter": False,
        "model_rationales_used_as_labels": False,
        "selection": f"exact action/candidate membership; identity dedup; sha256 rank seed={seed}",
        "per_game_limit": per_game,
        "games": counts,
        "n_selected": sum(item["selected"] for item in counts.values()),
        "source_inputs": inputs,
        "output_sha256": hashes,
        "target_token_scan": {"tokens": list(TARGET_TOKENS), "found": target_leaks},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "source_sft_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=REPO_ROOT / "labeling/gpt54_skill_labeled",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "artifacts/source_only_game_sft",
    )
    parser.add_argument("--per-game", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if not args.source_root.is_dir() or args.per_game < 1:
        parser.error("source-root must exist and per-game must be positive")
    result = build(args.source_root, args.output_root, per_game=args.per_game, seed=args.seed)
    print(json.dumps({key: result[key] for key in ("dataset_kind", "n_selected", "games")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
