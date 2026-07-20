#!/usr/bin/env python3
"""Held-out ALFWorld evaluation with immutable one-shot admission scopes.

Run one shard per rollout GPU.  The evaluator never trains, admits, rewrites,
or reloads an artifact.  Official ALFWorld ``won`` is the only success signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.frozen_transfer_policy import (  # noqa: E402
    FrozenAdmissionGuard,
    StrictOpenAIClient,
    action_prompt,
    parse_exact_numbered_response,
    skill_prompt,
)


CONDITIONS = ("base", "game_sft", "base_harness", "game_sft_harness")
SPLITS = ("eval_in_distribution", "eval_out_of_distribution")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _official_won(info: Dict[str, Any]) -> bool:
    """Return ALFWorld's official terminal success bit and nothing else."""
    return bool(_first(info.get("won")))


def _task_identity(info: Dict[str, Any], episode_id: str) -> str:
    for key in ("gamefile", "game_file", "extra.gamefile"):
        value = info.get(key)
        if value:
            return str(_first(value))
    extra = info.get("extra")
    if isinstance(extra, dict) and extra.get("gamefile"):
        return str(_first(extra["gamefile"]))
    return episode_id


def _clean_observation(value: str) -> str:
    return str(value).split("\n\nAdmissible actions:", 1)[0].strip()


def _choose_action(
    *,
    condition: str,
    client: StrictOpenAIClient,
    base_model: str,
    action_adapter: str,
    skill_adapter: str,
    guard: FrozenAdmissionGuard,
    goal: str,
    observation: str,
    admissible: Sequence[str],
    recent_actions: Sequence[str],
) -> tuple[str | None, Dict[str, Any]]:
    trace: Dict[str, Any] = {
        "condition": condition,
        "n_official_admissible": len(admissible),
        "artifact_hashes": guard.artifact_hashes if "harness" in condition else [],
    }
    action_model = action_adapter if condition in {"game_sft", "game_sft_harness"} else base_model

    active_skill = None
    candidate_actions = list(admissible)
    if condition == "base_harness":
        guarded = guard.filter_actions(admissible)
        candidate_actions = [item.command for item in guarded]
        trace["n_scope_admissible"] = len(candidate_actions)
        trace["scope_operators"] = sorted({item.parsed.operator for item in guarded})
        if not candidate_actions:
            trace["abstain_reason"] = "NO_ADMITTED_ACTION_IN_CURRENT_STATE"
            return None, trace
    elif condition == "game_sft_harness":
        guarded = guard.filter_actions(admissible)
        bindings = guard.available_bindings(guarded)
        trace["n_scope_admissible"] = len(guarded)
        trace["scope_operators"] = sorted({item.parsed.operator for item in guarded})
        if not guarded or not bindings:
            trace["abstain_reason"] = "NO_ADMITTED_ACTION_IN_CURRENT_STATE"
            return None, trace
        prompt = skill_prompt(
            goal=goal,
            observation=observation,
            candidates=bindings,
            recent_actions=recent_actions,
        )
        reply = ""
        try:
            reply, usage = client.complete(model=skill_adapter, prompt=prompt)
        except Exception as exc:
            raise RuntimeError(f"SKILL_ENDPOINT_FAILURE:{type(exc).__name__}:{exc}") from exc
        try:
            selected = parse_exact_numbered_response(reply, kind="skill", n=len(bindings))
        except ValueError as exc:  # received model hallucination: safe abstention
            trace.update(
                skill_prompt_sha256=_prompt_hash(prompt),
                skill_reply=reply[:500],
                abstain_reason=f"SKILL_SELECTION_FAILED:{type(exc).__name__}:{exc}",
            )
            return None, trace
        binding = bindings[selected]
        active_skill = binding.source_skill_name
        guarded = guard.filter_actions(admissible, operator=binding.operator)
        candidate_actions = [item.command for item in guarded]
        trace.update(
            skill_prompt_sha256=_prompt_hash(prompt),
            skill_reply=reply[:500],
            skill_usage=usage,
            selected_source_skill=binding.source_skill_name,
            selected_operator=binding.operator,
            selected_artifact_hash=binding.artifact.artifact_hash,
            n_selected_operator_actions=len(candidate_actions),
        )
        if not candidate_actions:
            trace["abstain_reason"] = "SELECTED_SKILL_HAS_NO_ALLOWED_ACTION"
            return None, trace

    if not candidate_actions:
        trace["abstain_reason"] = "NO_OFFICIAL_ADMISSIBLE_ACTION"
        return None, trace
    prompt = action_prompt(
        domain="alfworld",
        goal=goal,
        observation=observation,
        actions=candidate_actions,
        active_skill=active_skill,
        recent_actions=recent_actions,
    )
    reply = ""
    try:
        reply, usage = client.complete(model=action_model, prompt=prompt)
    except Exception as exc:
        raise RuntimeError(f"ACTION_ENDPOINT_FAILURE:{type(exc).__name__}:{exc}") from exc
    try:
        selected = parse_exact_numbered_response(reply, kind="action", n=len(candidate_actions))
    except ValueError as exc:  # no random/default action fallback
        trace.update(
            action_prompt_sha256=_prompt_hash(prompt),
            action_reply=reply[:500],
            abstain_reason=f"ACTION_SELECTION_FAILED:{type(exc).__name__}:{exc}",
        )
        return None, trace
    action = candidate_actions[selected]
    if action not in admissible:
        raise AssertionError("numbered candidate escaped official admissible list")
    trace.update(
        action_prompt_sha256=_prompt_hash(prompt),
        action_reply=reply[:500],
        action_usage=usage,
        selected_action=action,
    )
    return action, trace


def run_shard(args: argparse.Namespace) -> Dict[str, Any]:
    from env_wrappers.alfworld_nl_wrapper import make_alfworld_env

    guard = FrozenAdmissionGuard.from_files(
        manifest_path=args.admission_manifest,
        binding_config_path=args.binding_config,
    )
    endpoint = args.endpoints[args.shard_index % len(args.endpoints)]
    client = StrictOpenAIClient(endpoint, timeout_s=args.request_timeout)
    # Every shard uses the same task-order seed, resets through the same
    # sequence, and evaluates only its modulo-assigned indices.  This gives
    # disjoint coverage without sacrificing paired tasks across conditions.
    shard_seed = args.seed
    env = make_alfworld_env(
        split=args.split,
        max_steps=args.max_steps,
        config_path=str(args.config),
        random_seed=shard_seed,
    )
    assigned = list(range(args.shard_index, args.episodes, args.num_shards))
    rows: List[Dict[str, Any]] = []
    try:
        # Advance the shared deterministic order without opening skipped
        # TextWorld games. Each worker then opens only its modulo slice.
        env.skip_games(args.shard_index)
        for local_index, global_index in enumerate(assigned):
            started = time.monotonic()
            observation, info = env.reset()
            goal = _clean_observation(observation)
            episode_id = f"{args.split}-seed{shard_seed}-shard{args.shard_index}-local{len(rows)}"
            task_id = _task_identity(info, episode_id)
            actions: List[str] = []
            traces: List[Dict[str, Any]] = []
            score = 0.0
            terminated = False
            truncated = False
            success = _official_won(info)
            error = None
            abstain_reason = None
            while not (success or terminated or truncated) and len(actions) < args.max_steps:
                admissible = [str(item) for item in info.get("action_names", [])]
                try:
                    action, trace = _choose_action(
                        condition=args.condition,
                        client=client,
                        base_model=args.base_model,
                        action_adapter=args.action_adapter,
                        skill_adapter=args.skill_adapter,
                        guard=guard,
                        goal=goal,
                        observation=_clean_observation(observation),
                        admissible=admissible,
                        recent_actions=actions,
                    )
                except Exception as exc:
                    error = f"POLICY_ERROR:{type(exc).__name__}:{exc}"
                    break
                traces.append(trace)
                if action is None:
                    abstain_reason = str(trace.get("abstain_reason") or "UNSPECIFIED_ABSTENTION")
                    break
                observation, reward, terminated, truncated, info = env.step(action)
                actions.append(action)
                score = max(score, float(reward))
                success = _official_won(info)
            rows.append({
                "global_episode_index": global_index,
                "episode_id": episode_id,
                "task_id": task_id,
                "split": args.split,
                "seed": shard_seed,
                "condition": args.condition,
                "success": bool(success),
                "official_score": score,
                "steps": len(actions),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "abstained": abstain_reason is not None,
                "abstain_reason": abstain_reason,
                "error": error,
                "actions": actions,
                "traces": traces,
                "wall_time_s": time.monotonic() - started,
            })
            print(
                f"[{args.condition} {args.split} shard={args.shard_index}] "
                f"{len(rows)}/{len(assigned)} success={success} steps={len(actions)} "
                f"abstain={abstain_reason}",
                flush=True,
            )
            if local_index + 1 < len(assigned):
                env.skip_games(args.num_shards - 1)
    finally:
        env.close()
        client.close()

    valid = [row for row in rows if row["error"] is None]
    result = {
        "schema_version": 1,
        "protocol": "strict_frozen_one_shot_alfworld_v1",
        "condition": args.condition,
        "split": args.split,
        "episodes_planned_total": args.episodes,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_seed": shard_seed,
        "base_model": args.base_model,
        "action_adapter": args.action_adapter if "game_sft" in args.condition else None,
        "skill_adapter": args.skill_adapter if args.condition == "game_sft_harness" else None,
        "endpoint": endpoint,
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "admission_manifest": str(args.admission_manifest.resolve()),
        "admission_manifest_sha256": _sha256(args.admission_manifest),
        "binding_config_sha256": _sha256(args.binding_config),
        "artifact_hashes": guard.artifact_hashes,
        "target_gradient_updates": 0,
        "git_commit_at_launch": _git_commit(),
        "summary": {
            "n_rows": len(rows),
            "n_valid": len(valid),
            "n_errors": len(rows) - len(valid),
            "n_abstained": sum(bool(row["abstained"]) for row in valid),
            "success_rate": (
                sum(bool(row["success"]) for row in valid) / len(valid) if valid else 0.0
            ),
            "mean_steps": mean([row["steps"] for row in valid]) if valid else 0.0,
            "mean_wall_time_s": mean([row["wall_time_s"] for row in valid]) if valid else 0.0,
        },
        "rows": rows,
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--action-adapter", default="action_taking")
    parser.add_argument("--skill-adapter", default="skill_selection")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/alfworld_pick_and_place_config.yaml",
    )
    parser.add_argument(
        "--admission-manifest", type=Path,
        default=(
            REPO_ROOT / "artifacts/admission/alfworld/"
            "manifest-c92a05274bacf88286a05a1be48c8d6bd48da6285038be7f01b1682d014e7cd1-eaa1bbf4214ff0d3.json"
        ),
    )
    parser.add_argument(
        "--binding-config", type=Path,
        default=REPO_ROOT / "configs/alfworld_one_shot_bindings.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard parameters")
    for path in (args.config, args.admission_manifest, args.binding_config):
        if not path.is_file():
            parser.error(f"required frozen input missing: {path}")
    return args


def main() -> int:
    args = _parse_args()
    result = run_shard(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0 if result["summary"]["n_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
