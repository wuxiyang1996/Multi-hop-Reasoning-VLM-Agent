"""Matched multi-horizon source qualification for causal-effect options."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .causal_effect_options import (
    CLASS_CONTEXTUAL,
    CLASS_STABLE,
    validate_causal_effect_option_artifact,
)
from .contracts import stable_hash
from .visual_intervention_receipts import (
    FrozenVisualSnapshot,
    file_sha256,
    observable_sha256,
    validate_plan,
)


QUALIFICATION_VERSION = "CAUSAL_EFFECT_SOURCE_QUALIFICATION_V1"
HORIZON = 8
MODES = ("COMMON_HASH_CONTINUATION", "FULL_TREATMENT_REGIME")
TREATMENTS = (
    "AUTHENTIC_EFFECT_STRUCTURE",
    "SHUFFLED_EFFECT_STRUCTURE",
    "ALL_ACTION_HASH_RANDOM",
    "DISCOVERY_ACTION_MARGINAL",
    "REPEAT_SOURCE_ACTION",
)
GATE_CONTROLS = (
    "SHUFFLED_EFFECT_STRUCTURE",
    "ALL_ACTION_HASH_RANDOM",
    "DISCOVERY_ACTION_MARGINAL",
)


class QualificationEnvironment(Protocol):
    action_names: list[str]

    def reset(self, *, seed: int): ...

    def step(self, action: str): ...

    def close(self) -> None: ...


def _choice(items: Sequence[str], *identity: Any) -> str:
    if not items:
        raise ValueError("cannot choose from an empty action pool")
    index = int(stable_hash(identity)[:16], 16) % len(items)
    return items[index]


def _class_members(classes: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = defaultdict(list)
    for action, effect_class in sorted(classes.items()):
        result[str(effect_class)].append(str(action))
    return {key: tuple(value) for key, value in result.items()}


def treatment_action(
    artifact: Mapping[str, Any],
    snapshot: FrozenVisualSnapshot,
    *,
    treatment: str,
    mode: str,
    horizon_step: int,
) -> str:
    """Deterministically realize one frozen treatment action."""

    if treatment not in TREATMENTS or mode not in MODES:
        raise ValueError("unsupported treatment or estimand mode")
    native = tuple(snapshot.native_actions)
    if mode == "COMMON_HASH_CONTINUATION" and horizon_step > 0:
        return _choice(native, snapshot.snapshot_id, mode, "COMMON", horizon_step)
    if treatment == "ALL_ACTION_HASH_RANDOM":
        pool = native
    elif treatment == "DISCOVERY_ACTION_MARGINAL":
        counts = artifact["source_grounding"]["source_policy_action_counts"]
        pool = tuple(
            action
            for action in sorted(native)
            for _ in range(int(counts.get(action, 0)))
        )
        if not pool:
            raise ValueError("discovery action marginal is empty")
    elif treatment == "REPEAT_SOURCE_ACTION":
        pool = (snapshot.source_action,)
    else:
        classes = (
            artifact["source_grounding"]["action_classes"]
            if treatment == "AUTHENTIC_EFFECT_STRUCTURE"
            else artifact["shuffled_control"]["action_classes"]
        )
        members = _class_members(classes)
        # The symbolic controller alternates a stable effect basis with a
        # contextual probe.  Persistent-null actions are never selected.
        selected_class = CLASS_STABLE if horizon_step % 2 == 0 else CLASS_CONTEXTUAL
        pool = members[selected_class]
    return _choice(
        pool, snapshot.snapshot_id, treatment, mode, horizon_step,
    )


def _structured_lives(info: Mapping[str, Any]) -> int | None:
    structured = info.get("structured_state")
    if not isinstance(structured, dict):
        return None
    ram = structured.get("ram_watch")
    if not isinstance(ram, dict) or not isinstance(ram.get("lives"), int):
        return None
    return int(ram["lives"])


def run_qualification_trajectory(
    artifact: Mapping[str, Any],
    snapshot: FrozenVisualSnapshot,
    *,
    treatment: str,
    mode: str,
    env_factory: Callable[[str, int], QualificationEnvironment],
    max_episode_steps: int,
) -> dict[str, Any]:
    env = env_factory(snapshot.game, max_episode_steps)
    try:
        observation, info = env.reset(seed=snapshot.episode_seed)
        for prefix_action in snapshot.prefix_actions:
            observation, _reward, terminated, truncated, info = env.step(prefix_action)
            if terminated or truncated:
                raise RuntimeError("prefix terminated before qualification fork")
        before_hash = observable_sha256(observation)
        if before_hash != snapshot.expected_observable_sha256:
            raise RuntimeError("qualification fork observable mismatch")
        if tuple(str(item) for item in env.action_names) != snapshot.native_actions:
            raise RuntimeError("qualification native actions changed")
        before_lives = _structured_lives(info)
        actions: list[str] = []
        rewards: list[float] = []
        terminated = truncated = False
        for horizon_step in range(HORIZON):
            action = treatment_action(
                artifact, snapshot, treatment=treatment, mode=mode,
                horizon_step=horizon_step,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        returns = {
            f"h{horizon}": sum(rewards[:horizon])
            for horizon in (1, 2, 4, 8)
        }
        body: dict[str, Any] = {
            "qualification_version": QUALIFICATION_VERSION,
            "status": "INTERVENTION_OBSERVED",
            "artifact_sha256": artifact["artifact_sha256"],
            "snapshot_id": snapshot.snapshot_id,
            "episode_id": snapshot.episode_id,
            "episode_seed": snapshot.episode_seed,
            "split": snapshot.split,
            "fork_step": snapshot.step,
            "expected_observable_sha256": snapshot.expected_observable_sha256,
            "before_observable_sha256": before_hash,
            "treatment": treatment,
            "mode": mode,
            "action_trace": actions,
            "reward_trace": rewards,
            "returns": returns,
            "positive": {
                key: value > 0 for key, value in returns.items()
            },
            "before_lives": before_lives,
            "after_lives": _structured_lives(info),
            "after_observable_sha256": observable_sha256(observation),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        return body | {"receipt_sha256": stable_hash(body)}
    finally:
        env.close()


def summarize_source_qualification(
    receipts: Sequence[Mapping[str, Any]],
    *,
    minimum_mean_margin: float = 0.0,
    minimum_paired_net_wins: int = 1,
) -> dict[str, Any]:
    valid = [row for row in receipts if row.get("status") == "INTERVENTION_OBSERVED"]
    summaries: dict[str, Any] = {}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[(str(row["mode"]), str(row["treatment"]))].append(row)
    for mode in MODES:
        summaries[mode] = {}
        for treatment in TREATMENTS:
            rows = grouped[(mode, treatment)]
            summaries[mode][treatment] = {
                "n": len(rows),
                "mean_return": {
                    horizon: (
                        sum(float(row["returns"][horizon]) for row in rows) / len(rows)
                        if rows else None
                    )
                    for horizon in ("h1", "h2", "h4", "h8")
                },
                "positive_rate": {
                    horizon: (
                        sum(bool(row["positive"][horizon]) for row in rows) / len(rows)
                        if rows else None
                    )
                    for horizon in ("h1", "h2", "h4", "h8")
                },
                "life_loss_rate": (
                    sum(
                        row.get("before_lives") is not None
                        and row.get("after_lives") is not None
                        and int(row["after_lives"]) < int(row["before_lives"])
                        for row in rows
                    ) / len(rows)
                    if rows else None
                ),
            }

    primary_mode = "FULL_TREATMENT_REGIME"
    def episode_h8(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            values[str(row["episode_id"])].append(float(row["returns"]["h8"]))
        return {
            episode_id: sum(episode_values) / len(episode_values)
            for episode_id, episode_values in values.items()
        }

    # Episodes, not the two snapshots sampled from each episode, are the
    # independent paired units.  Snapshot-level pairing would be
    # pseudoreplication and could overstate a four-episode qualification result.
    authentic_rows = episode_h8(
        grouped[(primary_mode, "AUTHENTIC_EFFECT_STRUCTURE")]
    )
    control_gates = {}
    for control in GATE_CONTROLS:
        control_rows = episode_h8(grouped[(primary_mode, control)])
        common = sorted(set(authentic_rows) & set(control_rows))
        diffs = [
            authentic_rows[key] - control_rows[key]
            for key in common
        ]
        authentic_mean = summaries[primary_mode]["AUTHENTIC_EFFECT_STRUCTURE"][
            "mean_return"
        ]["h8"]
        control_mean = summaries[primary_mode][control]["mean_return"]["h8"]
        mean_margin = (
            float(authentic_mean) - float(control_mean)
            if authentic_mean is not None and control_mean is not None else None
        )
        net_wins = sum(value > 0 for value in diffs) - sum(value < 0 for value in diffs)
        passed = (
            len(common) == len(authentic_rows) > 0
            and mean_margin is not None
            and mean_margin > minimum_mean_margin
            and net_wins >= minimum_paired_net_wins
        )
        control_gates[control] = {
            "paired_unit": "SOURCE_EPISODE",
            "paired_n": len(common),
            "mean_h8_margin": mean_margin,
            "paired_wins": sum(value > 0 for value in diffs),
            "paired_ties": sum(value == 0 for value in diffs),
            "paired_losses": sum(value < 0 for value in diffs),
            "paired_net_wins": net_wins,
            "passed": passed,
        }
    qualification_passed = all(
        row["passed"] for row in control_gates.values()
    )
    return {
        "summary_version": QUALIFICATION_VERSION,
        "primary_endpoint": "FULL_TREATMENT_REGIME_H8_CUMULATIVE_REWARD",
        "thresholds": {
            "minimum_mean_margin_strictly_greater_than": minimum_mean_margin,
            "minimum_paired_net_wins": minimum_paired_net_wins,
        },
        "condition_summaries": summaries,
        "control_gates": control_gates,
        "qualification_passed": qualification_passed,
        "next_step": (
            "RUN_HELD_OUT_SOURCE_CONFIRMATION"
            if qualification_passed else "STOP_BEFORE_HELD_OUT_AND_TARGET"
        ),
    }


def collect_source_qualification(
    plan: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    split: str,
    output_dir: Path,
    env_factory: Callable[[str, int], QualificationEnvironment],
    workers: int,
    runtime_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_causal_effect_option_artifact(artifact)
    if artifact.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("artifact and replay plan differ")
    if split not in ("qualification", "held_out"):
        raise ValueError("source gate uses qualification or held_out only")
    if workers < 1:
        raise ValueError("workers must be positive")
    snapshots = sorted(
        (row for row in validate_plan(plan) if row.split == split),
        key=lambda row: (row.episode_id, row.selection_rank_sha256),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    jobs = [
        (snapshot, mode, treatment)
        for snapshot in snapshots
        for mode in MODES
        for treatment in TREATMENTS
    ]
    max_steps = int(plan["selection"]["max_episode_steps"])

    def execute(job):
        snapshot, mode, treatment = job
        try:
            return run_qualification_trajectory(
                artifact, snapshot, treatment=treatment, mode=mode,
                env_factory=env_factory, max_episode_steps=max_steps,
            )
        except Exception as exc:
            body = {
                "qualification_version": QUALIFICATION_VERSION,
                "status": "INTERVENTION_FAILED",
                "artifact_sha256": artifact["artifact_sha256"],
                "snapshot_id": snapshot.snapshot_id,
                "episode_id": snapshot.episode_id,
                "split": snapshot.split,
                "treatment": treatment,
                "mode": mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return body | {"receipt_sha256": stable_hash(body)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(execute, jobs))
    receipts_path = output_dir / "receipts.jsonl"
    receipts_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in receipts),
        encoding="utf-8",
    )
    status_counts = dict(sorted(Counter(row["status"] for row in receipts).items()))
    if status_counts != {"INTERVENTION_OBSERVED": len(jobs)}:
        summary = {
            "qualification_passed": False,
            "next_step": "STOP_PROTOCOL_FAILURE",
            "status_counts": status_counts,
        }
    else:
        summary = summarize_source_qualification(receipts)
        summary["status_counts"] = status_counts
    body = {
        "qualification_version": QUALIFICATION_VERSION,
        "claim_boundary": "SOURCE_QUALIFICATION_ONLY_NO_TARGET_EVIDENCE",
        "split": split,
        "plan_sha256": plan["plan_sha256"],
        "artifact_sha256": artifact["artifact_sha256"],
        "snapshot_count": len(snapshots),
        "trajectory_count": len(jobs),
        "receipts_file": receipts_path.name,
        "receipts_sha256": file_sha256(receipts_path),
        "runtime_receipt": dict(runtime_receipt or {}),
        "summary": summary,
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
