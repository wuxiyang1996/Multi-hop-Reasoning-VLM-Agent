from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_HORIZONS = (1, 2, 4, 8)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _event_index(evidence_dir: Path) -> dict[str, dict[str, Any]]:
    episodes: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"seed": None, "skills": {}, "rewards": {}, "responses": []}
    )
    for row in _read_jsonl(evidence_dir / "events.jsonl"):
        episode = episodes[str(row["episode_id"])]
        payload = row.get("payload") or {}
        if row.get("kind") == "RESET":
            episode["seed"] = int(payload["requested_seed"])
        elif row.get("kind") == "AGENT_PROPOSAL_SET":
            skill_id = payload.get("selected_skill_id")
            if skill_id:
                episode["skills"][int(payload["step"])] = str(skill_id)
        elif row.get("kind") == "ENVIRONMENT_STEP":
            episode["rewards"][int(payload["step"])] = float(payload["reward"])
        elif row.get("kind") == "AGENT_RESPONSE":
            episode["responses"].append(str(payload.get("raw_response", "")))
    return dict(episodes)


def _hint_exclusion_status(evidence_dir: Path) -> str:
    run_dir = evidence_dir.parent
    receipt_path = run_dir / "source_overlay_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        return (
            "EXCLUDED_WITH_RECEIPT"
            if receipt.get("human_policy_hints_excluded") is True
            else "NOT_EXCLUDED_WITH_RECEIPT"
        )
    provenance_path = run_dir / "source_provenance_at_start.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        files = "\n".join(str(path) for path in (provenance.get("files") or {}))
        if "source_no_human_policy_hints.patch" in files:
            return "EXCLUDED_WITH_PROVENANCE"
    return "NO_EXCLUSION_RECEIPT"


def summarize_source_evidence(
    evidence_dir: str | Path,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    root = Path(evidence_dir)
    episodes = _event_index(root)
    episode_rows = _read_jsonl(root / "episodes.jsonl")
    returns = {str(row["episode_id"]): float(row["total_reward"]) for row in episode_rows}
    skill_counts: Counter[str] = Counter()
    continuous_edges: Counter[str] = Counter()
    immediate_support: Counter[str] = Counter()
    horizon_support = {int(horizon): Counter() for horizon in horizons}
    positive_events = 0
    total_steps = 0
    hint_language = 0
    for episode in episodes.values():
        skills = episode["skills"]
        rewards = episode["rewards"]
        positive_steps = {step for step, reward in rewards.items() if reward > 0}
        positive_events += len(positive_steps)
        total_steps += len(rewards)
        hint_language += sum(
            "critical action" in response.lower()
            or "prefer action 1" in response.lower()
            for response in episode["responses"]
        )
        for step, skill_id in skills.items():
            skill_counts[skill_id] += 1
            immediate_support[skill_id] += step in positive_steps
            for horizon in horizons:
                horizon_support[int(horizon)][skill_id] += any(
                    step <= reward_step < step + int(horizon)
                    for reward_step in positive_steps
                )
            if skills.get(step + 1) == skill_id:
                continuous_edges[skill_id] += 1
    skills = []
    for skill_id, selected_steps in skill_counts.most_common():
        skills.append({
            "skill_id": skill_id,
            "selected_steps": selected_steps,
            "continuous_edges": continuous_edges[skill_id],
            "positive_reward_support": {
                f"h{horizon}": horizon_support[int(horizon)][skill_id]
                for horizon in horizons
            },
        })
    return {
        "evidence_dir": str(root),
        "episodes": len(episode_rows),
        "episode_returns": [returns[str(row["episode_id"])] for row in episode_rows],
        "mean_episode_return": (
            sum(returns.values()) / len(returns) if returns else 0.0
        ),
        "environment_steps": total_steps,
        "positive_reward_events": positive_events,
        "positive_reward_density": positive_events / total_steps if total_steps else 0.0,
        "human_hint_exclusion": _hint_exclusion_status(root),
        "agent_responses_with_hint_language": hint_language,
        "skills": skills,
    }


def paired_return_diagnostic(
    authentic_dir: str | Path,
    skill_off_dir: str | Path,
) -> dict[str, Any]:
    authentic_root, off_root = Path(authentic_dir), Path(skill_off_dir)
    authentic_events = _event_index(authentic_root)
    off_events = _event_index(off_root)
    authentic_returns = {
        str(row["episode_id"]): float(row["total_reward"])
        for row in _read_jsonl(authentic_root / "episodes.jsonl")
    }
    off_returns = {
        str(row["episode_id"]): float(row["total_reward"])
        for row in _read_jsonl(off_root / "episodes.jsonl")
    }
    authentic_by_seed = {
        int(row["seed"]): authentic_returns[episode_id]
        for episode_id, row in authentic_events.items()
    }
    off_by_seed = {
        int(row["seed"]): off_returns[episode_id]
        for episode_id, row in off_events.items()
    }
    common = sorted(set(authentic_by_seed) & set(off_by_seed))
    differences = [authentic_by_seed[seed] - off_by_seed[seed] for seed in common]
    return {
        "common_seeds": common,
        "authentic_returns": [authentic_by_seed[seed] for seed in common],
        "skill_off_returns": [off_by_seed[seed] for seed in common],
        "paired_differences": differences,
        "mean_paired_difference": sum(differences) / len(differences) if differences else 0.0,
        "positive_zero_negative": {
            "positive": sum(value > 0 for value in differences),
            "zero": sum(value == 0 for value in differences),
            "negative": sum(value < 0 for value in differences),
        },
        "claim_limit": (
            "Seed pairing does not make independently sampled model trajectories causal. "
            "This is a discovery diagnostic only."
        ),
    }


def diagnose_matched_gate(
    evidence_dir: str | Path,
    phase7_report: str | Path,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    root = Path(evidence_dir)
    source = summarize_source_evidence(root, horizons)
    report = json.loads(Path(phase7_report).read_text())
    records = _read_jsonl(root / "matched_policy_records.jsonl")
    replays = _read_jsonl(root / "matched_policy_replays.jsonl")
    authentic = [row for row in records if row.get("treatment") == "G_PLUS_S"]
    source_skill_id = str(report["source_skill_id"])
    source_row = next(
        (row for row in source["skills"] if row["skill_id"] == source_skill_id),
        None,
    )
    prompt_deltas = [
        int(snapshot["prompt_tokens"]["G_PLUS_S"])
        - int(snapshot["prompt_tokens"]["G_PLUS_RANDOM"])
        for snapshot in report["snapshots"]
    ]
    discovery_nodes = set(report["discovery_graph"]["nodes"])
    unseen_nodes = {
        split: len({
            snapshot["node_id"]
            for snapshot in report["snapshots"]
            if snapshot["split"] == split and snapshot["node_id"] not in discovery_nodes
        })
        for split in ("qualification", "heldout")
    }
    replay_fields = set().union(*(row.keys() for row in replays)) if replays else set()
    has_delayed_return = any(
        key in replay_fields for key in ("return_h2", "return_h4", "return_h8")
    )
    classifications = ["MATCHED_POLICY_INFLUENCE_ESTABLISHED"]
    if not has_delayed_return:
        classifications.append("DELAYED_TREATMENT_VALUE_NOT_IDENTIFIED")
    if source_row and source_row["positive_reward_support"].get("h1", 0) == 0:
        classifications.append("SELECTED_SKILL_HAS_ZERO_IMMEDIATE_REWARD_SUPPORT")
    if report["gates"].get("SOURCE_GRAPH_SUPPORTED") is not True:
        classifications.append("EXACT_EFFECT_GRAPH_NOT_RECURRENT")
    if report["gates"].get("SOURCE_VALUE_SUPPORTED") is not True:
        classifications.append("SOURCE_VALUE_NOT_SUPPORTED_AT_OBSERVED_HORIZON")
    return {
        "evidence": source,
        "source_skill_id": source_skill_id,
        "matched_snapshots": len(authentic),
        "replay_rows": len(replays),
        "observed_treatment_horizon": 1,
        "delayed_treatment_return_fields_present": has_delayed_return,
        "selected_skill_reward_support_on_live_authentic_path": source_row,
        "prompt_token_delta_authentic_minus_random": {
            "minimum": min(prompt_deltas),
            "maximum": max(prompt_deltas),
            "mean": sum(prompt_deltas) / len(prompt_deltas),
            "exactly_matched": sum(delta == 0 for delta in prompt_deltas),
            "snapshots": len(prompt_deltas),
        },
        "unseen_effect_nodes_relative_to_discovery": unseen_nodes,
        "blind_recurrence": report["blind_recurrence"],
        "gates": report["gates"],
        "classifications": classifications,
        "diagnosis": (
            "The current run proves policy influence, not useful reasoning transfer. "
            "Its one-step replay cannot identify delayed value, and the structurally "
            "selected skill has no immediate positive-reward support."
        ),
    }


def build_failure_diagnosis(
    config: Mapping[str, Any],
    matched_evidence: str | Path,
    phase7_report: str | Path,
    no_hint_evidence: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    legacy = []
    for spec in config["games"]:
        authentic = summarize_source_evidence(spec["authentic_evidence"])
        skill_off = summarize_source_evidence(spec["skill_off_evidence"])
        legacy.append({
            "game": spec["game"],
            "authentic": authentic,
            "skill_off": skill_off,
            "return_diagnostic": paired_return_diagnostic(
                spec["authentic_evidence"], spec["skill_off_evidence"]
            ),
        })
    clean = {
        game: summarize_source_evidence(path)
        for game, path in sorted((no_hint_evidence or {}).items())
    }
    return {
        "schema_version": "SOURCE_GATE_FAILURE_DIAGNOSIS_V1",
        "matched_gate": diagnose_matched_gate(matched_evidence, phase7_report),
        "legacy_six_game_discovery": legacy,
        "fresh_no_hint_discovery": clean,
        "failure_tree": {
            "current_class": "MEASUREMENT_AND_REPRESENTATION_FAILURE_BEFORE_TRANSFER",
            "source_weight_or_context_influence": "SUPPORTED",
            "source_value": "UNIDENTIFIED_BEYOND_ONE_STEP_AND_UNSUPPORTED_AT_ONE_STEP",
            "explicit_reasoning_motif": "NOT_SUPPORTED",
            "far_domain_transfer": "NOT_YET_AUTHORIZED",
        },
        "next_minimum_experiment": {
            "selection": (
                "Use discovery-only lineage plus official reward support; freeze before "
                "qualification and do not inspect skill names or action semantics."
            ),
            "replay_horizons": list(DEFAULT_HORIZONS),
            "estimands": [
                "first-action treatment followed by one common frozen continuation policy",
                "full treatment policy regime for the same fixed horizon",
            ],
            "controls": ["B", "G_MINUS_S", "G_PLUS_S", "G_PLUS_RANDOM"],
            "stop_rule": (
                "If no fresh no-hint candidate has held-out value and blind recurrence, "
                "stop the explicit transferable-backbone claim."
            ),
        },
    }


__all__ = [
    "DEFAULT_HORIZONS",
    "summarize_source_evidence",
    "paired_return_diagnostic",
    "diagnose_matched_gate",
    "build_failure_diagnosis",
]
