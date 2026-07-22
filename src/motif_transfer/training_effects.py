from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


TREATMENTS = ("B", "G_MINUS_S", "G_PLUS_S", "G_PLUS_RANDOM")
SPLITS = ("discovery", "qualification", "heldout")


def _stable_hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _cmp(left: float, right: float) -> int:
    return (left > right) - (left < right)


def _snapshot_id(row: Mapping[str, Any]) -> str:
    return f"seed={int(row['episode_seed'])}:step={int(row['step'])}"


def validate_matched_bundle(evidence_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(evidence_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = manifest.get("matched_policy_treatments") or {}
    records_path = root / str(metadata.get("records_file", "matched_policy_records.jsonl"))
    replays_path = root / str(metadata.get("replays_file", "matched_policy_replays.jsonl"))
    if metadata.get("records_sha256") != _file_hash(records_path):
        raise ValueError("matched policy record hash mismatch")
    if metadata.get("replays_sha256") != _file_hash(replays_path):
        raise ValueError("matched policy replay hash mismatch")
    records = _read_jsonl(records_path)
    replays = _read_jsonl(replays_path)
    replay_hash_fields = (
        "intervention_id", "seed", "prefix_actions",
        "expected_fork_state_sha256", "replayed_fork_state_sha256",
        "alternative_action", "admissible_actions_sha256",
        "alternative_next_state_sha256", "status", "failure_codes",
    )
    for row in replays:
        unsigned = {key: row[key] for key in replay_hash_fields}
        if row.get("receipt_sha256") != _stable_hash(unsigned):
            raise ValueError("matched replay receipt hash mismatch")
    replay_ids = {str(row["receipt_sha256"]) for row in replays}
    if len(replay_ids) != len(replays):
        raise ValueError("duplicate matched replay receipt")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[_snapshot_id(row)].append(row)
        if row.get("replay_receipt_sha256") not in replay_ids:
            raise ValueError("unresolved replay receipt")
        if row.get("replay_status") != "INTERVENTION_OBSERVED":
            raise ValueError("matched replay was not observed")
        if row.get("parser_fallback") is True:
            raise ValueError("parser fallback in matched policy record")
        if row.get("prompt_sha256") != _stable_hash(row.get("prompt", "")):
            raise ValueError("prompt hash mismatch")
        if row.get("raw_response_sha256") != _stable_hash(row.get("raw_response", "")):
            raise ValueError("response hash mismatch")
    for snapshot_id, rows in grouped.items():
        by_treatment = {str(row["treatment"]): row for row in rows}
        if len(rows) != 4 or set(by_treatment) != set(TREATMENTS):
            raise ValueError(f"unbalanced snapshot {snapshot_id}")
        invariant_keys = (
            "episode_id", "episode_seed", "step", "source_skill_id",
            "prefix_actions", "before_observable_sha256", "native_actions",
            "native_actions_sha256",
        )
        for key in invariant_keys:
            if len({_stable_hash(row.get(key)) for row in rows}) != 1:
                raise ValueError(f"snapshot invariant mismatch: {snapshot_id}:{key}")
        if by_treatment["B"].get("prompt") != by_treatment["G_MINUS_S"].get("prompt"):
            raise ValueError(f"weight contrast prompt mismatch: {snapshot_id}")
        if by_treatment["B"].get("requested_adapter") is not None:
            raise ValueError(f"base treatment requested an adapter: {snapshot_id}")
        for name in ("G_MINUS_S", "G_PLUS_S", "G_PLUS_RANDOM"):
            if by_treatment[name].get("requested_adapter") != "action_taking":
                raise ValueError(f"game treatment adapter mismatch: {snapshot_id}:{name}")
            if by_treatment[name].get("used_adapter") != "action_taking":
                raise ValueError(f"game adapter fallback: {snapshot_id}:{name}")
        if by_treatment["G_PLUS_S"].get("context_skill_id") != metadata.get("source_skill_id"):
            raise ValueError(f"authentic context mismatch: {snapshot_id}")
        random_id = by_treatment["G_PLUS_RANDOM"].get("context_skill_id")
        if not random_id or random_id == metadata.get("source_skill_id"):
            raise ValueError(f"invalid random-skill control: {snapshot_id}")
    if int(metadata.get("snapshot_count", -1)) != len(grouped):
        raise ValueError("manifest snapshot count mismatch")
    return manifest, records


def _make_snapshot(rows: Iterable[Mapping[str, Any]], split: str) -> dict[str, Any]:
    by = {str(row["treatment"]): row for row in rows}
    base, masked = by["B"], by["G_MINUS_S"]
    authentic, random = by["G_PLUS_S"], by["G_PLUS_RANDOM"]
    signature = {
        "weight_action": base["parsed_action"] != masked["parsed_action"],
        "weight_state": base["after_observable_sha256"] != masked["after_observable_sha256"],
        "skill_action": masked["parsed_action"] != authentic["parsed_action"],
        "skill_state": masked["after_observable_sha256"] != authentic["after_observable_sha256"],
        "auth_action": authentic["parsed_action"] != random["parsed_action"],
        "auth_state": authentic["after_observable_sha256"] != random["after_observable_sha256"],
        "weight_reward_cmp": _cmp(float(masked["replay_reward"]), float(base["replay_reward"])),
        "skill_reward_cmp": _cmp(float(authentic["replay_reward"]), float(masked["replay_reward"])),
        "auth_reward_cmp": _cmp(float(authentic["replay_reward"]), float(random["replay_reward"])),
    }
    return {
        "snapshot_id": _snapshot_id(authentic),
        "episode_id": authentic["episode_id"],
        "episode_seed": int(authentic["episode_seed"]),
        "step": int(authentic["step"]),
        "split": split,
        "signature": signature,
        "node_id": _stable_hash(signature),
        "actions": {name: by[name]["parsed_action"] for name in TREATMENTS},
        "rewards": {name: float(by[name]["replay_reward"]) for name in TREATMENTS},
        "prompt_tokens": {name: int(by[name].get("prompt_tokens", 0)) for name in TREATMENTS},
        "completion_tokens": {
            name: int(by[name].get("completion_tokens", 0)) for name in TREATMENTS
        },
    }


def analyze_training_effects(evidence_dir: str | Path) -> dict[str, Any]:
    manifest, records = validate_matched_bundle(evidence_dir)
    replay_rows = _read_jsonl(
        Path(evidence_dir) / manifest["matched_policy_treatments"]["replays_file"]
    )
    rewards = {
        str(row["receipt_sha256"]): float(row.get("alternative_reward", 0.0))
        for row in replay_rows
    }
    for row in records:
        row["replay_reward"] = rewards[str(row["replay_receipt_sha256"])]

    seeds = sorted({int(row["episode_seed"]) for row in records})
    split_by_seed = {seed: SPLITS[index % 3] for index, seed in enumerate(seeds)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[_snapshot_id(row)].append(row)
    snapshots = [
        _make_snapshot(rows, split_by_seed[int(rows[0]["episode_seed"])])
        for _, rows in sorted(grouped.items())
    ]

    nodes_by_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    edges_by_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        nodes_by_split[snapshot["split"]][snapshot["node_id"]] += 1
        by_episode[str(snapshot["episode_id"])].append(snapshot)
    for episode in by_episode.values():
        ordered = sorted(episode, key=lambda row: row["step"])
        for left, right in zip(ordered, ordered[1:]):
            if right["step"] != left["step"] + 1 or right["split"] != left["split"]:
                continue
            edge = f"{left['node_id']}->{right['node_id']}"
            edges_by_split[left["split"]][edge] += 1

    discovery_nodes = set(nodes_by_split["discovery"])
    discovery_edges = set(edges_by_split["discovery"])
    recurrence = {}
    for split in ("qualification", "heldout"):
        node_total = sum(nodes_by_split[split].values())
        edge_total = sum(edges_by_split[split].values())
        recurrence[split] = {
            "node_exact_support": sum(
                count for node, count in nodes_by_split[split].items()
                if node in discovery_nodes
            ),
            "node_total": node_total,
            "edge_exact_support": sum(
                count for edge, count in edges_by_split[split].items()
                if edge in discovery_edges
            ),
            "edge_total": edge_total,
        }

    split_stats = {}
    for split in SPLITS:
        rows = [row for row in snapshots if row["split"] == split]
        split_stats[split] = {
            "snapshots": len(rows),
            "weight_effects": sum(
                row["signature"]["weight_action"] or row["signature"]["weight_state"]
                for row in rows
            ),
            "skill_effects": sum(
                row["signature"]["skill_action"] or row["signature"]["skill_state"]
                for row in rows
            ),
            "authenticity_effects": sum(
                row["signature"]["auth_action"] or row["signature"]["auth_state"]
                for row in rows
            ),
            "mean_rewards": {
                name: (sum(row["rewards"][name] for row in rows) / len(rows) if rows else 0.0)
                for name in TREATMENTS
            },
            "mean_prompt_tokens": {
                name: (
                    sum(row["prompt_tokens"][name] for row in rows) / len(rows)
                    if rows else 0.0
                )
                for name in TREATMENTS
            },
        }

    causal_supported = all(
        split_stats[split]["skill_effects"] > 0
        and split_stats[split]["authenticity_effects"] > 0
        for split in SPLITS
    )
    graph_supported = bool(discovery_edges) and all(
        recurrence[split]["node_total"] > 0
        and recurrence[split]["node_exact_support"] == recurrence[split]["node_total"]
        and recurrence[split]["edge_total"] > 0
        and recurrence[split]["edge_exact_support"] == recurrence[split]["edge_total"]
        for split in ("qualification", "heldout")
    )
    heldout_rewards = split_stats["heldout"]["mean_rewards"]
    value_supported = (
        split_stats["heldout"]["snapshots"] > 0
        and heldout_rewards["G_PLUS_S"] > heldout_rewards["G_MINUS_S"]
        and heldout_rewards["G_PLUS_S"] > heldout_rewards["G_PLUS_RANDOM"]
    )
    authority_order_safe = all(
        row.get("sampling_order") == "AUTHENTIC_FIRST_SHADOW_AFTER_V1"
        for row in records
    )
    gates = {
        "SOURCE_AUTHORITY_ORDER_SAFE": authority_order_safe,
        "SOURCE_CAUSAL_SUPPORTED": causal_supported,
        "SOURCE_GRAPH_SUPPORTED": graph_supported,
        "SOURCE_VALUE_SUPPORTED": value_supported,
    }
    gates["PHASE7_PASS"] = all(gates.values())
    return {
        "schema_version": 1,
        "source_skill_id": manifest["matched_policy_treatments"]["source_skill_id"],
        "split_rule": "ascending_episode_seed_round_robin_discovery_qualification_heldout_v1",
        "split_by_seed": {str(key): value for key, value in split_by_seed.items()},
        "snapshot_count": len(snapshots),
        "split_stats": split_stats,
        "discovery_graph": {
            "nodes": dict(nodes_by_split["discovery"]),
            "edges": dict(edges_by_split["discovery"]),
        },
        "blind_recurrence": recurrence,
        "gates": gates,
        "claim_boundary": (
            "Causal differences show weight/context influence only. SOURCE_VALUE_SUPPORTED "
            "additionally requires held-out official one-step reward advantage; no result "
            "alone establishes far-domain transfer."
        ),
        "snapshots": snapshots,
    }


__all__ = ["TREATMENTS", "SPLITS", "validate_matched_bundle", "analyze_training_effects"]
