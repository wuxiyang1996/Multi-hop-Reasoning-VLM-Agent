from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


TREATMENTS = ("B", "G_MINUS_S", "G_PLUS_S", "G_PLUS_RANDOM")
MODES = ("COMMON_G_MINUS_S_CONTINUATION", "FULL_TREATMENT_REGIME")
HORIZONS = (1, 2, 4, 8)


def stable_hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_id(row: Mapping[str, Any]) -> str:
    return f"seed={int(row['episode_seed'])}:step={int(row['step'])}"


def choose_lineage_snapshots(
    records: Sequence[Mapping[str, Any]],
    split_by_seed: Mapping[int, str],
    *,
    maximum_per_split: int,
) -> dict[str, list[str]]:
    """Choose long candidate spans without reading actions, rewards, or skill text."""

    authentic = [row for row in records if row.get("treatment") == "G_PLUS_S"]
    by_split_episode: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in authentic:
        split = split_by_seed[int(row["episode_seed"])]
        by_split_episode[split][str(row["episode_id"])].append(row)
    selected: dict[str, list[str]] = {}
    for split in ("discovery", "qualification", "heldout"):
        spans = []
        for episode_id, episode_rows in by_split_episode[split].items():
            ordered = sorted(episode_rows, key=lambda row: int(row["step"]))
            current = []
            for row in ordered:
                if current and int(row["step"]) != int(current[-1]["step"]) + 1:
                    spans.append(current)
                    current = []
                current.append(row)
            if current:
                spans.append(current)
        spans.sort(key=lambda span: (
            -len(span),
            stable_hash((split, span[0]["episode_id"], int(span[0]["step"]))),
        ))
        chosen = []
        for span in spans:
            for row in span:
                if len(chosen) >= maximum_per_split:
                    break
                chosen.append(snapshot_id(row))
            if len(chosen) >= maximum_per_split:
                break
        selected[split] = chosen
    return selected


def extract_policy_prefix(prompt: str) -> str:
    marker = "Game state:"
    position = prompt.find(marker)
    if position < 0:
        raise ValueError("matched prompt lacks Game state marker")
    return prompt[:position].rstrip()


def cumulative_returns(step_rewards: Sequence[float]) -> dict[str, float]:
    if not step_rewards:
        raise ValueError("at least the treatment first action must be observed")
    result = {}
    for horizon in HORIZONS:
        result[f"h{horizon}"] = float(sum(step_rewards[:horizon]))
    return result


def analyze_multihorizon_rows(
    rows: Sequence[Mapping[str, Any]],
    split_by_seed: Mapping[int, str],
) -> dict[str, Any]:
    observed = [row for row in rows if row.get("status") == "INTERVENTION_OBSERVED"]
    stats: dict[str, dict[str, Any]] = {}
    for split in ("discovery", "qualification", "heldout"):
        split_rows = [
            row for row in observed if split_by_seed[int(row["episode_seed"])] == split
        ]
        stats[split] = {}
        for mode in MODES:
            mode_rows = [row for row in split_rows if row["mode"] == mode]
            stats[split][mode] = {
                f"h{horizon}": {
                    treatment: (
                        sum(
                            float(row["cumulative_returns"][f"h{horizon}"])
                            for row in mode_rows if row["treatment"] == treatment
                        ) / sum(row["treatment"] == treatment for row in mode_rows)
                        if any(row["treatment"] == treatment for row in mode_rows)
                        else None
                    )
                    for treatment in TREATMENTS
                }
                for horizon in HORIZONS
            }
    def _advantage(split: str, mode: str) -> bool:
        values = stats[split][mode]["h8"]
        return (
            values["G_PLUS_S"] is not None
            and values["G_PLUS_S"] > values["G_MINUS_S"]
            and values["G_PLUS_S"] > values["G_PLUS_RANDOM"]
        )
    gates = {
        "QUALIFICATION_COMMON_H8_VALUE": _advantage(
            "qualification", "COMMON_G_MINUS_S_CONTINUATION"
        ),
        "HELDOUT_COMMON_H8_VALUE": _advantage(
            "heldout", "COMMON_G_MINUS_S_CONTINUATION"
        ),
        "QUALIFICATION_FULL_H8_VALUE": _advantage(
            "qualification", "FULL_TREATMENT_REGIME"
        ),
        "HELDOUT_FULL_H8_VALUE": _advantage(
            "heldout", "FULL_TREATMENT_REGIME"
        ),
    }
    gates["SOURCE_H8_VALUE_SUPPORTED"] = all(gates.values())
    return {
        "split_stats": stats,
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({str(row.get("status")) for row in rows})
        },
        "gates": gates,
        "primary_horizon": 8,
        "claim_boundary": (
            "h1/h2/h4 are diagnostics. The frozen primary gate is h8 and requires "
            "authentic advantage in both common-continuation and full-regime estimands "
            "on qualification and heldout."
        ),
    }


__all__ = [
    "TREATMENTS", "MODES", "HORIZONS", "stable_hash", "file_hash",
    "snapshot_id", "choose_lineage_snapshots", "extract_policy_prefix",
    "cumulative_returns", "analyze_multihorizon_rows",
]
