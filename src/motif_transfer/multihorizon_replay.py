from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


TREATMENTS = ("B", "G_MINUS_S", "G_PLUS_S", "G_PLUS_RANDOM")
MODES = ("COMMON_G_MINUS_S_CONTINUATION", "FULL_TREATMENT_REGIME")
HORIZONS = (1, 2, 4, 8)
SPLITS = ("discovery", "qualification", "held_out")


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
    for split in SPLITS:
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
    split_by_seed: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Score paired multi-horizon interventions with complete-cell enforcement.

    A treatment mean is never computed from an unbalanced subset.  Every selected
    snapshot must contain exactly one row for each of the four treatments under
    each estimand.  ``split`` embedded in a runtime row takes precedence; the
    optional seed mapping is retained for old artifacts.
    """

    def _split(row: Mapping[str, Any]) -> str:
        value = row.get("split")
        if value is None:
            if split_by_seed is None:
                raise ValueError("row lacks split and no split_by_seed was supplied")
            value = split_by_seed[int(row["episode_seed"])]
        value = str(value)
        if value == "heldout":
            # Read compatibility only.  New receipts always emit held_out.
            value = "held_out"
        if value not in SPLITS:
            raise ValueError(f"unsupported split: {value}")
        return value

    normalized = [dict(row) | {"split": _split(row)} for row in rows]
    observed = [
        row for row in normalized if row.get("status") == "INTERVENTION_OBSERVED"
    ]
    selected_snapshots = {
        (str(row["episode_id"]), int(row.get("fork_step", row.get("step", -1))))
        for row in normalized
    }
    cells: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observed:
        cells[(
            str(row["episode_id"]),
            int(row.get("fork_step", row.get("step", -1))),
            str(row["mode"]),
        )].append(row)

    complete_cells: dict[tuple[str, int, str], dict[str, Mapping[str, Any]]] = {}
    invalid_cells: list[dict[str, Any]] = []
    for episode_id, fork_step in sorted(selected_snapshots):
        for mode in MODES:
            key = (episode_id, fork_step, mode)
            cell_rows = cells.get(key, [])
            by_treatment: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in cell_rows:
                by_treatment[str(row["treatment"])].append(row)
            if set(by_treatment) == set(TREATMENTS) and all(
                len(by_treatment[treatment]) == 1 for treatment in TREATMENTS
            ):
                complete_cells[key] = {
                    treatment: by_treatment[treatment][0]
                    for treatment in TREATMENTS
                }
            else:
                invalid_cells.append({
                    "episode_id": episode_id,
                    "fork_step": fork_step,
                    "mode": mode,
                    "treatment_counts": {
                        treatment: len(by_treatment.get(treatment, ()))
                        for treatment in TREATMENTS
                    },
                })

    stats: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        stats[split] = {}
        for mode in MODES:
            paired = [
                treatment_rows
                for treatment_rows in complete_cells.values()
                if str(next(iter(treatment_rows.values()))["split"]) == split
                and str(next(iter(treatment_rows.values()))["mode"]) == mode
            ]
            horizon_stats = {}
            for horizon in HORIZONS:
                field = f"h{horizon}"
                means = {
                    treatment: (
                        sum(
                            float(cell[treatment]["cumulative_returns"][field])
                            for cell in paired
                        ) / len(paired)
                        if paired else None
                    )
                    for treatment in TREATMENTS
                }
                paired_deltas = {
                    control: (
                        sum(
                            float(cell["G_PLUS_S"]["cumulative_returns"][field])
                            - float(cell[control]["cumulative_returns"][field])
                            for cell in paired
                        ) / len(paired)
                        if paired else None
                    )
                    for control in ("G_MINUS_S", "G_PLUS_RANDOM")
                }
                horizon_stats[field] = {
                    "mean_return": means,
                    "authentic_paired_mean_delta": paired_deltas,
                }
            stats[split][mode] = {
                "complete_snapshots": len(paired),
                "horizons": horizon_stats,
            }

    def _advantage(split: str, mode: str) -> bool:
        cell = stats[split][mode]
        values = cell["horizons"]["h8"]["authentic_paired_mean_delta"]
        return (
            cell["complete_snapshots"] > 0
            and values["G_MINUS_S"] is not None
            and values["G_PLUS_RANDOM"] is not None
            and values["G_MINUS_S"] > 0
            and values["G_PLUS_RANDOM"] > 0
        )

    blind_cells_complete = not any(
        row for row in invalid_cells
        if any(
            candidate.get("episode_id") == row["episode_id"]
            and candidate.get("split") in {"qualification", "held_out"}
            for candidate in normalized
        )
    )
    gates = {
        "BLIND_CELLS_COMPLETE": blind_cells_complete,
        "QUALIFICATION_COMMON_H8_VALUE": _advantage(
            "qualification", "COMMON_G_MINUS_S_CONTINUATION"
        ),
        "HELDOUT_COMMON_H8_VALUE": _advantage(
            "held_out", "COMMON_G_MINUS_S_CONTINUATION"
        ),
        "QUALIFICATION_FULL_H8_VALUE": _advantage(
            "qualification", "FULL_TREATMENT_REGIME"
        ),
        "HELDOUT_FULL_H8_VALUE": _advantage(
            "held_out", "FULL_TREATMENT_REGIME"
        ),
    }
    gates["SOURCE_H8_VALUE_SUPPORTED"] = all(gates.values())
    return {
        "split_stats": stats,
        "status_counts": {
            status: sum(row.get("status") == status for row in normalized)
            for status in sorted({str(row.get("status")) for row in normalized})
        },
        "selected_snapshots": len(selected_snapshots),
        "complete_cells": len(complete_cells),
        "invalid_cells": invalid_cells,
        "gates": gates,
        "primary_horizon": 8,
        "claim_boundary": (
            "h1/h2/h4 are diagnostics. The frozen primary gate is h8 and requires "
            "authentic advantage in both common-continuation and full-regime estimands "
            "on qualification and heldout."
        ),
    }


__all__ = [
    "TREATMENTS", "MODES", "HORIZONS", "SPLITS", "stable_hash", "file_hash",
    "snapshot_id", "choose_lineage_snapshots", "extract_policy_prefix",
    "cumulative_returns", "analyze_multihorizon_rows",
]
