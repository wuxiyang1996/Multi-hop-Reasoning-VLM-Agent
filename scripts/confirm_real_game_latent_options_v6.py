#!/usr/bin/env python3
"""Fresh cross-game confirmation of the frozen anonymous option ontology.

The V5 candidate was selected on Sokoban, Tetris, and 2048.  This evaluator
applies it without refitting to the pre-existing Candy Crush and Super Mario
held-out episode roles.  Target-game discovery episodes are used only to set a
reward location/scale, never to update clusters or value coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import Ridge


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_latent_options import (  # noqa: E402
    SourceOptionRow,
    extract_structural_episode,
    mse_summary,
    reward_normalizer,
    target_values,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(payload: dict[str, Any], field: str) -> None:
    body = dict(payload)
    claimed = body.pop(field, None)
    if claimed != stable_hash(body):
        raise ValueError(f"{field} mismatch")


def _experiences(source: dict[str, Any], item_id: str) -> list[dict[str, Any]]:
    artifact = source["artifacts"][item_id]
    path = Path(artifact["path"])
    if _sha256(path) != artifact["sha256"]:
        raise ValueError(f"source episode drift: {item_id}")
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("experiences")
    if not isinstance(rows, list):
        raise ValueError(f"missing experiences: {item_id}")
    return rows


def _rows(
    source: dict[str, Any], *, games: Sequence[str], role: str,
    normalizers: dict[str, tuple[float, float]],
) -> tuple[SourceOptionRow, ...]:
    output: list[SourceOptionRow] = []
    for game in games:
        mean, scale = normalizers[game]
        for item_id in source["games"][game]["roles"][role]:
            output.extend(extract_structural_episode(
                _experiences(source, item_id), game=game, episode_id=item_id,
                reward_mean=mean, reward_scale=scale,
            ))
    return tuple(output)


def _normalizers(
    source: dict[str, Any], games: Sequence[str],
) -> dict[str, tuple[float, float]]:
    return {
        game: reward_normalizer([
            _experiences(source, item_id)
            for item_id in source["games"][game]["roles"]["discovery"]
        ])
        for game in games
    }


def _cluster_ids(rows: Sequence[SourceOptionRow], candidate: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray([row.effect_features for row in rows], dtype=np.float64)
    mean = np.asarray(candidate["effect_scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(candidate["effect_scaler"]["scale"], dtype=np.float64)
    centers = np.asarray(candidate["cluster_centers"], dtype=np.float64)
    standardized = (matrix - mean) / scale
    distances = np.sum((standardized[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def _design(
    rows: Sequence[SourceOptionRow], candidate: dict[str, Any], *,
    corruption: str | None = None,
) -> np.ndarray:
    option_ids = _cluster_ids(rows, candidate)
    cluster_count = int(candidate["cluster_count"])
    if corruption == "phase_permuted":
        option_ids = (option_ids + 1) % cluster_count
    elif corruption == "within_episode_shift":
        option_ids = option_ids.copy()
        start = 0
        while start < len(rows):
            end = start + 1
            while end < len(rows) and rows[end].episode_id == rows[start].episode_id:
                end += 1
            option_ids[start:end] = np.roll(option_ids[start:end], max(1, (end - start) // 3))
            start = end
    elif corruption is not None:
        raise ValueError(f"unknown corruption: {corruption}")
    previous = []
    for index, row in enumerate(rows):
        same_episode = index > 0 and rows[index - 1].episode_id == row.episode_id
        previous.append(int(option_ids[index - 1]) if same_episode else cluster_count)
    return np.column_stack((
        np.asarray([row.context_features for row in rows], dtype=np.float64),
        np.eye(cluster_count, dtype=np.float64)[option_ids],
        np.eye(cluster_count + 1, dtype=np.float64)[previous],
    ))


def _predict(design: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
    model = candidate["value_model"]
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    intercept = np.asarray(model["intercept"], dtype=np.float64)
    return design @ coefficients.T + intercept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "FROZEN_BEFORE_CONFIRMATION_HELDOUT_READ":
        raise SystemExit("confirmation protocol is not frozen")
    for relative, expected in config["integrity"]["file_sha256"].items():
        if _sha256(REPO / relative) != expected:
            raise SystemExit(f"frozen dependency changed: {relative}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if _sha256(args.manifest) != config["manifest_file_sha256"]:
        raise SystemExit("manifest drift")
    _self_hash(manifest, "manifest_sha256")
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    _self_hash(candidate, "artifact_sha256")
    if candidate["artifact_sha256"] != config["candidate_artifact_sha256"]:
        raise SystemExit("candidate drift")
    if int(candidate["cluster_count"]) != int(config["fixed_cluster_count"]):
        raise SystemExit("cluster count changed")
    source = manifest["source"]["legacy_real_game_rollouts"]
    train_games = tuple(config["train_games"])
    confirmation_games = tuple(config["confirmation_games"])
    if set(train_games) & set(confirmation_games):
        raise SystemExit("confirmation games overlap candidate train games")

    # The only confirmation-game calibration is reward location/scale from the
    # already frozen discovery role.  Clusters and value parameters remain fixed.
    confirmation_normalizers = _normalizers(source, confirmation_games)
    held_out = _rows(
        source, games=confirmation_games, role="held_out",
        normalizers=confirmation_normalizers,
    )
    expected = target_values(held_out)
    authentic = mse_summary(expected, _predict(_design(held_out, candidate), candidate))
    controls = {
        name: mse_summary(
            expected,
            _predict(_design(held_out, candidate, corruption=corruption), candidate),
        )
        for name, corruption in (
            ("phase_permuted", "phase_permuted"),
            ("within_episode_option_shift", "within_episode_shift"),
        )
    }

    train_normalizers = {
        game: (
            float(candidate["reward_normalizers_discovery_only"][game]["mean"]),
            float(candidate["reward_normalizers_discovery_only"][game]["scale"]),
        )
        for game in train_games
    }
    discovery = _rows(
        source, games=train_games, role="discovery", normalizers=train_normalizers,
    )
    marginal = Ridge(alpha=1.0).fit(
        np.asarray([row.context_features for row in discovery], dtype=np.float64),
        target_values(discovery),
    )
    controls["marginal_context"] = mse_summary(
        expected,
        marginal.predict(np.asarray([row.context_features for row in held_out], dtype=np.float64)),
    )
    improvement = {
        name: (metric["aggregate"] - authentic["aggregate"]) / metric["aggregate"]
        for name, metric in controls.items()
    }
    per_game = {}
    for game in confirmation_games:
        subset = tuple(row for row in held_out if row.game == game)
        subset_expected = target_values(subset)
        authentic_game = mse_summary(
            subset_expected, _predict(_design(subset, candidate), candidate),
        )
        marginal_game = mse_summary(
            subset_expected,
            marginal.predict(np.asarray([row.context_features for row in subset], dtype=np.float64)),
        )
        per_game[game] = {
            "rows": len(subset),
            "episodes": len(source["games"][game]["roles"]["held_out"]),
            "authentic": authentic_game,
            "marginal_context": marginal_game,
            "authentic_improvement_over_marginal": (
                marginal_game["aggregate"] - authentic_game["aggregate"]
            ) / marginal_game["aggregate"],
            "cluster_counts": np.bincount(
                _cluster_ids(subset, candidate), minlength=int(candidate["cluster_count"]),
            ).tolist(),
        }
    minimum = float(config["formal_gates"]["minimum_relative_improvement"])
    gates = {
        "candidate_self_hash_valid": True,
        "disjoint_confirmation_games": not (set(train_games) & set(confirmation_games)),
        "exact_heldout_episode_count": sum(
            len(source["games"][game]["roles"]["held_out"])
            for game in confirmation_games
        ) == int(config["formal_gates"]["expected_heldout_episodes"]),
        "pooled_above_all_controls": all(value >= minimum for value in improvement.values()),
        "both_games_above_marginal": all(
            value["authentic_improvement_over_marginal"] > 0 for value in per_game.values()
        ),
        "anonymous_options_non_degenerate": all(
            sum(count > 0 for count in value["cluster_counts"]) >= 3
            for value in per_game.values()
        ),
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "real-game-latent-options-v6-confirmation",
        "status": "FRESH_CROSS_GAME_LATENT_ONTOLOGY_CONFIRMED" if passed else "FRESH_CROSS_GAME_LATENT_ONTOLOGY_NOT_CONFIRMED",
        "claim_boundary": config["claim_boundary"],
        "candidate_artifact_sha256": candidate["artifact_sha256"],
        "config_sha256": _sha256(args.config),
        "manifest_file_sha256": _sha256(args.manifest),
        "train_games": list(train_games),
        "confirmation_games": list(confirmation_games),
        "heldout_rows": len(held_out),
        "authentic": authentic,
        "controls": controls,
        "relative_improvement_over_controls": improvement,
        "per_game": per_game,
        "gates": gates,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
