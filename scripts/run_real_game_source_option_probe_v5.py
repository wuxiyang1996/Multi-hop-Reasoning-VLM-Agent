#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import Ridge


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.real_game_latent_options import (  # noqa: E402
    CONTEXT_FEATURE_NAMES,
    EFFECT_FEATURE_NAMES,
    extract_structural_episode,
    fit_effect_clusters,
    fit_value_model,
    mse_summary,
    option_design,
    reward_normalizer,
    target_values,
)
from motif_transfer.real_game_multitarget_manifest import (  # noqa: E402
    file_sha256,
    stable_hash,
)


DEFAULT_GAMES = ("sokoban", "tetris", "twenty_forty_eight")
SEED = 20260810


def _rounded(value: np.ndarray | list | tuple, decimals: int = 12) -> list:
    return np.round(np.asarray(value, dtype=np.float64), decimals=decimals).tolist()


def _source_payload(manifest: dict) -> dict:
    return manifest["source"]["legacy_real_game_rollouts"]


def _experiences(source: dict, item_id: str) -> list[dict]:
    artifact = source["artifacts"][item_id]
    payload = json.loads(Path(artifact["path"]).read_text())
    rows = payload.get("experiences")
    if not isinstance(rows, list):
        raise ValueError(f"source episode has no experiences list: {item_id}")
    return rows


def _normalizers(source: dict, games: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    output = {}
    for game in games:
        episodes = [
            _experiences(source, item_id)
            for item_id in source["games"][game]["roles"]["discovery"]
        ]
        output[game] = reward_normalizer(episodes)
    return output


def _rows(
    source: dict,
    *,
    games: tuple[str, ...],
    role: str,
    normalizers: dict[str, tuple[float, float]],
) -> tuple:
    output = []
    for game in games:
        mean, scale = normalizers[game]
        for item_id in source["games"][game]["roles"][role]:
            output.extend(extract_structural_episode(
                _experiences(source, item_id),
                game=game,
                episode_id=item_id,
                reward_mean=mean,
                reward_scale=scale,
            ))
    return tuple(output)


def _serialize_candidate(
    *,
    games: tuple[str, ...],
    normalizers: dict[str, tuple[float, float]],
    scaler,
    clusterer,
    value_model,
    selected_cluster_count: int,
    manifest_sha256: str,
    runtime_hashes: dict[str, str],
) -> dict:
    payload = {
        "schema_version": 1,
        "artifact_role": "FROZEN_TARGET_QUALIFICATION_CANDIDATE",
        "claim_limit": "Development source probe; not a confirmatory source-held-out result",
        "games": list(games),
        "manifest_sha256": manifest_sha256,
        "seed": SEED,
        "effect_feature_names": list(EFFECT_FEATURE_NAMES),
        "context_feature_names": list(CONTEXT_FEATURE_NAMES),
        "reward_normalizers_discovery_only": {
            game: {"mean": round(mean, 12), "scale": round(scale, 12)}
            for game, (mean, scale) in normalizers.items()
        },
        "cluster_count": selected_cluster_count,
        "effect_scaler": {
            "mean": _rounded(scaler.mean_),
            "scale": _rounded(scaler.scale_),
        },
        "cluster_centers": _rounded(clusterer.cluster_centers_),
        "value_model": {
            "alpha": float(value_model.alpha),
            "coefficients": _rounded(value_model.coef_),
            "intercept": _rounded(value_model.intercept_),
        },
        "runtime_hashes": runtime_hashes,
        "forbidden_source_fields": [
            "intentions",
            "skills",
            "skill_candidates",
            "skill_chosen_idx",
            "skill_reasoning",
            "summary",
            "summary_state",
        ],
    }
    payload["artifact_sha256"] = stable_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "configs/real_game_multitarget_v5_manifest.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/real_game_multitarget_neurosymbolic_v5.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "runs/real_game_multitarget_neurosymbolic_v5/source_development",
    )
    parser.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    games = tuple(args.games)
    source = _source_payload(manifest)
    normalizers = _normalizers(source, games)
    discovery = _rows(source, games=games, role="discovery", normalizers=normalizers)
    qualification = _rows(source, games=games, role="qualification", normalizers=normalizers)
    held_out = _rows(source, games=games, role="held_out", normalizers=normalizers)

    qualification_candidates = []
    fitted = {}
    for cluster_count in range(3, 9):
        scaler, clusterer = fit_effect_clusters(
            discovery, cluster_count=cluster_count, seed=SEED
        )
        model = fit_value_model(
            option_design(discovery, scaler=scaler, clusterer=clusterer),
            target_values(discovery),
        )
        metrics = mse_summary(
            target_values(qualification),
            model.predict(option_design(qualification, scaler=scaler, clusterer=clusterer)),
        )
        qualification_candidates.append({"cluster_count": cluster_count, "mse": metrics})
        fitted[cluster_count] = (scaler, clusterer, model)
    selected = min(
        qualification_candidates,
        key=lambda row: (row["mse"]["aggregate"], row["cluster_count"]),
    )["cluster_count"]
    scaler, clusterer, model = fitted[selected]
    expected = target_values(held_out)
    authentic = mse_summary(
        expected,
        model.predict(option_design(held_out, scaler=scaler, clusterer=clusterer)),
    )
    controls = {
        name: mse_summary(
            expected,
            model.predict(option_design(
                held_out,
                scaler=scaler,
                clusterer=clusterer,
                corruption=corruption,
            )),
        )
        for name, corruption in (
            ("phase_permuted", "phase_permuted"),
            ("within_episode_option_shift", "within_episode_shift"),
        )
    }
    marginal_model = Ridge(alpha=1.0).fit(
        np.asarray([row.context_features for row in discovery]),
        target_values(discovery),
    )
    controls["marginal_context"] = mse_summary(
        expected,
        marginal_model.predict(np.asarray([row.context_features for row in held_out])),
    )
    rng = np.random.default_rng(SEED)
    shuffled_values = target_values(discovery)[rng.permutation(len(discovery))]
    value_shuffle_model = fit_value_model(
        option_design(discovery, scaler=scaler, clusterer=clusterer), shuffled_values
    )
    controls["value_shuffle"] = mse_summary(
        expected,
        value_shuffle_model.predict(
            option_design(held_out, scaler=scaler, clusterer=clusterer)
        ),
    )
    improvements = {
        name: (control["aggregate"] - authentic["aggregate"]) / control["aggregate"]
        for name, control in controls.items()
    }
    per_game = {}
    for game in games:
        subset = tuple(row for row in held_out if row.game == game)
        per_game[game] = mse_summary(
            target_values(subset),
            model.predict(option_design(subset, scaler=scaler, clusterer=clusterer)),
        )

    runtime_hashes = {
        "config": file_sha256(args.config),
        "manifest": file_sha256(args.manifest),
        "module": file_sha256(REPO / "src/motif_transfer/real_game_latent_options.py"),
        "runner": file_sha256(Path(__file__)),
    }
    candidate = _serialize_candidate(
        games=games,
        normalizers=normalizers,
        scaler=scaler,
        clusterer=clusterer,
        value_model=model,
        selected_cluster_count=selected,
        manifest_sha256=manifest["manifest_sha256"],
        runtime_hashes=runtime_hashes,
    )
    report = {
        "schema_version": 1,
        "experiment": "real_game_multitarget_neurosymbolic_v5_source_development",
        "status": "DEVELOPMENT_PASS_FRESH_CONFIRMATION_REQUIRED"
        if all(value >= 0.10 for value in improvements.values())
        else "DEVELOPMENT_FAIL",
        "claim_limit": (
            "The exact compiler was frozen after a first development read of the episode-held-out "
            "metrics. This authorizes target qualification diagnostics, not a confirmatory transfer claim."
        ),
        "games": list(games),
        "row_counts": {
            "discovery": len(discovery),
            "qualification": len(qualification),
            "held_out": len(held_out),
        },
        "qualification_candidates": qualification_candidates,
        "selected_cluster_count": selected,
        "cluster_counts_discovery": np.bincount(
            clusterer.predict(scaler.transform(np.asarray([
                row.effect_features for row in discovery
            ]))),
            minlength=selected,
        ).tolist(),
        "held_out": {
            "authentic": authentic,
            "controls": controls,
            "relative_improvement_over_controls": improvements,
            "per_game_authentic": per_game,
        },
        "candidate_artifact_sha256": candidate["artifact_sha256"],
        "runtime_hashes": runtime_hashes,
    }
    report["report_sha256"] = stable_hash(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frozen_candidate.json").write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "development_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "status": report["status"],
        "selected_cluster_count": selected,
        "authentic_aggregate_mse": authentic["aggregate"],
        "control_aggregate_mse": {
            name: value["aggregate"] for name, value in controls.items()
        },
        "relative_improvement_over_controls": improvements,
        "candidate_artifact_sha256": candidate["artifact_sha256"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
