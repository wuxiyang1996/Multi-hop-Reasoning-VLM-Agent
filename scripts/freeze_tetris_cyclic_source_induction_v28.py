#!/usr/bin/env python3
"""Freeze fresh official-Tetris source splits before collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


OFFICIAL_ENV = Path(
    "/fs/gamma-projects/vlm-robot/GamingAgent/gamingagent/envs/"
    "custom_04_tetris/tetrisEnv.py"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen config: {output}")
    paths = {
        "cyclic_inducer": "src/motif_transfer/cyclic_identity_induction.py",
        "collector": "scripts/collect_tetris_cyclic_source_reserve_v28.py",
        "analyzer": "scripts/analyze_tetris_cyclic_source_induction_v28.py",
        "old_source_artifact": "runs/tetris_rotation_group_v1/source_artifact.json",
        "target_utility_summary": (
            "docs/results/tir_rotation_counterfactual_v2_summary.json"
        ),
    }
    seeds = list(range(281001, 281097))
    body = {
        "schema_version": "tetris-cyclic-source-induction-v28-protocol",
        "status": "FROZEN_BEFORE_FRESH_SOURCE_COLLECTION",
        "claim_boundary": (
            "Fresh source-only official-Tetris intervention forks induce an "
            "anonymous composition-to-identity constraint without source "
            "action tokens, target data, or a supplied inverse formula. The "
            "existing fresh TIR result is used only as extensionally identical "
            "utility context; it is not relabeled as a new prospective target "
            "run for the V28 artifact."
        ),
        "source_namespace": "tetris-cyclic-source-reserve-v28",
        "official_tetris_environment": str(OFFICIAL_ENV),
        "official_tetris_environment_file_sha256": _sha(OFFICIAL_ENV),
        **paths,
        **{
            f"{field}_file_sha256": _sha(REPO / path)
            for field, path in paths.items()
        },
        "dependency_fields": {
            "official_tetris_environment_file_sha256": (
                "official_tetris_environment"
            ),
            **{
                f"{field}_file_sha256": field for field in paths
            },
        },
        "source_splits": {
            "development": seeds[:48],
            "qualification": seeds[48:72],
            "reserve": seeds[72:],
        },
        "split_rule": (
            "Consecutive committed seeds 281001..281096; first 48 "
            "development, next 24 qualification, final 24 reserve. "
            "Only official dynamics with observed cyclic order four are "
            "retained. Reserve remains unopened until qualification passes."
        ),
        "minimum_induction_episodes": 2,
        "minimum_development_episodes": 12,
        "minimum_qualification_episodes": 6,
        "acquisition_order_namespace": (
            "tetris-cyclic-source-v28-acquisition-order"
        ),
        "outputs": {
            "development": (
                "runs/tetris_cyclic_source_induction_v28/development.json"
            ),
            "qualification": (
                "runs/tetris_cyclic_source_induction_v28/qualification.json"
            ),
            "reserve": "runs/tetris_cyclic_source_induction_v28/reserve.json",
        },
        "output": "docs/results/tetris_cyclic_source_induction_v28.json",
        "frozen_gates": {
            "unique_composition_to_identity_hypothesis": True,
            "zero_raw_source_action_tokens": True,
            "zero_target_data_for_source_induction": True,
            "qualification_all_forks_classified": True,
            "permuted_effect_and_terminal_controls_abstain": True,
            "reserve_all_forks_classified": True,
        },
    }
    config = body | {"config_sha256": stable_hash(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "configs/tetris_cyclic_source_induction_v28.json",
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), indent=2))


if __name__ == "__main__":
    main()
