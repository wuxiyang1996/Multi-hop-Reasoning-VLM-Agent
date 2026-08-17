#!/usr/bin/env python3
"""Launch the independent V13 ALFWorld replication with frozen V11 logic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_alfworld_unified_goal_acquisition_v11 as frozen  # noqa: E402
from motif_transfer.alfworld_env import (  # noqa: E402
    ALFWorldTextBatchEnvironment as _FrozenBatchEnvironment,
)


V13_STATUS = "FROZEN_BEFORE_ANY_ALFWORLD_V13_RESERVE_RESET_OR_OUTCOME"


class _ValidTrainBatchEnvironment(_FrozenBatchEnvironment):
    """Transport valid_train while leaving the frozen V11 runner untouched."""

    def __init__(self, *args, data_path: str, split: str, **kwargs) -> None:
        if split != "train":
            raise ValueError("V13 transport expects the frozen runner train token")
        self._v13_data_root = Path(data_path).resolve()
        super().__init__(
            *args, data_path=data_path, split=split, **kwargs,
        )

    def reset(self):
        observation = super().reset()
        actual = Path(self.resolved_game_file).resolve()
        relative = actual.relative_to(
            self._v13_data_root / "json_2.1.1" / "valid_train"
        )
        # V11 computes its authorization identity relative to train.  Expose a
        # virtual path with that prefix during execution, then normalize every
        # episode back to the preregistered split-relative ID before writing.
        self.resolved_game_file = str(
            self._v13_data_root / "json_2.1.1" / "train" / relative
        )
        return observation


def _normalize_report(
    report: dict, config: dict,
) -> dict:
    fake_root = (
        Path(str(config["alfworld_data"])) / "json_2.1.1" / "train"
    ).resolve()
    actual_root = (
        Path(str(config["alfworld_data"])) / "json_2.1.1" / "valid_train"
    ).resolve()
    expected_hashes = dict(config["task_file_sha256"])
    observed_ids: set[str] = set()
    for rows in report["episodes"].values():
        for episode in rows:
            body = dict(episode)
            body.pop("episode_sha256", None)
            relative = str(
                Path(str(body["task_id"])).resolve().relative_to(fake_root)
            )
            actual = actual_root / relative
            actual_sha = hashlib.sha256(actual.read_bytes()).hexdigest()
            if expected_hashes.get(relative) != actual_sha:
                raise ValueError(f"V13 physical task mismatch: {relative}")
            body["task_id"] = relative
            episode.clear()
            episode.update(body | {"episode_sha256": frozen.stable_hash(body)})
            observed_ids.add(relative)
    expected_ids = set(map(str, config["task_ids"]))
    report_body = dict(report)
    report_body.pop("report_sha256", None)
    report_body["schema_version"] = (
        "alfworld-unified-goal-acquisition-report-v13"
    )
    report_body["role"] = "independent_formal_replication"
    report_body["physical_data_split"] = "valid_train"
    report_body["v11_frozen_action_runtime_reused"] = True
    report_body["v13_transport_only"] = (
        "VALID_TRAIN_DATASET_PATH_AND_SPLIT_RELATIVE_ID_NORMALIZATION"
    )
    report_body["gates"] = dict(report_body["gates"]) | {
        "episodes_match_preregistered_task_ids": observed_ids == expected_ids,
        "physical_task_hashes_verified": observed_ids == set(expected_hashes),
    }
    return report_body | {"report_sha256": frozen.stable_hash(report_body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = frozen._read(args.config)
    frozen._self_hash(config, "config_sha256")
    if config.get("status") != V13_STATUS:
        raise ValueError("expected a frozen ALFWorld V13 replication config")
    launcher_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if config.get("v13_launcher_file_sha256") != launcher_sha:
        raise ValueError("V13 launcher changed after the replication was frozen")
    transport_config = REPO / str(config["valid_train_transport_config"])
    transport_sha = hashlib.sha256(transport_config.read_bytes()).hexdigest()
    if config.get("valid_train_transport_config_file_sha256") != transport_sha:
        raise ValueError("valid_train transport config changed after freeze")

    # The only compatibility change is the preregistration status token.  The
    # imported V11 run function and all of its hash-checked dependencies remain
    # the action-selection and evaluation implementation.
    frozen.FORMAL_STATUS = V13_STATUS
    frozen.ALFWorldTextBatchEnvironment = _ValidTrainBatchEnvironment
    report = _normalize_report(frozen.run(args.config), config)
    output = REPO / str(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "summaries": report["summaries"],
        "paired": report["paired"],
        "authority_calls": sum(
            map(len, report["authority_receipts"].values())
        ),
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
