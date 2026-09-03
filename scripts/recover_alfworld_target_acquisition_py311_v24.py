#!/usr/bin/env python3
"""Recover the V23 report after its post-episode selection-schema error.

The frozen V23 runner completed every episode before looking up a V18-era
selection metadata field that V23 renamed.  This wrapper leaves the frozen
runner, policy, programs, thresholds, task identities, and artifacts untouched.
It supplies only a read-time alias for that metadata field and labels the
result as a deterministic reconstruction, rather than a second fresh run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_alfworld_target_acquisition_fresh_v19 as frozen  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402


DEFAULT_CONFIG = (
    REPO / "configs/alfworld_target_acquisition_py311_v23/formal.json"
)
DEFAULT_OUTPUT = (
    REPO / "runs/alfworld_target_acquisition_py311_v23/"
    "report_deterministic_reconstruction.json"
)
EXPECTED_FROZEN_RUNNER_SHA256 = (
    "bc05364847edacd3daa0c6ffe43624abb8921657add7e018c3a767f1129aeaab"
)
OLD_FIELD = "selection_used_observation_walkthrough_or_policy_outcome"
NEW_FIELD = (
    "selection_used_compiler_solvability_observation_or_policy_outcome"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _SelectionSchemaAlias(dict[str, Any]):
    """Expose one legacy lookup without changing serialized hash content."""

    def __getitem__(self, key: str) -> Any:
        if key == OLD_FIELD and key not in self and NEW_FIELD in self:
            return super().__getitem__(NEW_FIELD)
        return super().__getitem__(key)


def _install_read_alias(selection_path: Path) -> None:
    original_read = frozen._read

    def read_with_alias(path: Path) -> dict[str, Any]:
        value = original_read(path)
        if path.resolve() == selection_path.resolve():
            return _SelectionSchemaAlias(value)
        return value

    frozen._read = read_with_alias


def recover(config_path: Path) -> dict[str, Any]:
    frozen_runner = Path(frozen.__file__).resolve()
    if _sha(frozen_runner) != EXPECTED_FROZEN_RUNNER_SHA256:
        raise ValueError("the frozen V19 runner changed; recovery is invalid")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["v19_runner_file_sha256"] != EXPECTED_FROZEN_RUNNER_SHA256:
        raise ValueError("V23 does not name the expected frozen runner")
    if Path(REPO / config["output"]).exists():
        raise ValueError("a primary V23 report exists; recovery is unnecessary")

    selection_path = REPO / str(config["selection"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if OLD_FIELD in selection or selection.get(NEW_FIELD) is not False:
        raise ValueError("V23 selection does not match the one-field schema bug")
    _install_read_alias(selection_path)
    report = frozen.run(config_path)

    body = dict(report)
    body.pop("report_sha256", None)
    body["role"] = (
        "deterministic_reconstruction_of_completed_prospective_execution"
    )
    body["status"] = (
        "ALFWORLD_SOURCE_ACQUISITION_VALUE_VALIDATED_BY_DETERMINISTIC_"
        "RECONSTRUCTION"
        if all(body["gates"].values())
        else "ALFWORLD_SOURCE_ACQUISITION_VALUE_RECONSTRUCTION_FAILED"
    )
    body["recovery_audit"] = {
        "original_execution_role": (
            "prospective_execution_untouched_mechanism_replication"
        ),
        "original_execution_completed_all_condition_episodes": True,
        "original_failure_stage": "post_episode_report_assembly",
        "original_exception": (
            "KeyError: "
            "'selection_used_observation_walkthrough_or_policy_outcome'"
        ),
        "freshness_applies_to_original_execution_only": True,
        "reconstruction_reuses_consumed_deterministic_tasks": True,
        "policy_program_threshold_or_task_change": False,
        "read_time_metadata_alias_only": {OLD_FIELD: NEW_FIELD},
        "frozen_runner_sha256": EXPECTED_FROZEN_RUNNER_SHA256,
        "recovery_wrapper_sha256": _sha(Path(__file__).resolve()),
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output if args.output.is_absolute() else REPO / args.output
    report = recover(config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "summaries": report["summaries"],
        "paired": report["paired"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
