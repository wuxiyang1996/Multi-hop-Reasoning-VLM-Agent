#!/usr/bin/env python3
"""Materialize the bundled ALFWorld artifacts and a portable replay config."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.audit_four_domain_release_v1 import audit_release  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(
    *,
    manifest_path: Path,
    output_dir: Path,
    alfworld_config: Path,
    alfworld_data: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    alfworld_config = alfworld_config.resolve()
    alfworld_data = alfworld_data.resolve()
    if not alfworld_config.is_file():
        raise ValueError(f"ALFWorld config is absent: {alfworld_config}")
    if not (alfworld_data / "json_2.1.1" / "valid_unseen").is_dir():
        raise ValueError(f"ALFWorld valid_unseen data is absent: {alfworld_data}")
    audit = audit_release(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    role_names = {
        "alfworld_frozen_candidate": "alfworld_candidate.json",
        "alfworld_development_report": "alfworld_development_report.json",
        "alfworld_final_report": "alfworld_reference_final_report.json",
    }
    materialized: dict[str, Path] = {}
    for spec in manifest["bundled_artifacts"]:
        role = str(spec["role"])
        destination = output_dir / role_names[role]
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite release file: {destination}")
        source = (REPO / str(spec["path"])).resolve()
        with gzip.open(source, "rb") as handle:
            raw = handle.read()
        if hashlib.sha256(raw).hexdigest() != spec["uncompressed_sha256"]:
            raise ValueError(f"bundled artifact changed during materialization: {role}")
        destination.write_bytes(raw)
        materialized[role] = destination

    frozen_config_path = (
        REPO / manifest["alfworld_replay"]["frozen_config"]
    ).resolve()
    replay_config = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    replay_output = output_dir / "alfworld_replay_report.json"
    replay_config_path = output_dir / "alfworld_portable_replay_config.json"
    if replay_config_path.exists() or replay_output.exists():
        raise FileExistsError("refusing to overwrite a prior portable replay")
    replay_config["target"]["alfworld_config"] = str(alfworld_config)
    replay_config["target"]["alfworld_data"] = str(alfworld_data)
    replay_config["target"]["artifact"] = str(
        materialized["alfworld_frozen_candidate"]
    )
    replay_config["target"]["qualification_report"] = str(replay_output)
    replay_config["development_evidence"]["qualification_report"] = str(
        materialized["alfworld_development_report"]
    )
    replay_config["portable_reproduction"] = {
        "kind": "RESOURCE_PATH_RETARGETED_FROZEN_REPLAY",
        "source_frozen_config": str(frozen_config_path.relative_to(REPO)),
        "source_frozen_config_sha256": _sha256(frozen_config_path),
        "release_manifest_sha256": _sha256(manifest_path),
        "reference_final_report": str(
            materialized["alfworld_final_report"]
        ),
        "scientific_parameters_changed": False,
        "machine_local_resource_paths_changed": True,
    }
    replay_config_path.write_text(
        json.dumps(replay_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runner = (REPO / manifest["alfworld_replay"]["runner"]).resolve()
    python = os.environ.get("ALFWORLD_PYTHON", "python")
    command = " ".join(map(shlex.quote, (
        python,
        str(runner),
        "--config",
        str(replay_config_path),
    )))
    return {
        "status": "PORTABLE_ALFWORLD_REPLAY_MATERIALIZED",
        "release_audit_sha256": audit["audit_sha256"],
        "output_dir": str(output_dir),
        "replay_config": str(replay_config_path),
        "reference_final_report": str(materialized["alfworld_final_report"]),
        "replay_output": str(replay_output),
        "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "configs/four_domain_neurosymbolic_release_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alfworld-config",
        type=Path,
        default=os.environ.get("ALFWORLD_CONFIG"),
    )
    parser.add_argument(
        "--alfworld-data",
        type=Path,
        default=os.environ.get("ALFWORLD_DATA"),
    )
    args = parser.parse_args()
    if args.alfworld_config is None or args.alfworld_data is None:
        raise SystemExit(
            "set ALFWORLD_CONFIG and ALFWORLD_DATA or pass both path arguments"
        )
    print(json.dumps(materialize(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        alfworld_config=args.alfworld_config,
        alfworld_data=args.alfworld_data,
    ), indent=2))


if __name__ == "__main__":
    main()
