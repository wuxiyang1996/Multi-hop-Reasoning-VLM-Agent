#!/usr/bin/env python3
"""Verify preregistered game→ALFWorld bindings against one real demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.skill_admission import (  # noqa: E402
    BindingCandidate,
    FrozenAdmissionStore,
    StrictOneShotAdmission,
    target_demo_receipt_from_dict,
)
from skill_bank.program_ir import canonical_program_from_dict  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument(
        "--programs", type=Path,
        default=REPO_ROOT / "artifacts/source_evidence_index/source_programs.jsonl",
    )
    parser.add_argument(
        "--bindings", type=Path,
        default=REPO_ROOT / "configs/alfworld_one_shot_bindings.json",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "artifacts/admission/alfworld",
    )
    args = parser.parse_args()

    demo = target_demo_receipt_from_dict(json.loads(args.demo.read_text(encoding="utf-8")))
    config = json.loads(args.bindings.read_text(encoding="utf-8"))
    programs = []
    with args.programs.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                programs.append(canonical_program_from_dict(json.loads(line)))
    by_identity = {(program.source_games[0], program.name): program for program in programs}
    verifier = StrictOneShotAdmission()
    store = FrozenAdmissionStore(args.output_root)
    rows = []
    for spec in config.get("bindings", []):
        identity = (str(spec["source_game"]), str(spec["source_skill_name"]))
        program = by_identity.get(identity)
        if program is None:
            raise SystemExit(f"source program not found: {identity}")
        candidate = BindingCandidate(
            candidate_id=str(spec["candidate_id"]),
            source_program_id=program.program_id,
            source_program_hash=program.content_hash(),
            source_step_id=str(spec["source_step_id"]),
            target_domain=str(config["target_domain"]),
            task_family=str(config["task_family"]),
            target_operator=str(spec["target_operator"]),
            argument_types=dict(spec.get("argument_types") or {}),
            source_effect=str(spec["source_effect"]),
            proposal_source=str(config.get("proposal_policy") or "untrusted"),
        )
        artifact = verifier.admit(program=program, candidates=[candidate], demo=demo)
        path = store.freeze(artifact)
        rows.append({
            "candidate_id": candidate.candidate_id,
            "program_id": program.program_id,
            "status": artifact.status.value,
            "artifact_hash": artifact.artifact_hash,
            # The manifest lives beside its immutable artifacts.  Store a
            # relative path so a frozen experiment can run from a clean
            # worktree or archive instead of silently reaching into the
            # directory where admission happened.
            "artifact_path": path.name,
            "failure_codes": artifact.failure_codes,
        })
    manifest = {
        "schema_version": 1,
        "demo_id": demo.demo_id,
        "demo_hash": demo.content_hash(),
        "one_shot": True,
        "target_gradient_updates": 0,
        "bindings": rows,
    }
    binding_set_hash = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest["binding_set_hash"] = binding_set_hash
    manifest_path = args.output_root / (
        f"manifest-{demo.content_hash()}-{binding_set_hash[:16]}.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0 if all(row["status"] in {"ADMITTED", "CONDITIONAL"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
