#!/usr/bin/env python3
"""Freeze the WebShop V17 one-shot formal protocol after qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from motif_transfer.webshop_structural_transfer_v17 import CONDITIONS  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / (
        "configs/webshop_structural_v17_frozen.json"
    ))
    parser.add_argument("--qualification-report", type=Path, default=REPO / (
        "runs/webshop_structural_transfer_v17_qualification/report.json"
    ))
    parser.add_argument("--source-artifact", type=Path, default=REPO / (
        "runs/sokoban_relational_structural_v2/artifact.json"
    ))
    parser.add_argument("--source-confirmation", type=Path, default=REPO / (
        "runs/sokoban_relational_structural_v2/fresh_confirmation_report.json"
    ))
    parser.add_argument("--target-function", type=Path, default=REPO / (
        "runs/webshop_structural_transfer_v17_development/target_function.json"
    ))
    parser.add_argument("--target-grounder", type=Path, default=REPO / (
        "runs/webshop_structural_transfer_v17_development/low_sample_grounder.json"
    ))
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--maximum-output-tokens", type=int, default=1200)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--maximum-steps", type=int, default=12)
    parser.add_argument("--output", type=Path, default=REPO / (
        "configs/webshop_structural_v17_formal_protocol.json"
    ))
    args = parser.parse_args()

    manifest = _read(args.manifest)
    qualification = _read(args.qualification_report)
    schema_match = re.fullmatch(
        r"webshop-structural-v(\d+)-reserve-v1",
        str(manifest.get("schema_version")),
    )
    if schema_match is None:
        raise SystemExit("unsupported structural reserve schema")
    reserve_version = f"V{schema_match.group(1)}"
    version_lower = reserve_version.lower()
    if manifest.get("status") != (
        f"FROZEN_BEFORE_ANY_{reserve_version}_PROVIDER_CALL_OR_OUTCOME"
    ):
        raise SystemExit(f"{reserve_version} reserve is not frozen")
    if manifest.get("formal_outcomes_read_or_run") is not False:
        raise SystemExit("V17 formal reserve is no longer sealed")
    if qualification.get("status") != (
        f"{reserve_version}_TRANSPORT_QUALIFICATION_PASSED"
    ):
        raise SystemExit(f"{reserve_version} qualification did not pass")
    if not all((qualification.get("gates") or {}).values()):
        raise SystemExit("V17 qualification gates are incomplete")
    qualification_parameters = {
        "model": args.model,
        "maximum_steps": args.maximum_steps,
        "maximum_output_tokens": args.maximum_output_tokens,
        "candidate_count": args.candidate_count,
        "schema_retries": 3,
    }
    mismatches = {
        key: {"qualification": qualification.get(key), "formal": value}
        for key, value in qualification_parameters.items()
        if qualification.get(key) != value
    }
    if mismatches:
        raise SystemExit(
            f"qualification/formal execution parameter mismatch: {mismatches}"
        )
    tasks = [row["task_id"] for row in manifest["roles"]["formal_reserve"]]
    body = {
        "schema_version": (
            f"webshop-source-structural-{version_lower}-formal-protocol-v1"
        ),
        "status": f"FROZEN_BEFORE_{reserve_version}_FORMAL_EXECUTION",
        "claim_boundary": (
            "One-shot execution on 32 native synthetic option-relation goals "
            f"that are ASIN/semantics-disjoint from {reserve_version} "
            "qualification, every manifest-declared prior reserve, and the "
            "human-goal pool."
        ),
        "tasks": tasks,
        "conditions": list(CONDITIONS),
        "model": args.model,
        "maximum_steps": args.maximum_steps,
        "maximum_output_tokens": args.maximum_output_tokens,
        "candidate_count": args.candidate_count,
        "schema_retries": 3,
        "manifest_file_sha256": file_sha256(args.manifest),
        "source_artifact_file_sha256": file_sha256(args.source_artifact),
        "source_confirmation_file_sha256": file_sha256(args.source_confirmation),
        "target_function_file_sha256": file_sha256(args.target_function),
        "target_grounder_file_sha256": file_sha256(args.target_grounder),
        "qualification_report_file_sha256": file_sha256(args.qualification_report),
        "controller_file_sha256": file_sha256(REPO / (
            "src/motif_transfer/webshop_structural_transfer_v17.py"
        )),
        "runner_file_sha256": file_sha256(REPO / (
            "scripts/run_webshop_structural_transfer_v17.py"
        )),
        "success_gates": {
            "all_receipts_complete": True,
            "matched_initial_state_hashes": True,
            "authentic_source_admitted_every_episode": True,
            "controls_never_receive_source_authority": True,
            "zero_source_authorized_unsafe_commits": True,
            "strict_success_gain_over_neural": True,
            "zero_strict_success_losses_vs_neural": True,
            "pass_success_not_below_neural": True,
            "mean_reward_not_below_neural": True,
            "paired_reward_net_not_negative": True,
            "strictly_beats_terminal_permutation_and_generic_scaffold": True,
            "matches_source_free_target_native_ceiling": True,
            "source_vs_neural_two_sided_exact_sign_p_at_most": 0.05,
            "source_vs_permuted_two_sided_exact_sign_p_at_most": 0.05,
        },
        "decision_rule": (
            "Run every task-condition cell once. Do not repair, replace, or "
            "rerun a formal receipt after any formal outcome is observed."
        ),
        "qualification_report_sha256": qualification["report_sha256"],
        "formal_outcomes_read_or_run": False,
    }
    artifact = body | {"protocol_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": artifact["status"], "tasks": len(tasks),
        "qualification_report_sha256": qualification["report_sha256"],
        "protocol_sha256": artifact["protocol_sha256"],
        "formal_outcomes_read_or_run": False,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
