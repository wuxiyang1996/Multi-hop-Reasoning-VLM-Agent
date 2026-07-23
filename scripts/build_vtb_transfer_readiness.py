#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import Lifecycle  # noqa: E402
from motif_transfer.transfer_matrix import (  # noqa: E402
    REQUIRED_TARGET_CONDITIONS,
    TransferExperimentSpec,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fail-closed VTB motif-transfer readiness receipt.")
    parser.add_argument("--config", type=Path, default=REPO / "configs/vtb_motif_transfer_v2.json")
    parser.add_argument("--manifest", type=Path, default=REPO / "configs/vtb_single_turn_manifest_v2.json")
    parser.add_argument("--runtime-audit", type=Path,
                        default=REPO / "docs/results/vtb_official_runtime_audit_v2.json")
    parser.add_argument("--source-diagnosis", type=Path,
                        default=REPO / "docs/results/source_gate_failure_diagnosis_v1.json")
    parser.add_argument("--output", type=Path,
                        default=REPO / "docs/results/vtb_transfer_readiness_v2.json")
    args = parser.parse_args()

    config = _read(args.config)
    manifest = _read(args.manifest)
    runtime = _read(args.runtime_audit)["audit"]
    source = _read(args.source_diagnosis)["failure_tree"]
    conditions = tuple(config.get("conditions") or [])
    source_candidate = config["source_gate"].get("current_source_candidate")
    other_candidate = config["source_gate"].get("current_other_game_control")
    source_supported = bool(
        source_candidate
        and config["source_gate"].get("current_status") == "SOURCE_SUPPORTED"
        and source.get("explicit_reasoning_motif") == "SOURCE_SUPPORTED"
    )
    lifecycle = Lifecycle.SOURCE_SUPPORTED if source_supported else Lifecycle.CANDIDATE
    models = config.get("model_identities") or {}
    spec = TransferExperimentSpec(
        experiment_id=str(config["experiment_id"]),
        target_cell=str(config["target_cell"]),
        target_manifest_sha256=_sha(args.manifest),
        source_candidate_id=str((source_candidate or {}).get("candidate_id") or ""),
        source_candidate_sha256=str((source_candidate or {}).get("sha256") or ""),
        source_lifecycle=lifecycle,
        decision_model=str(models.get("decision_agent") or ""),
        harness_model=str(models.get("motif_harness_agent") or ""),
        judge_model=str(models.get("judge") or ""),
        tool_contract_sha256=str(runtime.get("tool_contract_sha256") or ""),
        action_or_tool_budget=int(config["official_protocol"]["tool_call_cap"]),
        conditions=conditions,
    )
    # Diagnostic construction itself must still satisfy the frozen matrix schema.
    spec.validate(diagnostic_only=True)
    blockers = []
    if conditions != REQUIRED_TARGET_CONDITIONS:
        blockers.append("six-condition matrix changed")
    if manifest.get("official_commit") != config["official_protocol"].get("commit"):
        blockers.append("manifest/config official commit mismatch")
    if not runtime.get("paper_faithful_full_tool_ready"):
        blockers.append("VTB official full-tool runtime is not ready")
    if not source_supported:
        blockers.append("no frozen SOURCE_SUPPORTED authentic game motif")
    if not other_candidate:
        blockers.append("no frozen SOURCE_SUPPORTED other-game control motif")
    if not all((spec.decision_model, spec.harness_model, spec.judge_model)):
        blockers.append("one or more model identities are unset")

    confirmatory_ready = not blockers and spec.confirmatory_ready
    payload = {
        "schema_version": 2,
        "status": "CONFIRMATORY_READY" if confirmatory_ready else "BLOCKED",
        "confirmatory_ready": confirmatory_ready,
        "diagnostic_mode_authorized": True,
        "diagnostic_claim_label": "UNQUALIFIED_SOURCE_DIAGNOSTIC",
        "config_sha256": _sha(args.config),
        "manifest_sha256": _sha(args.manifest),
        "runtime_audit_sha256": _sha(args.runtime_audit),
        "source_diagnosis_sha256": _sha(args.source_diagnosis),
        "source_failure_class": source.get("current_class"),
        "source_transfer_authority": source.get("far_domain_transfer"),
        "official_full_tool_ready": runtime.get("paper_faithful_full_tool_ready"),
        "online_interposition_runner_implemented": (
            REPO / "scripts/run_vtb_interposed_single_turn.py"
        ).is_file(),
        "source_treatment_compiler_implemented": (
            REPO / "scripts/compile_vtb_treatments.py"
        ).is_file(),
        "exact_request_common_randomness_implemented": (
            REPO / "src/motif_transfer/exact_request_cache.py"
        ).is_file(),
        "matched_adaptation_smoke": "docs/results/vtb_interposition_matched_smoke_summary_v2.json",
        "blockers": blockers,
        "experiment_spec": spec.to_json(),
        "interpretation": (
            "The apparatus can falsify structural motif transfer once both gates pass. "
            "Until then, target runs test plumbing or unqualified hypotheses, not game-to-target transfer."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "confirmatory_ready": confirmatory_ready,
        "blockers": blockers,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
