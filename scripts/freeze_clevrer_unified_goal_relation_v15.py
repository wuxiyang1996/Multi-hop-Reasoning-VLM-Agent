#!/usr/bin/env python3
"""Freeze the untouched CLEVRER V15 reserve after development passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.clevrer_unified_goal_relation import (  # noqa: E402
    TARGET_INTERFACE,
    source_goal_relation_envelope,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.video_proof_grounder import (  # noqa: E402
    validate_v14_artifact,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _reserve_exposure(sample_ids: list[str]) -> list[str]:
    command = ["rg", "-l", "-F"]
    for sample_id in sample_ids:
        command.extend(("-e", sample_id))
    command.extend((str(REPO / "runs"), str(REPO / "docs/results")))
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "reserve exposure scan failed")
    return sorted(filter(None, result.stdout.splitlines()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-report", type=Path,
        default=REPO / "docs/results/clevrer_unified_goal_relation_v15_development.json",
    )
    parser.add_argument(
        "--source-artifact", type=Path,
        default=REPO / "runs/sokoban_goal_relation_macro_v3/artifact.json",
    )
    parser.add_argument(
        "--source-confirmation", type=Path,
        default=REPO / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json",
    )
    parser.add_argument(
        "--grounder-artifact", type=Path,
        default=REPO / "runs/clevrer_sokoban_proof_v14_training/frozen_proof_grounder.json",
    )
    parser.add_argument(
        "--grounder-training-report", type=Path,
        default=REPO / "runs/clevrer_sokoban_proof_v14_training/training_report.json",
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=REPO / "configs/clevrer_sokoban_proof_v14_splits.json",
    )
    parser.add_argument(
        "--official-root", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/datasets/CLEVRER-official"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "configs/clevrer_unified_goal_relation_v15_reserve.json",
    )
    args = parser.parse_args()

    development = _read(args.development_report)
    development_body = dict(development)
    claimed = development_body.pop("report_sha256", None)
    if claimed != stable_hash(development_body):
        raise ValueError("V15 development report self-hash mismatch")
    if development.get("status") != "CLEVRER_UNIFIED_V15_DEVELOPMENT_GATE_PASSED":
        raise ValueError("cannot freeze V15 reserve after a failed development gate")

    source = _read(args.source_artifact)
    confirmation = _read(args.source_confirmation)
    inducer = REPO / "src/motif_transfer/source_goal_relation_induction.py"
    adapter = REPO / "src/motif_transfer/clevrer_unified_goal_relation.py"
    runner = REPO / "scripts/run_clevrer_unified_goal_relation_v15.py"
    envelope = source_goal_relation_envelope(
        source, confirmation, inducer_artifact_sha256=_sha(inducer),
    )
    if envelope.envelope_sha256 != development["source_program"]["envelope_sha256"]:
        raise ValueError("development/source envelope mismatch")

    grounder = _read(args.grounder_artifact)
    _, _, _, threshold = validate_v14_artifact(grounder)
    training = _read(args.grounder_training_report)
    if training.get("status") != "V14_PROOF_GROUNDER_DEVELOPMENT_GATE_PASSED":
        raise ValueError("frozen target grounder lacks a passed training report")
    manifest = _read(args.split_manifest)
    reserve = list(manifest["benchmarks"]["clevrer"]["splits"]["reserve"])
    if len(reserve) != 360 or len(set(reserve)) != 360:
        raise ValueError("expected 360 unique frozen reserve IDs")
    exposure = _reserve_exposure(reserve)
    if exposure:
        raise ValueError(f"reserve outcome exposure detected: {exposure}")

    lineage_paths = {
        "development_report": _relative(args.development_report),
        "source_artifact": _relative(args.source_artifact),
        "source_confirmation": _relative(args.source_confirmation),
        "source_inducer": _relative(inducer),
        "adapter": _relative(adapter),
        "runner": _relative(runner),
        "v14_helper_runner": "scripts/run_clevrer_sokoban_proof_v14.py",
        "split_manifest": _relative(args.split_manifest),
        "grounder_artifact": _relative(args.grounder_artifact),
        "grounder_training_report": _relative(args.grounder_training_report),
        "proof_grounder_module": "src/motif_transfer/video_proof_grounder.py",
        "proof_receipts_module": "src/motif_transfer/clevrer_proof_receipts.py",
        "query_compiler_module": "src/motif_transfer/clevrer_query_compiler.py",
        "base_feature_module": "src/motif_transfer/video_recovery_cate.py",
        "unified_harness_module": "src/motif_transfer/unified_neurosymbolic_harness.py",
        "unified_runtime_module": "src/motif_transfer/unified_transfer_runtime.py",
    }
    frozen_lineage = {
        key: _sha(REPO / path) for key, path in lineage_paths.items()
    }
    paired = development["paired_authentic"]
    body = {
        "schema_version": "clevrer-unified-goal-relation-v15-reserve-config",
        "status": "FROZEN_BEFORE_CLEVRER_V15_RESERVE_OUTCOMES",
        "claim_boundary": (
            "Prospective reserve test of a template-free recurrent relation "
            "program induced only from Sokoban state/action/effect/next_state "
            "tuples, compiled through the unified fail-closed harness into a "
            "CLEVRER-native paired event-graph proof grounder and native "
            "representation switch. Supports this fixed synthetic CLEVRER "
            "setup only; does not establish natural-video transfer or prove "
            "source provenance is necessary versus an extensionally identical "
            "target-written controller."
        ),
        "source": {
            "artifact": _relative(args.source_artifact),
            "confirmation": _relative(args.source_confirmation),
            "artifact_sha256": source["artifact_sha256"],
            "confirmation_sha256": confirmation["report_sha256"],
            "contract_sha256": envelope.contract.contract_sha256,
            "envelope_sha256": envelope.envelope_sha256,
            "named_policy_template_used": False,
            "target_data_read": False,
        },
        "development": {
            "report": _relative(args.development_report),
            "report_sha256": development["report_sha256"],
            "calibration": {
                "utility_vs_neural": paired["target_explicit_no_recovery"],
                "authenticity_vs_source_permuted": paired[
                    "permuted_uplift_cate_recover"
                ],
            },
            "target_base_nonconfirmatory_warning": (
                "Development authentic-vs-target-base was 15W/10L and not "
                "significant; reserve requires strictly positive matched net "
                "wins and cannot retune the frozen threshold."
            ),
        },
        "grounder": {
            "artifact": _relative(args.grounder_artifact),
            "training_report": _relative(args.grounder_training_report),
            "artifact_sha256": grounder["artifact_sha256"],
            "decision_threshold": float(threshold),
            "runtime_gold_or_official_program_read": False,
        },
        "target": {
            "interface": TARGET_INTERFACE,
            "split_manifest": _relative(args.split_manifest),
            "official_root": str(args.official_root.resolve()),
            "role": "reserve",
            "reserve_ids_in_config": False,
            "reserve_id_count": len(reserve),
            "prior_result_exposure_matches": exposure,
            "explicit_relation_prediction_directory": "with_edge_supervision_old",
            "trajectory_prediction_directory": "without_edge_supervision",
        },
        "conditions": [
            "neural_only_explicit_relation",
            "authentic_source_induced_goal_relation",
            "target_base_receipt_recovery",
            "generic_error_scaffold",
            "source_permuted_uplift",
            "shuffled_proof_binding",
            "source_inverted_effect",
            "target_trajectory_only",
            "target_native_representation_ceiling",
        ],
        "gates": {
            "expected_samples": 360,
            "minimum_authentic_recoveries": 20,
            "causal_control_conditions": [
                "neural_only_explicit_relation",
                "target_base_receipt_recovery",
                "generic_error_scaffold",
                "source_permuted_uplift",
                "shuffled_proof_binding",
                "source_inverted_effect",
            ],
            "minimum_utility_net_wins": 5,
            "maximum_utility_exact_p": 0.05,
            "minimum_authenticity_net_wins": 5,
            "maximum_authenticity_exact_p": 0.05,
            "threshold_retuning_after_reserve": False,
            "reserve_failure_stops_claim": True,
        },
        "lineage_paths": lineage_paths,
        "frozen_lineage": frozen_lineage,
        "cost_budget": {
            "external_provider_calls_allowed": 0,
            "external_provider_cost_usd_allowed": 0.0,
            "use_local_official_predictions_only": True,
        },
    }
    config = body | {"config_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": config["status"], "reserve_ids": len(reserve),
        "exposure_matches": exposure, "decision_threshold": threshold,
        "cost_budget": config["cost_budget"], "config_sha256": config["config_sha256"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
