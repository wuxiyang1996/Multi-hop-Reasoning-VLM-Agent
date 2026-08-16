#!/usr/bin/env python3
"""Freeze a fresh V3 cohort after the V2 candidate-validation failure."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    read_object,
    validate_manifest as validate_phase1_manifest,
)
from motif_transfer.phase2_webshop_utility_v1 import (  # noqa: E402
    FAILED_STATUS,
    SCHEMA,
    STATUS,
    file_sha256,
    validate_manifest as validate_phase2_manifest,
    validate_self_hash,
)
from motif_transfer.webshop_candidate_failclosed_v3 import (  # noqa: E402
    FALLBACK_SCHEMA,
    SAFE_FALLBACK_ACTION,
)
from motif_transfer.webshop_search_automaton_v16 import CONDITIONS  # noqa: E402
from motif_transfer.webshop_semantic_reserve import (  # noqa: E402
    require_semantic_reserve,
)
from scripts.freeze_phase2_webshop_utility_v2 import (  # noqa: E402
    _assert_identities_absent,
    _human_asins,
    _select_fresh_goals,
    _write_once,
)


OUTPUT = REPO / "configs/phase2_webshop_utility_v3/manifest.json"
PHASE1_MANIFEST = REPO / "configs/phase1_direct_prospective_v1/manifest.json"
PHASE1_AUDIT = REPO / "docs/results/phase1_direct_prospective_24_of_24_v4.json"
HISTORICAL_GOALS = REPO / "configs/webshop_synthetic_unique_v14_frozen.json"
V1_MANIFEST = REPO / "configs/phase2_webshop_utility_v1/manifest.json"
V2_MANIFEST = REPO / "configs/phase2_webshop_utility_v2/manifest.json"
V2_REPORT = REPO / "runs/phase2_webshop_utility_v2/report.json"
RUNTIME_DRYCHECK = REPO / "docs/results/phase2_webshop_runtime_v3_drycheck.json"
TARGET_GROUNDER = REPO / "docs/results/webshop_neural_symbolic_v9_frozen_grounder.json"
PRIOR_WEBSHOP_REPORT = REPO / "runs/webshop_search_automaton_v16_formal/report.json"
FORMAL_SIZE = 32
SEED_LABEL = "phase2-webshop-six-source-utility-v3-failclosed"


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Phase-2 V3 manifest already frozen: {OUTPUT}")

    phase1 = read_object(PHASE1_MANIFEST)
    validate_phase1_manifest(phase1, repo=REPO)
    phase1_audit = read_object(PHASE1_AUDIT)
    validate_self_hash(phase1_audit, "audit_sha256")
    if phase1_audit.get("status") != "DIRECT_PROSPECTIVE_24_OF_24_VALIDATED":
        raise RuntimeError("Phase-1 24/24 audit did not pass")

    historical_goals = read_object(HISTORICAL_GOALS)
    validate_self_hash(historical_goals, "artifact_sha256")
    v1_manifest = read_object(V1_MANIFEST)
    validate_phase2_manifest(v1_manifest, repo=REPO)
    v2_manifest = read_object(V2_MANIFEST)
    validate_phase2_manifest(v2_manifest, repo=REPO)
    v2_report = read_object(V2_REPORT)
    validate_self_hash(v2_report, "report_sha256")
    failed_gates = sorted(key for key, value in v2_report["gates"].items() if not value)
    if (
        v2_report.get("status") != FAILED_STATUS
        or v2_report.get("manifest_sha256") != v2_manifest.get("manifest_sha256")
        or failed_gates != ["all_receipts_complete"]
        or v2_report["summaries"]["authentic_search_automaton_plus_target"]["failures"] != 1
        or v2_report["summaries"]["target_native_search_ceiling"]["failures"] != 1
    ):
        raise RuntimeError("V2 did not fail solely at the diagnosed candidate-validation gate")

    runtime_drycheck = read_object(RUNTIME_DRYCHECK)
    validate_self_hash(runtime_drycheck, "drycheck_sha256")
    if runtime_drycheck.get("status") != "PHASE2_WEBSHOP_RUNTIME_V3_DRYCHECK_PASSED":
        raise RuntimeError("V3 runtime/fallback dry-check did not pass")
    if str(Path(sys.executable).resolve()) != runtime_drycheck["python_executable"]:
        raise RuntimeError("freezer is not running under the dry-checked interpreter")

    target_grounder = read_object(TARGET_GROUNDER)
    if not target_grounder.get("preflight_passed"):
        raise RuntimeError("target neural grounder did not pass")
    prior_report = read_object(PRIOR_WEBSHOP_REPORT)
    validate_self_hash(prior_report, "report_sha256")
    if prior_report.get("status") != "FRESH_FORMAL_TRANSFER_GATE_PASSED":
        raise RuntimeError("prior WebShop target adapter qualification did not pass")

    vendor_root = Path(
        "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/vendor/WebShop"
    )
    products_path = vendor_root / "data/items_shuffle_1000.json"
    human_path = vendor_root / "data/items_human_ins.json"
    engine_path = vendor_root / "web_agent_site/engine/engine.py"
    goal_path = vendor_root / "web_agent_site/engine/goal.py"
    app_path = vendor_root / "web_agent_site/app.py"
    for path in (products_path, human_path, engine_path, goal_path, app_path):
        if not path.is_file():
            raise RuntimeError(f"missing WebShop runtime file: {path}")
    sys.path.insert(0, str(vendor_root))
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals

    seed_hash = stable_hash({
        "seed_label": SEED_LABEL,
        "phase1_audit_sha256": phase1_audit["audit_sha256"],
        "v2_report_sha256": v2_report["report_sha256"],
        "fallback_schema": FALLBACK_SCHEMA,
    })
    goal_seed = int(seed_hash[:8], 16)
    random.seed(goal_seed)
    products, _, prices, _ = load_products(
        filepath=str(products_path), human_goals=False,
    )
    random.seed(goal_seed)
    goals = get_goals(products, prices, human_goals=False)
    random.seed(goal_seed)
    random.shuffle(goals)

    consumed = [
        dict(row)
        for role in ("development", "formal_reserve")
        for row in historical_goals["roles"][role]
    ]
    consumed.extend(
        dict(row["target_task"])
        for row in phase1["cells"] if row["target_domain"] == "webshop"
    )
    consumed.extend(dict(row) for row in v1_manifest["tasks"])
    consumed.extend(dict(row) for row in v2_manifest["tasks"])
    consumed.extend(
        {"asin": asin, "instruction_text": "excluded human goal"}
        for asin in sorted(_human_asins(products_path, human_path))
    )
    tasks = _select_fresh_goals(goals, consumed=consumed, count=FORMAL_SIZE)
    for row in tasks:
        row["target_identity"] = (
            row["target_identity"]
            .replace("phase2v2", "phase2v3")
            .format(goal_seed=goal_seed)
        )
    _assert_identities_absent(tasks)
    semantic_audit = require_semantic_reserve(
        tasks,
        consumed_rows=consumed,
        required_unique_goals=FORMAL_SIZE,
        require_asin_disjointness=True,
        require_unique_candidate_asins=True,
    )

    sources = {}
    for game in SOURCE_GAMES:
        source = phase1["sources"][game]
        sources[game] = {
            "artifact": source["artifact"],
            "artifact_sha256": source["artifact_sha256"],
            "artifact_file_sha256": source["artifact_file_sha256"],
        }
    for index, row in enumerate(tasks):
        game = SOURCE_GAMES[index % len(SOURCE_GAMES)]
        row.update({
            "source_game": game,
            "source_artifact": sources[game]["artifact"],
            "source_artifact_sha256": sources[game]["artifact_sha256"],
            "source_artifact_file_sha256": sources[game]["artifact_file_sha256"],
        })

    runtime_files = (
        "src/motif_transfer/phase2_webshop_utility_v1.py",
        "src/motif_transfer/webshop_candidate_failclosed_v3.py",
        "src/motif_transfer/webshop_search_automaton_v16.py",
        "src/motif_transfer/webshop_coverage_transfer_v14.py",
        "src/motif_transfer/webshop_constraint_coverage_v14.py",
        "src/motif_transfer/webshop_unique_goal_server_v14.py",
        "scripts/check_phase2_webshop_runtime_v3.py",
        "scripts/freeze_phase2_webshop_utility_v2.py",
        "scripts/freeze_phase2_webshop_utility_v3.py",
        "scripts/run_phase2_webshop_utility_v1.py",
        "scripts/run_phase2_webshop_utility_v3.py",
        "scripts/run_webshop_direct_server_v1.py",
        "scripts/run_webshop_search_automaton_v16.py",
        "scripts/run_webshop_neural_symbolic_v9.py",
        "scripts/run_webshop_transfer_qualification_v5.py",
    )
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "attempt_version": "v3_after_v2_candidate_validation_failure",
        "claim_boundary": (
            "One-shot aggregate causal utility test of the common policy supported by "
            "six independent game lineages on 32 new, product-disjoint WebShop goals. "
            "Per-game task counts are replication strata, not powered per-game studies."
        ),
        "selection_read_target_outcome": False,
        "historical_target_outcome_reuse_allowed": False,
        "source_assignment_read_target_semantics": False,
        "source_assignment_rule": "SOURCE_GAMES manifest order round-robin before V3 outcomes",
        "candidate_generation_failure_policy": {
            "schema_version": FALLBACK_SCHEMA,
            "trigger": "all frozen model schema retries fail target-native action validation",
            "action": SAFE_FALLBACK_ACTION,
            "task_or_goal_information_used": False,
            "source_information_used": False,
            "condition_information_used": False,
            "transport_errors_caught": False,
            "rationale": "Convert malformed model output into an audited safe policy outcome, not a runtime failure.",
        },
        "seed_derivation": {
            "label": SEED_LABEL,
            "phase1_audit_sha256": phase1_audit["audit_sha256"],
            "v2_report_sha256": v2_report["report_sha256"],
            "fallback_schema": FALLBACK_SCHEMA,
            "seed_hash": seed_hash,
            "rule": "int(first eight hex digits, base16)",
        },
        "goal_seed": goal_seed,
        "server_goal_count": len(goals),
        "number_of_registered_tasks_required": 1 + max(
            int(row["server_goal_index"]) for row in tasks
        ),
        "prior_consumed_goal_manifest": str(V2_MANIFEST.relative_to(REPO)),
        "prior_consumed_goal_manifest_file_sha256": file_sha256(V2_MANIFEST),
        "earlier_consumed_goal_manifest": str(V1_MANIFEST.relative_to(REPO)),
        "earlier_consumed_goal_manifest_file_sha256": file_sha256(V1_MANIFEST),
        "historical_consumed_goal_manifest": str(HISTORICAL_GOALS.relative_to(REPO)),
        "historical_consumed_goal_manifest_file_sha256": file_sha256(HISTORICAL_GOALS),
        "predecessor_report": str(V2_REPORT.relative_to(REPO)),
        "predecessor_report_sha256": v2_report["report_sha256"],
        "predecessor_report_file_sha256": file_sha256(V2_REPORT),
        "predecessor_failed_gates": failed_gates,
        "runtime_drycheck": str(RUNTIME_DRYCHECK.relative_to(REPO)),
        "runtime_drycheck_sha256": runtime_drycheck["drycheck_sha256"],
        "runtime_drycheck_file_sha256": file_sha256(RUNTIME_DRYCHECK),
        "execution_interpreter": runtime_drycheck["python_executable"],
        "prior_target_adapter_qualification": str(PRIOR_WEBSHOP_REPORT.relative_to(REPO)),
        "prior_target_adapter_qualification_file_sha256": file_sha256(PRIOR_WEBSHOP_REPORT),
        "parent_phase1_manifest": str(PHASE1_MANIFEST.relative_to(REPO)),
        "parent_phase1_manifest_sha256": phase1["manifest_sha256"],
        "parent_phase1_manifest_file_sha256": file_sha256(PHASE1_MANIFEST),
        "parent_phase1_audit": str(PHASE1_AUDIT.relative_to(REPO)),
        "parent_phase1_audit_sha256": phase1_audit["audit_sha256"],
        "semantic_independence_audit": semantic_audit,
        "sources": sources,
        "target_grounder": str(TARGET_GROUNDER.relative_to(REPO)),
        "target_grounder_file_sha256": file_sha256(TARGET_GROUNDER),
        "conditions": list(CONDITIONS),
        "tasks": tasks,
        "parameters": {
            "model": "openai/gpt-4.1-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "maximum_output_tokens": 1200,
            "maximum_steps": 12,
            "candidate_count": 5,
            "schema_retries": 3,
            "timeout_seconds": 180,
        },
        "one_shot_execution_rule": (
            "Each target-condition cell may be reset/executed once. Existing valid "
            "receipts may be resumed; a start marker without a receipt is consumed "
            "incomplete and must not be rerun."
        ),
        "success_gates": {
            "exact_32x5_receipt_matrix": True,
            "all_receipt_hashes_valid": True,
            "all_receipts_complete": True,
            "matched_initial_state_hashes": True,
            "all_six_source_lineages_exercised": True,
            "all_three_symbolic_actions_exercised": True,
            "zero_authentic_unsafe_commits": True,
            "authentic_strict_success_gain_over_raw": True,
            "authentic_vs_raw_significant": "paired exact two-sided p <= .05",
            "zero_strict_negative_transfer_vs_raw": True,
            "authentic_pass_success_not_below_raw": True,
            "authentic_mean_reward_not_below_raw": True,
            "authentic_reward_pairing_net_nonnegative": True,
            "authentic_significantly_beats_event_permuted": True,
            "authentic_significantly_beats_ledger_blind": True,
            "authentic_matches_target_native_ceiling_exactly": True,
            "every_lineage_has_zero_strict_losses_vs_raw": True,
        },
        "runtime_file_sha256": {
            relative: file_sha256(REPO / relative) for relative in runtime_files
        },
        "vendor_runtime_file_sha256": {
            str(products_path): file_sha256(products_path),
            str(human_path): file_sha256(human_path),
            str(engine_path): file_sha256(engine_path),
            str(goal_path): file_sha256(goal_path),
            str(app_path): file_sha256(app_path),
        },
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write_once(OUTPUT, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "attempt_version": manifest["attempt_version"],
        "goal_seed": goal_seed,
        "tasks": len(tasks),
        "source_counts": {
            game: sum(row["source_game"] == game for row in tasks)
            for game in SOURCE_GAMES
        },
        "semantic_independence_passed": semantic_audit["passed"],
        "manifest_sha256": manifest["manifest_sha256"],
        "output": str(OUTPUT.relative_to(REPO)),
    }, indent=2))


if __name__ == "__main__":
    main()
