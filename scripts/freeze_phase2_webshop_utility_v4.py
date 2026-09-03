#!/usr/bin/env python3
"""Freeze fresh V4 goals after proving and removing server initialization races."""

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
    file_sha256,
    validate_manifest as validate_phase2_manifest,
    validate_self_hash,
)
from motif_transfer.webshop_candidate_failclosed_v3 import (  # noqa: E402
    FALLBACK_SCHEMA,
)
from motif_transfer.webshop_semantic_reserve import (  # noqa: E402
    require_semantic_reserve,
)
from scripts.freeze_phase2_webshop_utility_v2 import (  # noqa: E402
    _assert_identities_absent,
    _human_asins,
    _select_fresh_goals,
    _write_once,
)


OUTPUT = REPO / "configs/phase2_webshop_utility_v4/manifest.json"
PHASE1_MANIFEST = REPO / "configs/phase1_direct_prospective_v1/manifest.json"
PHASE1_AUDIT = REPO / "docs/results/phase1_direct_prospective_24_of_24_v4.json"
HISTORICAL_GOALS = REPO / "configs/webshop_synthetic_unique_v14_frozen.json"
V1_MANIFEST = REPO / "configs/phase2_webshop_utility_v1/manifest.json"
V2_MANIFEST = REPO / "configs/phase2_webshop_utility_v2/manifest.json"
V3_MANIFEST = REPO / "configs/phase2_webshop_utility_v3/manifest.json"
V3_FAILED_PREFLIGHT = REPO / "docs/results/phase2_webshop_utility_v3_preflight.json"
V3_SINGLETHREAD_DIAGNOSTIC = (
    REPO / "docs/results/phase2_webshop_utility_v3_singlethread_diagnostic.json"
)
SEED_LABEL = "phase2-webshop-six-source-utility-v4-singlethread-failclosed"
FORMAL_SIZE = 32


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Phase-2 V4 manifest already frozen: {OUTPUT}")
    if (REPO / "runs/phase2_webshop_utility_v3/started").exists():
        raise RuntimeError("V3 formal execution unexpectedly started")

    phase1 = read_object(PHASE1_MANIFEST)
    validate_phase1_manifest(phase1, repo=REPO)
    phase1_audit = read_object(PHASE1_AUDIT)
    validate_self_hash(phase1_audit, "audit_sha256")
    historical = read_object(HISTORICAL_GOALS)
    validate_self_hash(historical, "artifact_sha256")
    manifests = {
        "v1": read_object(V1_MANIFEST),
        "v2": read_object(V2_MANIFEST),
        "v3": read_object(V3_MANIFEST),
    }
    for manifest in manifests.values():
        validate_phase2_manifest(manifest, repo=REPO)

    failed_preflight = read_object(V3_FAILED_PREFLIGHT)
    validate_self_hash(failed_preflight, "preflight_sha256")
    diagnostic = read_object(V3_SINGLETHREAD_DIAGNOSTIC)
    validate_self_hash(diagnostic, "preflight_sha256")
    if (
        failed_preflight.get("status") != "PHASE2_WEBSHOP_LIVE_PREFLIGHT_FAILED"
        or failed_preflight.get("manifest_sha256") != manifests["v3"]["manifest_sha256"]
        or failed_preflight.get("gates", {}).get("zero_actions") is not True
        or failed_preflight.get("gates", {}).get("zero_provider_calls") is not True
        or failed_preflight.get("gates", {}).get("zero_outcomes_read") is not True
        or diagnostic.get("status") != "PHASE2_WEBSHOP_SINGLETHREAD_PREFLIGHT_PASSED"
        or diagnostic.get("manifest_sha256") != manifests["v3"]["manifest_sha256"]
        or diagnostic.get("mismatches") != []
    ):
        raise RuntimeError("V3 did not exhibit the diagnosed threaded initialization race")

    runtime_drycheck = read_object(REPO / manifests["v3"]["runtime_drycheck"])
    validate_self_hash(runtime_drycheck, "drycheck_sha256")
    if runtime_drycheck.get("status") != "PHASE2_WEBSHOP_RUNTIME_V3_DRYCHECK_PASSED":
        raise RuntimeError("candidate fail-closed runtime was not qualified")
    if str(Path(sys.executable).resolve()) != runtime_drycheck["python_executable"]:
        raise RuntimeError("freezer is not running under the dry-checked interpreter")

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
        "v3_failed_preflight_sha256": failed_preflight["preflight_sha256"],
        "v3_singlethread_diagnostic_sha256": diagnostic["preflight_sha256"],
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
        for row in historical["roles"][role]
    ]
    consumed.extend(
        dict(row["target_task"])
        for row in phase1["cells"] if row["target_domain"] == "webshop"
    )
    for manifest in manifests.values():
        consumed.extend(dict(row) for row in manifest["tasks"])
    consumed.extend(
        {"asin": asin, "instruction_text": "excluded human goal"}
        for asin in sorted(_human_asins(products_path, human_path))
    )
    tasks = _select_fresh_goals(goals, consumed=consumed, count=FORMAL_SIZE)
    for row in tasks:
        row["target_identity"] = (
            row["target_identity"]
            .replace("phase2v2", "phase2v4")
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

    for index, row in enumerate(tasks):
        game = SOURCE_GAMES[index % len(SOURCE_GAMES)]
        source = manifests["v3"]["sources"][game]
        row.update({
            "source_game": game,
            "source_artifact": source["artifact"],
            "source_artifact_sha256": source["artifact_sha256"],
            "source_artifact_file_sha256": source["artifact_file_sha256"],
        })

    runtime_files = (
        "src/motif_transfer/phase2_webshop_utility_v1.py",
        "src/motif_transfer/webshop_candidate_failclosed_v3.py",
        "src/motif_transfer/webshop_search_automaton_v16.py",
        "src/motif_transfer/webshop_coverage_transfer_v14.py",
        "src/motif_transfer/webshop_constraint_coverage_v14.py",
        "src/motif_transfer/webshop_unique_goal_server_v14.py",
        "scripts/freeze_phase2_webshop_utility_v2.py",
        "scripts/freeze_phase2_webshop_utility_v4.py",
        "scripts/run_phase2_webshop_utility_v1.py",
        "scripts/run_phase2_webshop_utility_v3.py",
        "scripts/run_phase2_webshop_utility_v4.py",
        "scripts/run_webshop_direct_server_v4.py",
        "scripts/run_webshop_search_automaton_v16.py",
        "scripts/run_webshop_neural_symbolic_v9.py",
        "scripts/run_webshop_transfer_qualification_v5.py",
        "scripts/verify_phase2_webshop_utility_v4.py",
    )
    body = dict(manifests["v3"])
    body.pop("manifest_sha256", None)
    body.update({
        "attempt_version": "v4_singlethread_server_and_failclosed_candidates",
        "selection_read_target_outcome": False,
        "historical_target_outcome_reuse_allowed": False,
        "source_assignment_read_target_semantics": False,
        "source_assignment_rule": "SOURCE_GAMES manifest order round-robin before V4 outcomes",
        "server_concurrency_policy": {
            "threaded": False,
            "reason": "Eliminate lazy goal-initialization races proven by V3 paired preflights.",
        },
        "seed_derivation": {
            "label": SEED_LABEL,
            "phase1_audit_sha256": phase1_audit["audit_sha256"],
            "v3_failed_preflight_sha256": failed_preflight["preflight_sha256"],
            "v3_singlethread_diagnostic_sha256": diagnostic["preflight_sha256"],
            "fallback_schema": FALLBACK_SCHEMA,
            "seed_hash": seed_hash,
            "rule": "int(first eight hex digits, base16)",
        },
        "goal_seed": goal_seed,
        "server_goal_count": len(goals),
        "number_of_registered_tasks_required": 1 + max(
            int(row["server_goal_index"]) for row in tasks
        ),
        "prior_consumed_goal_manifest": str(V3_MANIFEST.relative_to(REPO)),
        "prior_consumed_goal_manifest_file_sha256": file_sha256(V3_MANIFEST),
        "earlier_consumed_goal_manifest": str(V2_MANIFEST.relative_to(REPO)),
        "earlier_consumed_goal_manifest_file_sha256": file_sha256(V2_MANIFEST),
        "predecessor_failed_preflight": str(V3_FAILED_PREFLIGHT.relative_to(REPO)),
        "predecessor_failed_preflight_sha256": failed_preflight["preflight_sha256"],
        "predecessor_failed_preflight_file_sha256": file_sha256(V3_FAILED_PREFLIGHT),
        "singlethread_diagnostic": str(V3_SINGLETHREAD_DIAGNOSTIC.relative_to(REPO)),
        "singlethread_diagnostic_sha256": diagnostic["preflight_sha256"],
        "singlethread_diagnostic_file_sha256": file_sha256(V3_SINGLETHREAD_DIAGNOSTIC),
        "semantic_independence_audit": semantic_audit,
        "tasks": tasks,
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
    })
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
