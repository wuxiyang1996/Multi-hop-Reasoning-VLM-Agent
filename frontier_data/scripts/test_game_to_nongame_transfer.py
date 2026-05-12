#!/usr/bin/env python
"""Test cross-domain skill transfer: game → non-game via harness.

Uses the existing harness pipeline (FewShotAdapter + predicate translator
+ skill_bank_bridge) to attempt forward-binding of game multi-task
abstract skills into non-game target tasks.

Two modes:
  --offline   Structural validation only (no LLM). Checks protocol
              compatibility, predicate translation coverage, and
              hop-walker execution via the harness adapters.
  --online    LLM re-grounding via bind_abstract_to_task's prompt
              pipeline, then harness validation.

Output: a transfer matrix JSON + console report.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
for p in [str(REPO), str(REPO.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from skill_bank.shared_abstract_bank import (
    BoundConcreteSkill, ProtocolStep, SharedAbstractSkill,
    TwoLayerSkillStore,
)

logger = logging.getLogger("game_to_nongame_transfer")

FRONTIER_BANK = REPO / "frontier_data" / "output" / "shared_skill_bank"
PROD_BANK = REPO / "shared_skill_bank" / "_latest"

NON_GAME_TARGETS = [
    "miniwob", "webshop",
    "tir_bench", "visual_toolbench",
    "siv_bench", "video_holmes",
]

GAME_COHORTS = {"gymv_game", "env_wr_game"}


def load_bank(bank_root: Path) -> TwoLayerSkillStore:
    """Load the TwoLayerSkillStore from disk."""
    store = TwoLayerSkillStore(bank_root)
    store.abstract.load()
    return store


def find_game_multi_task_abstracts(
    bank: TwoLayerSkillStore, min_tasks: int = 2,
) -> List[SharedAbstractSkill]:
    """Return abstracts with lineage spanning ≥ min_tasks game tasks."""
    results = []
    for abstract in bank.abstract.records:
        game_tasks = set()
        for L in abstract.lineage:
            if L.cohort in GAME_COHORTS or L.task.startswith("Temporal_"):
                game_tasks.add(L.task)
        if len(game_tasks) >= min_tasks:
            results.append(abstract)
    return results


def _try_import_bridge():
    """Import harness bridge components; return None on import error."""
    try:
        from harness.skill_bank_bridge import (
            binding_to_skill_record,
            lineage_to_demos,
            task_to_harness_domain,
            validate_binding_via_harness,
        )
        from harness.predicate_translator import (
            PREDICATE_TRANSLATIONS,
            translate_skill_contract,
        )
        return {
            "binding_to_skill_record": binding_to_skill_record,
            "lineage_to_demos": lineage_to_demos,
            "task_to_harness_domain": task_to_harness_domain,
            "validate_binding_via_harness": validate_binding_via_harness,
            "PREDICATE_TRANSLATIONS": PREDICATE_TRANSLATIONS,
            "translate_skill_contract": translate_skill_contract,
        }
    except Exception as exc:
        logger.warning("Could not import harness bridge: %s", exc)
        return None


def _structural_bind(
    abstract: SharedAbstractSkill,
    target_task: str,
    target_domain: str,
) -> BoundConcreteSkill:
    """Create a structural (offline) binding by copying the abstract
    protocol and translating predicates where possible."""
    steps = []
    for ps in abstract.protocol_steps:
        steps.append(ProtocolStep(
            op=ps.op,
            payload=dict(ps.payload or {}),
            slot_types=dict(ps.slot_types or {}),
            preconditions=list(ps.preconditions or []),
            effects_add=list(ps.effects_add or []),
            effects_del=list(ps.effects_del or []),
            evidence_role=ps.evidence_role,
            notes=ps.notes,
        ))

    binding = BoundConcreteSkill(
        concrete_skill_id=f"xfer.{abstract.abstract_skill_id}@{target_task}",
        abstract_skill_id=abstract.abstract_skill_id,
        task=target_task,
        name=f"{abstract.name} (game→{target_task})",
        protocol=steps,
        contract={
            "preconditions": [],
            "postconditions": [],
            "eff_add": [],
            "eff_del": [],
        },
        binding_status="PENDING",
        binding_source="cross_domain_transfer_test",
        raw_skill_id=abstract.abstract_skill_id,
        sub_episodes=[],
        schema_version=2,
    )
    return binding


def offline_transfer_test(
    abstracts: List[SharedAbstractSkill],
    bank: TwoLayerSkillStore,
    targets: List[str],
    bridge: Optional[Dict],
) -> Dict[str, Any]:
    """Run offline structural transfer for each (abstract, target) pair.

    Returns a results dict with per-pair diagnostics.
    """
    results = {
        "mode": "offline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_abstracts": len(abstracts),
        "n_targets": len(targets),
        "matrix": {},
        "summary": {},
    }

    task_to_domain = bridge["task_to_harness_domain"] if bridge else lambda t: "unknown"
    pred_trans = bridge.get("PREDICATE_TRANSLATIONS", {}) if bridge else {}

    for abstract in abstracts:
        aid = abstract.abstract_skill_id
        sig = abstract.template_signature
        source_tasks = sorted(set(
            L.task for L in abstract.lineage
            if L.cohort in GAME_COHORTS or L.task.startswith("Temporal_")
        ))

        for target in targets:
            key = f"{aid}→{target}"
            target_domain = task_to_domain(target)

            binding = _structural_bind(abstract, target, target_domain)

            diag: Dict[str, Any] = {
                "abstract_id": aid,
                "template_signature": sig,
                "source_tasks": source_tasks,
                "target_task": target,
                "target_domain": target_domain,
                "n_protocol_steps": len(binding.protocol),
            }

            # Check predicate translation coverage
            trans_key = ("gymv", target_domain)
            if trans_key in pred_trans:
                trans_table = pred_trans[trans_key]
                diag["predicate_translation"] = "available"
                diag["translation_table_size"] = len(trans_table)

                source_preds = set()
                for ps in abstract.protocol_steps:
                    for e in (ps.effects_add or []):
                        if isinstance(e, dict):
                            source_preds.add(e.get("type", ""))
                    for e in (ps.effects_del or []):
                        if isinstance(e, dict):
                            source_preds.add(e.get("type", ""))
                source_preds.discard("")

                translated = source_preds & set(trans_table.keys())
                dropped = set()
                for p in translated:
                    if not trans_table[p]:
                        dropped.add(p)

                diag["source_predicates"] = sorted(source_preds)
                diag["translatable_predicates"] = sorted(translated - dropped)
                diag["dropped_predicates"] = sorted(dropped)
                diag["untranslated_predicates"] = sorted(source_preds - set(trans_table.keys()))
                diag["translation_coverage"] = (
                    len(translated - dropped) / max(1, len(source_preds))
                )
            else:
                diag["predicate_translation"] = "no_table"
                diag["translation_coverage"] = 0.0

            # Try harness validation if bridge available
            verdict = None
            if bridge and bridge.get("validate_binding_via_harness"):
                try:
                    verdict, harness_diag = bridge["validate_binding_via_harness"](
                        candidate_binding=binding,
                        abstract=abstract,
                        bank=bank,
                    )
                    diag["harness_verdict"] = harness_diag.get("verdict", "UNKNOWN")
                    diag["harness_pass_rate"] = harness_diag.get("pass_rate")
                    diag["harness_n_demos"] = harness_diag.get("n_demos", 0)
                    diag["harness_reason"] = harness_diag.get("reason", "")
                except Exception as exc:
                    diag["harness_verdict"] = "ERROR"
                    diag["harness_error"] = str(exc)[:200]
                    verdict = None
            else:
                diag["harness_verdict"] = "SKIPPED"

            diag["transfer_feasible"] = (
                diag.get("translation_coverage", 0) > 0
                or diag.get("harness_verdict") == "VALIDATED"
            )

            results["matrix"][key] = diag

    # Build summary
    by_target = defaultdict(lambda: {"total": 0, "feasible": 0, "validated": 0})
    by_abstract = defaultdict(lambda: {"total": 0, "feasible": 0, "validated": 0})
    for key, diag in results["matrix"].items():
        t = diag["target_task"]
        a = diag["abstract_id"]
        by_target[t]["total"] += 1
        by_abstract[a]["total"] += 1
        if diag.get("transfer_feasible"):
            by_target[t]["feasible"] += 1
            by_abstract[a]["feasible"] += 1
        if diag.get("harness_verdict") == "VALIDATED":
            by_target[t]["validated"] += 1
            by_abstract[a]["validated"] += 1

    results["summary"]["by_target"] = dict(by_target)
    results["summary"]["by_abstract"] = dict(by_abstract)
    results["summary"]["total_pairs"] = len(results["matrix"])
    results["summary"]["feasible_pairs"] = sum(
        1 for d in results["matrix"].values() if d.get("transfer_feasible")
    )
    results["summary"]["validated_pairs"] = sum(
        1 for d in results["matrix"].values()
        if d.get("harness_verdict") == "VALIDATED"
    )

    return results


def print_transfer_matrix(results: Dict[str, Any]):
    """Pretty-print the transfer matrix."""
    matrix = results["matrix"]
    summary = results["summary"]

    abstracts = sorted(set(d["abstract_id"] for d in matrix.values()))
    targets = sorted(set(d["target_task"] for d in matrix.values()))

    print("\n" + "=" * 80)
    print("GAME → NON-GAME SKILL TRANSFER MATRIX")
    print("=" * 80)

    # Header
    header = f"{'Abstract':<30}"
    for t in targets:
        header += f" {t[:12]:>12}"
    print(header)
    print("-" * len(header))

    for a in abstracts:
        row = f"{a:<30}"
        for t in targets:
            key = f"{a}→{t}"
            d = matrix.get(key, {})
            verdict = d.get("harness_verdict", "?")
            coverage = d.get("translation_coverage", 0)
            if verdict == "VALIDATED":
                cell = "✓ VALID"
            elif verdict == "REJECTED":
                cell = "✗ REJECT"
            elif coverage > 0:
                cell = f"~{coverage:.0%}"
            else:
                cell = "—"
            row += f" {cell:>12}"
        print(row)

    print("-" * len(header))
    print(f"\nTotal: {summary['total_pairs']} pairs | "
          f"Feasible: {summary['feasible_pairs']} | "
          f"Validated: {summary['validated_pairs']}")

    print("\n--- By Target Task ---")
    for t in targets:
        info = summary["by_target"].get(t, {})
        print(f"  {t:<20} feasible={info.get('feasible',0)}/{info.get('total',0)} "
              f"validated={info.get('validated',0)}")

    print("\n--- By Abstract Skill ---")
    for a in abstracts:
        info = summary["by_abstract"].get(a, {})
        print(f"  {a:<30} feasible={info.get('feasible',0)}/{info.get('total',0)} "
              f"validated={info.get('validated',0)}")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bank-root",
        help="Path to shared_skill_bank (default: auto-detect frontier_data or _latest)",
    )
    p.add_argument(
        "--targets", nargs="+", default=NON_GAME_TARGETS,
        help="Non-game target tasks",
    )
    p.add_argument(
        "--min-game-tasks", type=int, default=2,
        help="Minimum game tasks in lineage to qualify as transferable",
    )
    p.add_argument(
        "--output", default=str(REPO / "frontier_data" / "output" / "transfer_matrix.json"),
        help="Output JSON path",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    bank_root = Path(args.bank_root) if args.bank_root else None
    if bank_root is None:
        for candidate in [FRONTIER_BANK, PROD_BANK]:
            if (candidate / "abstract.jsonl").exists():
                bank_root = candidate
                break
    if bank_root is None or not (bank_root / "abstract.jsonl").exists():
        logger.error("No shared_skill_bank found. Supply --bank-root.")
        sys.exit(1)

    logger.info("Loading bank from %s", bank_root)
    bank = load_bank(bank_root)

    abstracts = find_game_multi_task_abstracts(bank, min_tasks=args.min_game_tasks)
    if not abstracts and args.min_game_tasks > 1:
        logger.info("No multi-task abstracts with ≥%d tasks. Trying min=1 to include single-game skills...", args.min_game_tasks)
        abstracts = find_game_multi_task_abstracts(bank, min_tasks=1)
    logger.info("Found %d game multi-task abstracts (≥%d tasks)", len(abstracts), args.min_game_tasks)

    if not abstracts:
        logger.warning("No multi-task game abstracts found. Try --min-game-tasks 1")
        sys.exit(0)

    for a in abstracts:
        tasks = sorted(set(L.task for L in a.lineage if L.cohort in GAME_COHORTS or L.task.startswith("Temporal_")))
        logger.info("  %s  sig=%s  tasks=%s", a.abstract_skill_id, a.template_signature, tasks)

    bridge = _try_import_bridge()
    if bridge:
        logger.info("Harness bridge loaded — will run structural validation")
    else:
        logger.info("Harness bridge unavailable — predicate analysis only")

    results = offline_transfer_test(abstracts, bank, args.targets, bridge)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Wrote transfer matrix to %s", out_path)

    print_transfer_matrix(results)


if __name__ == "__main__":
    main()
