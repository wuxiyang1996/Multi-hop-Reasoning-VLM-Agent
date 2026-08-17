#!/usr/bin/env python3
"""Freeze outcome-blind future-task probes for the unified transfer runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


ALFWORLD_DATA = Path(
    "/fs/gamma-projects/vlm-robot/"
    "Multi-hop-Reasoning-VLM-Agent-github-main/.cache/alfworld_data"
)
OUTPUT = REPO / "configs/phase5_unified_applicability_v1_frozen.json"
TASK_PATTERN = re.compile(r"pick_two_obj_and_place-[^\"\s]+/game\.tw-pddl")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _executed_alfworld_tasks() -> tuple[set[str], int]:
    executed: set[str] = set()
    scanned = 0
    for path in sorted((REPO / "runs").rglob("*.json")):
        if any(token in path.name.lower() for token in (
            "enumeration", "manifest", "plan",
        )):
            continue
        scanned += 1
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                executed.update(TASK_PATTERN.findall(line))
    return executed, scanned


def _configured_task_mentions(task_id: str) -> list[str]:
    matches = []
    for base in ("configs", "runs", "docs"):
        for path in sorted((REPO / base).rglob("*.json")):
            try:
                if task_id in path.read_text(encoding="utf-8", errors="ignore"):
                    matches.append(str(path.relative_to(REPO)))
            except OSError:
                continue
    return matches


def _route_rows() -> list[dict[str, Any]]:
    webshop = _read(REPO / "docs/results/webshop_structural_v21_formal_compact.json")
    discovery = _read(REPO / "runs/discoveryworld_structural_transfer_v1_matched/report.json")
    tir = _read(REPO / "runs/tir_maze_structural_transfer_v3/heldout_report.json")
    alfworld = _read(REPO / "runs/alfworld_structural_transfer_v2_extension/report.json")
    source_relational = _read(REPO / "runs/sokoban_relational_structural_v2/artifact.json")
    webshop_grounder = _read(
        REPO / "runs/webshop_structural_transfer_v17_development/low_sample_grounder.json"
    )
    dw_manifest = _read(REPO / "configs/discoveryworld_structural_transfer_v1/manifest.json")
    alf_manifest = _read(
        REPO / "configs/alfworld_structural_transfer_v2_extension/manifest.json"
    )
    put_near = dw_manifest["source_programs"]["put_near"]
    return [
        {
            "route_id": "sokoban-relational-to-webshop-v21",
            "target_domain": "webshop",
            "target_interface": "product_option_relation_search",
            "required_capabilities": [
                "candidate_relations", "native_search", "unique_terminal_binding",
            ],
            "source_program_sha256": source_relational["artifact_sha256"],
            "source_program_induced_from_interventions": True,
            "source_program_qualified": True,
            "target_grounder_sha256": webshop_grounder["artifact_sha256"],
            "target_executor_sha256": _sha(
                REPO / "src/motif_transfer/webshop_structural_transfer_v17.py"
            ),
            "target_grounder_id": "webshop.low_sample_relation_grounder.v17",
            "target_executor_id": "webshop.structural_target_executor.v21",
            "evidence_report_sha256": webshop["full_report_content_sha256"],
            "utility_vs_neural": {"wins": 16, "losses": 0, "ties": 16},
            "authenticity_vs_source_permuted": {
                "wins": 16, "losses": 0, "ties": 16,
            },
            "evidence": {
                "path": "docs/results/webshop_structural_v21_formal_compact.json",
                "file_sha256": _sha(
                    REPO / "docs/results/webshop_structural_v21_formal_compact.json"
                ),
                "status": webshop["status"],
            },
        },
        {
            "route_id": "minigrid-put-near-to-discoveryworld-easy-v1",
            "target_domain": "discoveryworld",
            "target_interface": "proteomics_easy_counted_partial_order",
            "required_capabilities": [
                "counted_relations", "native_actions", "unique_commit_binding",
            ],
            "source_program_sha256": put_near["program_sha256"],
            "source_program_induced_from_interventions": True,
            "source_program_qualified": True,
            "target_grounder_sha256": dw_manifest["target_development_report"][
                "grounder_sha256"
            ],
            "target_executor_sha256": _sha(
                REPO / "src/motif_transfer/discoveryworld_structural_runtime_v1.py"
            ),
            "target_grounder_id": "discoveryworld.counted_relation_grounder.v1",
            "target_executor_id": "discoveryworld.structural_target_executor.v1",
            "evidence_report_sha256": discovery["report_sha256"],
            "utility_vs_neural": {"wins": 9, "losses": 0, "ties": 3},
            "authenticity_vs_source_permuted": {
                "wins": 9, "losses": 0, "ties": 3,
            },
            "evidence": {
                "path": "runs/discoveryworld_structural_transfer_v1_matched/report.json",
                "file_sha256": _sha(
                    REPO / "runs/discoveryworld_structural_transfer_v1_matched/report.json"
                ),
                "status": discovery["status"],
            },
        },
        {
            "route_id": "sokoban-relational-to-tir-maze-v3",
            "target_domain": "tir",
            "target_interface": "single_image_maze_relational_path",
            "required_capabilities": [
                "direction_binding", "pixel_graph", "unique_goal_path",
            ],
            "source_program_sha256": source_relational["artifact_sha256"],
            "source_program_induced_from_interventions": True,
            "source_program_qualified": True,
            "target_grounder_sha256": stable_hash({
                "model": "openai/gpt-4.1-mini",
                "binding_module_file_sha256": _sha(
                    REPO / "src/motif_transfer/tir_maze_topology.py"
                ),
                "interface": "R_L_U_D_START_GOAL_PIXEL_GRAPH",
            }),
            "target_executor_sha256": _sha(
                REPO / "src/motif_transfer/tir_maze_topology.py"
            ),
            "target_grounder_id": "tir.openrouter_direction_pixel_grounder.v3",
            "target_executor_id": "tir.relational_path_executor.v3",
            "evidence_report_sha256": tir["report_sha256"],
            "utility_vs_neural": {"wins": 6, "losses": 0, "ties": 12},
            "authenticity_vs_source_permuted": {
                "wins": 6, "losses": 0, "ties": 12,
            },
            "evidence": {
                "path": "runs/tir_maze_structural_transfer_v3/heldout_report.json",
                "file_sha256": _sha(
                    REPO / "runs/tir_maze_structural_transfer_v3/heldout_report.json"
                ),
                "status": tir["status"],
            },
        },
        {
            "route_id": "minigrid-put-near-to-alfworld-multiplicity-v2",
            "target_domain": "alfworld",
            "target_interface": "multiplicity_add_remove_sequence",
            "required_capabilities": [
                "entity_binding", "multiplicity_two", "native_actions",
                "operator_effect_grounding",
            ],
            "source_program_sha256": alf_manifest["source_induced"]["program_sha256"],
            "source_program_induced_from_interventions": True,
            "source_program_qualified": True,
            "target_grounder_sha256": alf_manifest["grounder"]["grounder_sha256"],
            "target_executor_sha256": _sha(
                REPO / "src/motif_transfer/alfworld_structural_runtime_v1.py"
            ),
            "target_grounder_id": "alfworld.structural_grounder.v1",
            "target_executor_id": "alfworld.structural_target_executor.v2",
            "evidence_report_sha256": alfworld["report_sha256"],
            "utility_vs_neural": {"wins": 2, "losses": 0, "ties": 70},
            "authenticity_vs_source_permuted": {
                "wins": 1, "losses": 1, "ties": 70,
            },
            "evidence": {
                "path": "runs/alfworld_structural_transfer_v2_extension/report.json",
                "file_sha256": _sha(
                    REPO / "runs/alfworld_structural_transfer_v2_extension/report.json"
                ),
                "status": alfworld["status"],
            },
        },
    ]


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite frozen probes: {OUTPUT}")
    root = ALFWORLD_DATA / "json_2.1.1" / "train"
    executed, scanned = _executed_alfworld_tasks()
    alfworld_tasks = sorted(
        path.relative_to(root).as_posix()
        for path in root.glob("pick_two_obj_and_place-*/trial_*/game.tw-pddl")
        if path.relative_to(root).as_posix() not in executed
    )
    if len(alfworld_tasks) != 8:
        raise SystemExit(
            "expected the eight post-extension execution-untouched ALFWorld "
            f"multiplicity tasks; found {len(alfworld_tasks)}"
        )
    alfworld_rows = []
    for task_id in alfworld_tasks:
        path = root / task_id
        alfworld_rows.append({
            "task_id": task_id,
            "task_file_sha256": _sha(path),
            "prior_execution_occurrences": int(task_id in executed),
            "formal_outcome_read": False,
        })

    discoveryworld_rows = [
        ("proteomics.easy.seed213", "proteomics_easy_counted_partial_order"),
        ("proteomics.easy.seed214", "proteomics_easy_counted_partial_order"),
        ("proteomics.normal.seed5", "proteomics_normal_adjacency"),
        ("proteomics.normal.seed6", "proteomics_normal_adjacency"),
        ("space_sick.normal.seed5", "space_sick_normal_dialogue_feeding"),
        ("space_sick.normal.seed6", "space_sick_normal_dialogue_feeding"),
    ]
    prior_mentions = {
        task_id: _configured_task_mentions(task_id)
        for task_id, _ in discoveryworld_rows
    }
    if any(prior_mentions.values()):
        raise SystemExit(f"DiscoveryWorld probe already mentioned: {prior_mentions}")

    body: dict[str, Any] = {
        "schema_version": "phase5-unified-applicability-freeze-v1",
        "status": "FROZEN_BEFORE_ANY_PROBE_TARGET_RESET_OR_OUTCOME",
        "claim_boundary": (
            "Outcome-blind pre-execution applicability and calibrated routing "
            "audit only. It does not claim success on these unopened tasks."
        ),
        "routes": _route_rows(),
        "future_probes": {
            "discoveryworld": [
                {
                    "task_id": task_id,
                    "target_interface": interface,
                    "prior_json_mentions": prior_mentions[task_id],
                    "target_reset_before_freeze": False,
                    "formal_outcome_read": False,
                }
                for task_id, interface in discoveryworld_rows
            ],
            "alfworld": alfworld_rows,
        },
        "integrity": {
            "alfworld_execution_json_files_scanned": scanned,
            "alfworld_data_root": str(root),
            "alfworld_execution_untouched_tasks": len(alfworld_rows),
            "discoveryworld_prior_json_mentions": 0,
            "freezer_file_sha256": _sha(Path(__file__)),
            "runtime_file_sha256": _sha(
                REPO / "src/motif_transfer/unified_transfer_runtime.py"
            ),
        },
        "selection_rule": {
            "alfworld": (
                "all remaining train/pick_two_obj_and_place task files absent "
                "from non-enumeration, non-manifest, non-plan run JSON"
            ),
            "discoveryworld": (
                "predeclared new task identities spanning exact Easy interface "
                "and two structurally different Normal interfaces"
            ),
            "outcome_used": False,
        },
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "routes": len(manifest["routes"]),
        "discoveryworld_probes": len(discoveryworld_rows),
        "alfworld_execution_untouched_tasks": len(alfworld_rows),
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
