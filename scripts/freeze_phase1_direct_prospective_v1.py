#!/usr/bin/env python3
"""Freeze 24 disjoint direct source-game/target-domain execution cells.

Selection uses target identities and schema only.  It never consults an
answer, reward, world state, model response, or prior target outcome for a
selected reserve item.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SCHEMA,
    SOURCE_GAMES,
    STATUS,
    TARGET_DOMAINS,
    file_sha256,
)
from motif_transfer.search_automaton_transfer_v16 import (  # noqa: E402
    SourceSearchAutomaton,
)
from motif_transfer.webshop_semantic_reserve import (  # noqa: E402
    require_semantic_reserve,
)


RUN_ROOT = REPO / "runs/phase1_direct_prospective_v1"
CONFIG_ROOT = REPO / "configs/phase1_direct_prospective_v1"
PARENT_SOURCE = (
    REPO / "runs/phase1_common_search_ir_formal_v1/"
    "common_search_automaton_artifact.json"
)
ALF_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/"
    ".cache/alfworld_data/json_2.1.1/valid_unseen"
)
TIR_DATASET = Path(
    "/fs/gamma-projects/vlm-robot/datasets/TIR-Bench/TIR-Bench.json"
)
WEBSHOP_VENDOR = Path(
    "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/vendor/WebShop"
)
WEBSHOP_SEED = 20260815


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_artifacts() -> dict[str, dict[str, Any]]:
    parent = _read(PARENT_SOURCE)
    SourceSearchAutomaton(parent)
    parent_file_hash = file_sha256(PARENT_SOURCE)
    lineage_by_game = {
        str(row["game"]): dict(row) for row in parent["source_lineages"]
    }
    if tuple(lineage_by_game) != SOURCE_GAMES:
        raise ValueError("parent source lineage order changed")
    receipts: dict[str, dict[str, Any]] = {}
    for game in SOURCE_GAMES:
        lineage = lineage_by_game[game]
        body = {
            "schema_version": parent["schema_version"],
            "status": parent["status"],
            "target_authorized": True,
            "claim_boundary": (
                "ONE_FORMALLY_QUALIFIED_PHASE1_SOURCE_GAME_LINEAGE;"
                "COMMON_SYMBOLIC_EVENT_ROUTING_ONLY;TARGET_NATIVE_NEURAL_"
                "GROUNDING_REQUIRED;NO_SOURCE_NATIVE_ACTION_OR_CANDIDATE_ID"
            ),
            "canonical_policy_sha256": parent["canonical_policy_sha256"],
            "learned_policy": dict(parent["learned_policy"]),
            "source_lineage": {
                "kind": "ONE_INDEPENDENT_PHASE1_GAME_FORMAL_LINEAGE",
                "game": game,
                "parent_consensus_artifact_sha256": parent["artifact_sha256"],
                "parent_consensus_artifact_file_sha256": parent_file_hash,
                "formal_evidence": lineage,
            },
            "source_lineages": [lineage],
            "transfer_contract": dict(parent["transfer_contract"]),
        }
        artifact = body | {"artifact_sha256": stable_hash(body)}
        path = RUN_ROOT / "source_artifacts" / f"{game}.json"
        _write_once(path, artifact)
        SourceSearchAutomaton(artifact)
        receipts[game] = {
            "artifact": str(path.relative_to(REPO)),
            "artifact_sha256": artifact["artifact_sha256"],
            "artifact_file_sha256": file_sha256(path),
            "parent_consensus_artifact_sha256": parent["artifact_sha256"],
            "formal_lineage_report_file_sha256": lineage["report_file_sha256"],
        }
    return receipts


def _all_trial_ids_in_prior_evidence() -> set[str]:
    result = subprocess.run(
        ["rg", "-o", r"trial_T[0-9_]+", "runs", "docs/results"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return set(re.findall(r"trial_T[0-9_]+", result.stdout))


def _select_alfworld() -> list[dict[str, Any]]:
    prior = _all_trial_ids_in_prior_evidence()
    candidates = []
    for path in ALF_ROOT.rglob("game.tw-pddl"):
        relative = str(path.relative_to(ALF_ROOT))
        if path.parent.name in prior:
            continue
        family = relative.split("-", 1)[0]
        candidates.append({
            "task_id": relative,
            "family": family,
            "trial_id": path.parent.name,
            "game_file_sha256": file_sha256(path),
        })
    candidates.sort(key=lambda row: stable_hash(
        "phase1-direct-v1|alfworld|" + str(row["task_id"])
    ))
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_family.setdefault(str(row["family"]), []).append(row)
    selected = [rows[0] for _, rows in sorted(by_family.items())]
    selected_ids = {str(row["task_id"]) for row in selected}
    selected.extend(
        row for row in candidates if str(row["task_id"]) not in selected_ids
    )
    selected = selected[: len(SOURCE_GAMES)]
    if len(selected) != len(SOURCE_GAMES):
        raise ValueError("fewer than six fresh ALFWorld valid_unseen tasks")
    return selected


def _tir_used_ids(maze_ids: set[str]) -> set[str]:
    used = set()
    pattern = re.compile(r'"sample_id"\s*:\s*"?([0-9]+)')
    for root in (REPO / "runs", REPO / "docs/results"):
        for path in root.rglob("*"):
            if not path.is_file() or "tir" not in str(path).lower():
                continue
            if path.stat().st_size > 200_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            used.update(
                match.group(1) for match in pattern.finditer(text)
                if match.group(1) in maze_ids
            )
    return used


def _select_tir() -> list[dict[str, Any]]:
    # Only the ID and schema fields influence selection.  Prompt, image and
    # answer payloads remain outside the selection key.
    rows = _read(TIR_DATASET)
    maze_ids = {
        str(row["id"]) for row in rows
        if row.get("task") == "maze" and not row.get("image_2")
    }
    used = _tir_used_ids(maze_ids)
    selected = sorted(
        maze_ids - used,
        key=lambda sample_id: stable_hash(
            "phase1-direct-v1|tirbench|" + sample_id
        ),
    )[: len(SOURCE_GAMES)]
    if len(selected) != len(SOURCE_GAMES):
        raise ValueError("fewer than six fresh TIR maze samples")
    return [{"task_id": sample_id} for sample_id in selected]


def _prior_discovery_text() -> str:
    chunks = []
    for root in (REPO / "runs", REPO / "docs/results"):
        for path in root.rglob("*"):
            if not path.is_file() or "discovery" not in str(path).lower():
                continue
            if path.stat().st_size > 50_000_000:
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
    return "\n".join(chunks)


def _select_discoveryworld() -> list[dict[str, Any]]:
    prior = _prior_discovery_text()
    selected = []
    # The prior prospective replication established eligibility for every one
    # of ten Proteomics Easy seeds, whereas only six of ten Space Sick seeds
    # exposed the predeclared native-commit fork.  Use the higher-coverage
    # interface here because the unit under test is source-lineage routing,
    # not a new DiscoveryWorld theme comparison.
    for seed in range(21, 27):
        scenario = "Proteomics"
        task_id = f"proteomics.easy.seed{seed}"
        if task_id in prior:
            raise ValueError(f"DiscoveryWorld reserve identity was executed: {task_id}")
        selected.append({
            "task_id": task_id,
            "scenario": scenario,
            "difficulty": "Easy",
            "seed": seed,
        })
    return selected


def _iter_json_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_json_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_json_values(nested)
    else:
        yield value


def _prior_webshop_identities() -> tuple[set[str], set[str]]:
    asins: set[str] = set()
    instructions: set[str] = set()
    for root in (REPO / "configs", REPO / "runs"):
        for path in root.rglob("*webshop*.json"):
            if not path.is_file() or path.stat().st_size > 100_000_000:
                continue
            try:
                payload = _read(path)
            except (OSError, json.JSONDecodeError):
                continue
            def visit(value: Any) -> None:
                if isinstance(value, Mapping):
                    if value.get("asin"):
                        asins.add(str(value["asin"]))
                    if value.get("instruction_text"):
                        instructions.add(str(value["instruction_text"]).strip().lower())
                    for nested in value.values():
                        visit(nested)
                elif isinstance(value, list):
                    for nested in value:
                        visit(nested)
            visit(payload)
    return asins, instructions


def _select_webshop() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    products_path = WEBSHOP_VENDOR / "data/items_shuffle_1000.json"
    human_path = WEBSHOP_VENDOR / "data/items_human_ins.json"
    sys.path.insert(0, str(WEBSHOP_VENDOR))
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals

    prior_asins, prior_instructions = _prior_webshop_identities()
    products_payload = _read(products_path)
    human = _read(human_path)
    prior_asins.update(
        str(row["asin"]) for row in products_payload
        if str(row.get("asin")) in human
    )
    random.seed(WEBSHOP_SEED)
    products, _, prices, _ = load_products(
        filepath=str(products_path), human_goals=False
    )
    random.seed(WEBSHOP_SEED)
    goals = get_goals(products, prices, human_goals=False)
    random.seed(WEBSHOP_SEED)
    random.shuffle(goals)
    selected = []
    used_asins = set(prior_asins)
    used_instructions = set(prior_instructions)
    for index, goal in enumerate(goals):
        asin = str(goal["asin"])
        instruction = str(goal["instruction_text"]).strip().lower()
        if asin in used_asins or instruction in used_instructions:
            continue
        selected.append({
            "task_id": f"webshop.{index}",
            "server_goal_index": index,
            "asin": asin,
            "instruction_text": str(goal["instruction_text"]),
            "goal": goal,
            "goal_sha256": stable_hash(goal),
        })
        used_asins.add(asin)
        used_instructions.add(instruction)
        if len(selected) == len(SOURCE_GAMES):
            break
    if len(selected) != len(SOURCE_GAMES):
        raise ValueError("fewer than six fresh WebShop goal identities")
    audit = require_semantic_reserve(
        selected,
        consumed_rows=[
            {"asin": asin, "instruction_text": "prior identity"}
            for asin in sorted(prior_asins)
        ],
        required_unique_goals=len(SOURCE_GAMES),
        require_asin_disjointness=True,
        require_unique_candidate_asins=True,
    )
    goal_manifest_body = {
        "schema_version": 1,
        "artifact_role": "OUTCOME_BLIND_PHASE1_DIRECT_WEBSHOP_RESERVE_V1",
        "status": "FROZEN_BEFORE_ANY_PROVIDER_CALL_OR_OUTCOME",
        "claim_boundary": (
            "Six fresh native synthetic WebShop goal identities, one per source "
            "lineage, disjoint by ASIN and instruction from prior WebShop configs."
        ),
        "selection_rule": (
            "Seed native synthetic goal generation with 20260815 and take the "
            "first six identities whose ASIN and instruction do not occur in "
            "prior WebShop configurations or result artifacts."
        ),
        "goal_seed": WEBSHOP_SEED,
        "server_goal_count": len(goals),
        "number_of_registered_tasks_required": 1 + max(
            int(row["server_goal_index"]) for row in selected
        ),
        "roles": {"development": [], "formal_reserve": selected},
        "preflight": {"selected_vs_prior": audit},
        "runtime_contract": {
            "server_launcher": "scripts/run_webshop_direct_server_v1.py",
            "goal_mode": "native_synthetic",
            "human_goal_mode_forbidden": True,
        },
        "runtime_hashes": {
            "products": file_sha256(products_path),
            "human_instructions": file_sha256(human_path),
            "vendor_engine": file_sha256(
                WEBSHOP_VENDOR / "web_agent_site/engine/engine.py"
            ),
            "vendor_goal": file_sha256(
                WEBSHOP_VENDOR / "web_agent_site/engine/goal.py"
            ),
        },
    }
    goal_manifest = goal_manifest_body | {
        "artifact_sha256": stable_hash(goal_manifest_body)
    }
    return selected, goal_manifest


def _webshop_protocols(
    sources: Mapping[str, Mapping[str, Any]],
    selected: list[dict[str, Any]],
    goal_manifest_path: Path,
) -> dict[str, str]:
    qualification = (
        REPO / "runs/webshop_search_automaton_v16_development_"
        "gpt41mini_anytime/report.json"
    )
    target_grounder = (
        REPO / "docs/results/webshop_neural_symbolic_v9_frozen_grounder.json"
    )
    protocol_paths = {}
    for game, task in zip(SOURCE_GAMES, selected):
        source_path = REPO / str(sources[game]["artifact"])
        protocol = {
            "status": "FROZEN_BEFORE_FORMAL_EXECUTION",
            "tasks": [str(task["task_id"])],
            "conditions": [
                "raw_target_only",
                "authentic_search_automaton_plus_target",
                "event_binding_permuted_control",
                "ledger_blind_control",
                "target_native_search_ceiling",
            ],
            "qualification_report_file_sha256": file_sha256(qualification),
            "source_artifact_file_sha256": file_sha256(source_path),
            "target_grounder_file_sha256": file_sha256(target_grounder),
            "goal_manifest_file_sha256": file_sha256(goal_manifest_path),
            "controller_file_sha256": file_sha256(
                REPO / "src/motif_transfer/webshop_search_automaton_v16.py"
            ),
            "coverage_controller_file_sha256": file_sha256(
                REPO / "src/motif_transfer/webshop_coverage_transfer_v14.py"
            ),
            "runner_file_sha256": file_sha256(
                REPO / "scripts/run_webshop_search_automaton_v16.py"
            ),
            "model": "qwen/qwen3.5-35b-a3b",
            "maximum_output_tokens": 3200,
            "maximum_steps": 12,
            "source_game": game,
            "target_goal_sha256": task["goal_sha256"],
            "selected_target_previously_executed": False,
        }
        path = CONFIG_ROOT / "webshop_protocols" / f"{game}.json"
        _write_once(path, protocol)
        protocol_paths[game] = str(path.relative_to(REPO))
    return protocol_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=CONFIG_ROOT / "manifest.json",
    )
    args = parser.parse_args()
    if args.output.exists() or RUN_ROOT.exists() or CONFIG_ROOT.exists():
        raise SystemExit("direct prospective V1 outputs already exist; refusing overwrite")

    required_runtime = [
        "src/motif_transfer/direct_prospective_matrix_v1.py",
        "scripts/freeze_phase1_direct_prospective_v1.py",
        "scripts/run_phase1_direct_webshop_v1.py",
        "scripts/run_webshop_direct_server_v1.py",
        "scripts/run_phase1_direct_alfworld_v1.py",
        "scripts/run_phase1_direct_tirbench_v1.py",
        "scripts/run_phase1_direct_discoveryworld_v1.py",
        "scripts/audit_phase1_direct_prospective_v1.py",
    ]
    missing = [relative for relative in required_runtime if not (REPO / relative).is_file()]
    if missing:
        raise SystemExit(f"runtime must exist before freeze: {missing}")

    sources = _source_artifacts()
    webshop, goal_manifest = _select_webshop()
    goal_manifest_path = CONFIG_ROOT / "webshop_goal_manifest.json"
    _write_once(goal_manifest_path, goal_manifest)
    webshop_protocols = _webshop_protocols(sources, webshop, goal_manifest_path)
    selected = {
        "webshop": webshop,
        "alfworld": _select_alfworld(),
        "discoveryworld": _select_discoveryworld(),
        "tirbench": _select_tir(),
    }
    cells = []
    for domain in TARGET_DOMAINS:
        for game, task in zip(SOURCE_GAMES, selected[domain]):
            row = {
                "cell_id": f"{game}__to__{domain}",
                "source_game": game,
                "target_domain": domain,
                "target_task_id": str(task["task_id"]),
                "target_task": task,
                "target_task_multiplicity": 1,
                "selected_target_previously_executed": False,
                "source_artifact": sources[game]["artifact"],
                "source_artifact_sha256": sources[game]["artifact_sha256"],
            }
            if domain == "webshop":
                row["domain_protocol"] = webshop_protocols[game]
            cells.append(row)

    # Configs used by the unmodified target-native DiscoveryWorld collection
    # and fork freezer.  Source-specific online routing is added only by the
    # direct runner after these files are frozen.
    discovery_manifest = {
        "schema_version": "discoveryworld-phase1-direct-reserve-v1",
        "status": "FROZEN_BEFORE_TARGET_RESET",
        "official_environment_commit": "4f85d5217f2b8a87c1ca02f47007d75306c3fae2",
        "roles": {"formal_reserve": [row for row in selected["discoveryworld"]]},
    }
    discovery_manifest["manifest_sha256"] = stable_hash(discovery_manifest)
    discovery_manifest_path = CONFIG_ROOT / "discoveryworld_manifest.json"
    _write_once(discovery_manifest_path, discovery_manifest)
    discovery_target_config = {
        "schema_version": "discoveryworld-target-only-phase1-direct-v1",
        "claim_boundary": (
            "Six never-executed Proteomics Easy instances, seeds21-26, "
            "target-only acquisition before source routing."
        ),
        "manifest": str(discovery_manifest_path.relative_to(REPO)),
        "manifest_role": "formal_reserve",
        "model": {
            "api_key_name": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "maximum_output_tokens": 1200,
            "model": "qwen/qwen3.5-35b-a3b",
            "provider": "openrouter",
            "schema_attempts": 3,
            "temperature": 0,
        },
        "runtime": {"include_vision": False, "maximum_steps": 96},
    }
    discovery_target_path = CONFIG_ROOT / "discoveryworld_target_only.json"
    _write_once(discovery_target_path, discovery_target_config)
    old_protocol = _read(REPO / "configs/discoveryworld_sokoban_replication_v1_protocol.json")
    discovery_protocol = dict(old_protocol)
    discovery_protocol.update({
        "schema_version": "discoveryworld-search-automaton-direct-v1",
        "status": "FORMAL_RESERVE_FROZEN_BEFORE_OPEN",
        "claim_boundary": (
            "Direct prospective lineage-specific search-automaton routing on "
            "six fresh DiscoveryWorld Easy seeds21-23."
        ),
        "evaluation_stage": "FORMAL_RESERVE",
        "target_baseline_config": str(discovery_target_path.relative_to(REPO)),
        "task_ids": [row["task_id"] for row in selected["discoveryworld"]],
        "operational_disclosure": (
            "No scientific changes or selective retries after the first selected "
            "target reset; transport retries preserve request identity."
        ),
    })
    discovery_protocol_path = CONFIG_ROOT / "discoveryworld_protocol.json"
    _write_once(discovery_protocol_path, discovery_protocol)

    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_boundary": (
            "Direct prospective operational validation of one independently "
            "qualified Phase-1 game lineage in each of 24 source×target cells. "
            "This does not claim 24 independently powered efficacy estimates; "
            "domain-level efficacy remains supported by the prior frozen matched "
            "evaluations."
        ),
        "selection_read_target_outcome": False,
        "historical_target_outcome_reuse_allowed": False,
        "cell_execution_definition": (
            "One unique fresh target identity, full domain matched-condition "
            "execution, and online admitted route receipts containing the cell's "
            "lineage-specific source artifact hash."
        ),
        "sources": sources,
        "targets": {
            "webshop": {
                "selection": selected["webshop"],
                "goal_seed": WEBSHOP_SEED,
                "goal_manifest": str(goal_manifest_path.relative_to(REPO)),
                "goal_manifest_file_sha256": file_sha256(goal_manifest_path),
            },
            "alfworld": {
                "selection": selected["alfworld"],
                "split": "eval_out_of_distribution",
                "valid_unseen_root": str(ALF_ROOT),
                "target_grounder": (
                    "runs/procedural_game_alfworld_v1_development/"
                    "frozen_candidate_artifact.json"
                ),
            },
            "discoveryworld": {
                "selection": selected["discoveryworld"],
                "target_manifest": str(discovery_manifest_path.relative_to(REPO)),
                "target_config": str(discovery_target_path.relative_to(REPO)),
                "protocol": str(discovery_protocol_path.relative_to(REPO)),
            },
            "tirbench": {
                "selection": selected["tirbench"],
                "dataset": str(TIR_DATASET),
                "dataset_file_sha256": file_sha256(TIR_DATASET),
                "model": {
                    "provider": "openrouter",
                    "id": "openai/gpt-4.1-mini",
                    "base_url": "https://openrouter.ai/api/v1",
                    "maximum_output_tokens": 1200,
                    "max_side": 768,
                    "jpeg_quality": 90,
                },
            },
        },
        "cells": cells,
        "conditions": {
            "webshop": [
                "raw_target_only", "authentic_search_automaton_plus_target",
                "event_binding_permuted_control", "ledger_blind_control",
                "target_native_search_ceiling",
            ],
            "alfworld": [
                "raw_target_only", "authentic_search_automaton_plus_target",
                "event_binding_permuted_control", "ledger_blind_control",
                "target_native_search_ceiling",
            ],
            "discoveryworld": list(discovery_protocol["conditions"]),
            "tirbench": [
                "raw_target_only", "authentic_search_automaton_plus_target",
                "event_binding_permuted_control",
                "ledger_blind_repeat_first_control",
                "commit_availability_only_control",
                "target_native_exhaustive_ceiling",
            ],
        },
        "runtime_file_sha256": {
            relative: file_sha256(REPO / relative) for relative in required_runtime
        },
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write_once(args.output, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "cells": len(cells),
        "unique_source_artifacts": len({
            row["source_artifact_sha256"] for row in cells
        }),
        "unique_target_tasks": sum(
            len({row["task_id"] for row in selected[domain]})
            for domain in TARGET_DOMAINS
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
