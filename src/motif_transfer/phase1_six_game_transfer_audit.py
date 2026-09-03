"""Fail-closed audit for Phase-1 game to four-target transfer claims.

The audit distinguishes three facts that are easy to conflate:

* a game has recorded trajectories or discovered skill labels;
* a target has an executable neural grounding for a symbolic IR; and
* the game independently induced and qualified the exact IR used on target.

Only the third fact, together with a positive target result, authorizes a
game-to-target transfer cell.  Target results from a different source lineage
are retained as readiness evidence, never relabelled as Phase-1 transfer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_GAMES = (
    "tetris",
    "candy_crush",
    "gymv_streets_of_rage_2",
    "gymv_strider",
    "gymv_columns",
    "gymv_thunder_force_iii",
)
TARGETS = ("webshop", "alfworld", "discoveryworld", "tirbench")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _phase1_rollout_summary(evidence_dir: Path) -> dict[str, Any]:
    required = ("manifest.json", "episodes.jsonl", "events.jsonl")
    missing = [name for name in required if not (evidence_dir / name).is_file()]
    if missing:
        return {
            "complete": False,
            "missing": missing,
            "episodes": 0,
            "agent_decisions": 0,
            "selected_skill_ids": [],
        }

    episodes = list(_read_jsonl(evidence_dir / "episodes.jsonl"))
    kinds: Counter[str] = Counter()
    selected_skills: set[str] = set()
    for row in _read_jsonl(evidence_dir / "events.jsonl"):
        kind = str(row.get("kind", ""))
        kinds[kind] += 1
        if kind == "AGENT_PROPOSAL_SET":
            skill_id = (row.get("payload") or {}).get("selected_skill_id")
            if skill_id:
                selected_skills.add(str(skill_id))
    return {
        "complete": True,
        "episodes": len(episodes),
        "agent_decisions": kinds["AGENT_DECISION"],
        "environment_steps": kinds["ENVIRONMENT_STEP"],
        "selected_skill_ids": sorted(selected_skills),
        "manifest_sha256": _file_sha256(evidence_dir / "manifest.json"),
        "events_sha256": _file_sha256(evidence_dir / "events.jsonl"),
        "episodes_sha256": _file_sha256(evidence_dir / "episodes.jsonl"),
    }


def _microcontroller_rows(path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        grouped[str(row.get("game"))].append(row)

    result: dict[str, dict[str, Any]] = {}
    for game, rows in sorted(grouped.items()):
        snapshots = {str(row.get("snapshot_id")) for row in rows}
        split_mode_cells = {
            (str(row.get("split")), str(row.get("mode"))) for row in rows
        }
        result[game] = {
            "matched_snapshots": len(snapshots),
            "treatment_trajectories": len(rows),
            "complete_split_mode_cells": len(split_mode_cells),
            "invalid_trajectories": sum(
                row.get("status") != "INTERVENTION_OBSERVED" for row in rows
            ),
        }
    return result


def build_phase1_six_game_four_target_audit(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    paths = {
        "phase1_matrix": root / "configs/phase1_skill_internal_matrix.json",
        "source_microcontroller_config": root / "configs/source_microcontroller_v1.json",
        "phase1_execution_summary": root / "docs/results/phase1_skill_execution_sets_v1_summary.json",
        "source_gate_diagnosis": root / "docs/results/source_gate_failure_diagnosis_v1.json",
        "source_microcontroller_summary": root / "docs/results/source_microcontroller_v1_summary.json",
        "source_microcontroller_rows": root / "runs/source_microcontroller_v1_gymv/execution/microcontroller_rows.jsonl",
        "candy_receipt_gate": root / "runs/real_game_to_alfworld_intervention_v1_expanded/source_gate.json",
        "candy_grounder_gate": root / "runs/real_game_to_alfworld_intervention_v1_expanded/source_grounder_gate.json",
        "candy_transfer_gate": root / "runs/real_game_to_alfworld_intervention_v1_expanded/transfer_gate.json",
        "thunder_visual_qualification": root / "docs/results/source_visual_effect_qualification_v1_summary.json",
        "thunder_value_smoke": root / "docs/results/source_neurosymbolic_value_smoke_v1_summary.json",
        "four_target_summary": root / "docs/results/search_automaton_transfer_v16_summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing audit inputs: " + ", ".join(missing))

    matrix = _read_json(paths["phase1_matrix"])
    micro_config = _read_json(paths["source_microcontroller_config"])
    execution_summary = _read_json(paths["phase1_execution_summary"])
    source_diagnosis = _read_json(paths["source_gate_diagnosis"])
    micro_summary = _read_json(paths["source_microcontroller_summary"])
    candy_receipt = _read_json(paths["candy_receipt_gate"])
    candy_grounder = _read_json(paths["candy_grounder_gate"])
    candy_transfer = _read_json(paths["candy_transfer_gate"])
    thunder_visual = _read_json(paths["thunder_visual_qualification"])
    thunder_value = _read_json(paths["thunder_value_smoke"])
    target_summary = _read_json(paths["four_target_summary"])
    micro_rows = _microcontroller_rows(paths["source_microcontroller_rows"])

    matrix_rows = matrix.get("games") or []
    matrix_games = tuple(str(row.get("game")) for row in matrix_rows)
    if matrix_games != EXPECTED_GAMES:
        raise ValueError(
            f"Phase-1 game identity/order changed: {matrix_games!r}"
        )
    replayable_games = set(map(str, micro_config.get("replayable_games") or []))

    legacy_by_game = {
        str(row.get("game")): row
        for row in source_diagnosis.get("legacy_six_game_discovery") or []
    }
    if set(legacy_by_game) != set(EXPECTED_GAMES):
        raise ValueError("source diagnosis does not cover exactly six Phase-1 games")

    source_games: dict[str, Any] = {}
    for row in matrix_rows:
        game = str(row["game"])
        evidence_dir = root / str(row["authentic_evidence"])
        rollout = _phase1_rollout_summary(evidence_dir)
        legacy = legacy_by_game[game]
        source_stage = {
            "recorded_rollouts_complete": bool(rollout["complete"]),
            "historical_skill_context_present": bool(
                rollout["selected_skill_ids"]
            ),
            "human_hint_exclusion_receipt": (
                (legacy.get("authentic") or {}).get("human_hint_exclusion")
                == "EXCLUDED_WITH_RECEIPT"
            ),
            "matched_intervention_receipts": game in micro_rows,
            "independent_source_ir_frozen": False,
            "independent_source_ir_value_qualified": False,
            "exact_v16_search_automaton_lineage": False,
        }
        diagnostics: list[dict[str, Any]] = []
        if game in micro_rows:
            diagnostics.append({
                "experiment": "JOINT_PROGRESS_STALL_MICROCONTROLLER_V1",
                "scope": "FOUR_GYMV_GAMES_JOINT_NOT_INDEPENDENT",
                "status": micro_summary.get("status"),
                "source_gate_passed": bool(
                    (micro_summary.get("gates") or {}).get(
                        "SOURCE_MICROCONTROLLER_SUPPORTED"
                    )
                ),
                **micro_rows[game],
            })
        if game == "candy_crush":
            source_stage["matched_intervention_receipts"] = (
                candy_receipt.get("status") == "SOURCE_GATE_PASSED"
            )
            diagnostics.append({
                "experiment": "CANDY_NATIVE_ACTION_EFFECT_V1",
                "receipt_gate": candy_receipt.get("status"),
                "neural_grounder_gate": candy_grounder.get("status"),
                "transfer_gate": candy_transfer.get("status"),
                "target_conditions_executed": len(
                    candy_transfer.get("conditions_executed") or []
                ),
            })
        if game == "gymv_thunder_force_iii":
            diagnostics.extend((
                {
                    "experiment": "THUNDER_VISUAL_EFFECT_STRUCTURE_V1",
                    "status": thunder_visual.get("verdict"),
                    "source_gate_passed": bool(
                        thunder_visual.get("qualification_passed")
                    ),
                },
                {
                    "experiment": "THUNDER_SKILL_CONTEXT_H8_SMOKE_V1",
                    "status": thunder_value.get("status"),
                    "source_gate_passed": bool(
                        (thunder_value.get("gates") or {}).get(
                            "SOURCE_H8_VALUE_SUPPORTED"
                        )
                    ),
                },
            ))

        source_games[game] = {
            "status": "SOURCE_NOT_QUALIFIED_FOR_TARGET_TRANSFER",
            "rollout": rollout,
            "source_stage": source_stage,
            "best_available_diagnostics": diagnostics,
            "blocking_reason": (
                "No independently induced, blind-value-qualified Phase-1 IR "
                "matches the frozen IR exercised by the four target adapters."
            ),
        }

    target_domains: dict[str, Any] = {}
    raw_domains = target_summary.get("domains") or {}
    if set(raw_domains) != set(TARGETS):
        raise ValueError("four-target summary domain set changed")
    for target in TARGETS:
        domain = raw_domains[target]
        target_domains[target] = {
            "status": domain.get("status"),
            "evidence_tier": domain.get("evidence_tier"),
            "mechanism_gates_passed": bool(domain.get("all_gates_passed")),
            "source_lineage": "SOKOBAN_SEARCH_AUTOMATON_V16",
            "source_artifact_sha256": (
                target_summary.get("source") or {}
            ).get("artifact_sha256"),
            "phase1_lineage": False,
            "conditional_readiness": (
                "Reusable only if a Phase-1 game independently induces and "
                "qualifies an alpha-isomorphic three-action search IR."
            ),
        }

    cells: dict[str, dict[str, Any]] = {}
    for game in EXPECTED_GAMES:
        cells[game] = {}
        for target in TARGETS:
            cells[game][target] = {
                "status": "BLOCKED_AT_SOURCE_GATE_TARGET_NOT_EXECUTED",
                "validated": False,
                "source_gate_passed": False,
                "target_mechanism_available_for_other_source": target_domains[target][
                    "mechanism_gates_passed"
                ],
                "exact_source_artifact_lineage_match": False,
                "target_provider_calls_for_this_phase1_lineage": 0,
            }

    body: dict[str, Any] = {
        "schema_version": "phase1-six-game-four-target-transfer-audit-v1",
        "status": "ZERO_OF_24_PHASE1_TRANSFER_CELLS_VALIDATED",
        "claim_boundary": (
            "Existing target results use the Sokoban V16 source artifact. They "
            "demonstrate target-side transport readiness, not transfer from any "
            "of the six Phase-1 games."
        ),
        "source_games": source_games,
        "target_domains": target_domains,
        "cells": cells,
        "aggregate": {
            "source_games": len(EXPECTED_GAMES),
            "target_domains": len(TARGETS),
            "total_cells": len(EXPECTED_GAMES) * len(TARGETS),
            "phase1_source_qualified_games": sum(
                bool(row["source_stage"]["independent_source_ir_value_qualified"])
                for row in source_games.values()
            ),
            "validated_phase1_transfer_cells": 0,
            "target_mechanism_ready_cells_conditional_on_same_ir": sum(
                bool(target_domains[target]["mechanism_gates_passed"])
                for _game in EXPECTED_GAMES for target in TARGETS
            ),
            "joint_six_game_observational_candidate_passed": bool(
                (micro_summary.get("gates") or {}).get(
                    "SOURCE_MICROCONTROLLER_SUPPORTED"
                )
            ),
            "phase1_execution_inventory": execution_summary.get("totals"),
        },
        "compositional_validation_contract": {
            "purpose": (
                "Avoid 24 redundant target runs without weakening lineage."
            ),
            "required_source_proofs": (
                "Each of six games independently induces the same alpha-isomorphic "
                "IR and passes blind matched intervention/value controls."
            ),
            "required_link_proof": (
                "Canonicalized per-game IR hashes and routing truth tables match "
                "the frozen IR used by every target adapter."
            ),
            "required_target_proofs": (
                "The common frozen IR passes paired authentic/control evaluation "
                "once per target with target-native neural grounding."
            ),
            "current_source_proofs_passed": 0,
            "current_target_mechanism_proofs_passed": sum(
                bool(target_domains[target]["mechanism_gates_passed"])
                for target in TARGETS
            ),
            "authorized_6x4_claim": False,
        },
        "next_gate": {
            "name": "PHASE1_PER_GAME_COMMON_IR_SOURCE_QUALIFICATION",
            "target_data_must_remain_unread": True,
            "required_actions": [
                "EXPLORE_UNTRIED",
                "BACKTRACK_REPLAN",
                "COMMIT_VERIFY",
            ],
            "required_controls": [
                "EVENT_BINDING_PERMUTED",
                "LEDGER_BLIND",
                "STATIC_ACTION",
                "HASH_RANDOM",
            ],
            "stop_rule": (
                "Do not run a Phase-1 target cell until that game's source IR "
                "passes blind qualification and held-out matched value gates."
            ),
        },
        "input_sha256": {
            name: _file_sha256(path) for name, path in sorted(paths.items())
        },
    }
    return body | {"audit_sha256": _stable_hash(body)}

