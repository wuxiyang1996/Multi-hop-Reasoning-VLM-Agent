"""Source-only qualification gate for a three-way transfer controller.

V15 asks whether source evidence identifies three distinct interventions before
any target-domain adapter is written:

* ready and positive effect -> COMMIT, then VERIFY;
* infeasible branch -> BACKTRACK/REPLAN;
* feasible but untried branch -> EXPLORE.

The gate deliberately does not import WebShop code or inspect target outcomes.
It audits the frozen Sokoban effect program and matched real-game
SWITCH-versus-PERSIST forks, and fails closed when a branch is merely specified
or target-authored rather than source-qualified.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SWITCH = "ALWAYS_SWITCH"
PERSIST = "ALWAYS_PERSIST"
COMMON_CONTINUATION = "COMMON_HASH_CONTINUATION"
FULL_REGIME = "FULL_TREATMENT_REGIME"
REQUIRED_SPLITS = ("qualification", "held_out")
REQUIRED_MODES = (COMMON_CONTINUATION, FULL_REGIME)


def file_sha256(path: Path) -> str:
    """Return a content hash without interpreting the source artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"expected a JSON object at {path}:{line_number}"
                )
            rows.append(payload)
    return rows


def matched_switch_effects(
    rows: Iterable[Mapping[str, Any]], *, horizon: int = 8,
) -> list[dict[str, Any]]:
    """Pair matched SWITCH/PERSIST forks and compute SWITCH - PERSIST return.

    A missing or duplicated intervention is an integrity error, not a neutral
    observation. Other registered controls are ignored because this function's
    estimand is specifically the evidence for a distinct EXPLORE intervention.
    """

    horizon_key = f"h{horizon}"
    grouped: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    for row in rows:
        treatment = str(row.get("treatment"))
        if treatment not in {SWITCH, PERSIST}:
            continue
        if row.get("status") != "INTERVENTION_OBSERVED":
            raise ValueError(
                "matched source row is not an observed intervention: "
                f"{row.get('snapshot_id')} {treatment}"
            )
        key = (
            str(row.get("split")),
            str(row.get("mode")),
            str(row.get("snapshot_id")),
        )
        if treatment in grouped[key]:
            raise ValueError(f"duplicate matched treatment for {key}: {treatment}")
        grouped[key][treatment] = row

    effects: list[dict[str, Any]] = []
    for (split, mode, snapshot_id), treatments in sorted(grouped.items()):
        missing = {SWITCH, PERSIST} - set(treatments)
        if missing:
            raise ValueError(
                f"incomplete matched source cell {(split, mode, snapshot_id)}: "
                f"missing {sorted(missing)}"
            )
        switch_row = treatments[SWITCH]
        persist_row = treatments[PERSIST]
        identity_fields = ("game", "event", "fork_step", "episode_seed")
        mismatched = [
            field for field in identity_fields
            if switch_row.get(field) != persist_row.get(field)
        ]
        if mismatched:
            raise ValueError(
                f"mismatched fork identity for {snapshot_id}: {mismatched}"
            )
        try:
            switch_return = float(switch_row["cumulative_returns"][horizon_key])
            persist_return = float(persist_row["cumulative_returns"][horizon_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid {horizon_key} return for matched cell {snapshot_id}"
            ) from exc
        delta = switch_return - persist_return
        effects.append(
            {
                "split": split,
                "mode": mode,
                "snapshot_id": snapshot_id,
                "game": str(switch_row.get("game")),
                "event": str(switch_row.get("event")),
                "horizon": horizon,
                "switch_return": switch_return,
                "persist_return": persist_return,
                "switch_minus_persist": delta,
                "winner": (
                    "SWITCH" if delta > 0 else "PERSIST" if delta < 0 else "TIE"
                ),
            }
        )
    return effects


def summarize_effects(effects: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in effects:
        grouped[(str(row["split"]), str(row["mode"]))].append(row)

    summaries: dict[str, Any] = {}
    for split in REQUIRED_SPLITS:
        for mode in REQUIRED_MODES:
            rows = grouped.get((split, mode), [])
            deltas = [float(row["switch_minus_persist"]) for row in rows]
            key = f"{split}.{mode.lower()}"
            summaries[key] = {
                "cells": len(rows),
                "switch_better": sum(delta > 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "persist_better": sum(delta < 0 for delta in deltas),
                "mean_switch_minus_persist": (
                    sum(deltas) / len(deltas) if deltas else None
                ),
                "games_with_switch_win": sorted(
                    {str(row["game"]) for row in rows
                     if float(row["switch_minus_persist"]) > 0}
                ),
            }
    return summaries


def _has_rule(program: Mapping[str, Any], *, selection: str) -> bool:
    return any(
        isinstance(rule, Mapping) and rule.get("select") == selection
        for rule in program.get("rules", [])
    )


def build_source_triage_report(
    *,
    sokoban_artifact: Mapping[str, Any],
    sokoban_confirmation: Mapping[str, Any],
    microcontroller_summary: Mapping[str, Any],
    microcontroller_rows: Iterable[Mapping[str, Any]],
    topology_artifact: Mapping[str, Any] | None = None,
    topology_confirmation: Mapping[str, Any] | None = None,
    input_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the source-only V15 gate report from frozen source evidence."""

    program = sokoban_artifact.get("program", {})
    if not isinstance(program, Mapping):
        raise ValueError("Sokoban artifact has no structured program")
    option_counts = sokoban_confirmation.get("optimal_option_counts", {})
    if not isinstance(option_counts, Mapping):
        raise ValueError("Sokoban confirmation has no option counts")
    condition_metrics = sokoban_confirmation.get("condition_metrics", {})
    if not isinstance(condition_metrics, Mapping):
        raise ValueError("Sokoban confirmation has no condition metrics")

    authentic = condition_metrics.get("authentic_effect_guard", {})
    control_names = (
        "commit_availability_only",
        "inverted_effect_guard",
        "position_occupancy_prior",
    )
    control_accuracies = [
        float(condition_metrics[name]["accuracy"])
        for name in control_names
        if isinstance(condition_metrics.get(name), Mapping)
        and "accuracy" in condition_metrics[name]
    ]
    authentic_accuracy = float(authentic.get("accuracy", 0.0))
    commit_qualified = bool(
        sokoban_confirmation.get("source_gate_passed")
        and int(option_counts.get("COMMIT", 0)) > 0
        and int(option_counts.get("POSITION", 0)) > 0
        and len(control_accuracies) == len(control_names)
        and authentic_accuracy > max(control_accuracies)
    )

    # REPLAN appears in control flow, but the frozen source confirmation's
    # action space and labels contain only COMMIT/POSITION. Do not treat a
    # written safety fallback as an intervention-qualified third option.
    replan_labeled_examples = int(option_counts.get("REPLAN_OR_ABSTAIN", 0))
    topology_program = (
        topology_artifact.get("program", {})
        if isinstance(topology_artifact, Mapping) else {}
    )
    topology_refute_rule = bool(
        isinstance(topology_program, Mapping)
        and _has_rule(topology_program, selection="REFUTE_SEQUENCE")
    )
    topology_recognition_gate = bool(
        isinstance(topology_confirmation, Mapping)
        and topology_confirmation.get("source_gate_passed")
    )
    backtrack_qualified = bool(
        _has_rule(program, selection="REPLAN_OR_ABSTAIN")
        and replan_labeled_examples > 0
    )

    effects = matched_switch_effects(microcontroller_rows, horizon=8)
    effect_summary = summarize_effects(effects)
    qualification_common = effect_summary[
        f"qualification.{COMMON_CONTINUATION.lower()}"
    ]
    heldout_common = effect_summary[
        f"held_out.{COMMON_CONTINUATION.lower()}"
    ]
    complete_matched_splits = bool(
        qualification_common["cells"] > 0 and heldout_common["cells"] > 0
    )
    explore_qualified = bool(
        complete_matched_splits
        and qualification_common["switch_better"] > 0
        and heldout_common["switch_better"] > 0
        and qualification_common["mean_switch_minus_persist"] > 0
        and heldout_common["mean_switch_minus_persist"] > 0
    )

    branches = {
        "READY_AND_POSITIVE_EFFECT__COMMIT_VERIFY": {
            "source_qualified": commit_qualified,
            "evidence": {
                "fresh_confirmation_examples": int(
                    sum(int(value) for value in option_counts.values())
                ),
                "optimal_option_counts": dict(option_counts),
                "authentic_accuracy": authentic_accuracy,
                "maximum_control_accuracy": (
                    max(control_accuracies) if control_accuracies else None
                ),
            },
        },
        "INFEASIBLE__BACKTRACK_REPLAN": {
            "source_qualified": backtrack_qualified,
            "evidence": {
                "rule_written_in_source_program": _has_rule(
                    program, selection="REPLAN_OR_ABSTAIN"
                ),
                "intervention_labeled_examples": replan_labeled_examples,
                "topology_refute_sequence_rule_written": topology_refute_rule,
                "topology_recognition_gate_passed": topology_recognition_gate,
                "topology_backtrack_intervention_examples": 0,
                "topology_evidence_boundary": (
                    topology_confirmation.get("claim_boundary")
                    if isinstance(topology_confirmation, Mapping) else None
                ),
                "reason_if_failed": (
                    None if backtrack_qualified else
                    "REPLAN_OR_REFUTE_IS_RECOGNITION_CONTROL_FLOW_NOT_A_"
                    "QUALIFIED_BACKTRACK_INTERVENTION"
                ),
            },
        },
        "FEASIBLE_AND_UNTRIED__EXPLORE": {
            "source_qualified": explore_qualified,
            "evidence": {
                "matched_estimand": "ALWAYS_SWITCH_MINUS_ALWAYS_PERSIST_H8",
                "matched_cells": len(effects),
                "split_mode_summary": effect_summary,
                "prior_controller_status": microcontroller_summary.get("status"),
                "reason_if_failed": (
                    None if explore_qualified else
                    "NO_POSITIVE_SWITCH_EFFECT_IN_BLIND_COMMON_CONTINUATION"
                ),
            },
        },
    }
    gate_passed = all(branch["source_qualified"] for branch in branches.values())
    failed_branches = [
        name for name, branch in branches.items()
        if not branch["source_qualified"]
    ]
    return {
        "schema_version": 1,
        "experiment": "SOURCE_TRIAGE_GATE_V15",
        "status": (
            "SOURCE_TRIAGE_GATE_V15_PASSED"
            if gate_passed else "SOURCE_TRIAGE_GATE_V15_FAILED_CLOSED"
        ),
        "claim_boundary": (
            "SOURCE_ONLY_AUDIT; NO_TARGET_OUTCOMES_READ; NO_WEBSHOP_V15_"
            "CONTROLLER_AUTHORIZED_UNLESS_ALL_THREE_BRANCHES_PASS"
        ),
        "proposed_controller": [
            "INFEASIBLE -> BACKTRACK_REPLAN",
            "FEASIBLE_AND_UNTRIED -> EXPLORE",
            "READY_AND_POSITIVE_EFFECT -> COMMIT_THEN_VERIFY",
        ],
        "source_gate_passed": gate_passed,
        "failed_branches": failed_branches,
        "branches": branches,
        "matched_effect_cells": effects,
        "target_execution": {
            "target_domain": "webshop",
            "target_files_read": [],
            "provider_calls": 0,
            "formal_reserve_opened": False,
            "authorized": gate_passed,
            "decision": (
                "AUTHORIZE_TARGET_ADAPTER"
                if gate_passed else "STOP_BEFORE_TARGET_ADAPTER"
            ),
        },
        "diagnosis": [
            "Sokoban qualifies a two-option COMMIT/POSITION effect guard.",
            "The source confirmation contains no labeled REPLAN/BACKTRACK option.",
            "The topology executor confirms sequence recognition/refutation, but "
            "does not compare a BACKTRACK intervention with alternatives.",
            "Matched real-game SWITCH/PERSIST forks do not identify a recurring "
            "positive EXPLORE effect on the blind common-continuation estimand.",
            "Splitting target POSITION into BACKTRACK and EXPLORE would therefore "
            "be target-authored, not demonstrated source skill transfer.",
        ],
        "next_source_experiment": {
            "name": "RELATIVE_FEASIBILITY_AND_NOVELTY_FORKS_V16",
            "required_treatments": ["BACKTRACK", "EXPLORE_UNTRIED", "COMMIT"],
            "required_design": (
                "REAL_REPLAYABLE_GAME_STATES_WITH_MATCHED_ACTION_FORKS_AND_"
                "BLIND_HELD_OUT_GAMES_OR_LEVELS"
            ),
            "target_adaptation_allowed": (
                "ONLY_AFTER_EACH_BRANCH_HAS_NONZERO_SUPPORT_AND_AUTHENTIC_"
                "ROUTING_BEATS_STATIC_AND_PERMUTED_CONTROLS"
            ),
        },
        "input_provenance": dict(input_provenance or {}),
    }
