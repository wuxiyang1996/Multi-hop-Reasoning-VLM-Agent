"""Anonymous source-controller compilation and target-native execution.

This module keeps two pieces of the video system deliberately separate:

* the *controller* is induced from source-game ``(state, action, effect,
  next_state)`` ledgers and is represented only by content-addressed state
  deltas; and
* the target-native adapter decides how a grounded video candidate is
  attempted and how its observable qualification is encoded as HIGH/LOW.

The universal typed video VM is not claimed as learned.  In particular, names
such as ``PRESENCE`` or ``FIRST_EVENT`` belong to the target-native executor,
not to the anonymous controller compiled here.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .phase3_source_induction import operators_from_program, route_state


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_operator(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute identity from behavior, ignoring every supplied label."""

    preconditions = sorted(
        (
            {"field": str(item["field"]), "value": item["value"]}
            for item in row.get("preconditions") or ()
        ),
        key=lambda item: (item["field"], str(item["value"])),
    )
    state_delta = sorted(
        ([str(item[0]), str(item[1])] for item in row.get("state_delta") or ()),
        key=lambda item: (item[0], item[1]),
    )
    identity = {"preconditions": preconditions, "state_delta": state_delta}
    return {
        "operator_id": f"OP_{stable_hash(identity)[:16]}",
        **identity,
    }


def compile_anonymous_source_controller(
    *, root: Path, lineage_directory: Path,
) -> dict[str, Any]:
    """Compile all qualified source lineages without trusting operator names.

    Operator IDs are recomputed from preconditions and observed state deltas.
    Transitions are counted from held-out closed-loop route sequences, while
    abstention is accepted only when every lineage uses the same fail-closed
    rule and its effect-shuffled controller fails on authentic held-out data.
    """

    directory = lineage_directory if lineage_directory.is_absolute() else root / lineage_directory
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError("no source lineage reports")

    operator_support: Counter[str] = Counter()
    transition_support: Counter[tuple[str, str]] = Counter()
    canonical_rows: dict[str, dict[str, Any]] = {}
    lineage_receipts: list[dict[str, Any]] = []
    abstention_rules: list[Mapping[str, Any]] = []
    total_heldout = authentic_successes = shuffled_successes = 0

    for path in paths:
        report = _load_object(path)
        program = report.get("authentic_program") or {}
        heldout = report.get("heldout") or {}
        authentic = heldout.get("authentic_closed_loop") or {}
        shuffled = heldout.get("shuffled_program_closed_loop_on_authentic_effects") or {}
        gates = report.get("gates") or {}
        qualified = (
            report.get("status") == "SOURCE_ONLY_INDUCTION_HELDOUT_VALIDATED"
            and program.get("status") == "SOURCE_INDUCED_PROGRAM_QUALIFIED"
            and "NO_TARGET_DATA" in str(program.get("induction_authority"))
            and program.get("operator_vocabulary") == "CONTENT_ADDRESSED_STATE_DELTAS_ONLY"
            and all(bool(value) for value in gates.values())
            and float(authentic.get("success_rate", 0.0)) == 1.0
            and float(shuffled.get("success_rate", 1.0)) == 0.0
        )
        if not qualified:
            lineage_receipts.append({
                "lineage_id": stable_hash({"file_sha256": _file_sha256(path)})[:16],
                "status": "SOURCE_ABSTAINED",
                "file_sha256": _file_sha256(path),
            })
            continue

        supplied_to_canonical: dict[str, str] = {}
        for raw in program.get("operators") or ():
            canonical = _canonical_operator(raw)
            supplied_to_canonical[str(raw.get("operator_id"))] = canonical["operator_id"]
            canonical_rows[canonical["operator_id"]] = canonical

        route_counts = (heldout.get("authentic") or {}).get("operator_route_counts") or {}
        for supplied, count in route_counts.items():
            if str(supplied) not in supplied_to_canonical:
                raise ValueError(f"held-out route references unknown operator in {path}")
            operator_support[supplied_to_canonical[str(supplied)]] += int(count)

        for ledger in authentic.get("per_ledger") or ():
            supplied_route = [str(value) for value in ledger.get("route_operator_ids") or ()]
            route = [supplied_to_canonical[value] for value in supplied_route]
            transition_support.update(zip(route, route[1:]))

        abstention = program.get("abstention_rule") or {}
        abstention_rules.append(abstention)
        total_heldout += int(authentic.get("ledgers", 0))
        authentic_successes += int(authentic.get("successes", 0))
        shuffled_successes += int(shuffled.get("successes", 0))
        lineage_receipts.append({
            "lineage_id": stable_hash({
                "program_sha256": program.get("program_sha256"),
                "file_sha256": _file_sha256(path),
            })[:16],
            "status": "SOURCE_HELDOUT_QUALIFIED",
            "file_sha256": _file_sha256(path),
            "program_sha256": str(program.get("program_sha256")),
            "heldout_ledgers": int(authentic.get("ledgers", 0)),
            "authentic_successes": int(authentic.get("successes", 0)),
            "effect_shuffled_successes": int(shuffled.get("successes", 0)),
        })

    qualified = [row for row in lineage_receipts if row["status"] == "SOURCE_HELDOUT_QUALIFIED"]
    if not qualified:
        raise ValueError("no qualified source lineages")
    first_rule = dict(abstention_rules[0])
    consensus_abstention = all(dict(row) == first_rule for row in abstention_rules)

    operators = []
    for operator_id, row in sorted(canonical_rows.items()):
        operators.append({**row, "heldout_route_support": operator_support[operator_id]})
    transitions = [
        {"from_operator_id": left, "to_operator_id": right, "heldout_support": count}
        for (left, right), count in sorted(transition_support.items())
    ]
    gates = {
        "all_discovered_lineages_qualified": len(qualified) == len(lineage_receipts),
        "at_least_three_independent_source_lineages": len(qualified) >= 3,
        "anonymous_operator_inventory_nonempty": bool(operators),
        "heldout_authentic_closed_loop_exact": authentic_successes == total_heldout > 0,
        "heldout_effect_shuffle_destructive": shuffled_successes == 0 and total_heldout > 0,
        "observed_transition_graph_nonempty": bool(transitions),
        "fail_closed_abstention_consensus": consensus_abstention,
    }
    body = {
        "schema_version": "anonymous-source-video-harness-v1",
        "status": "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED" if all(gates.values()) else "ANONYMOUS_SOURCE_VIDEO_HARNESS_ABSTAINED",
        "authority": "SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_AND_HELDOUT_CONTROLS_ONLY",
        "target_data_read": False,
        "operator_identity": "CONTENT_HASH_OF_PRECONDITIONS_AND_STATE_DELTA;SUPPLIED_NAMES_IGNORED",
        "operators": operators,
        "transitions": transitions,
        "abstention_rule": first_rule if consensus_abstention else {"kind": "ABSTAIN_NO_CONSENSUS"},
        "lineage_receipts": lineage_receipts,
        "heldout_control": {
            "authentic_successes": authentic_successes,
            "effect_shuffled_successes": shuffled_successes,
            "total_ledgers": total_heldout,
        },
        "gates": gates,
        "claim_boundary": {
            "source_induced": "operator instances, transition counts, and abstention behavior",
            "designer_specified": "universal typed VM and target-native grounding/binding",
        },
    }
    body["artifact_sha256"] = stable_hash(body)
    return body


def route_grounded_candidate(
    controller: Mapping[str, Any], *, candidate_qualified: bool,
) -> tuple[str, ...]:
    """Run the anonymous source controller over one target-native candidate.

    The adapter exposes only an observed HIGH/LOW effect.  It does not expose
    the target answer or benchmark identity to the source program.
    """

    if controller.get("status") != "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED":
        return ("ABSTAIN",)
    program = {
        "operators": [
            {
                "operator_id": row["operator_id"],
                "preconditions": row["preconditions"],
                "state_delta": row["state_delta"],
                "discovery_support": 1,
                "discovery_precision": 1.0,
                "qualification_support": 1,
                "qualification_precision": 1.0,
            }
            for row in controller.get("operators") or ()
        ]
    }
    operators = operators_from_program(program)
    initial = {
        "active_presence": "ABSENT", "active_effect": "NONE",
        "has_untried": True, "terminal": False, "suspended": False,
    }
    attempt = route_state(operators, initial)
    if attempt is None:
        return ("ABSTAIN",)
    observed = {
        "active_presence": "PRESENT",
        "active_effect": "HIGH" if candidate_qualified else "LOW",
        "has_untried": False, "terminal": False, "suspended": False,
    }
    decision = route_state(operators, observed)
    if decision is None:
        return (attempt.operator_id, "ABSTAIN")
    disposition = "COMMIT" if candidate_qualified else "FALLBACK"
    return (attempt.operator_id, decision.operator_id, disposition)


__all__ = ["compile_anonymous_source_controller", "route_grounded_candidate"]
