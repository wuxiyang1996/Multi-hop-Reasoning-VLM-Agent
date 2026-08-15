"""Exhaustive receipt relineage from V16 to an equivalent six-game policy."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import stable_hash
from .search_automaton_transfer_v16 import (
    RoutedDecision,
    SourceSearchAutomaton,
)


DECISION_KEYS = frozenset({
    "domain",
    "episode_id",
    "decision_index",
    "target_event",
    "source_action",
    "native_action_id",
    "native_action",
    "admitted",
    "reason",
    "source_artifact_sha256",
    "target_evidence_sha256",
    "receipt_sha256",
})


def walk_routed_decisions(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, Mapping):
        if DECISION_KEYS <= set(value):
            yield {key: value[key] for key in DECISION_KEYS}
        for child in value.values():
            yield from walk_routed_decisions(child)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            yield from walk_routed_decisions(child)


def validate_self_hashed_report(report: Mapping[str, Any]) -> None:
    body = dict(report)
    claimed = body.pop("report_sha256", None)
    if claimed != stable_hash(body):
        raise ValueError("target report self-hash mismatch")


def relineage_decision(
    row: Mapping[str, Any],
    *,
    old_source: SourceSearchAutomaton,
    new_source: SourceSearchAutomaton,
) -> dict[str, Any]:
    old = RoutedDecision(**{key: row[key] for key in DECISION_KEYS})
    if not old.validate():
        raise ValueError("historical routed-decision self-hash mismatch")
    if old.source_artifact_sha256 != old_source.artifact_sha256:
        raise ValueError("historical routed decision has mixed source lineage")
    if old.source_action != old_source.policy.get(old.target_event):
        raise ValueError("historical routed decision violates source policy")
    if old_source.policy != new_source.policy:
        raise ValueError("old and new source policies are not identical")
    body = {key: row[key] for key in DECISION_KEYS if key != "receipt_sha256"}
    body["source_artifact_sha256"] = new_source.artifact_sha256
    relined = body | {"receipt_sha256": stable_hash(body)}
    unchanged = {
        key: old_value
        for key, old_value in row.items()
        if key not in {"source_artifact_sha256", "receipt_sha256"}
    } == {
        key: new_value
        for key, new_value in relined.items()
        if key not in {"source_artifact_sha256", "receipt_sha256"}
    }
    if not unchanged:
        raise AssertionError("relineage changed target behavior")
    if not RoutedDecision(**relined).validate():
        raise AssertionError("relined routed-decision hash is invalid")
    return relined


def summarize_domain_relineage(
    *,
    domain: str,
    target_report: Mapping[str, Any],
    target_report_path: Path,
    supplemental_receipts: Sequence[Mapping[str, Any]],
    supplemental_paths: Sequence[Path],
    old_source: SourceSearchAutomaton,
    new_source: SourceSearchAutomaton,
    evidence_tier: str,
) -> dict[str, Any]:
    validate_self_hashed_report(target_report)
    if target_report.get("source_artifact_sha256") != old_source.artifact_sha256:
        raise ValueError(f"{domain}: target report/source lineage mismatch")
    gates = dict(target_report.get("gates") or {})
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError(f"{domain}: historical target gates did not all pass")
    for receipt in supplemental_receipts:
        if "receipt_sha256" in receipt:
            body = dict(receipt)
            claimed = body.pop("receipt_sha256")
            if claimed != stable_hash(body):
                raise ValueError(f"{domain}: supplemental receipt hash mismatch")
    payloads = [target_report, *supplemental_receipts]
    historical = [row for payload in payloads for row in walk_routed_decisions(payload)]
    if not historical:
        raise ValueError(f"{domain}: no routed decisions available")
    relined = [
        relineage_decision(
            row, old_source=old_source, new_source=new_source
        )
        for row in historical
    ]
    actions = Counter(
        str(row["source_action"])
        for row in relined if row["source_action"] is not None
    )
    event_counts = Counter(str(row["target_event"]) for row in relined)
    historical_outcomes = {
        key: target_report[key]
        for key in (
            "summaries", "paired", "historical_success_counts", "mapping"
        )
        if key in target_report
    }
    body = {
        "domain": domain,
        "status": "PROGRAM_EQUIVALENT_RELINEAGE_PASSED",
        "evidence_tier": evidence_tier,
        "direct_new_target_execution": False,
        "historical_target_status": target_report["status"],
        "historical_target_report_path": str(target_report_path.resolve()),
        "historical_target_report_file_sha256": __import__("hashlib").sha256(
            target_report_path.read_bytes()
        ).hexdigest(),
        "historical_target_report_sha256": target_report["report_sha256"],
        "historical_outcomes": historical_outcomes,
        "historical_outcomes_sha256": stable_hash(historical_outcomes),
        "supplemental_receipt_count": len(supplemental_receipts),
        "supplemental_receipt_set_sha256": stable_hash([
            __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            for path in supplemental_paths
        ]),
        "old_source_artifact_sha256": old_source.artifact_sha256,
        "new_source_artifact_sha256": new_source.artifact_sha256,
        "routed_decisions": len(relined),
        "admitted_decisions": sum(bool(row["admitted"]) for row in relined),
        "source_abstentions": sum(
            row["source_action"] is None for row in relined
        ),
        "source_action_counts": dict(sorted(actions.items())),
        "target_event_counts": dict(sorted(event_counts.items())),
        "relined_receipt_set_sha256": stable_hash([
            row["receipt_sha256"] for row in relined
        ]),
        "historical_target_gates": gates,
        "gates": {
            "old_and_new_symbolic_policies_identical": (
                old_source.policy == new_source.policy
            ),
            "all_historical_target_gates_passed": all(gates.values()),
            "all_historical_route_receipts_valid": True,
            "all_relined_route_receipts_valid": True,
            "all_target_events_admissions_and_native_actions_unchanged": True,
            "all_three_symbolic_actions_exercised": (
                set(actions) == set(new_source.policy.values())
            ),
        },
        "claim_boundary": (
            "NO_NEW_TARGET_EXECUTION;INHERITS_HISTORICAL_TARGET_OUTCOMES_ONLY_"
            "THROUGH_EXHAUSTIVE_IDENTICAL_POLICY_ROUTE_RELINEAGE"
        ),
    }
    if not all(body["gates"].values()):
        body["status"] = "PROGRAM_EQUIVALENT_RELINEAGE_FAILED"
    return body | {"domain_report_sha256": stable_hash(body)}


__all__ = [
    "relineage_decision",
    "summarize_domain_relineage",
    "validate_self_hashed_report",
    "walk_routed_decisions",
]
