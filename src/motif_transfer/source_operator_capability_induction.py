"""Induce target-independent VM capabilities from sealed source evidence.

This module deliberately does not import an AGQA compiler and does not inspect
target examples.  It translates *validated structural facts* in source
artifacts into conservative operator capabilities.  The translation rules are
generic (cardinality, order, guarded branching, and effect ranking), and every
authorization carries its source lineage and gate receipts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import stable_hash


QUALIFIED_FUNCTION = "SOURCE_DOMAIN_FUNCTION_QUALIFIED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class CapabilityEvidence:
    operation: str
    authorized: bool
    structural_rule: str
    evidence_classes: tuple[str, ...]
    receipt_sha256s: tuple[str, ...]
    source_support: int
    reason: str


def _capability(
    operation: str,
    *,
    authorized: bool,
    rule: str,
    classes: Sequence[str] = (),
    receipts: Sequence[str] = (),
    support: int = 0,
    reason: str,
) -> CapabilityEvidence:
    return CapabilityEvidence(
        operation=operation,
        authorized=authorized,
        structural_rule=rule,
        evidence_classes=tuple(sorted(set(classes))),
        receipt_sha256s=tuple(sorted(set(receipts))),
        source_support=int(support),
        reason=reason,
    )


def _confirmed_binding_evidence(
    artifact: Mapping[str, Any], confirmation: Mapping[str, Any],
) -> tuple[bool, int, tuple[str, ...]]:
    gates = confirmation.get("gates") or {}
    metrics = confirmation.get("metrics") or {}
    artifact_hash = str(artifact.get("artifact_sha256") or "")
    required = (
        confirmation.get("status") == "SOURCE_GOAL_ACQUISITION_FRESH_VALIDATED",
        confirmation.get("source_gate_passed") is True,
        confirmation.get("claim_boundary") == "FRESH_SOURCE_ONLY;NO_TARGET_EVIDENCE",
        confirmation.get("artifact_sha256") == artifact_hash,
        artifact.get("target_data_read") is False,
        artifact.get("named_controller_template_used") is False,
        all(gates.get(name) is True for name in (
            "heldout_transition_conformance",
            "authentic_beats_shuffled_effect_conformance",
            "binding_onset_predicts_relation",
            "authentic_effect_binding_exact",
            "shuffled_effect_binding_rejected",
            "single_positive_binding_cardinality_induced",
            "source_only_lineage",
            "no_named_controller_template",
        )),
        float(metrics.get("binding_to_relation_precision", 0.0)) == 1.0,
        int(metrics.get("shuffled_effect_bindings", -1)) == 0,
    )
    support = min(
        int(metrics.get("binding_onsets", 0)),
        int(metrics.get("binding_to_relation_transitions", 0)),
    )
    receipts = tuple(filter(None, (
        artifact_hash,
        str(confirmation.get("source_dataset_sha256") or ""),
    )))
    return all(required) and support > 0, support, receipts


def _qualified_temporal_evidence(
    manifest: Mapping[str, Any], program_rows: Sequence[tuple[Mapping[str, Any], str]],
) -> dict[str, Any]:
    qualified: list[Mapping[str, Any]] = []
    receipts: list[str] = []
    total_examples = 0
    authentic_correct = 0
    shuffled_correct = 0
    endpoint_values: set[int] = set()
    categorical_guard_sets = 0
    multi_preconditions = 0
    branches = 0
    for wrapper, file_hash in program_rows:
        program = wrapper.get("source_function_program") or {}
        gates = program.get("qualification_gates") or {}
        authentic = program.get("qualification_metrics") or {}
        shuffled = program.get("qualification_shuffled_effect_metrics") or {}
        valid = (
            program.get("status") == QUALIFIED_FUNCTION
            and program.get("target_data_read") is False
            and program.get("source_identity_used_as_feature") is False
            and all(gates.values())
            and int(authentic.get("examples", 0)) > 0
            and float(authentic.get("accuracy", 0.0)) > float(shuffled.get("accuracy", 1.0))
        )
        if not valid:
            continue
        qualified.append(program)
        receipts.extend(filter(None, (file_hash, str(program.get("program_sha256") or ""))))
        total_examples += int(authentic["examples"])
        authentic_correct += int(authentic.get("correct", 0))
        shuffled_correct += int(shuffled.get("correct", 0))
        endpoint_values.update(int(v) for v in (program.get("shared_ir") or {}).get(
            "effect_endpoints", {}
        ).values())
        operators = program.get("operators") or []
        multi_preconditions += sum(
            len((operator.get("preconditions") or {})) >= 2 for operator in operators
        )
        graph = (program.get("transition_graph") or {}).get("transitions") or []
        guards = {str(row.get("guard") or "") for row in graph}
        if {"OBSERVED_EFFECT_HIGH", "OBSERVED_EFFECT_LOW", "OBSERVED_EFFECT_UNKNOWN"} <= guards:
            categorical_guard_sets += 1
        branches += len(graph)
    return {
        "source_only_manifest": "NO_TARGET" in str(manifest.get("claim_boundary") or ""),
        "qualified_programs": len(qualified),
        "total_examples": total_examples,
        "authentic_correct": authentic_correct,
        "shuffled_correct": shuffled_correct,
        "effect_endpoints": sorted(endpoint_values),
        "categorical_guard_sets": categorical_guard_sets,
        "multi_preconditions": multi_preconditions,
        "transition_branches": branches,
        "receipts": tuple(sorted(set(receipts))),
    }


def induce_capabilities(
    *,
    acquisition_artifact: Mapping[str, Any],
    acquisition_confirmation: Mapping[str, Any],
    temporal_manifest: Mapping[str, Any],
    temporal_programs: Sequence[tuple[Mapping[str, Any], str]],
) -> dict[str, Any]:
    """Return a source-only capability artifact; target data is not an input."""

    binding_ok, binding_support, binding_receipts = _confirmed_binding_evidence(
        acquisition_artifact, acquisition_confirmation,
    )
    temporal = _qualified_temporal_evidence(temporal_manifest, temporal_programs)
    temporal_ok = (
        temporal["source_only_manifest"]
        and temporal["qualified_programs"] >= 3
        and temporal["authentic_correct"] > temporal["shuffled_correct"]
    )
    ordered_endpoints = len(temporal["effect_endpoints"]) >= 2
    branching = temporal_ok and temporal["categorical_guard_sets"] >= 1
    conjunction = temporal_ok and temporal["multi_preconditions"] >= 1
    temporal_receipts = temporal["receipts"]

    rows = [
        _capability(op, authorized=binding_ok, rule="confirmed_positive_binding_cardinality",
                    classes=("CARDINALITY_CHANGE",), receipts=binding_receipts,
                    support=binding_support, reason="fresh held-out binding cardinality beats shuffled effects")
        for op in ("CARDINALITY", "EXISTS", "UNIQUE")
    ]
    rows += [
        _capability(op, authorized=binding_ok, rule="confirmed_typed_relation_binding",
                    classes=("TYPED_RELATION_BINDING",), receipts=binding_receipts,
                    support=binding_support, reason="typed entity-relation binding predicts the next relation update")
        for op in ("FILTER_EQ", "PROJECT")
    ]
    rows += [
        _capability(op, authorized=temporal_ok, rule="qualified_effect_score_ranking",
                    classes=("EFFECT_RANKING",), receipts=temporal_receipts,
                    support=temporal["total_examples"], reason="qualified effect argmax beats shuffled effect binding")
        for op in ("ARGMAX", "COMPARE")
    ]
    rows += [
        _capability(op, authorized=temporal_ok and ordered_endpoints,
                    rule="ordered_multihorizon_effect_observation",
                    classes=("ORDERED_ENDPOINTS", "EFFECT_RANKING"), receipts=temporal_receipts,
                    support=temporal["total_examples"],
                    reason="qualified source functions compare effects at multiple ordered endpoints")
        for op in ("INTERVAL_OF", "TEMPORAL_SELECT", "FIRST", "LAST")
    ]
    rows += [
        _capability("CHOOSE", authorized=branching, rule="guarded_effect_transition_branch",
                    classes=("GUARDED_BRANCH",), receipts=temporal_receipts,
                    support=temporal["transition_branches"], reason="observed-effect guards lead to distinct transitions"),
        _capability("AND", authorized=conjunction, rule="joint_observed_preconditions",
                    classes=("CONJUNCTIVE_GUARD",), receipts=temporal_receipts,
                    support=temporal["multi_preconditions"], reason="source operator fires only under jointly satisfied measured preconditions"),
        _capability("NOT", authorized=branching, rule="complementary_observed_effect_guard",
                    classes=("COMPLEMENTARY_GUARD",), receipts=temporal_receipts,
                    support=temporal["categorical_guard_sets"], reason="source transition graph separates satisfied, failed, and unknown guards"),
        _capability("XOR", authorized=branching, rule="mutually_exclusive_categorical_guards",
                    classes=("EXCLUSIVE_GUARDS",), receipts=temporal_receipts,
                    support=temporal["categorical_guard_sets"], reason="high/low/unknown observations are mutually exclusive source branches"),
    ]
    # No source artifact demonstrates merging two independently grounded sets.
    rows += [
        _capability(op, authorized=False, rule="independent_set_composition_required",
                    reason="no qualified source intervention evidence", support=0)
        for op in ("UNION", "INTERSECTION", "ARGMIN")
    ]
    capabilities = {row.operation: asdict(row) for row in rows}
    authorized = sorted(name for name, row in capabilities.items() if row["authorized"])
    binding_edges = {
        ("FILTER_EQ", "UNIQUE"), ("UNIQUE", "PROJECT"),
        ("FILTER_EQ", "EXISTS"), ("PROJECT", "FILTER_EQ"),
        ("PROJECT", "EXISTS"), ("UNIQUE", "COMPARE"),
        ("PROJECT", "COMPARE"), ("PROJECT", "CHOOSE"),
    } if binding_ok else set()
    temporal_edges = {
        ("INTERVAL_OF", "TEMPORAL_SELECT"),
        ("PROJECT", "INTERVAL_OF"), ("ARGMAX", "INTERVAL_OF"),
        ("TEMPORAL_SELECT", "UNIQUE"), ("TEMPORAL_SELECT", "EXISTS"),
        ("TEMPORAL_SELECT", "FIRST"), ("TEMPORAL_SELECT", "LAST"),
        ("FILTER_EQ", "FIRST"), ("FILTER_EQ", "LAST"),
        ("EXISTS", "FIRST"), ("EXISTS", "LAST"),
        ("FIRST", "UNIQUE"), ("LAST", "UNIQUE"),
        ("FIRST", "CHOOSE"), ("LAST", "CHOOSE"),
        ("FILTER_EQ", "ARGMAX"), ("PROJECT", "ARGMAX"),
        ("ARGMAX", "COMPARE"), ("COMPARE", "CHOOSE"),
        ("EXISTS", "COMPARE"),
    } if temporal_ok else set()
    guarded_edges = {
        ("EXISTS", "AND"), ("EXISTS", "XOR"),
        ("XOR", "FIRST"), ("XOR", "LAST"),
    } if branching and conjunction else set()
    composition_edges = sorted(binding_edges | temporal_edges | guarded_edges)
    body = {
        "schema_version": "source-induced-typed-operator-capabilities-v1",
        "status": "SOURCE_CAPABILITIES_INDUCED" if binding_ok and temporal_ok else "SOURCE_CAPABILITIES_ABSTAINED",
        "induction_authority": "SEALED_SOURCE_INTERVENTION_AND_TRANSITION_ARTIFACTS_ONLY",
        "target_data_read": False,
        "binding_evidence_passed": binding_ok,
        "temporal_evidence_passed": temporal_ok,
        "source_evidence_summary": temporal,
        "authorized_operators": authorized,
        "authorized_compositions": [list(edge) for edge in composition_edges],
        "composition_evidence": {
            "confirmed_relation_binding_transition_graph": [list(edge) for edge in sorted(binding_edges)],
            "qualified_multihorizon_guarded_transition_graph": [list(edge) for edge in sorted(temporal_edges)],
            "exclusive_and_conjunctive_guard_graph": [list(edge) for edge in sorted(guarded_edges)],
        },
        "capabilities": capabilities,
    }
    body["artifact_sha256"] = stable_hash(body)
    return body


def induce_from_paths(
    *, acquisition_artifact_path: Path, acquisition_confirmation_path: Path,
    temporal_manifest_path: Path,
) -> dict[str, Any]:
    manifest = _load(temporal_manifest_path)
    # .../<repo>/configs/<freeze>/manifest.json.  Resolve first so behavior is
    # identical for relative and absolute CLI inputs.
    root = temporal_manifest_path.resolve().parents[3]
    programs = []
    for receipt in manifest.get("source_receipts") or ():
        path = root / str(receipt["program_path"])
        actual_hash = _sha256(path)
        if actual_hash != receipt.get("program_file_sha256"):
            raise ValueError(f"sealed source program hash mismatch: {path}")
        programs.append((_load(path), actual_hash))
    return induce_capabilities(
        acquisition_artifact=_load(acquisition_artifact_path),
        acquisition_confirmation=_load(acquisition_confirmation_path),
        temporal_manifest=manifest,
        temporal_programs=programs,
    )


__all__ = ["CapabilityEvidence", "induce_capabilities", "induce_from_paths"]
