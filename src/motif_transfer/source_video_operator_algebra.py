"""Target-independent operator algebra induced from qualified game evidence.

The output vocabulary describes observable structure, never AGQA/CLEVRER DSL
tokens.  Target adapters may bind a primitive only by an exact typed signature.
This module reads source artifacts and held-out source controls only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import stable_hash


@dataclass(frozen=True)
class StructuralPrimitive:
    name: str
    input_types: tuple[str, ...]
    output_type: str
    evidence_family: str
    source_domains: tuple[str, ...]
    support: int
    artifact_sha256s: tuple[str, ...]
    control: str


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _primitive(
    name: str,
    inputs: Sequence[str],
    output: str,
    family: Mapping[str, Any],
    *,
    support: int,
    hashes: Sequence[str],
    control: str,
) -> StructuralPrimitive:
    if support <= 0:
        raise ValueError(f"non-positive source support for {name}")
    return StructuralPrimitive(
        name=name,
        input_types=tuple(inputs),
        output_type=output,
        evidence_family=str(family["family_id"]),
        source_domains=tuple(str(x) for x in family["source_domains"]),
        support=int(support),
        artifact_sha256s=tuple(sorted(set(str(x) for x in hashes if x))),
        control=control,
    )


def _typed_spatial(root: Path, family: Mapping[str, Any]) -> list[StructuralPrimitive]:
    path = root / str(family["report"])
    report = _load(path)
    aggregate = report.get("aggregate") or {}
    gates = report.get("gates") or {}
    if not (
        report.get("status") == "SOURCE_STRUCTURAL_FRESH_VALIDATED"
        and "NO_CROSS_DOMAIN_TARGET_CLAIM" in str(report.get("claim_boundary"))
        and all(gates.values())
        and float(aggregate.get("authentic_binding_accuracy", 0)) == 1.0
        and float(aggregate.get("shuffled_binding_accuracy", 1)) == 0.0
        and int(aggregate.get("correct_program_selections", 0))
        == int(aggregate.get("success_paths", -1))
    ):
        return []
    support = int(aggregate["operator_occurrences"])
    hashes = (_file_sha(path), str(report.get("report_sha256") or ""))
    control = "typed-effect binding shuffle and source-program permutation"
    return [
        _primitive("ENTITY_BIND", ("EntitySet",), "BoundEntity", family,
                   support=support, hashes=hashes, control=control),
        _primitive("STATE_MUTATION", ("BoundEntity", "State"), "State", family,
                   support=support, hashes=hashes, control=control),
        _primitive("RELATION_ADD", ("BoundEntity", "Entity"), "Relation", family,
                   support=support, hashes=hashes, control=control),
        _primitive("GUARDED_SEQUENCE", ("PredicateSet", "TransitionSeq"), "TransitionSeq", family,
                   support=int(aggregate["success_paths"]), hashes=hashes, control=control),
        _primitive("PRECONDITION_DEPENDENCY", ("Effect", "Effect"), "CausalEdge", family,
                   support=int(aggregate["success_paths"]), hashes=hashes, control=control),
    ]


def _goal_acquisition(root: Path, family: Mapping[str, Any]) -> list[StructuralPrimitive]:
    artifact_path = root / str(family["artifact"])
    report_path = root / str(family["report"])
    artifact, report = _load(artifact_path), _load(report_path)
    metrics, gates = report.get("metrics") or {}, report.get("gates") or {}
    if not (
        report.get("status") == "SOURCE_GOAL_ACQUISITION_FRESH_VALIDATED"
        and report.get("source_gate_passed") is True
        and report.get("artifact_sha256") == artifact.get("artifact_sha256")
        and artifact.get("target_data_read") is False
        and artifact.get("named_controller_template_used") is False
        and all(gates.values())
        and int(metrics.get("shuffled_effect_bindings", -1)) == 0
    ):
        return []
    support = min(int(metrics.get("binding_onsets", 0)),
                  int(metrics.get("binding_to_relation_transitions", 0)))
    hashes = (_file_sha(artifact_path), _file_sha(report_path),
              str(artifact.get("artifact_sha256") or ""))
    control = "held-out next-effect shuffle"
    return [
        _primitive("SET_CARDINALITY", ("EntitySet",), "Natural", family,
                   support=support, hashes=hashes, control=control),
        _primitive("PRESENCE", ("EntitySet",), "Truth3", family,
                   support=support, hashes=hashes, control=control),
        _primitive("UNIQUE_BINDING", ("EntitySet",), "BoundEntity", family,
                   support=support, hashes=hashes, control=control),
        _primitive("RELATION_PROJECT", ("BoundEntity", "RelationSet"), "EntitySet", family,
                   support=support, hashes=hashes, control=control),
        _primitive("GOAL_RELATION_TEST", ("RelationSet", "GoalRelation"), "Truth3", family,
                   support=support, hashes=hashes, control=control),
    ]


def _topology(root: Path, family: Mapping[str, Any]) -> list[StructuralPrimitive]:
    artifact_path = root / str(family["artifact"])
    report_path = root / str(family["report"])
    artifact, report = _load(artifact_path), _load(report_path)
    metrics = report.get("condition_metrics") or {}
    canonical = metrics.get("canonical_topology_executor") or {}
    controls = [metrics.get(name) or {} for name in (
        "direction_permuted_executor", "phase_reversed_executor", "sequence_length_marginal"
    )]
    if not (
        report.get("status") == "SOURCE_TOPOLOGY_EXECUTOR_CONFIRMED"
        and report.get("source_gate_passed") is True
        and report.get("artifact_sha256") == artifact.get("artifact_sha256")
        and float(canonical.get("accuracy", 0)) == 1.0
        and all(float(row.get("accuracy", 1)) == 0.0 for row in controls)
    ):
        return []
    support = int(canonical.get("examples", 0))
    hashes = (_file_sha(artifact_path), _file_sha(report_path),
              str(artifact.get("artifact_sha256") or ""))
    control = "direction permutation, phase reversal, and length marginal"
    return [
        _primitive("ORDERED_PATH", ("TransitionSet",), "TransitionSeq", family,
                   support=support, hashes=hashes, control=control),
        _primitive("FIRST_EVENT", ("TransitionSeq",), "Event", family,
                   support=support, hashes=hashes, control=control),
        _primitive("LAST_EVENT", ("TransitionSeq",), "Event", family,
                   support=support, hashes=hashes, control=control),
    ]


def _search(root: Path, family: Mapping[str, Any]) -> list[StructuralPrimitive]:
    artifact_path = root / str(family["artifact"])
    report_path = root / str(family["report"])
    artifact, report = _load(artifact_path), _load(report_path)
    policies = report.get("fresh_policy_metrics") or {}
    authentic = policies.get("authentic_learned_event_policy") or {}
    permuted = policies.get("event_binding_permuted") or {}
    blind = policies.get("ledger_blind_repeat_first") or {}
    if not (
        report.get("artifact_sha256") == artifact.get("artifact_sha256")
        and "NO_TARGET_DATA_READ" in str(report.get("claim_boundary"))
        and float(authentic.get("success_rate", 0)) == 1.0
        and float(permuted.get("success_rate", 1)) == 0.0
        and float(blind.get("success_rate", 1)) < 1.0
    ):
        return []
    support = int(authentic.get("states", 0))
    hashes = (_file_sha(artifact_path), _file_sha(report_path),
              str(artifact.get("artifact_sha256") or ""))
    control = "event-binding permutation and ledger-blind policy"
    return [
        _primitive("COUNTERFACTUAL_REFUTATION", ("ExpectedEffect", "ObservedEffect"), "Truth3", family,
                   support=support, hashes=hashes, control=control),
        _primitive("ALTERNATIVE_ADVANCE", ("CandidateLedger", "Truth3"), "CandidateLedger", family,
                   support=support, hashes=hashes, control=control),
        _primitive("VERIFIED_COMMIT", ("ExpectedEffect", "ObservedEffect"), "Decision", family,
                   support=support, hashes=hashes, control=control),
    ]


def _cyclic(root: Path, family: Mapping[str, Any]) -> list[StructuralPrimitive]:
    path = root / str(family["report"])
    report = _load(path)
    gates = report.get("qualification_gates") or {}
    reserve = report.get("reserve") or {}
    controls = report.get("controls") or {}
    if not (
        report.get("status") == "THIRD_PROGRAM_FAMILY_SOURCE_RESERVE_VALIDATED"
        and all(gates.values())
        and int(reserve.get("correct", 0)) == int(reserve.get("total", -1))
        and bool(reserve.get("all_forks_classified"))
        and int(reserve.get("false_positive_support", -1)) == 0
        and controls.get("both_controls_abstain") is True
    ):
        return []
    support = int(reserve["total"])
    hashes = (_file_sha(path), str(report.get("report_sha256") or ""))
    control = "terminal-label and recovery-binding permutation"
    return [
        _primitive("STATE_EQUIVALENCE", ("State", "State"), "Truth3", family,
                   support=support, hashes=hashes, control=control),
        _primitive("EFFECT_COMPOSE", ("Effect", "Effect"), "Effect", family,
                   support=support, hashes=hashes, control=control),
        _primitive("IDENTITY_TEST", ("Effect",), "Truth3", family,
                   support=support, hashes=hashes, control=control),
        _primitive("INVERSE_RECOVERY", ("Effect",), "Effect", family,
                   support=support, hashes=hashes, control=control),
    ]


def _temporal(root: Path, family: Mapping[str, Any]) -> tuple[list[StructuralPrimitive], list[dict[str, Any]]]:
    path = root / str(family["manifest"])
    manifest = _load(path)
    if not (
        manifest.get("status") == "FROZEN_BEFORE_ANY_RESERVE_PLAN_OR_INTERVENTION_OUTCOME"
        and "NO_TARGET_CLAIM" in str(manifest.get("claim_boundary"))
    ):
        return [], []
    qualified: list[Mapping[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    hashes = [_file_sha(path), str(manifest.get("manifest_sha256") or "")]
    total_examples = correct = shuffled = branches = 0
    qualified_games: list[str] = []
    endpoints: set[int] = set()
    for row in manifest.get("source_receipts") or []:
        program_path = root / str(row["program_path"])
        wrapper = _load(program_path)
        program = wrapper.get("source_function_program") or {}
        metrics = program.get("qualification_metrics") or {}
        shuffled_metrics = program.get("qualification_shuffled_effect_metrics") or {}
        is_qualified = (
            row.get("qualification_status") == "SOURCE_DOMAIN_FUNCTION_QUALIFIED"
            and program.get("status") == "SOURCE_DOMAIN_FUNCTION_QUALIFIED"
            and program.get("target_data_read") is False
            and program.get("source_identity_used_as_feature") is False
            and all((program.get("qualification_gates") or {}).values())
            and float(metrics.get("accuracy", 0)) > float(shuffled_metrics.get("accuracy", 1))
        )
        if not is_qualified:
            abstentions.append({
                "source": row.get("source_game"),
                "status": "SOURCE_ABSTAINED",
                "program_sha256": row.get("source_function_program_sha256"),
            })
            continue
        qualified.append(row)
        qualified_games.append(str(row["source_game"]))
        hashes.extend((_file_sha(program_path), str(program.get("program_sha256") or "")))
        total_examples += int(metrics.get("examples", 0))
        correct += int(metrics.get("correct", 0))
        shuffled += int(shuffled_metrics.get("correct", 0))
        endpoints.update(int(v) for v in (program.get("shared_ir") or {}).get("effect_endpoints", {}).values())
        branches += len((program.get("transition_graph") or {}).get("transitions") or [])
    if len(qualified) < 3 or correct <= shuffled or len(endpoints) < 2:
        return [], abstentions

    # Independently repeated matched forks provide a target-free equality
    # experiment over multi-horizon effect measures.  Authentic pairs share a
    # snapshot/candidate and differ only in repeat index.  The destructive
    # control uses a fixed cyclic candidate derangement within each snapshot.
    authentic_equal = authentic_total = deranged_equal = deranged_total = 0
    rows_root = root / str(family["heldout_rows_root"])
    for game in qualified_games:
        rows_path = rows_root / game / "rows.jsonl"
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            grouped[(str(row["snapshot_id"]), str(row["candidate_id"]))].append(row)
        snapshots: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for (snapshot_id, _), repeats in grouped.items():
            repeats.sort(key=lambda row: int(row["repeat_index"]))
            if len(repeats) < 2:
                continue
            authentic_total += 1
            authentic_equal += (
                repeats[0]["cumulative_returns"] == repeats[1]["cumulative_returns"]
            )
            snapshots[snapshot_id].append(repeats[0])
        for candidates in snapshots.values():
            candidates.sort(key=lambda row: str(row["candidate_id"]))
            if len(candidates) < 2:
                continue
            for index, row in enumerate(candidates):
                other = candidates[(index + 1) % len(candidates)]
                deranged_total += 1
                deranged_equal += (
                    row["cumulative_returns"] == other["cumulative_returns"]
                )
        hashes.append(_file_sha(rows_path))
    authentic_rate = authentic_equal / authentic_total if authentic_total else 0.0
    deranged_rate = deranged_equal / deranged_total if deranged_total else 1.0
    equality_qualified = (
        authentic_total >= 100
        and authentic_rate >= 0.99
        and authentic_rate - deranged_rate >= 0.25
    )
    domains = dict(family)
    domains["source_domains"] = [str(x["source_game"]) for x in qualified]
    control = "held-out full-effect-vector shuffle; failed sources retained as abstentions"
    primitives = [
        _primitive("EFFECT_SCORE", ("Effect" ,), "Scalar", domains,
                   support=total_examples, hashes=hashes, control=control),
        _primitive("EFFECT_ARGMAX", ("EffectSet",), "Effect", domains,
                   support=total_examples, hashes=hashes, control=control),
        _primitive("ORDERED_ENDPOINTS", ("EffectTrace",), "EventSeq", domains,
                   support=total_examples, hashes=hashes, control=control),
        _primitive("INTERVAL_MEASURE", ("Event", "Event"), "Interval", domains,
                   support=total_examples, hashes=hashes, control=control),
        _primitive("GUARDED_BRANCH", ("Truth3", "Transition", "Transition"), "Transition", domains,
                   support=branches, hashes=hashes, control=control),
        _primitive("LOGICAL_NOT", ("Truth3",), "Truth3", domains,
                   support=branches, hashes=hashes, control=control),
        _primitive("LOGICAL_AND", ("Truth3", "Truth3"), "Truth3", domains,
                   support=branches, hashes=hashes, control=control),
        _primitive("LOGICAL_XOR", ("Truth3", "Truth3"), "Truth3", domains,
                   support=branches, hashes=hashes, control=control),
    ]
    if equality_qualified:
        primitives.append(
            _primitive(
                "MEASURE_EQUIVALENCE", ("Measure", "Measure"), "Truth3", domains,
                support=authentic_total, hashes=hashes,
                control=(
                    "matched-repeat effect measures versus fixed candidate derangement; "
                    f"authentic={authentic_equal}/{authentic_total}; "
                    f"deranged={deranged_equal}/{deranged_total}"
                ),
            )
        )
    return primitives, abstentions


_LOADERS = {
    "typed_spatial_report": _typed_spatial,
    "goal_acquisition": _goal_acquisition,
    "topology": _topology,
    "search_automaton": _search,
    "cyclic_identity": _cyclic,
}


def induce_source_video_algebra(*, root: Path, catalog: Mapping[str, Any]) -> dict[str, Any]:
    disclosure = catalog.get("selection_disclosure") or {}
    forbidden = disclosure.get("forbidden_selection_features") or []
    if catalog.get("status") != "FROZEN_SOURCE_SELECTION_RULE_BEFORE_NEW_TARGET_RESERVE":
        raise ValueError("source catalog is not frozen")
    if not forbidden or disclosure.get("new_target_reserves_read") is not False:
        raise ValueError("source-selection authority is not target blind")
    primitives: list[StructuralPrimitive] = []
    abstentions: list[dict[str, Any]] = []
    family_receipts: list[dict[str, Any]] = []
    for family in catalog.get("families") or []:
        kind = str(family.get("evidence_kind") or "")
        if kind == "temporal_manifest":
            rows, family_abstentions = _temporal(root, family)
            abstentions.extend(family_abstentions)
        else:
            loader = _LOADERS.get(kind)
            if loader is None:
                raise ValueError(f"unknown source evidence kind: {kind}")
            rows = loader(root, family)
        family_receipts.append({
            "family_id": family["family_id"],
            "evidence_kind": kind,
            "qualified": bool(rows),
            "primitive_count": len(rows),
        })
        primitives.extend(rows)
    names = [row.name for row in primitives]
    if len(names) != len(set(names)):
        raise ValueError("structural primitive names must be unique")
    required_families = {str(x["family_id"]) for x in catalog.get("families") or []}
    observed_families = {row["family_id"] for row in family_receipts}
    # Composition is induced by type-safe closure over independently qualified
    # primitives.  This does not assert that every cross-family program was
    # observed verbatim; it authorizes only output-to-input connections whose
    # types match exactly.  The matched permuted control rotates output types.
    composition_edges = sorted({
        (left.name, right.name)
        for left in primitives
        for right in primitives
        if left.name != right.name and left.output_type in right.input_types
    })
    body = {
        "schema_version": "source-video-operator-algebra-v1",
        "status": "SOURCE_VIDEO_OPERATOR_ALGEBRA_QUALIFIED"
        if required_families == observed_families and all(x["qualified"] for x in family_receipts)
        else "SOURCE_VIDEO_OPERATOR_ALGEBRA_ABSTAINED",
        "authority": "SOURCE_ARTIFACTS_AND_HELDOUT_SOURCE_CONTROLS_ONLY",
        "target_data_read": False,
        "target_dsl_tokens_used_for_induction": False,
        "catalog_sha256": stable_hash(catalog),
        "catalog_file_sha256": _file_sha(root / "configs/full_video_source_catalog_v1.json"),
        "family_receipts": family_receipts,
        "source_abstentions": abstentions,
        "primitives": [asdict(row) for row in primitives],
        "primitive_names": sorted(names),
        "composition_rule": "EXACT_OUTPUT_TO_INPUT_TYPE_CLOSURE",
        "composition_edges": [list(edge) for edge in composition_edges],
        "composition_edge_count": len(composition_edges),
    }
    body["artifact_sha256"] = stable_hash(body)
    return body


__all__ = ["StructuralPrimitive", "induce_source_video_algebra"]
