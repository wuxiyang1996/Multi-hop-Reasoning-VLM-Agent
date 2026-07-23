"""Receipt-grounded, non-semantic reasoning motifs for far-domain transfer.

The protocol event alphabet is deliberately *not* a skill ontology.  A motif
retains the source policy's observed decision structure (partition/topology,
proposal-set cardinality, selection, post-transition update, and replay-fork
receipts).  Agent-authored prose and legacy mega-skill labels remain untrusted
lineage metadata and never determine admission.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


PROTOCOL_EVENT_ALPHABET = (
    "PROPOSE", "EXECUTE", "OBSERVE", "UPDATE", "BRANCH", "TERMINATE",
)


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _require_digest(value: str, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase sha256 digest")


@dataclass(frozen=True)
class LegacyMegaSkillLineage:
    """An old mega-skill is an index/prior, never executable evidence."""

    mega_skill_id: str
    source_artifact_sha256: str
    member_refs: tuple[str, ...]
    legacy_template_signature: str
    authority: str = "LINEAGE_RETRIEVAL_ONLY"

    @classmethod
    def from_record(
        cls, record: Mapping[str, Any], *, source_artifact_sha256: str,
    ) -> "LegacyMegaSkillLineage":
        _require_digest(source_artifact_sha256, "source_artifact_sha256")
        members = tuple(
            f"{row.get('task', '')}::{row.get('skill_id', '')}"
            for row in record.get("members") or ()
        )
        return cls(
            mega_skill_id=str(record.get("mega_skill_id") or ""),
            source_artifact_sha256=source_artifact_sha256,
            member_refs=members,
            legacy_template_signature=str(
                record.get("template_signature")
                or record.get("reasoning_plan")
                or ""
            ),
        )

    def validate(self) -> None:
        if not self.mega_skill_id:
            raise ValueError("legacy mega-skill identity is empty")
        _require_digest(self.source_artifact_sha256, "source_artifact_sha256")
        if self.authority != "LINEAGE_RETRIEVAL_ONLY":
            raise ValueError("legacy mega-skill cannot gain execution authority")


@dataclass(frozen=True)
class DecisionStepSignature:
    """Observable policy-control shape; no action or domain semantics."""

    transition_id: str
    proposal_count: int | None
    proposal_decision: str | None
    selected_ordinal: int | None
    post_verdict: str | None
    continuation_decision: str | None
    proposal_receipt_sha256: str | None
    verdict_receipt_sha256: str | None

    def transferable_view(self) -> Mapping[str, Any]:
        return {
            "proposal_count": self.proposal_count,
            "proposal_decision": self.proposal_decision,
            "selected_ordinal": self.selected_ordinal,
            "post_verdict": self.post_verdict,
            "continuation_decision": self.continuation_decision,
        }


@dataclass(frozen=True)
class CausalMotifNode:
    node_id: str
    steps: tuple[DecisionStepSignature, ...]

    def transferable_view(self) -> Mapping[str, Any]:
        return {
            "node_id": self.node_id,
            "steps": [item.transferable_view() for item in self.steps],
        }


@dataclass(frozen=True)
class CausalMotifEdge:
    source_node_id: str
    target_node_id: str
    intervention_receipt_sha256s: tuple[str, ...]
    # Retained for audit only. It is never part of the verified fingerprint.
    untrusted_agent_kind: str
    untrusted_agent_claim: Mapping[str, Any]

    def verified_view(self) -> Mapping[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "n_intervention_receipts": len(self.intervention_receipt_sha256s),
            "intervention_receipt_sha256s": list(self.intervention_receipt_sha256s),
        }


@dataclass(frozen=True)
class ReceiptGroundedCausalMotif:
    motif_id: str
    source_hypothesis_hash: str
    source_program_hash: str
    source_reasoning_trace_sha256: str | None
    nodes: tuple[CausalMotifNode, ...]
    edges: tuple[CausalMotifEdge, ...]
    legacy_lineage: tuple[LegacyMegaSkillLineage, ...] = ()
    schema_version: int = 1

    def validate(self) -> None:
        _require_digest(self.source_hypothesis_hash, "source_hypothesis_hash")
        _require_digest(self.source_program_hash, "source_program_hash")
        if self.source_reasoning_trace_sha256 is not None:
            _require_digest(
                self.source_reasoning_trace_sha256,
                "source_reasoning_trace_sha256",
            )
        if len(self.nodes) < 2:
            raise ValueError("a causal motif requires at least two nodes")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)) or any(not item for item in node_ids):
            raise ValueError("causal motif node IDs must be non-empty and unique")
        transition_ids = [
            step.transition_id for node in self.nodes for step in node.steps
        ]
        if not transition_ids or len(transition_ids) != len(set(transition_ids)):
            raise ValueError("transition receipts must be non-empty and unique")
        for transition_id in transition_ids:
            _require_digest(transition_id, "transition_id")
        known_nodes = set(node_ids)
        for edge in self.edges:
            if edge.source_node_id not in known_nodes or edge.target_node_id not in known_nodes:
                raise ValueError("causal motif edge references an unknown node")
            for receipt in edge.intervention_receipt_sha256s:
                _require_digest(receipt, "intervention_receipt_sha256")
        for lineage in self.legacy_lineage:
            lineage.validate()

    def protocol_projection(self) -> tuple[str, ...]:
        """The intentionally uniform logging layer, never the skill itself."""
        events: list[str] = []
        for node in self.nodes:
            for step in node.steps:
                events.extend(("PROPOSE", "EXECUTE", "OBSERVE", "UPDATE"))
                if step.continuation_decision in {"REPLAN", "ABSTAIN"}:
                    events.append("BRANCH")
        events.append("TERMINATE")
        return tuple(events)

    def transferable_view(self) -> Mapping[str, Any]:
        """Content-free structure exposed to one-shot binding proposals."""
        self.validate()
        return {
            "motif_id": self.motif_id,
            "nodes": [node.transferable_view() for node in self.nodes],
            "edges": [edge.verified_view() for edge in self.edges],
            "source_receipt_refs": {
                "source_hypothesis_hash": self.source_hypothesis_hash,
                "source_program_hash": self.source_program_hash,
                "source_reasoning_trace_sha256": self.source_reasoning_trace_sha256,
            },
            "legacy_lineage_ids": [item.mega_skill_id for item in self.legacy_lineage],
            "claim_limit": (
                "Topology and decision events are observed; source/target semantics and "
                "Agent-authored causal explanations are not verified."
            ),
        }

    def causal_fingerprint(self) -> str:
        """Excludes actions, prose, game predicates, and legacy family names."""
        return _hash({
            "nodes": [node.transferable_view() for node in self.nodes],
            "edges": [edge.verified_view() for edge in self.edges],
        })

    def content_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))


@dataclass(frozen=True)
class MotifAntiCollapseAudit:
    motif_hash: str
    status: str
    checks: Mapping[str, bool]
    failure_codes: tuple[str, ...]
    protocol_projection_sha256: str
    causal_fingerprint_sha256: str
    claim_limit: str


@dataclass(frozen=True)
class MatchedEnvironmentOutcome:
    """One condition in a pre-registered, state-matched environment fork."""

    comparison_id: str
    treatment: str
    initial_state_sha256: str
    prefix_sha256: str
    policy_identity_sha256: str
    budget_sha256: str
    official_success: bool
    official_score: float
    valid_execution: bool = True


@dataclass(frozen=True)
class MatchedContrastReport:
    claim: str
    status: str
    authentic_wins: Mapping[str, int]
    control_wins: Mapping[str, int]
    ties: Mapping[str, int]
    matched_comparisons: int
    failure_codes: tuple[str, ...]
    claim_limit: str


def evaluate_matched_environment_contrasts(
    outcomes: Sequence[MatchedEnvironmentOutcome], *, claim: str,
) -> MatchedContrastReport:
    """Evaluate exact paired wins; do not use an LLM verdict as ground truth.

    This is deliberately a pilot evidence state, not a significance test. A
    paper-level claim still needs a pre-registered sample size and uncertainty
    analysis.
    """
    controls_by_claim = {
        "source_attribution": (
            "skill_disabled", "generic_protocol", "shuffled_topology", "other_source",
        ),
        "target_incremental_value": (
            "target_only", "generic_protocol", "shuffled_topology", "other_source",
        ),
    }
    if claim not in controls_by_claim:
        raise ValueError("unknown matched contrast claim")
    controls = controls_by_claim[claim]
    groups: dict[str, dict[str, MatchedEnvironmentOutcome]] = {}
    failures: list[str] = []
    for row in outcomes:
        if row.treatment not in {"authentic", *controls}:
            continue
        by_treatment = groups.setdefault(row.comparison_id, {})
        if row.treatment in by_treatment:
            failures.append(f"DUPLICATE_TREATMENT:{row.comparison_id}:{row.treatment}")
        by_treatment[row.treatment] = row
    authentic_wins = {name: 0 for name in controls}
    control_wins = {name: 0 for name in controls}
    ties = {name: 0 for name in controls}
    matched = 0
    identity_fields = (
        "initial_state_sha256", "prefix_sha256", "policy_identity_sha256", "budget_sha256",
    )
    for comparison_id, rows in sorted(groups.items()):
        missing = {"authentic", *controls} - set(rows)
        if missing:
            failures.append(
                f"INCOMPLETE_MATCHED_SET:{comparison_id}:{','.join(sorted(missing))}"
            )
            continue
        if not all(row.valid_execution for row in rows.values()):
            failures.append(f"INVALID_EXECUTION:{comparison_id}")
            continue
        authentic = rows["authentic"]
        if any(
            getattr(row, field) != getattr(authentic, field)
            for row in rows.values() for field in identity_fields
        ):
            failures.append(f"MATCH_IDENTITY_MISMATCH:{comparison_id}")
            continue
        matched += 1
        for control in controls:
            candidate = rows[control]
            # Official success is primary; official score breaks equal-success ties.
            authentic_key = (authentic.official_success, authentic.official_score)
            control_key = (candidate.official_success, candidate.official_score)
            if authentic_key > control_key:
                authentic_wins[control] += 1
            elif control_key > authentic_key:
                control_wins[control] += 1
            else:
                ties[control] += 1
    supported = (
        matched > 0
        and not failures
        and all(authentic_wins[name] > control_wins[name] for name in controls)
    )
    contradicted = any(control_wins[name] > authentic_wins[name] for name in controls)
    status = (
        "PILOT_SUPPORTED" if supported
        else "PILOT_CONTRADICTED" if contradicted and not failures
        else "INCONCLUSIVE"
    )
    return MatchedContrastReport(
        claim=claim,
        status=status,
        authentic_wins=authentic_wins,
        control_wins=control_wins,
        ties=ties,
        matched_comparisons=matched,
        failure_codes=tuple(failures),
        claim_limit=(
            "Uses official matched environment outcomes only. PILOT_SUPPORTED is not "
            "a statistical or general cross-domain transfer claim."
        ),
    )


def _step_signature(row: Mapping[str, Any]) -> DecisionStepSignature:
    transition_id = str(row.get("transition_id") or "")
    proposal_receipt = row.get("action_proposal_receipt") or {}
    proposal = proposal_receipt.get("proposal_set") or {}
    proposals = list(proposal.get("proposals") or ())
    selected_id = proposal.get("selected_proposal_id")
    selected_ordinal = next(
        (
            index for index, item in enumerate(proposals)
            if str(item.get("proposal_id")) == str(selected_id)
        ),
        None,
    )
    verdict_receipt = row.get("post_transition_verdict_receipt") or {}
    verdict = verdict_receipt.get("verdict") or {}
    return DecisionStepSignature(
        transition_id=transition_id,
        proposal_count=len(proposals) if proposal else None,
        proposal_decision=str(proposal.get("decision")) if proposal else None,
        selected_ordinal=selected_ordinal,
        post_verdict=str(verdict.get("verdict")) if verdict else None,
        continuation_decision=str(verdict.get("decision")) if verdict else None,
        proposal_receipt_sha256=(
            str(row.get("action_proposal_event_sha256"))
            if row.get("action_proposal_event_sha256") else None
        ),
        verdict_receipt_sha256=(
            str(row.get("post_transition_verdict_event_sha256"))
            if row.get("post_transition_verdict_event_sha256") else None
        ),
    )


def compile_causal_motif(
    graph: Mapping[str, Any], *,
    legacy_lineage: Sequence[LegacyMegaSkillLineage] = (),
) -> ReceiptGroundedCausalMotif:
    """Compile a validated source graph without interpreting its prose."""
    nodes = tuple(
        CausalMotifNode(
            node_id=str(node.get("node_id") or ""),
            steps=tuple(_step_signature(row) for row in node.get("observed_transitions") or ()),
        )
        for node in graph.get("nodes") or ()
    )
    edges = tuple(
        CausalMotifEdge(
            source_node_id=str(edge.get("source_node_id") or ""),
            target_node_id=str(edge.get("target_node_id") or ""),
            intervention_receipt_sha256s=tuple(
                str(item) for item in edge.get("intervention_receipt_sha256s") or ()
            ),
            untrusted_agent_kind=str(edge.get("kind") or ""),
            untrusted_agent_claim=dict(edge.get("agent_claim") or {}),
        )
        for edge in graph.get("edges") or ()
    )
    hypothesis_hash = str(graph.get("source_hypothesis_hash") or "")
    motif = ReceiptGroundedCausalMotif(
        motif_id="causal." + hypothesis_hash[:24],
        source_hypothesis_hash=hypothesis_hash,
        source_program_hash=str(graph.get("source_program_hash") or ""),
        source_reasoning_trace_sha256=(
            str(graph["source_reasoning_trace_sha256"])
            if graph.get("source_reasoning_trace_sha256") else None
        ),
        nodes=nodes,
        edges=edges,
        legacy_lineage=tuple(legacy_lineage),
    )
    motif.validate()
    return motif


def audit_motif_anti_collapse(
    motif: ReceiptGroundedCausalMotif,
) -> MotifAntiCollapseAudit:
    """Fail closed until structure exceeds the generic protocol projection.

    Passing this audit makes a motif a *candidate*, not a verified transferable
    skill. Source attribution and target matched-fork evidence are separate gates.
    """
    motif.validate()
    steps = [step for node in motif.nodes for step in node.steps]
    complete_decision_receipts = all(
        step.proposal_count is not None
        and step.proposal_receipt_sha256 is not None
        and step.verdict_receipt_sha256 is not None
        and step.continuation_decision is not None
        for step in steps
    )
    receipt_grounded_edges = any(
        edge.intervention_receipt_sha256s for edge in motif.edges
    )
    observed_control_variation = (
        len({
            (
                step.proposal_count,
                step.selected_ordinal,
                step.post_verdict,
                step.continuation_decision,
            )
            for step in steps
        }) > 1
        or any(
            step.continuation_decision in {"REPLAN", "ABSTAIN"}
            for step in steps
        )
    )
    topology_is_more_than_event_chain = (
        len(motif.nodes) >= 2
        and bool(motif.edges)
        and any(
            edge.source_node_id != edge.target_node_id
            for edge in motif.edges
        )
    )
    checks = {
        "multiple_receipted_nodes": len(motif.nodes) >= 2 and all(node.steps for node in motif.nodes),
        "complete_closed_loop_decision_receipts": complete_decision_receipts,
        "receipt_grounded_control_edge": receipt_grounded_edges,
        "observed_control_variation": observed_control_variation,
        "topology_exceeds_uniform_event_chain": topology_is_more_than_event_chain,
        # Empirical source attribution is intentionally not inferred here.
        "source_attribution_requires_matched_intervention": False,
        # Empirical target value is intentionally not inferred here.
        "target_incremental_value_requires_matched_forks": False,
    }
    structural_keys = tuple(checks)[:5]
    structurally_specific = all(checks[key] for key in structural_keys)
    failures = tuple(key.upper() for key, passed in checks.items() if not passed)
    return MotifAntiCollapseAudit(
        motif_hash=motif.content_hash(),
        status=(
            "STRUCTURALLY_SPECIFIC_CANDIDATE"
            if structurally_specific else "GENERIC_OR_UNDERDETERMINED"
        ),
        checks=checks,
        failure_codes=failures,
        protocol_projection_sha256=_hash(motif.protocol_projection()),
        causal_fingerprint_sha256=motif.causal_fingerprint(),
        claim_limit=(
            "Structural specificity is not source attribution or transfer evidence; "
            "both require matched environment interventions."
        ),
    )


def motif_conditioning_view(
    motif: ReceiptGroundedCausalMotif, *, treatment: str, seed: int = 0,
) -> Mapping[str, Any]:
    """Create registered source controls without target semantics."""
    if treatment not in {"authentic", "generic_protocol", "shuffled_topology", "receipt_null"}:
        raise ValueError("unknown causal motif treatment")
    authentic = dict(motif.transferable_view())
    if treatment == "authentic":
        view = authentic
    elif treatment == "generic_protocol":
        view = {
            "motif_id": "generic-protocol-control",
            "protocol_events": list(motif.protocol_projection()),
            "claim_limit": "Uniform logging alphabet only; no source skill content.",
        }
    elif treatment == "receipt_null":
        view = {
            **authentic,
            "source_receipt_refs": None,
            "nodes": [
                {
                    **node,
                    "steps": [
                        {**step} for step in node["steps"]
                    ],
                }
                for node in authentic["nodes"]
            ],
            "edges": [
                {**edge, "intervention_receipt_sha256s": []}
                for edge in authentic["edges"]
            ],
        }
    else:
        node_ids = [node["node_id"] for node in authentic["nodes"]]
        if len(node_ids) < 2:
            raise ValueError("shuffled topology requires at least two nodes")
        offset = 1 + int(seed) % (len(node_ids) - 1)
        mapping = {
            node_id: node_ids[(index + offset) % len(node_ids)]
            for index, node_id in enumerate(node_ids)
        }
        view = {
            **authentic,
            "edges": [
                {
                    **edge,
                    "source_node_id": mapping[edge["source_node_id"]],
                    "target_node_id": mapping[edge["target_node_id"]],
                }
                for edge in authentic["edges"]
            ],
        }
    return {
        "treatment": treatment,
        "seed": int(seed),
        "conditioning": view,
        "conditioning_sha256": _hash(view),
    }


__all__ = [
    "CausalMotifEdge", "CausalMotifNode", "DecisionStepSignature",
    "LegacyMegaSkillLineage", "MotifAntiCollapseAudit",
    "MatchedContrastReport", "MatchedEnvironmentOutcome",
    "PROTOCOL_EVENT_ALPHABET", "ReceiptGroundedCausalMotif",
    "audit_motif_anti_collapse", "compile_causal_motif",
    "evaluate_matched_environment_contrasts", "motif_conditioning_view",
]
