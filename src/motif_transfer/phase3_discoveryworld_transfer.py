"""DiscoveryWorld-native grounding for the Phase-3 anonymous source IR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .discoveryworld_applicability_grounder_v4 import (
    TRANSPORT_SUFFIX,
    select_source_safe_candidate,
    source_applicability_complete,
)
from .discoveryworld_sokoban_transfer import (
    DiscoveryWorldGroundedCandidate,
    TARGET_BINDER_SYSTEM_PROMPT,
    TARGET_GROUNDER_SYSTEM_PROMPT,
    binder_prompt_payload,
    evidence_supported,
    grounder_prompt_payload,
    parse_grounded_candidates,
    parse_target_binding,
    positive_commit_effect_witnessed,
    select_candidate,
)
from .discoveryworld_policy import native_action_from_decision, target_native_facts
from .phase3_attempt_runtime import AnonymousAttemptRuntime, RuntimeDecision
from .phase3_source_portfolio import (
    permute_selected_effect_binding,
    select_source_program_portfolio,
)
from .phase3_typed_effect_induction import TYPED_EFFECTS


NEURAL_ONLY = "neural_only"
SOURCE_INDUCED = "source_induced"
SOURCE_PERMUTED = "source_permuted"
GENERIC_SCAFFOLD = "generic_scaffold"
TARGET_NATIVE_CEILING = "target_native_ceiling"
CONDITIONS = (
    NEURAL_ONLY,
    SOURCE_INDUCED,
    SOURCE_PERMUTED,
    GENERIC_SCAFFOLD,
    TARGET_NATIVE_CEILING,
)
MATCHED_CONDITIONS = CONDITIONS[1:]


PHASE3_TARGET_GROUNDER_SYSTEM_PROMPT = TARGET_GROUNDER_SYSTEM_PROMPT.replace(
    "Propose 2-4 compact, diverse native actions that are valid now.",
    "Propose exactly five compact, diverse native actions that are valid now: "
    "four distinct reversible POSITION actions and one bound COMMIT action. "
    "The four POSITION actions are an intervention set, not four paraphrases; "
    "use different native actions or arguments that test different ways of "
    "making the bound commit effect observable.",
) + (
    " Every proposed native action must pass the supplied current-state "
    "preconditions: use only known_actions, listed teleport locations/visible "
    "object UUIDs, and currently movable directions. A rejected action does "
    "not count toward the four POSITION interventions. The payload contains "
    "a phase3_position_action_catalog compiled from those native "
    "preconditions. Copy each POSITION action and arguments exactly from that "
    "catalog; actions outside it will be rejected. Prefer different action "
    "classes before using multiple arguments of one class."
) + (
    " For every candidate also return typed_effect_probabilities with exactly "
    "these four keys: EFFECT_BY_TRANSITION_1, EFFECT_BY_TRANSITION_4, "
    "EFFECT_BY_TRANSITION_8, and EXECUTABLE_TRANSITION_PERSISTENCE. Values "
    "must be neural estimates in [0,1] grounded only in the current target "
    "state and proposed native action. EFFECT_BY_TRANSITION_1 is useful "
    "progress directly after this action; EFFECT_BY_TRANSITION_4 and "
    "EFFECT_BY_TRANSITION_8 are useful progress enabled within a short or "
    "medium target-native follow-up; EXECUTABLE_TRANSITION_PERSISTENCE is "
    "the probability the intervention remains valid and safe through those "
    "follow-ups. Do not use or mention any source game."
) + TRANSPORT_SUFFIX

PHASE3_TARGET_BINDER_SYSTEM_PROMPT = TARGET_BINDER_SYSTEM_PROMPT + (
    " The payload may contain a phase3_target_binding_catalog compiled from "
    "the task's explicit target object type and current target-native facts. "
    "When present, copy target_uuid and target_name exactly from one catalog "
    "entry; use the scientific hypothesis only to choose among those entries. "
    "When phase3_commit_action_catalog is present, copy commit_action exactly "
    "from it; do not replace DROP with PUT or invent a container."
) + (
    " When phase3_acquisition_evidence is present, reassess the hypothesis "
    "from those raw target-native measurements instead of trusting a prior "
    "memory conclusion. For multivariate anomaly tasks, compare complete "
    "measurement vectors jointly: identify the tightest cluster and bind the "
    "remaining outlier. Never infer an anomaly from one component alone."
) + TRANSPORT_SUFFIX

_FORBIDDEN_OUTCOME_FIELDS = frozenset({
    "completed", "completedSuccessfully", "score", "maxScore",
    "scoreNormalized", "scoreCard", "official_success", "evaluation",
})

_PROTEOMICS_SUBJECT_RE = re.compile(
    r"investigate the (?P<subject>[^.\n]+)", re.IGNORECASE,
)
_PROTEOMICS_VALUE_RE = re.compile(
    r"-\s*(?P<feature>Protein\s+[^:]+):\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.IGNORECASE,
)


def extract_phase3_acquisition_evidence(
    reference: Mapping[str, Any], fork_step: int,
) -> tuple[Mapping[str, Any], ...]:
    """Compile raw pre-fork instrument readings without evaluator fields.

    The compiler only reads policy-visible ``last_action_message`` strings.
    Repeated measurements of the same entity are deduplicated, and no
    conclusion, task-progress value, transition outcome, or source artifact is
    copied.  The neural target binder remains responsible for interpreting the
    resulting target-native vectors.
    """

    rows: dict[str, Mapping[str, Any]] = {}
    steps = reference.get("steps") or ()
    for row in steps[:fork_step]:
        if not isinstance(row, Mapping):
            continue
        after = row.get("after_target_native_facts") or {}
        if not isinstance(after, Mapping):
            continue
        message = str(after.get("last_action_message") or "")
        subject_match = _PROTEOMICS_SUBJECT_RE.search(message)
        values = {
            match.group("feature").strip(): float(match.group("value"))
            for match in _PROTEOMICS_VALUE_RE.finditer(message)
        }
        if subject_match is None or not values:
            continue
        subject = subject_match.group("subject").strip()
        rows[subject.lower()] = {
            "subject": subject,
            "measurement_vector": values,
            "evidence_episode_step": int(row.get("episode_step") or 0),
            "evidence_source": "POLICY_VISIBLE_INSTRUMENT_MESSAGE",
        }
    return tuple(rows[key] for key in sorted(rows))


def phase3_acquisition_relations(
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Build an answer-free relation graph over target measurement vectors."""

    relations = []
    for left_index, left in enumerate(evidence):
        left_values = left.get("measurement_vector") or {}
        if not isinstance(left_values, Mapping):
            continue
        for right in evidence[left_index + 1:]:
            right_values = right.get("measurement_vector") or {}
            if not isinstance(right_values, Mapping):
                continue
            shared = sorted(set(left_values) & set(right_values))
            if not shared:
                continue
            squared_distance = sum(
                (float(left_values[key]) - float(right_values[key])) ** 2
                for key in shared
            )
            relations.append({
                "left_subject": str(left.get("subject") or ""),
                "right_subject": str(right.get("subject") or ""),
                "shared_features": shared,
                "squared_euclidean_distance": round(squared_distance, 8),
            })
    return tuple(sorted(
        relations,
        key=lambda row: (
            row["squared_euclidean_distance"], row["left_subject"],
            row["right_subject"],
        ),
    ))


def phase3_acquisition_outlier_candidates(
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Infer a unique one-outlier candidate from a three-vector relation graph.

    This is deliberately fail-closed: incomplete vectors, non-three-way tasks,
    or a tied nearest-pair relation produce no candidate.  It never consumes an
    evaluator label or a task outcome.
    """

    subjects = {
        str(row.get("subject") or "").strip()
        for row in evidence if str(row.get("subject") or "").strip()
    }
    relations = phase3_acquisition_relations(evidence)
    if len(subjects) != 3 or len(relations) != 3:
        return ()
    minimum = float(relations[0]["squared_euclidean_distance"])
    if sum(
        abs(float(row["squared_euclidean_distance"]) - minimum) <= 1e-12
        for row in relations
    ) != 1:
        return ()
    tight_pair = {
        str(relations[0]["left_subject"]),
        str(relations[0]["right_subject"]),
    }
    remaining = sorted(subjects - tight_pair)
    return tuple(remaining) if len(remaining) == 1 else ()


def outcome_blind_target_native_facts(observation) -> dict[str, Any]:
    """Remove evaluator/completion fields before any Phase-3 neural call."""

    facts = target_native_facts(observation)
    progress = []
    for row in facts.get("task_progress") or ():
        if not isinstance(row, Mapping):
            continue
        progress.append({
            key: value for key, value in row.items()
            if key not in _FORBIDDEN_OUTCOME_FIELDS
        })
    facts["task_progress"] = progress

    def assert_clean(value: Any, path: str = "target_native_facts") -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key) in _FORBIDDEN_OUTCOME_FIELDS:
                    raise ValueError(f"formal outcome field leaked at {path}.{key}")
                assert_clean(nested, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                assert_clean(nested, f"{path}[{index}]")

    assert_clean(facts)
    return facts


def phase3_binder_prompt_payload(
    observation, *, memory: str, hypotheses: Sequence[str],
    acquisition_evidence: Sequence[Mapping[str, Any]] = (),
    schema_error: str | None = None,
) -> dict[str, Any]:
    payload = binder_prompt_payload(
        observation, memory=memory, hypotheses=hypotheses,
        schema_error=schema_error,
    )
    payload["target_native_facts"] = outcome_blind_target_native_facts(observation)
    required_type, catalog = phase3_target_binding_catalog(
        observation, acquisition_evidence=acquisition_evidence,
    )
    payload["phase3_required_target_object_type"] = required_type
    payload["phase3_target_binding_catalog"] = list(catalog)
    payload["phase3_target_binding_catalog_is_exhaustive"] = bool(required_type)
    commit_catalog = phase3_commit_action_catalog(observation)
    payload["phase3_commit_action_catalog"] = list(commit_catalog)
    payload["phase3_commit_action_catalog_is_exhaustive"] = bool(commit_catalog)
    payload["phase3_acquisition_evidence"] = list(acquisition_evidence)
    payload["phase3_acquisition_relations"] = list(
        phase3_acquisition_relations(acquisition_evidence)
    )
    payload["phase3_acquisition_outlier_candidates"] = list(
        phase3_acquisition_outlier_candidates(acquisition_evidence)
    )
    payload["phase3_acquisition_rule"] = (
        "ONE_OUTLIER_AMONG_THREE_VECTORS_REMAINS_AFTER_UNIQUE_TIGHTEST_PAIR"
    )
    payload["phase3_acquisition_evidence_is_outcome_blind"] = True
    if acquisition_evidence:
        # The raw pre-fork observations are stronger evidence than a lossy and
        # potentially mistaken rollout summary.  Remove that conclusion from
        # the neural input instead of asking the model to overcome anchoring.
        payload["untrusted_scientific_memory"] = ""
        payload["running_hypotheses"] = []
        payload["phase3_prior_conclusion_superseded_by_raw_evidence"] = True
    payload["formal_outcome_fields_visible"] = False
    return payload


def phase3_target_binding_catalog(
    observation, *, acquisition_evidence: Sequence[Mapping[str, Any]] = (),
) -> tuple[str | None, tuple[Mapping[str, Any], ...]]:
    """Compile exact operands for an object type explicitly named by a task."""

    facts = outcome_blind_target_native_facts(observation)
    descriptions = " ".join(
        str(row.get("description") or "")
        for row in facts.get("task_progress") or ()
        if isinstance(row, Mapping)
    ).lower()
    required_type = (
        "statue" if str(observation.scenario).lower() == "proteomics"
        and "statue" in descriptions else None
    )
    outliers = phase3_acquisition_outlier_candidates(acquisition_evidence)
    objects: dict[tuple[int, str], Mapping[str, Any]] = {}
    for key in ("accessible_objects", "salient_relative_objects"):
        for row in facts.get(key) or ():
            if not isinstance(row, Mapping):
                continue
            uuid = row.get("uuid")
            name = str(row.get("name") or "")
            if not isinstance(uuid, int) or isinstance(uuid, bool) or not name:
                continue
            if required_type and required_type not in name.lower():
                continue
            if outliers and not any(
                subject.lower() in name.lower() for subject in outliers
            ):
                continue
            objects[(uuid, name)] = {"uuid": uuid, "name": name}
    return required_type, tuple(
        objects[key] for key in sorted(objects, key=lambda row: (row[1], row[0]))
    )


def phase3_commit_action_catalog(
    observation,
) -> tuple[Mapping[str, Any], ...]:
    """Compile explicit final-action operands from task text and inventory."""

    facts = outcome_blind_target_native_facts(observation)
    descriptions = " ".join(
        str(row.get("description") or "")
        for row in facts.get("task_progress") or ()
        if isinstance(row, Mapping)
    ).lower()
    proposals = []
    if (
        str(observation.scenario).lower() == "proteomics"
        and "drop" in descriptions
        and "DROP" in observation.known_actions
    ):
        for row in facts.get("inventory") or ():
            if (
                isinstance(row, Mapping)
                and isinstance(row.get("uuid"), int)
                and "flag" in str(row.get("name") or "").lower()
            ):
                proposals.append({"action": "DROP", "arg1": int(row["uuid"])})
    valid = {}
    for proposal in proposals:
        action = native_action_from_decision(proposal, observation)
        valid[stable_hash(action)] = action
    return tuple(valid[key] for key in sorted(valid))


def validate_phase3_target_binding_semantics(
    binding, observation, *,
    acquisition_evidence: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Enforce benchmark-native entity types stated explicitly by the task."""

    descriptions = " ".join(
        str(row.get("description") or "")
        for row in outcome_blind_target_native_facts(observation).get(
            "task_progress", ()
        )
        if isinstance(row, Mapping)
    ).lower()
    # Proteomics explicitly asks for the statue of the hypothesized species.
    # Binding the nearby animal with the same species token is a schema error,
    # not a policy choice.  The neural binder must repair to an exact supplied
    # statue UUID/name; the symbolic runtime never resolves object identity.
    if (
        str(observation.scenario).lower() == "proteomics"
        and "statue" in descriptions
        and "statue" not in str(binding.target_name).lower()
    ):
        raise ValueError(
            "Proteomics task requires a statue target; animal UUID/name is invalid"
        )
    required_type, catalog = phase3_target_binding_catalog(
        observation, acquisition_evidence=acquisition_evidence,
    )
    if required_type and {
        "uuid": int(binding.target_uuid), "name": str(binding.target_name),
    } not in catalog:
        raise ValueError(
            "target UUID/name must be copied from phase3_target_binding_catalog"
        )
    commit_catalog = phase3_commit_action_catalog(observation)
    if commit_catalog and dict(binding.commit_action) not in commit_catalog:
        raise ValueError(
            "commit_action must be copied from phase3_commit_action_catalog"
        )


def call_phase3_binder(
    backend, observation, *, memory: str, hypotheses: tuple[str, ...],
    attempts: int, acquisition_evidence: Sequence[Mapping[str, Any]] = (),
):
    """Bind target entities without exposing completion/evaluator fields."""

    schema_error = None
    audit = []
    relations = phase3_acquisition_relations(acquisition_evidence)
    outliers = phase3_acquisition_outlier_candidates(acquisition_evidence)
    acquisition_audit = {
        "acquisition_evidence_count": len(acquisition_evidence),
        "acquisition_evidence_sha256": stable_hash(list(acquisition_evidence)),
        "acquisition_relations_sha256": stable_hash(list(relations)),
        "acquisition_outlier_candidates": list(outliers),
        "prior_conclusion_superseded_by_raw_evidence": bool(
            acquisition_evidence
        ),
    }
    for attempt in range(attempts):
        payload = phase3_binder_prompt_payload(
            observation, memory=memory, hypotheses=hypotheses,
            acquisition_evidence=acquisition_evidence,
            schema_error=schema_error,
        )
        raw = backend.complete(
            "binder", PHASE3_TARGET_BINDER_SYSTEM_PROMPT, payload,
        )
        usage = dict(backend.last_usage or {})
        try:
            binding = parse_target_binding(raw, observation)
            validate_phase3_target_binding_semantics(
                binding, observation,
                acquisition_evidence=acquisition_evidence,
            )
            audit.append({
                "attempt": attempt + 1, "accepted": True,
                "cache_hit": bool(usage.get("cache_hit")),
                "formal_outcome_fields_visible": False,
            } | acquisition_audit)
            return binding, raw, audit
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
            audit.append({
                "attempt": attempt + 1, "accepted": False,
                "error": schema_error,
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "cache_hit": bool(usage.get("cache_hit")),
                "formal_outcome_fields_visible": False,
            } | acquisition_audit)
    raise RuntimeError(f"Phase-3 binder exhausted schema attempts: {audit}")


@dataclass(frozen=True)
class Phase3GroundedCandidate(DiscoveryWorldGroundedCandidate):
    typed_effect_probabilities: Mapping[str, float]


def _typed_probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    output = float(value)
    if not 0.0 <= output <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return output


def attach_phase3_typed_effects(
    bundle: Mapping[str, Any],
    candidates: Sequence[DiscoveryWorldGroundedCandidate],
    observation,
) -> tuple[Phase3GroundedCandidate, ...]:
    """Bind strict neural typed effects to already native-validated actions."""

    raw_rows = bundle.get("candidates")
    if not isinstance(raw_rows, list):
        raise ValueError("Phase-3 typed grounder omitted candidates")
    raw_by_action: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        try:
            action = native_action_from_decision(row, observation)
        except (KeyError, TypeError, ValueError):
            continue
        raw_by_action[stable_hash(action)] = row
    output = []
    for candidate in candidates:
        raw = raw_by_action.get(stable_hash(dict(candidate.action)))
        if raw is None:
            raise ValueError("accepted native candidate lost typed-effect binding")
        typed = raw.get("typed_effect_probabilities")
        if not isinstance(typed, Mapping) or set(typed) != set(TYPED_EFFECTS):
            raise ValueError(
                "typed_effect_probabilities must contain exactly "
                + ", ".join(TYPED_EFFECTS)
            )
        values = {
            effect_type: _typed_probability(typed[effect_type], effect_type)
            for effect_type in TYPED_EFFECTS
        }
        output.append(Phase3GroundedCandidate(
            **asdict(candidate), typed_effect_probabilities=values,
        ))
    return tuple(output)


def canonical_position_candidates(
    candidates: Sequence[DiscoveryWorldGroundedCandidate],
) -> tuple[DiscoveryWorldGroundedCandidate, ...]:
    """Canonicalize the set without imposing a target semantic preference."""

    positions = {row.candidate_sha256: row for row in candidates if row.target_role == "POSITION"}
    return tuple(sorted(
        positions.values(),
        key=lambda row: (
            stable_hash(dict(row.action)), row.candidate_sha256,
        ),
    ))


def phase3_candidate_set_complete(
    candidates: Sequence[DiscoveryWorldGroundedCandidate],
) -> bool:
    return bool(
        len(canonical_position_candidates(candidates)) == 4
        and sum(row.target_role == "COMMIT" for row in candidates) == 1
    )


def phase3_position_action_catalog(
    observation, target_binding,
) -> tuple[Mapping[str, Any], ...]:
    """Compile currently valid reversible operands for neural scoring.

    This is target-native precondition compilation, not a policy: it does not
    rank actions or attach effect values.  The neural grounder still selects
    and scores four interventions from the catalog.
    """

    proposals: list[dict[str, Any]] = []
    known = observation.known_actions
    if "TELEPORT_TO_OBJECT" in known:
        proposals.append({
            "action": "TELEPORT_TO_OBJECT",
            "arg1": int(target_binding.target_uuid),
        })
    if "MOVE_DIRECTION" in known:
        for direction in (
            (observation.ui.get("agentLocation") or {}).get(
                "directions_you_can_move", ()
            )
        ):
            proposals.append({"action": "MOVE_DIRECTION", "arg1": direction})
    if "ROTATE_DIRECTION" in known:
        for direction in ("north", "east", "south", "west"):
            proposals.append({"action": "ROTATE_DIRECTION", "arg1": direction})
    if "TELEPORT_TO_LOCATION" in known:
        for location in sorted(observation.teleport_locations):
            proposals.append({
                "action": "TELEPORT_TO_LOCATION", "arg1": location,
            })

    valid: dict[str, Mapping[str, Any]] = {}
    for proposal in proposals:
        try:
            action = native_action_from_decision(proposal, observation)
        except (KeyError, TypeError, ValueError):
            continue
        valid[stable_hash(action)] = action
    return tuple(valid[key] for key in sorted(valid))


def call_phase3_grounder(
    backend, observation, *, memory: str, hypotheses: tuple[str, ...],
    recent: list[dict[str, Any]], target_binding, attempts: int,
):
    """Acquire a multiplicity-complete target candidate set without outcomes."""

    schema_error = None
    audit = []
    for attempt in range(attempts):
        payload = grounder_prompt_payload(
            observation, memory=memory, hypotheses=hypotheses, recent=recent,
            target_binding=target_binding, schema_error=schema_error,
        )
        payload["target_native_facts"] = outcome_blind_target_native_facts(
            observation
        )
        payload["formal_outcome_fields_visible"] = False
        payload["phase3_position_action_catalog"] = list(
            phase3_position_action_catalog(observation, target_binding)
        )
        payload["phase3_position_catalog_is_exhaustive"] = True
        raw = backend.complete("grounder", PHASE3_TARGET_GROUNDER_SYSTEM_PROMPT, payload)
        usage = dict(backend.last_usage or {})
        try:
            bundle, base_candidates = parse_grounded_candidates(raw, observation)
            candidates = attach_phase3_typed_effects(
                bundle, base_candidates, observation,
            )
            if not recent:
                if not phase3_candidate_set_complete(candidates):
                    positions = canonical_position_candidates(candidates)
                    commits = [
                        row for row in candidates if row.target_role == "COMMIT"
                    ]
                    rejections = bundle.get("candidate_parse_rejections") or []
                    raise ValueError(
                        "initial Phase-3 candidate set must contain four unique "
                        "POSITION interventions and one COMMIT after native "
                        "precondition validation; "
                        f"valid_positions={len(positions)}, "
                        f"valid_commits={len(commits)}, "
                        "replace each rejected or duplicate action; "
                        "candidate_parse_rejections="
                        f"{json.dumps(rejections, sort_keys=True)}"
                    )
            elif not source_applicability_complete(
                candidates, observation, target_binding,
            ):
                raise ValueError(
                    "recovery candidate set has neither a safe POSITION nor "
                    "a symbolically witnessed COMMIT"
                )
            audit.append({
                "attempt": attempt + 1, "accepted": True,
                "multiplicity_complete": not recent,
                "recovery_complete": bool(recent),
                "cache_hit": bool(usage.get("cache_hit")),
                "formal_outcome_fields_visible": False,
            })
            return bundle, candidates, raw, audit
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
            audit.append({
                "attempt": attempt + 1, "accepted": False,
                "multiplicity_complete": False, "error": schema_error,
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "cache_hit": bool(usage.get("cache_hit")),
                "formal_outcome_fields_visible": False,
            })
    raise RuntimeError(f"Phase-3 grounder exhausted multiplicity attempts: {audit}")


@dataclass(frozen=True)
class Phase3SelectionReceipt:
    schema_version: str
    condition: str
    selected_candidate_sha256: str
    selected_action: Mapping[str, Any]
    selected_role: str
    candidate_bundle_sha256: str
    source_artifact_sha256: str | None
    source_program_sha256: str | None
    source_profile_sha256: str | None
    source_prior_sha256: str | None
    applicability_receipt_sha256: str | None
    program_decision_receipt_sha256: str | None
    portfolio_receipt_sha256: str | None
    effect_binding_control_receipt_sha256: str | None
    portfolio_selection_receipt: Mapping[str, Any] | None
    effect_binding_control_receipt: Mapping[str, Any] | None
    positive_commit_effect_witnessed: bool
    evidence_supported: bool
    source_admitted: bool | None
    selection_reason: str
    target_outcome_read: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls, *, condition: str,
        candidates: Sequence[DiscoveryWorldGroundedCandidate],
        selected: DiscoveryWorldGroundedCandidate, observation, target_binding,
        runtime: AnonymousAttemptRuntime | None = None,
        decision: RuntimeDecision | None = None, source_admitted: bool | None = None,
        selection_reason: str, portfolio_receipt_sha256: str | None = None,
        effect_binding_control_receipt_sha256: str | None = None,
        portfolio_selection_receipt: Mapping[str, Any] | None = None,
        effect_binding_control_receipt: Mapping[str, Any] | None = None,
    ) -> "Phase3SelectionReceipt":
        body = {
            "schema_version": "phase3-discoveryworld-selection-v1",
            "condition": str(condition),
            "selected_candidate_sha256": selected.candidate_sha256,
            "selected_action": dict(selected.action),
            "selected_role": selected.target_role,
            "candidate_bundle_sha256": stable_hash([
                row.candidate_sha256 for row in candidates
            ]),
            "source_artifact_sha256": (
                runtime.artifact_sha256 if runtime is not None else None
            ),
            "source_program_sha256": (
                runtime.program_sha256 if runtime is not None else None
            ),
            "source_profile_sha256": (
                runtime.prior.source_profile_sha256 if runtime is not None else None
            ),
            "source_prior_sha256": (
                runtime.prior.prior_sha256 if runtime is not None else None
            ),
            "applicability_receipt_sha256": (
                runtime.applicability_receipt["applicability_receipt_sha256"]
                if runtime is not None else None
            ),
            "program_decision_receipt_sha256": (
                decision.receipt_sha256 if decision is not None else None
            ),
            "portfolio_receipt_sha256": portfolio_receipt_sha256,
            "effect_binding_control_receipt_sha256": (
                effect_binding_control_receipt_sha256
            ),
            "portfolio_selection_receipt": (
                dict(portfolio_selection_receipt)
                if portfolio_selection_receipt is not None else None
            ),
            "effect_binding_control_receipt": (
                dict(effect_binding_control_receipt)
                if effect_binding_control_receipt is not None else None
            ),
            "positive_commit_effect_witnessed": positive_commit_effect_witnessed(
                selected, observation, target_binding,
            ),
            "evidence_supported": evidence_supported(
                selected, observation, target_binding,
            ),
            "source_admitted": source_admitted,
            "selection_reason": str(selection_reason),
            "target_outcome_read": False,
        }
        return cls(receipt_sha256=stable_hash(body), **body)

    def validate(self) -> bool:
        body = asdict(self)
        claimed = body.pop("receipt_sha256")
        return claimed == stable_hash(body)


@dataclass
class _SourceArm:
    artifact: Mapping[str, Any]
    runtime: AnonymousAttemptRuntime | None = None
    initial_positions: tuple[DiscoveryWorldGroundedCandidate, ...] = ()
    first_call: bool = True
    fallen_back: bool = False
    permute_effect_binding: bool = False
    portfolio_admitted: bool = True
    portfolio_receipt_sha256: str | None = None
    effect_binding_control_receipt_sha256: str | None = None
    portfolio_selection_receipt: Mapping[str, Any] | None = None
    effect_binding_control_receipt: Mapping[str, Any] | None = None


class Phase3DiscoveryWorldSelector:
    """Stateful five-arm selector; never reads terminal or official outcome."""

    def __init__(
        self, *, authentic_artifact: Mapping[str, Any],
        permuted_artifact: Mapping[str, Any],
    ) -> None:
        self.source_arms = {
            SOURCE_INDUCED: _SourceArm(authentic_artifact),
            SOURCE_PERMUTED: _SourceArm(permuted_artifact),
        }

    @staticmethod
    def _action_ids(candidates):
        return [stable_hash(dict(row.action)) for row in candidates]

    @staticmethod
    def _typed_effects(candidates):
        return [
            dict(getattr(row, "typed_effect_probabilities", {}))
            for row in candidates
        ]

    def _arm_effects(self, arm: _SourceArm, positions):
        effects = self._typed_effects(positions)
        if not arm.permute_effect_binding:
            return effects
        program = arm.artifact.get("typed_effect_program")
        if not isinstance(program, Mapping):
            raise ValueError("effect-binding control requires a typed program")
        permuted, receipt = permute_selected_effect_binding(
            program, candidate_ids=self._action_ids(positions),
            candidate_effects=effects,
        )
        arm.effect_binding_control_receipt_sha256 = str(
            receipt["effect_binding_control_receipt_sha256"]
        )
        arm.effect_binding_control_receipt = receipt
        return list(permuted)

    @staticmethod
    def _current_effect(candidates, observation, target_binding) -> str:
        return "HIGH" if any(
            row.target_role == "COMMIT"
            and positive_commit_effect_witnessed(row, observation, target_binding)
            for row in candidates
        ) else "LOW"

    @staticmethod
    def _native_selection(condition, candidates, observation, target_binding):
        if condition == GENERIC_SCAFFOLD:
            selected, _ = select_candidate(
                "target_native_myopic", candidates, observation,
                target_binding=target_binding,
            )
            reason = "SOURCE_FREE_TARGET_CANDIDATE_RANKING"
        elif condition == TARGET_NATIVE_CEILING:
            selected, _ = select_source_safe_candidate(
                "authentic_sokoban_effect_plus_target", candidates, observation,
                target_binding=target_binding,
            )
            reason = "TARGET_NATIVE_EXACT_EFFECT_CEILING"
        else:
            raise ValueError(f"unsupported source-free condition: {condition}")
        receipt = Phase3SelectionReceipt.create(
            condition=condition, candidates=candidates, selected=selected,
            observation=observation, target_binding=target_binding,
            selection_reason=reason,
        )
        return selected, receipt

    def select(self, condition, candidates, observation, *, target_binding=None, **_):
        condition = str(condition)
        if condition in {GENERIC_SCAFFOLD, TARGET_NATIVE_CEILING}:
            return self._native_selection(
                condition, candidates, observation, target_binding,
            )
        if condition not in self.source_arms:
            raise ValueError(f"unsupported Phase-3 condition: {condition}")
        arm = self.source_arms[condition]
        positions = canonical_position_candidates(candidates)
        if arm.runtime is None:
            grounding_sha = stable_hash([asdict(row) for row in positions])
            arm.initial_positions = positions
            typed = isinstance(arm.artifact.get("typed_effect_program"), Mapping)
            candidate_ids = (
                self._action_ids(positions) if typed else
                [row.candidate_sha256 for row in positions]
            )
            arm.runtime = AnonymousAttemptRuntime(
                artifact=arm.artifact,
                candidate_ids=candidate_ids,
                target_grounding_sha256=grounding_sha,
                candidate_effects=self._arm_effects(arm, positions),
            )
        runtime = arm.runtime
        if (
            runtime.supports_dynamic_rebinding
            and not arm.first_call
            and not arm.fallen_back
        ):
            runtime.rebind_candidates(
                candidate_ids=self._action_ids(positions),
                candidate_effects=self._arm_effects(arm, positions),
                target_grounding_sha256=stable_hash([
                    asdict(row) for row in positions
                ]),
            )
        if arm.fallen_back or not runtime.admitted or not arm.portfolio_admitted:
            arm.fallen_back = True
            selected, _ = select_candidate(
                "target_native_myopic", candidates, observation,
                target_binding=target_binding,
            )
            decision = None
            reason = (
                "SOURCE_PORTFOLIO_ABSTAINED_TO_MATCHED_SOURCE_FREE_GROUNDER"
                if not arm.portfolio_admitted else
                "SOURCE_ABSTAINED_TO_MATCHED_SOURCE_FREE_GROUNDER"
            )
        else:
            effect = self._current_effect(candidates, observation, target_binding)
            decision = (
                runtime.resume_observed_prefix(effect)
                if arm.first_call else runtime.observe(effect)
            )
            arm.first_call = False
            if decision.kind == "TRIAL":
                if runtime.supports_dynamic_rebinding:
                    selected = {
                        stable_hash(dict(row.action)): row for row in positions
                    }.get(decision.candidate_id)
                else:
                    by_id = {
                        row.candidate_sha256: row for row in arm.initial_positions
                    }
                    desired = by_id[decision.candidate_id]
                    selected = {
                        stable_hash(dict(row.action)): row for row in candidates
                    }.get(stable_hash(dict(desired.action)))
                if selected is None:
                    # Candidate IDs are bound at the fork, but native
                    # preconditions can change after an intervention.  Never
                    # execute a stale source-selected action merely because it
                    # was valid in the initial state.  The current target
                    # grounder is the sole action-validity authority.
                    arm.fallen_back = True
                    selected, _ = select_candidate(
                        "target_native_myopic", candidates, observation,
                        target_binding=target_binding,
                    )
                    reason = (
                        "SOURCE_TRIAL_NOT_REGROUNDED_IN_CURRENT_STATE_"
                        "TO_MATCHED_SOURCE_FREE_GROUNDER"
                    )
                else:
                    reason = "SOURCE_INDUCED_ANONYMOUS_TRIAL_DELTA"
            elif decision.kind == "TERMINATE":
                witnessed = [
                    row for row in candidates
                    if row.target_role == "COMMIT"
                    and positive_commit_effect_witnessed(
                        row, observation, target_binding,
                    )
                ]
                if not witnessed:
                    raise RuntimeError(
                        "induced termination has no target-native effect witness"
                    )
                selected = max(witnessed, key=lambda row: (
                    row.positive_effect_probability,
                    row.prerequisite_probability,
                    row.candidate_sha256,
                ))
                reason = "SOURCE_INDUCED_ANONYMOUS_TERMINAL_DELTA"
            else:
                arm.fallen_back = True
                selected, _ = select_candidate(
                    "target_native_myopic", candidates, observation,
                    target_binding=target_binding,
                )
                reason = "SOURCE_PROGRAM_ABSTAINED_TO_MATCHED_SOURCE_FREE_GROUNDER"
        receipt = Phase3SelectionReceipt.create(
            condition=condition, candidates=candidates, selected=selected,
            observation=observation, target_binding=target_binding,
            runtime=runtime, decision=decision,
            source_admitted=not arm.fallen_back and runtime.admitted,
            selection_reason=reason,
            portfolio_receipt_sha256=arm.portfolio_receipt_sha256,
            effect_binding_control_receipt_sha256=(
                arm.effect_binding_control_receipt_sha256
            ),
            portfolio_selection_receipt=arm.portfolio_selection_receipt,
            effect_binding_control_receipt=arm.effect_binding_control_receipt,
        )
        return selected, receipt


class Phase3DiscoveryWorldPortfolioSelector(Phase3DiscoveryWorldSelector):
    """Select one source-induced program by target-native applicability."""

    def __init__(self, *, source_artifacts: Sequence[Mapping[str, Any]]) -> None:
        if len(source_artifacts) < 2:
            raise ValueError("source program portfolio requires multiple artifacts")
        self._portfolio_artifacts = tuple(source_artifacts)
        self.source_arms: dict[str, _SourceArm] = {}

    def _initialize_portfolio(self, candidates) -> None:
        positions = canonical_position_candidates(candidates)
        candidate_ids = self._action_ids(positions)
        effects = self._typed_effects(positions)
        grounding_sha = stable_hash([asdict(row) for row in positions])
        receipt = select_source_program_portfolio(
            self._portfolio_artifacts, candidate_ids=candidate_ids,
            candidate_effects=effects, target_grounding_sha256=grounding_sha,
        )
        selected_sha = receipt["selected_artifact_sha256"]
        selected = next((
            artifact for artifact in self._portfolio_artifacts
            if artifact.get("artifact_sha256") == selected_sha
        ), self._portfolio_artifacts[0])
        admitted = selected_sha is not None
        receipt_sha = str(receipt["portfolio_receipt_sha256"])
        self.source_arms = {
            SOURCE_INDUCED: _SourceArm(
                selected, portfolio_admitted=admitted,
                portfolio_receipt_sha256=receipt_sha,
                portfolio_selection_receipt=receipt,
            ),
            SOURCE_PERMUTED: _SourceArm(
                selected, permute_effect_binding=True,
                portfolio_admitted=admitted,
                portfolio_receipt_sha256=receipt_sha,
                portfolio_selection_receipt=receipt,
            ),
        }

    def select(self, condition, candidates, observation, **kwargs):
        if str(condition) in {SOURCE_INDUCED, SOURCE_PERMUTED} and not self.source_arms:
            self._initialize_portfolio(candidates)
        return super().select(condition, candidates, observation, **kwargs)


__all__ = [
    "CONDITIONS", "GENERIC_SCAFFOLD", "MATCHED_CONDITIONS", "NEURAL_ONLY",
    "PHASE3_TARGET_BINDER_SYSTEM_PROMPT", "PHASE3_TARGET_GROUNDER_SYSTEM_PROMPT",
    "Phase3DiscoveryWorldSelector",
    "Phase3DiscoveryWorldPortfolioSelector",
    "Phase3GroundedCandidate", "attach_phase3_typed_effects",
    "Phase3SelectionReceipt", "SOURCE_INDUCED", "SOURCE_PERMUTED",
    "TARGET_NATIVE_CEILING", "call_phase3_binder", "call_phase3_grounder",
    "canonical_position_candidates", "phase3_candidate_set_complete",
    "outcome_blind_target_native_facts", "phase3_binder_prompt_payload",
    "phase3_commit_action_catalog", "phase3_position_action_catalog",
    "phase3_target_binding_catalog",
    "validate_phase3_target_binding_semantics",
]
