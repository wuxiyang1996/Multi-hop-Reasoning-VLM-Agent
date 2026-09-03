"""Permission-bounded neural-symbolic skill retargeting.

The source artifact can select a canonical option.  The target Harness can ground
observations, realize an externally selected option, and report observable effects.
Official outcomes are carried by ``TargetStep`` and are never passed to the Harness
or skill interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from statistics import mean
from typing import Any, Mapping, Protocol, Sequence

from .contracts import stable_hash


class RetargetingReject(ValueError):
    """A frozen-artifact, authority, identity, or grounding check failed."""


class SkillArtifactKind(str, Enum):
    AUTHENTIC_SOURCE = "AUTHENTIC_SOURCE"
    SHUFFLED_CONTROL = "SHUFFLED_CONTROL"
    TARGET_ORACLE = "TARGET_ORACLE"


class RetargetingCondition(str, Enum):
    RAW_TARGET_ONLY = "raw_target_only"
    NULL_SKILL_SAME_HARNESS = "null_skill_same_harness"
    SHUFFLED_SOURCE_SKILL = "shuffled_source_skill"
    AUTHENTIC_SOURCE_SKILL = "authentic_source_skill"
    TARGET_ORACLE_SKILL = "target_oracle_skill"


CORE_CONDITIONS = frozenset({
    RetargetingCondition.NULL_SKILL_SAME_HARNESS,
    RetargetingCondition.SHUFFLED_SOURCE_SKILL,
    RetargetingCondition.AUTHENTIC_SOURCE_SKILL,
    RetargetingCondition.TARGET_ORACLE_SKILL,
})
REQUIRED_CONDITIONS = frozenset({RetargetingCondition.RAW_TARGET_ONLY, *CORE_CONDITIONS})


class SkillDecisionVerdict(str, Enum):
    SELECT = "SELECT"
    ABSTAIN = "ABSTAIN"
    TERMINATE = "TERMINATE"
    REJECT = "REJECT"


class ExperimentVerdict(str, Enum):
    MECHANISM_SUPPORTED = "MECHANISM_SUPPORTED"
    NO_ATTRIBUTABLE_GAIN = "NO_ATTRIBUTABLE_GAIN"
    NEGATIVE_TRANSFER = "NEGATIVE_TRANSFER"
    INVALID_EXPERIMENT = "INVALID_EXPERIMENT"


def _unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    rows = tuple(map(str, values))
    if not rows or len(rows) != len(set(rows)):
        raise RetargetingReject(f"{label} must be non-empty and unique")
    return rows


_FORBIDDEN_OUTCOME_KEYS = frozenset({
    "reward", "official_reward", "official_score", "official_success",
})


def _forbidden_outcome_path(value: Any, path: str = "payload") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            child = f"{path}.{key}"
            if key_text in _FORBIDDEN_OUTCOME_KEYS:
                return child
            found = _forbidden_outcome_path(nested, child)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found = _forbidden_outcome_path(nested, f"{path}[{index}]")
            if found is not None:
                return found
    return None


@dataclass(frozen=True)
class SourceQualification:
    authentic_value: float
    shuffled_value: float
    marginal_value: float
    receipt_ids: tuple[str, ...]
    qualification_hash: str

    @classmethod
    def create(
        cls,
        *,
        authentic_value: float,
        shuffled_value: float,
        marginal_value: float,
        receipt_ids: Sequence[str],
    ) -> "SourceQualification":
        receipts = _unique(receipt_ids, "source qualification receipt ids")
        payload = {
            "authentic_value": float(authentic_value),
            "shuffled_value": float(shuffled_value),
            "marginal_value": float(marginal_value),
            "receipt_ids": list(receipts),
        }
        return cls(
            authentic_value=float(authentic_value),
            shuffled_value=float(shuffled_value),
            marginal_value=float(marginal_value),
            receipt_ids=receipts,
            qualification_hash=stable_hash(payload),
        )

    @property
    def passed(self) -> bool:
        return self.authentic_value > max(self.shuffled_value, self.marginal_value)

    def validate(self) -> bool:
        payload = {
            "authentic_value": self.authentic_value,
            "shuffled_value": self.shuffled_value,
            "marginal_value": self.marginal_value,
            "receipt_ids": list(self.receipt_ids),
        }
        return (
            bool(self.receipt_ids)
            and len(self.receipt_ids) == len(set(self.receipt_ids))
            and stable_hash(payload) == self.qualification_hash
        )


@dataclass(frozen=True)
class SymbolicOptionRule:
    rule_id: str
    option: str
    requires: tuple[str, ...]
    forbids: tuple[str, ...] = ()
    expected_add: tuple[str, ...] = ()
    expected_remove: tuple[str, ...] = ()
    recovery_option: str | None = None
    priority: int = 0

    def applies(self, facts: frozenset[str]) -> bool:
        return set(self.requires).issubset(facts) and not set(self.forbids).intersection(facts)

    def payload(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "option": self.option,
            "requires": list(self.requires),
            "forbids": list(self.forbids),
            "expected_add": list(self.expected_add),
            "expected_remove": list(self.expected_remove),
            "recovery_option": self.recovery_option,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class FrozenSkillArtifact:
    skill_id: str
    artifact_kind: SkillArtifactKind
    source_domain: str
    option_vocabulary: tuple[str, ...]
    predicate_vocabulary: tuple[str, ...]
    role_vocabulary: tuple[str, ...]
    rules: tuple[SymbolicOptionRule, ...]
    termination_facts: tuple[str, ...]
    source_lineage: tuple[str, ...]
    qualification: SourceQualification | None
    parent_artifact_hash: str | None
    control_option_permutation: tuple[tuple[str, str], ...]
    artifact_hash: str

    @classmethod
    def create(
        cls,
        *,
        skill_id: str,
        artifact_kind: SkillArtifactKind,
        source_domain: str,
        option_vocabulary: Sequence[str],
        predicate_vocabulary: Sequence[str],
        role_vocabulary: Sequence[str],
        rules: Sequence[SymbolicOptionRule],
        termination_facts: Sequence[str],
        source_lineage: Sequence[str] = (),
        qualification: SourceQualification | None = None,
        parent_artifact_hash: str | None = None,
        control_option_permutation: Sequence[tuple[str, str]] = (),
    ) -> "FrozenSkillArtifact":
        artifact = cls(
            skill_id=str(skill_id),
            artifact_kind=SkillArtifactKind(artifact_kind),
            source_domain=str(source_domain),
            option_vocabulary=_unique(option_vocabulary, "skill option vocabulary"),
            predicate_vocabulary=_unique(predicate_vocabulary, "skill predicate vocabulary"),
            role_vocabulary=_unique(role_vocabulary, "skill role vocabulary"),
            rules=tuple(rules),
            termination_facts=tuple(map(str, termination_facts)),
            source_lineage=tuple(map(str, source_lineage)),
            qualification=qualification,
            parent_artifact_hash=parent_artifact_hash,
            control_option_permutation=tuple(
                (str(source), str(target))
                for source, target in control_option_permutation
            ),
            artifact_hash="",
        )
        artifact = cls(**{**artifact.__dict__, "artifact_hash": stable_hash(artifact.payload())})
        artifact.assert_valid()
        return artifact

    def payload(self) -> dict[str, Any]:
        qualification = None
        if self.qualification is not None:
            qualification = {
                "authentic_value": self.qualification.authentic_value,
                "shuffled_value": self.qualification.shuffled_value,
                "marginal_value": self.qualification.marginal_value,
                "receipt_ids": list(self.qualification.receipt_ids),
                "qualification_hash": self.qualification.qualification_hash,
            }
        return {
            "skill_id": self.skill_id,
            "artifact_kind": self.artifact_kind.value,
            "source_domain": self.source_domain,
            "option_vocabulary": list(self.option_vocabulary),
            "predicate_vocabulary": list(self.predicate_vocabulary),
            "role_vocabulary": list(self.role_vocabulary),
            "rules": [rule.payload() for rule in self.rules],
            "termination_facts": list(self.termination_facts),
            "source_lineage": list(self.source_lineage),
            "qualification": qualification,
            "parent_artifact_hash": self.parent_artifact_hash,
            "control_option_permutation": [
                list(row) for row in self.control_option_permutation
            ],
        }

    @property
    def program_structure_hash(self) -> str:
        """Hash guards/effects/control shape while excluding option labels."""

        return stable_hash({
            "predicate_vocabulary": list(self.predicate_vocabulary),
            "role_vocabulary": list(self.role_vocabulary),
            "rule_structure": [
                {
                    "rule_id": rule.rule_id,
                    "requires": list(rule.requires),
                    "forbids": list(rule.forbids),
                    "expected_add": list(rule.expected_add),
                    "expected_remove": list(rule.expected_remove),
                    "has_recovery": rule.recovery_option is not None,
                    "priority": rule.priority,
                }
                for rule in self.rules
            ],
            "termination_facts": list(self.termination_facts),
            "source_lineage": list(self.source_lineage),
            "qualification_hash": (
                self.qualification.qualification_hash
                if self.qualification is not None else None
            ),
        })

    def assert_valid(self) -> None:
        if stable_hash(self.payload()) != self.artifact_hash:
            raise RetargetingReject("frozen skill artifact hash mismatch")
        if len(self.option_vocabulary) < 2 or len(self.rules) < 2:
            raise RetargetingReject("a transferable skill needs at least two options and rules")
        if len(self.rules) == 1 or len({rule.option for rule in self.rules}) < 2:
            raise RetargetingReject("skill has no observable option variation")
        if not self.termination_facts:
            raise RetargetingReject("skill requires an explicit termination condition")
        predicates = set(self.predicate_vocabulary)
        options = set(self.option_vocabulary)
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise RetargetingReject("skill rule ids must be unique")
        if not set(self.termination_facts).issubset(predicates):
            raise RetargetingReject("termination facts are outside the predicate vocabulary")
        for rule in self.rules:
            if rule.option not in options:
                raise RetargetingReject(f"rule {rule.rule_id} selects an unknown option")
            mentioned = set((*rule.requires, *rule.forbids, *rule.expected_add, *rule.expected_remove))
            if not mentioned.issubset(predicates):
                raise RetargetingReject(f"rule {rule.rule_id} uses an unknown predicate")
            if not rule.requires:
                raise RetargetingReject(f"rule {rule.rule_id} lacks a typed precondition")
            if not rule.expected_add and not rule.expected_remove:
                raise RetargetingReject(f"rule {rule.rule_id} lacks an expected effect")
            if rule.recovery_option is not None and rule.recovery_option not in options:
                raise RetargetingReject(f"rule {rule.rule_id} has an unknown recovery option")
        if self.artifact_kind == SkillArtifactKind.AUTHENTIC_SOURCE:
            if not self.source_lineage:
                raise RetargetingReject("authentic skill lacks source intervention lineage")
            if len(self.source_lineage) != len(set(self.source_lineage)):
                raise RetargetingReject("source intervention lineage contains duplicates")
            if self.qualification is None or not self.qualification.validate():
                raise RetargetingReject("authentic skill lacks a valid source qualification")
            if not self.qualification.passed:
                raise RetargetingReject("authentic skill failed the source value gate")
            if not set(self.qualification.receipt_ids).issubset(self.source_lineage):
                raise RetargetingReject(
                    "source qualification is not grounded in the frozen source lineage"
                )
            if self.parent_artifact_hash is not None:
                raise RetargetingReject("authentic skill cannot have a parent control artifact")
            if self.control_option_permutation:
                raise RetargetingReject("authentic skill cannot carry a control permutation")
        elif self.artifact_kind == SkillArtifactKind.SHUFFLED_CONTROL:
            if not self.parent_artifact_hash:
                raise RetargetingReject("shuffled control must identify its authentic parent")
            permutation = dict(self.control_option_permutation)
            if (
                len(permutation) != len(self.control_option_permutation)
                or set(permutation) != options
                or set(permutation.values()) != options
                or any(source == target for source, target in permutation.items())
            ):
                raise RetargetingReject(
                    "shuffled control requires an explicit option-label derangement"
                )
        elif self.artifact_kind == SkillArtifactKind.TARGET_ORACLE:
            if self.qualification is not None or self.source_lineage:
                raise RetargetingReject("target oracle cannot carry source qualification evidence")
            if self.parent_artifact_hash is not None or self.control_option_permutation:
                raise RetargetingReject("target oracle cannot carry shuffled-control provenance")

    def shuffled_control(self, *, skill_id: str | None = None) -> "FrozenSkillArtifact":
        self.assert_valid()
        if self.artifact_kind != SkillArtifactKind.AUTHENTIC_SOURCE:
            raise RetargetingReject("only an authentic source artifact can be shuffled")
        options = tuple(self.option_vocabulary)
        mapping = {option: options[(index + 1) % len(options)] for index, option in enumerate(options)}
        rules = tuple(
            SymbolicOptionRule(
                rule_id=rule.rule_id,
                option=mapping[rule.option],
                requires=rule.requires,
                forbids=rule.forbids,
                expected_add=rule.expected_add,
                expected_remove=rule.expected_remove,
                recovery_option=(
                    mapping[rule.recovery_option] if rule.recovery_option is not None else None
                ),
                priority=rule.priority,
            )
            for rule in self.rules
        )
        return FrozenSkillArtifact.create(
            skill_id=skill_id or f"{self.skill_id}-shuffled",
            artifact_kind=SkillArtifactKind.SHUFFLED_CONTROL,
            source_domain=self.source_domain,
            option_vocabulary=options,
            predicate_vocabulary=self.predicate_vocabulary,
            role_vocabulary=self.role_vocabulary,
            rules=rules,
            termination_facts=self.termination_facts,
            source_lineage=self.source_lineage,
            qualification=self.qualification,
            parent_artifact_hash=self.artifact_hash,
            control_option_permutation=tuple(mapping.items()),
        )


@dataclass(frozen=True)
class TargetHarnessArtifact:
    harness_id: str
    target_domain: str
    option_vocabulary: tuple[str, ...]
    predicate_vocabulary: tuple[str, ...]
    role_vocabulary: tuple[str, ...]
    adaptation_receipt_ids: tuple[str, ...]
    grounder_hash: str
    realizer_hash: str
    verifier_hash: str
    artifact_hash: str

    @classmethod
    def create(
        cls,
        *,
        harness_id: str,
        target_domain: str,
        option_vocabulary: Sequence[str],
        predicate_vocabulary: Sequence[str],
        role_vocabulary: Sequence[str],
        adaptation_receipt_ids: Sequence[str],
        grounder_hash: str,
        realizer_hash: str,
        verifier_hash: str,
    ) -> "TargetHarnessArtifact":
        artifact = cls(
            harness_id=str(harness_id),
            target_domain=str(target_domain),
            option_vocabulary=_unique(option_vocabulary, "Harness option vocabulary"),
            predicate_vocabulary=_unique(predicate_vocabulary, "Harness predicate vocabulary"),
            role_vocabulary=_unique(role_vocabulary, "Harness role vocabulary"),
            adaptation_receipt_ids=_unique(
                adaptation_receipt_ids, "target adaptation receipt ids"
            ),
            grounder_hash=str(grounder_hash),
            realizer_hash=str(realizer_hash),
            verifier_hash=str(verifier_hash),
            artifact_hash="",
        )
        artifact = cls(**{**artifact.__dict__, "artifact_hash": stable_hash(artifact.payload())})
        artifact.assert_valid()
        return artifact

    def payload(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "target_domain": self.target_domain,
            "option_vocabulary": list(self.option_vocabulary),
            "predicate_vocabulary": list(self.predicate_vocabulary),
            "role_vocabulary": list(self.role_vocabulary),
            "adaptation_receipt_ids": list(self.adaptation_receipt_ids),
            "grounder_hash": self.grounder_hash,
            "realizer_hash": self.realizer_hash,
            "verifier_hash": self.verifier_hash,
        }

    def assert_valid(self) -> None:
        if stable_hash(self.payload()) != self.artifact_hash:
            raise RetargetingReject("target Harness artifact hash mismatch")
        if not all((self.grounder_hash, self.realizer_hash, self.verifier_hash)):
            raise RetargetingReject("target Harness component hashes must be frozen")


@dataclass(frozen=True)
class TargetObservation:
    observation_id: str
    payload: Mapping[str, Any]
    native_actions: tuple[str, ...]
    terminal: bool = False

    @property
    def observation_hash(self) -> str:
        return stable_hash({
            "observation_id": self.observation_id,
            "payload": dict(self.payload),
            "native_actions": list(self.native_actions),
            "terminal": self.terminal,
        })


@dataclass(frozen=True)
class NativeActionCandidate:
    action: str
    within_option_score: float


@dataclass(frozen=True)
class OptionGrounding:
    option: str
    candidates: tuple[NativeActionCandidate, ...]


@dataclass(frozen=True)
class GroundingDraft:
    facts: frozenset[str]
    role_bindings: tuple[tuple[str, str], ...]
    option_groundings: tuple[OptionGrounding, ...]


@dataclass(frozen=True)
class GroundingFrame:
    observation_hash: str
    facts: frozenset[str]
    role_bindings: tuple[tuple[str, str], ...]
    option_groundings: tuple[OptionGrounding, ...]
    harness_hash: str
    frame_hash: str

    @property
    def available_options(self) -> tuple[str, ...]:
        return tuple(row.option for row in self.option_groundings if row.candidates)

    def candidates_for(self, option: str) -> tuple[NativeActionCandidate, ...]:
        matches = [row.candidates for row in self.option_groundings if row.option == option]
        if len(matches) != 1:
            raise RetargetingReject(f"option {option!r} is not grounded exactly once")
        return matches[0]


@dataclass(frozen=True)
class NativeActionDecision:
    selected_option: str
    action: str
    candidate_actions: tuple[str, ...]
    observation_hash: str
    frame_hash: str
    harness_hash: str
    decision_hash: str

    def validate(self) -> bool:
        payload = {
            "selected_option": self.selected_option,
            "action": self.action,
            "candidate_actions": list(self.candidate_actions),
            "observation_hash": self.observation_hash,
            "frame_hash": self.frame_hash,
            "harness_hash": self.harness_hash,
        }
        return self.action in self.candidate_actions and stable_hash(payload) == self.decision_hash


@dataclass(frozen=True)
class EffectReceipt:
    before_frame_hash: str
    after_frame_hash: str
    decision_hash: str
    added_facts: tuple[str, ...]
    removed_facts: tuple[str, ...]
    harness_hash: str
    receipt_hash: str

    def validate(self) -> bool:
        payload = {
            "before_frame_hash": self.before_frame_hash,
            "after_frame_hash": self.after_frame_hash,
            "decision_hash": self.decision_hash,
            "added_facts": list(self.added_facts),
            "removed_facts": list(self.removed_facts),
            "harness_hash": self.harness_hash,
        }
        return stable_hash(payload) == self.receipt_hash

    def supports(self, rule: SymbolicOptionRule) -> bool:
        return (
            set(rule.expected_add).issubset(self.added_facts)
            and set(rule.expected_remove).issubset(self.removed_facts)
        )


class TargetGrounder(Protocol):
    grounder_hash: str

    def ground(self, observation: TargetObservation) -> GroundingDraft: ...


class PermissionBoundTargetHarness:
    """A Harness with no option-selection method or official-outcome input."""

    def __init__(self, artifact: TargetHarnessArtifact, grounder: TargetGrounder):
        artifact.assert_valid()
        if grounder.grounder_hash != artifact.grounder_hash:
            raise RetargetingReject("grounder implementation does not match frozen artifact")
        self.artifact = artifact
        self.grounder = grounder

    def ground(self, observation: TargetObservation) -> GroundingFrame:
        self.artifact.assert_valid()
        forbidden = _forbidden_outcome_path(observation.payload)
        if forbidden is not None:
            raise RetargetingReject(
                f"Harness observation contains forbidden official-outcome field: {forbidden}"
            )
        draft = self.grounder.ground(observation)
        predicates = set(self.artifact.predicate_vocabulary)
        roles = set(self.artifact.role_vocabulary)
        options = set(self.artifact.option_vocabulary)
        if not set(draft.facts).issubset(predicates):
            raise RetargetingReject("grounder emitted an undeclared predicate")
        role_names = [name for name, _ in draft.role_bindings]
        if len(role_names) != len(set(role_names)) or not set(role_names).issubset(roles):
            raise RetargetingReject("grounder emitted duplicate or undeclared role bindings")
        grounded_options = [row.option for row in draft.option_groundings]
        if len(grounded_options) != len(set(grounded_options)):
            raise RetargetingReject("grounder emitted a duplicate option")
        if not set(grounded_options).issubset(options):
            raise RetargetingReject("grounder emitted an undeclared option")
        native = set(observation.native_actions)
        all_actions = [candidate.action for row in draft.option_groundings for candidate in row.candidates]
        if len(all_actions) != len(set(all_actions)):
            raise RetargetingReject("a native action may not be assigned across multiple options")
        if not set(all_actions).issubset(native):
            raise RetargetingReject("grounded action is not in the current native action set")
        payload = {
            "observation_hash": observation.observation_hash,
            "facts": sorted(draft.facts),
            "role_bindings": [list(row) for row in draft.role_bindings],
            "option_groundings": [
                {
                    "option": row.option,
                    "candidates": [
                        {"action": candidate.action, "within_option_score": candidate.within_option_score}
                        for candidate in row.candidates
                    ],
                }
                for row in draft.option_groundings
            ],
            "harness_hash": self.artifact.artifact_hash,
        }
        return GroundingFrame(
            observation_hash=observation.observation_hash,
            facts=draft.facts,
            role_bindings=draft.role_bindings,
            option_groundings=draft.option_groundings,
            harness_hash=self.artifact.artifact_hash,
            frame_hash=stable_hash(payload),
        )

    def realize(self, externally_selected_option: str, frame: GroundingFrame) -> NativeActionDecision:
        self.artifact.assert_valid()
        if frame.harness_hash != self.artifact.artifact_hash:
            raise RetargetingReject("grounding frame came from a different Harness")
        candidates = frame.candidates_for(externally_selected_option)
        if not candidates:
            raise RetargetingReject("externally selected option has no native realization")
        selected = max(candidates, key=lambda row: (row.within_option_score, row.action))
        actions = tuple(row.action for row in candidates)
        payload = {
            "selected_option": externally_selected_option,
            "action": selected.action,
            "candidate_actions": list(actions),
            "observation_hash": frame.observation_hash,
            "frame_hash": frame.frame_hash,
            "harness_hash": self.artifact.artifact_hash,
        }
        return NativeActionDecision(
            selected_option=externally_selected_option,
            action=selected.action,
            candidate_actions=actions,
            observation_hash=frame.observation_hash,
            frame_hash=frame.frame_hash,
            harness_hash=self.artifact.artifact_hash,
            decision_hash=stable_hash(payload),
        )

    def verify(
        self,
        before: GroundingFrame,
        decision: NativeActionDecision,
        after: GroundingFrame,
    ) -> EffectReceipt:
        if not decision.validate():
            raise RetargetingReject("native action decision hash mismatch")
        if before.harness_hash != self.artifact.artifact_hash \
                or after.harness_hash != self.artifact.artifact_hash:
            raise RetargetingReject("effect frames came from a different Harness")
        if decision.frame_hash != before.frame_hash:
            raise RetargetingReject("decision does not reference the before frame")
        added = tuple(sorted(after.facts - before.facts))
        removed = tuple(sorted(before.facts - after.facts))
        payload = {
            "before_frame_hash": before.frame_hash,
            "after_frame_hash": after.frame_hash,
            "decision_hash": decision.decision_hash,
            "added_facts": list(added),
            "removed_facts": list(removed),
            "harness_hash": self.artifact.artifact_hash,
        }
        return EffectReceipt(
            before_frame_hash=before.frame_hash,
            after_frame_hash=after.frame_hash,
            decision_hash=decision.decision_hash,
            added_facts=added,
            removed_facts=removed,
            harness_hash=self.artifact.artifact_hash,
            receipt_hash=stable_hash(payload),
        )


@dataclass(frozen=True)
class SkillDecision:
    verdict: SkillDecisionVerdict
    option: str | None
    rule_id: str | None
    artifact_hash: str | None
    reason: str


class FrozenSkillExecution:
    def __init__(self, artifact: FrozenSkillArtifact | None):
        if artifact is not None:
            artifact.assert_valid()
        self.artifact = artifact
        self._previous_rule: SymbolicOptionRule | None = None

    def select(
        self,
        frame: GroundingFrame,
        previous_effect: EffectReceipt | None,
    ) -> SkillDecision:
        if self.artifact is None:
            return SkillDecision(
                SkillDecisionVerdict.ABSTAIN, None, None, None, "NULL_SKILL",
            )
        self.artifact.assert_valid()
        if not set(frame.facts).issubset(self.artifact.predicate_vocabulary):
            return SkillDecision(
                SkillDecisionVerdict.REJECT, None, None, self.artifact.artifact_hash,
                "TARGET_FRAME_OUTSIDE_SKILL_PREDICATE_VOCABULARY",
            )
        if set(self.artifact.termination_facts).issubset(frame.facts):
            return SkillDecision(
                SkillDecisionVerdict.TERMINATE, None, None, self.artifact.artifact_hash,
                "FROZEN_TERMINATION_CONDITION",
            )
        if self._previous_rule is not None and previous_effect is not None:
            if not previous_effect.validate():
                return SkillDecision(
                    SkillDecisionVerdict.REJECT, None, self._previous_rule.rule_id,
                    self.artifact.artifact_hash, "PREVIOUS_EFFECT_HASH_MISMATCH",
                )
            if not previous_effect.supports(self._previous_rule):
                recovery = self._previous_rule.recovery_option
                if recovery is None:
                    return SkillDecision(
                        SkillDecisionVerdict.ABSTAIN, None, self._previous_rule.rule_id,
                        self.artifact.artifact_hash, "EXPECTED_EFFECT_REFUTED_NO_RECOVERY",
                    )
                if recovery not in frame.available_options:
                    return SkillDecision(
                        SkillDecisionVerdict.REJECT, recovery, self._previous_rule.rule_id,
                        self.artifact.artifact_hash, "RECOVERY_OPTION_NOT_GROUNDED",
                    )
                self._previous_rule = None
                return SkillDecision(
                    SkillDecisionVerdict.SELECT, recovery, None,
                    self.artifact.artifact_hash, "FAILURE_SPECIFIC_RECOVERY",
                )
        matches = [rule for rule in self.artifact.rules if rule.applies(frame.facts)]
        if not matches:
            self._previous_rule = None
            return SkillDecision(
                SkillDecisionVerdict.ABSTAIN, None, None, self.artifact.artifact_hash,
                "NO_TYPED_RULE_APPLIES",
            )
        best_priority = max(rule.priority for rule in matches)
        winners = [rule for rule in matches if rule.priority == best_priority]
        if len({rule.option for rule in winners}) != 1:
            return SkillDecision(
                SkillDecisionVerdict.REJECT, None, None, self.artifact.artifact_hash,
                "AMBIGUOUS_HIGHEST_PRIORITY_RULES",
            )
        selected = sorted(winners, key=lambda rule: rule.rule_id)[0]
        if selected.option not in frame.available_options:
            return SkillDecision(
                SkillDecisionVerdict.REJECT, selected.option, selected.rule_id,
                self.artifact.artifact_hash, "SELECTED_OPTION_NOT_GROUNDED",
            )
        self._previous_rule = selected
        return SkillDecision(
            SkillDecisionVerdict.SELECT, selected.option, selected.rule_id,
            self.artifact.artifact_hash, "TYPED_RULE_SELECTED",
        )


class TargetFallbackPolicy(Protocol):
    policy_hash: str

    def choose_option(self, frame: GroundingFrame) -> str: ...

    def choose_raw_action(self, observation: TargetObservation) -> str: ...


@dataclass(frozen=True)
class TargetStep:
    observation: TargetObservation
    reward: float
    official_success: bool
    official_score: float


class RetargetingEnvironment(Protocol):
    environment_hash: str

    def reset(self, episode_id: str) -> TargetObservation: ...

    def step(self, action: str) -> TargetStep: ...


@dataclass(frozen=True)
class RetargetingStepReceipt:
    step_index: int
    actor: str
    selected_option: str | None
    fallback_option: str | None
    native_action: str
    before_observation_hash: str
    after_observation_hash: str
    skill_artifact_hash: str | None
    harness_hash: str | None
    decision_hash: str
    effect_receipt_hash: str | None
    expected_effect_supported: bool | None
    reward: float
    official_success_after: bool
    official_score_after: float
    receipt_hash: str

    def validate(self) -> bool:
        payload = {key: value for key, value in self.__dict__.items() if key != "receipt_hash"}
        return stable_hash(payload) == self.receipt_hash


@dataclass(frozen=True)
class RetargetingEpisodeOutcome:
    pair_id: str
    condition: RetargetingCondition
    initial_observation_hash: str
    environment_hash: str
    budget: int
    fallback_policy_hash: str
    harness_hash: str | None
    skill_artifact_hash: str | None
    skill_artifact_kind: SkillArtifactKind | None
    skill_parent_artifact_hash: str | None
    skill_structure_hash: str | None
    valid: bool
    status: str
    official_success: bool
    official_score: float
    cumulative_reward: float
    steps: int
    receipts: tuple[RetargetingStepReceipt, ...]


def _expected_effect_supported(
    artifact: FrozenSkillArtifact | None,
    decision: SkillDecision,
    effect: EffectReceipt | None,
) -> bool | None:
    if artifact is None or decision.rule_id is None or effect is None:
        return None
    rule = next((row for row in artifact.rules if row.rule_id == decision.rule_id), None)
    return effect.supports(rule) if rule is not None else None


def run_retargeting_episode(
    *,
    environment: RetargetingEnvironment,
    episode_id: str,
    condition: RetargetingCondition,
    harness: PermissionBoundTargetHarness,
    fallback_policy: TargetFallbackPolicy,
    maximum_steps: int,
    skill_artifact: FrozenSkillArtifact | None = None,
) -> RetargetingEpisodeOutcome:
    """Run one condition without exposing official outcomes to skill/Harness calls."""

    condition = RetargetingCondition(condition)
    if maximum_steps <= 0:
        raise ValueError("maximum_steps must be positive")
    expected_kind = {
        RetargetingCondition.AUTHENTIC_SOURCE_SKILL: SkillArtifactKind.AUTHENTIC_SOURCE,
        RetargetingCondition.SHUFFLED_SOURCE_SKILL: SkillArtifactKind.SHUFFLED_CONTROL,
        RetargetingCondition.TARGET_ORACLE_SKILL: SkillArtifactKind.TARGET_ORACLE,
    }.get(condition)
    if expected_kind is None and skill_artifact is not None:
        raise RetargetingReject(f"condition {condition.value} must not receive a skill artifact")
    if expected_kind is not None:
        if skill_artifact is None or skill_artifact.artifact_kind != expected_kind:
            raise RetargetingReject(
                f"condition {condition.value} requires {expected_kind.value} artifact"
            )
        skill_artifact.assert_valid()
        harness_options = set(harness.artifact.option_vocabulary)
        harness_predicates = set(harness.artifact.predicate_vocabulary)
        harness_roles = set(harness.artifact.role_vocabulary)
        if set(skill_artifact.option_vocabulary) != harness_options:
            raise RetargetingReject("skill and target Harness option vocabularies differ")
        if set(skill_artifact.predicate_vocabulary) != harness_predicates:
            raise RetargetingReject("skill and target Harness predicate vocabularies differ")
        if set(skill_artifact.role_vocabulary) != harness_roles:
            raise RetargetingReject("skill and target Harness role vocabularies differ")
    observation = environment.reset(episode_id)
    initial_hash = observation.observation_hash
    execution = FrozenSkillExecution(skill_artifact)
    previous_effect: EffectReceipt | None = None
    receipts: list[RetargetingStepReceipt] = []
    cumulative_reward = 0.0
    official_success = False
    official_score = 0.0
    valid = True
    status = "BUDGET_EXHAUSTED"

    for step_index in range(maximum_steps):
        if observation.terminal:
            status = "TERMINAL"
            break
        selected_option: str | None = None
        fallback_option: str | None = None
        skill_decision = SkillDecision(
            SkillDecisionVerdict.ABSTAIN, None, None, None, "RAW_TARGET_ONLY",
        )
        frame: GroundingFrame | None = None
        native_decision: NativeActionDecision | None = None
        actor = "raw_target_policy"
        if condition == RetargetingCondition.RAW_TARGET_ONLY:
            action = fallback_policy.choose_raw_action(observation)
            if action not in observation.native_actions:
                raise RetargetingReject("raw target policy selected an inadmissible action")
            decision_hash = stable_hash({
                "actor": actor,
                "action": action,
                "observation_hash": observation.observation_hash,
                "policy_hash": fallback_policy.policy_hash,
            })
        else:
            frame = harness.ground(observation)
            fallback_option = fallback_policy.choose_option(frame)
            if fallback_option not in frame.available_options:
                raise RetargetingReject("target fallback selected an unavailable option")
            skill_decision = execution.select(frame, previous_effect)
            if skill_decision.verdict == SkillDecisionVerdict.REJECT:
                valid = False
                status = f"SKILL_REJECT:{skill_decision.reason}"
                break
            if skill_decision.verdict == SkillDecisionVerdict.TERMINATE:
                valid = False
                status = "EARLY_SKILL_TERMINATION"
                break
            if skill_decision.verdict == SkillDecisionVerdict.SELECT:
                selected_option = skill_decision.option
                actor = "source_skill" if condition != RetargetingCondition.TARGET_ORACLE_SKILL \
                    else "target_oracle_skill"
            else:
                selected_option = fallback_option
                actor = "target_fallback_policy"
            if selected_option is None:
                raise RetargetingReject("no component selected a canonical option")
            native_decision = harness.realize(selected_option, frame)
            if native_decision.selected_option != selected_option or not native_decision.validate():
                raise RetargetingReject("Harness realization crossed the selected option boundary")
            action = native_decision.action
            decision_hash = native_decision.decision_hash

        before_hash = observation.observation_hash
        target_step = environment.step(action)
        cumulative_reward += float(target_step.reward)
        official_success = bool(target_step.official_success)
        official_score = float(target_step.official_score)
        effect: EffectReceipt | None = None
        if frame is not None and native_decision is not None:
            after_frame = harness.ground(target_step.observation)
            effect = harness.verify(frame, native_decision, after_frame)
            if not effect.validate():
                raise RetargetingReject("effect receipt hash mismatch")
        effect_supported = _expected_effect_supported(skill_artifact, skill_decision, effect)
        payload = {
            "step_index": step_index,
            "actor": actor,
            "selected_option": selected_option,
            "fallback_option": fallback_option,
            "native_action": action,
            "before_observation_hash": before_hash,
            "after_observation_hash": target_step.observation.observation_hash,
            "skill_artifact_hash": (
                skill_artifact.artifact_hash if skill_artifact is not None else None
            ),
            "harness_hash": (
                harness.artifact.artifact_hash
                if condition != RetargetingCondition.RAW_TARGET_ONLY else None
            ),
            "decision_hash": decision_hash,
            "effect_receipt_hash": effect.receipt_hash if effect is not None else None,
            "expected_effect_supported": effect_supported,
            "reward": float(target_step.reward),
            "official_success_after": official_success,
            "official_score_after": official_score,
        }
        receipts.append(RetargetingStepReceipt(**payload, receipt_hash=stable_hash(payload)))
        previous_effect = effect
        observation = target_step.observation
        if observation.terminal:
            status = "TERMINAL"
            break

    if any(not row.validate() for row in receipts):
        raise RetargetingReject("episode contains an invalid step receipt")
    return RetargetingEpisodeOutcome(
        pair_id=str(episode_id),
        condition=condition,
        initial_observation_hash=initial_hash,
        environment_hash=environment.environment_hash,
        budget=int(maximum_steps),
        fallback_policy_hash=fallback_policy.policy_hash,
        harness_hash=(
            None if condition == RetargetingCondition.RAW_TARGET_ONLY
            else harness.artifact.artifact_hash
        ),
        skill_artifact_hash=(
            skill_artifact.artifact_hash if skill_artifact is not None else None
        ),
        skill_artifact_kind=(
            skill_artifact.artifact_kind if skill_artifact is not None else None
        ),
        skill_parent_artifact_hash=(
            skill_artifact.parent_artifact_hash if skill_artifact is not None else None
        ),
        skill_structure_hash=(
            skill_artifact.program_structure_hash if skill_artifact is not None else None
        ),
        valid=valid,
        status=status,
        official_success=official_success,
        official_score=official_score,
        cumulative_reward=cumulative_reward,
        steps=len(receipts),
        receipts=tuple(receipts),
    )


@dataclass(frozen=True)
class RetargetingExperimentReport:
    verdict: ExperimentVerdict
    reason: str
    metrics: Mapping[str, Any]
    outcomes: tuple[RetargetingEpisodeOutcome, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "outcomes": [
                {
                    **{key: value for key, value in row.__dict__.items() if key != "receipts"},
                    "condition": row.condition.value,
                    "skill_artifact_kind": (
                        row.skill_artifact_kind.value
                        if row.skill_artifact_kind is not None else None
                    ),
                    "receipts": [receipt.__dict__ for receipt in row.receipts],
                }
                for row in self.outcomes
            ],
        }


def evaluate_retargeting_experiment(
    outcomes: Sequence[RetargetingEpisodeOutcome],
    *,
    minimum_authentic_intervention_rate: float,
) -> RetargetingExperimentReport:
    rows = tuple(outcomes)
    if not 0.0 <= minimum_authentic_intervention_rate <= 1.0:
        raise ValueError("minimum intervention rate must be in [0, 1]")
    by_pair: dict[str, list[RetargetingEpisodeOutcome]] = {}
    for row in rows:
        by_pair.setdefault(row.pair_id, []).append(row)
    if not by_pair:
        return RetargetingExperimentReport(
            ExperimentVerdict.INVALID_EXPERIMENT, "no paired outcomes", {}, rows,
        )
    condition_kinds = {
        RetargetingCondition.RAW_TARGET_ONLY: None,
        RetargetingCondition.NULL_SKILL_SAME_HARNESS: None,
        RetargetingCondition.SHUFFLED_SOURCE_SKILL: SkillArtifactKind.SHUFFLED_CONTROL,
        RetargetingCondition.AUTHENTIC_SOURCE_SKILL: SkillArtifactKind.AUTHENTIC_SOURCE,
        RetargetingCondition.TARGET_ORACLE_SKILL: SkillArtifactKind.TARGET_ORACLE,
    }
    for pair_id, pair_rows in by_pair.items():
        names = {row.condition for row in pair_rows}
        if names != REQUIRED_CONDITIONS or len(pair_rows) != len(REQUIRED_CONDITIONS):
            return RetargetingExperimentReport(
                ExperimentVerdict.INVALID_EXPERIMENT,
                f"pair {pair_id} does not contain each required condition exactly once",
                {}, rows,
            )
        if any(not row.valid for row in pair_rows):
            return RetargetingExperimentReport(
                ExperimentVerdict.INVALID_EXPERIMENT,
                f"pair {pair_id} contains a rejected/invalid episode",
                {}, rows,
            )
        for row in pair_rows:
            expected_kind = condition_kinds[row.condition]
            if row.skill_artifact_kind != expected_kind:
                return RetargetingExperimentReport(
                    ExperimentVerdict.INVALID_EXPERIMENT,
                    f"pair {pair_id} has a condition/artifact-kind mismatch",
                    {}, rows,
                )
            if (row.skill_artifact_hash is None) != (expected_kind is None):
                return RetargetingExperimentReport(
                    ExperimentVerdict.INVALID_EXPERIMENT,
                    f"pair {pair_id} has a condition/artifact-hash mismatch",
                    {}, rows,
                )
            if row.condition == RetargetingCondition.RAW_TARGET_ONLY:
                if row.harness_hash is not None:
                    return RetargetingExperimentReport(
                        ExperimentVerdict.INVALID_EXPERIMENT,
                        f"pair {pair_id} raw condition unexpectedly used a Harness",
                        {}, rows,
                    )
            elif row.harness_hash is None:
                return RetargetingExperimentReport(
                    ExperimentVerdict.INVALID_EXPERIMENT,
                    f"pair {pair_id} core condition lacks a Harness hash",
                    {}, rows,
                )
            if row.steps != len(row.receipts) or any(
                not receipt.validate() for receipt in row.receipts
            ):
                return RetargetingExperimentReport(
                    ExperimentVerdict.INVALID_EXPERIMENT,
                    f"pair {pair_id} contains an invalid receipt chain",
                    {}, rows,
                )
            if row.receipts:
                first, last = row.receipts[0], row.receipts[-1]
                if first.before_observation_hash != row.initial_observation_hash:
                    return RetargetingExperimentReport(
                        ExperimentVerdict.INVALID_EXPERIMENT,
                        f"pair {pair_id} initial observation does not match its receipt chain",
                        {}, rows,
                    )
                if (
                    last.official_success_after != row.official_success
                    or not math.isclose(
                        last.official_score_after, row.official_score,
                        rel_tol=0.0, abs_tol=1e-12,
                    )
                    or not math.isclose(
                        sum(receipt.reward for receipt in row.receipts),
                        row.cumulative_reward,
                        rel_tol=0.0, abs_tol=1e-12,
                    )
                ):
                    return RetargetingExperimentReport(
                        ExperimentVerdict.INVALID_EXPERIMENT,
                        f"pair {pair_id} official outcome does not match its receipt chain",
                        {}, rows,
                    )
        identity_fields = (
            "initial_observation_hash", "environment_hash", "budget", "fallback_policy_hash",
        )
        for field in identity_fields:
            if len({getattr(row, field) for row in pair_rows}) != 1:
                return RetargetingExperimentReport(
                    ExperimentVerdict.INVALID_EXPERIMENT,
                    f"pair {pair_id} has {field} mismatch",
                    {}, rows,
                )
        core = [row for row in pair_rows if row.condition in CORE_CONDITIONS]
        harness_hashes = {row.harness_hash for row in core}
        if len(harness_hashes) != 1 or None in harness_hashes:
            return RetargetingExperimentReport(
                ExperimentVerdict.INVALID_EXPERIMENT,
                f"pair {pair_id} did not use one frozen Harness for all core conditions",
                {}, rows,
            )

    for field in ("environment_hash", "budget", "fallback_policy_hash"):
        if len({getattr(row, field) for row in rows}) != 1:
            return RetargetingExperimentReport(
                ExperimentVerdict.INVALID_EXPERIMENT,
                f"experiment uses more than one frozen {field}",
                {}, rows,
            )
    global_harness_hashes = {
        row.harness_hash for row in rows if row.condition in CORE_CONDITIONS
    }
    if len(global_harness_hashes) != 1:
        return RetargetingExperimentReport(
            ExperimentVerdict.INVALID_EXPERIMENT,
            "core conditions do not share one global target Harness",
            {}, rows,
        )
    skill_hashes: dict[RetargetingCondition, set[str | None]] = {
        condition: {row.skill_artifact_hash for row in group}
        for condition, group in {
            condition: [row for row in rows if row.condition == condition]
            for condition in REQUIRED_CONDITIONS
        }.items()
    }
    if any(len(hashes) != 1 for hashes in skill_hashes.values()):
        return RetargetingExperimentReport(
            ExperimentVerdict.INVALID_EXPERIMENT,
            "a condition changed skill artifacts across paired episodes",
            {}, rows,
        )
    authentic_hash = next(iter(skill_hashes[RetargetingCondition.AUTHENTIC_SOURCE_SKILL]))
    shuffled_hash = next(iter(skill_hashes[RetargetingCondition.SHUFFLED_SOURCE_SKILL]))
    oracle_hash = next(iter(skill_hashes[RetargetingCondition.TARGET_ORACLE_SKILL]))
    if None in (authentic_hash, shuffled_hash, oracle_hash) or len({
        authentic_hash, shuffled_hash, oracle_hash,
    }) != 3:
        return RetargetingExperimentReport(
            ExperimentVerdict.INVALID_EXPERIMENT,
            "authentic, shuffled, and oracle artifacts are not distinct and frozen",
            {}, rows,
        )
    shuffled_parents = {
        row.skill_parent_artifact_hash for row in rows
        if row.condition == RetargetingCondition.SHUFFLED_SOURCE_SKILL
    }
    if shuffled_parents != {authentic_hash}:
        return RetargetingExperimentReport(
            ExperimentVerdict.INVALID_EXPERIMENT,
            "shuffled control is not derived from the authentic frozen artifact",
            {}, rows,
        )
    authentic_structures = {
        row.skill_structure_hash for row in rows
        if row.condition == RetargetingCondition.AUTHENTIC_SOURCE_SKILL
    }
    shuffled_structures = {
        row.skill_structure_hash for row in rows
        if row.condition == RetargetingCondition.SHUFFLED_SOURCE_SKILL
    }
    if (
        len(authentic_structures) != 1
        or None in authentic_structures
        or authentic_structures != shuffled_structures
    ):
        return RetargetingExperimentReport(
            ExperimentVerdict.INVALID_EXPERIMENT,
            "shuffled control changed guards/effects instead of only option labels",
            {}, rows,
        )

    condition_rows = {
        condition: [row for row in rows if row.condition == condition]
        for condition in REQUIRED_CONDITIONS
    }
    summaries = {
        condition.value: {
            "episodes": len(group),
            "successes": sum(row.official_success for row in group),
            "success_rate": mean(float(row.official_success) for row in group),
            "mean_score": mean(row.official_score for row in group),
            "mean_steps": mean(row.steps for row in group),
        }
        for condition, group in condition_rows.items()
    }
    authentic_rows = condition_rows[RetargetingCondition.AUTHENTIC_SOURCE_SKILL]
    authentic_receipts = [receipt for row in authentic_rows for receipt in row.receipts]
    interventions = [
        receipt for receipt in authentic_receipts
        if receipt.actor == "source_skill"
        and receipt.selected_option != receipt.fallback_option
    ]
    supported_effects = [
        receipt for receipt in interventions
        if receipt.expected_effect_supported is not None
    ]
    intervention_rate = len(interventions) / max(1, len(authentic_receipts))
    applicability = (
        sum(bool(row.expected_effect_supported) for row in supported_effects)
        / len(supported_effects)
        if supported_effects else 0.0
    )
    paired = {
        row.pair_id: row for row in authentic_rows
    }
    null_by_pair = {
        row.pair_id: row
        for row in condition_rows[RetargetingCondition.NULL_SKILL_SAME_HARNESS]
    }
    rescues = sum(
        paired[pair_id].official_success and not null_by_pair[pair_id].official_success
        for pair_id in paired
    )
    regressions = sum(
        not paired[pair_id].official_success and null_by_pair[pair_id].official_success
        for pair_id in paired
    )
    auth_success = summaries[RetargetingCondition.AUTHENTIC_SOURCE_SKILL.value]["successes"]
    null_success = summaries[RetargetingCondition.NULL_SKILL_SAME_HARNESS.value]["successes"]
    shuffled_success = summaries[RetargetingCondition.SHUFFLED_SOURCE_SKILL.value]["successes"]
    oracle_success = summaries[RetargetingCondition.TARGET_ORACLE_SKILL.value]["successes"]
    metrics = {
        "summaries": summaries,
        "authentic_intervention_rate": intervention_rate,
        "minimum_authentic_intervention_rate": minimum_authentic_intervention_rate,
        "authentic_effect_applicability": applicability,
        "authentic_vs_null_paired_rescues": rescues,
        "authentic_vs_null_paired_regressions": regressions,
        "same_harness_verified": True,
        "paired_identity_verified": True,
    }
    if auth_success < null_success:
        return RetargetingExperimentReport(
            ExperimentVerdict.NEGATIVE_TRANSFER,
            "authentic source skill harms same-Harness official success",
            metrics,
            rows,
        )
    gates = (
        auth_success > null_success,
        auth_success > shuffled_success,
        oracle_success >= auth_success,
        intervention_rate >= minimum_authentic_intervention_rate,
        bool(interventions),
        applicability > 0.0,
    )
    if all(gates):
        return RetargetingExperimentReport(
            ExperimentVerdict.MECHANISM_SUPPORTED,
            "authentic frozen structure beats null and shuffled under one target Harness",
            metrics,
            rows,
        )
    return RetargetingExperimentReport(
        ExperimentVerdict.NO_ATTRIBUTABLE_GAIN,
        "the complete same-Harness attribution gates were not satisfied",
        metrics,
        rows,
    )


__all__ = [
    "CORE_CONDITIONS",
    "REQUIRED_CONDITIONS",
    "EffectReceipt",
    "ExperimentVerdict",
    "FrozenSkillArtifact",
    "FrozenSkillExecution",
    "GroundingDraft",
    "GroundingFrame",
    "NativeActionCandidate",
    "NativeActionDecision",
    "OptionGrounding",
    "PermissionBoundTargetHarness",
    "RetargetingCondition",
    "RetargetingEnvironment",
    "RetargetingEpisodeOutcome",
    "RetargetingExperimentReport",
    "RetargetingReject",
    "RetargetingStepReceipt",
    "SkillArtifactKind",
    "SkillDecision",
    "SkillDecisionVerdict",
    "SourceQualification",
    "SymbolicOptionRule",
    "TargetFallbackPolicy",
    "TargetGrounder",
    "TargetHarnessArtifact",
    "TargetObservation",
    "TargetStep",
    "evaluate_retargeting_experiment",
    "run_retargeting_episode",
]
