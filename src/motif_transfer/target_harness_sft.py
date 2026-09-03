"""Target-development supervision for the learned symbolic Harness controller.

Target-native grounders may expose probabilities in the unchanged four-field
typed-effect vocabulary.  Labels are still computed only by executing a frozen
source-induced program.  Native actions, domain identity, reward, and official
success are audit inputs and never enter the controller prompt or completion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .harness_controller_training import (
    anonymous_program_ir,
    execute_controller_step,
    initial_controller_state,
)
from .phase3_source_function_induction import QUALIFIED
from .phase3_typed_effect_induction import TYPED_EFFECTS


@dataclass(frozen=True)
class GroundedTargetState:
    """One target state after neural grounding and action anonymization."""

    target_domain: str
    target_task_sha256: str
    split: str
    state_receipt_sha256: str
    grounder_artifact_sha256: str
    candidates: tuple[Mapping[str, Any], ...]

    def validate(self) -> bool:
        ids = [str(row.get("candidate_id")) for row in self.candidates]
        return (
            self.split in {"train", "validation"}
            and len(ids) >= 2
            and len(ids) == len(set(ids))
            and all(value.startswith("C") for value in ids)
            and all(
                set(row.get("effects") or {}) == set(TYPED_EFFECTS)
                and all(
                    0.0 <= float(row["effects"][name]) <= 1.0
                    for name in TYPED_EFFECTS
                )
                for row in self.candidates
            )
        )


@dataclass(frozen=True)
class TargetHarnessSFTExample:
    """One controller example with target provenance kept out of the prompt."""

    example_id: str
    target_domain: str
    target_task_sha256: str
    source_family: str
    split: str
    objective: str
    control_variant: str
    pair_id: str
    input_payload: Mapping[str, Any]
    target_payload: Mapping[str, Any]
    evidence_receipt_ids: tuple[str, ...]
    derivation: str
    target_data_used: bool = True
    target_outcome_used: bool = False
    native_action_exposed: bool = False

    def validate(self) -> bool:
        body = asdict(self)
        claimed = body.pop("example_id")
        model_text = str(self.input_payload) + str(self.target_payload)
        return (
            claimed == stable_hash(body)
            and self.split in {"train", "validation"}
            and self.objective == "SELECT_OPERATOR"
            and bool(self.evidence_receipt_ids)
            and self.target_data_used is True
            and self.target_outcome_used is False
            and self.native_action_exposed is False
            and self.target_domain not in model_text
            and self.target_task_sha256 not in model_text
            and self.source_family not in model_text
            and self.control_variant not in model_text
            and all(token not in model_text for token in (
                "official_success", "official_reward", "selected_action",
                "expert_action", "native_actions", "target_domain",
            ))
        )


def _example(
    *, state: GroundedTargetState, source_family: str,
    control_variant: str, pair_id: str, input_payload: Mapping[str, Any],
    target_payload: Mapping[str, Any], program_receipt: str,
) -> TargetHarnessSFTExample:
    body = {
        "target_domain": state.target_domain,
        "target_task_sha256": state.target_task_sha256,
        "source_family": source_family,
        "split": state.split,
        "objective": "SELECT_OPERATOR",
        "control_variant": control_variant,
        "pair_id": pair_id,
        "input_payload": dict(input_payload),
        "target_payload": dict(target_payload),
        "evidence_receipt_ids": (
            state.state_receipt_sha256,
            state.grounder_artifact_sha256,
            str(program_receipt),
        ),
        "derivation": "FROZEN_SOURCE_PROGRAM_OVER_TARGET_NEURAL_GROUNDING",
        "target_data_used": True,
        "target_outcome_used": False,
        "native_action_exposed": False,
    }
    return TargetHarnessSFTExample(example_id=stable_hash(body), **body)


def _rotate_effects(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": str(row["candidate_id"]),
            "effects": dict(candidates[(index + 1) % len(candidates)]["effects"]),
        }
        for index, row in enumerate(candidates)
    ]


def _tied_effects(
    candidates: Sequence[Mapping[str, Any]], selected_id: str,
) -> list[dict[str, Any]]:
    selected = next(
        dict(row["effects"])
        for row in candidates if str(row["candidate_id"]) == selected_id
    )
    return [
        {"candidate_id": str(row["candidate_id"]), "effects": dict(selected)}
        for row in candidates
    ]


def _missing_effect(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = [
        {"candidate_id": str(row["candidate_id"]), "effects": dict(row["effects"])}
        for row in candidates
    ]
    output[0]["effects"].pop(TYPED_EFFECTS[-1])
    return output


def build_matched_target_pair(
    *, state: GroundedTargetState, source_family: str,
    program: Mapping[str, Any], program_receipt: str,
) -> tuple[TargetHarnessSFTExample, TargetHarnessSFTExample] | None:
    """Build one authentic target grounding and one executor-labelled control."""

    if not state.validate() or program.get("status") != QUALIFIED:
        return None
    candidates = [
        {"candidate_id": str(row["candidate_id"]), "effects": dict(row["effects"])}
        for row in state.candidates
    ]
    base_input = {
        "program": anonymous_program_ir(program),
        "symbolic_state": initial_controller_state(),
        "candidate_effects": candidates,
        "observed_effect": None,
    }
    authentic_target = execute_controller_step(
        program, state=base_input["symbolic_state"], candidates=candidates,
    )
    if authentic_target["decision"] != "EXECUTE_OPERATOR":
        return None
    pair_id = stable_hash({
        "state_receipt": state.state_receipt_sha256,
        "source_program": str(program_receipt),
    })
    authentic = _example(
        state=state, source_family=source_family,
        control_variant="AUTHENTIC_TARGET_NEURAL_GROUNDING",
        pair_id=pair_id, input_payload=base_input,
        target_payload=authentic_target, program_receipt=program_receipt,
    )
    selector = int(pair_id[:8], 16) % 3
    if selector == 0:
        variant = "TARGET_EFFECT_BINDING_PERMUTED"
        control_candidates = _rotate_effects(candidates)
    elif selector == 1:
        variant = "TARGET_FUNCTION_ARGMAX_TIED"
        control_candidates = _tied_effects(
            candidates, str(authentic_target["binding"]["active_candidate"]),
        )
    else:
        variant = "TARGET_EFFECT_SCHEMA_MISSING"
        control_candidates = _missing_effect(candidates)
    control_input = {
        **base_input,
        "candidate_effects": control_candidates,
    }
    control_target = execute_controller_step(
        program, state=control_input["symbolic_state"],
        candidates=control_candidates,
    )
    if control_target == authentic_target:
        variant = "TARGET_FUNCTION_ARGMAX_TIED"
        control_candidates = _tied_effects(
            candidates, str(authentic_target["binding"]["active_candidate"]),
        )
        control_input = {**base_input, "candidate_effects": control_candidates}
        control_target = execute_controller_step(
            program, state=control_input["symbolic_state"],
            candidates=control_candidates,
        )
    if control_target == authentic_target:
        raise AssertionError("matched target control did not change executor output")
    control = _example(
        state=state, source_family=source_family,
        control_variant=variant, pair_id=pair_id,
        input_payload=control_input, target_payload=control_target,
        program_receipt=program_receipt,
    )
    if not authentic.validate() or not control.validate():
        raise ValueError("invalid target Harness SFT pair")
    return authentic, control


__all__ = [
    "GroundedTargetState", "TargetHarnessSFTExample",
    "build_matched_target_pair",
]
