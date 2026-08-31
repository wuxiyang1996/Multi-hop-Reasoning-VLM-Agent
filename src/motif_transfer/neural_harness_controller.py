"""Fail-closed neural proposal layer for the symbolic Harness controller.

The neural model proposes one complete controller output.  Exact executor
comparison is allowed only in non-acting shadow qualification.  Live mode
checks the output schema, state ledger, and declared graph edge without
recomputing which grounded candidate should win.  It never repairs a neural
output or substitutes a Python-selected binding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence

from .contracts import stable_hash
from .harness_controller_training import (
    ABSTAIN,
    ACTIVE,
    ENTRY,
    TERMINAL,
    anonymous_program_ir,
    execute_controller_step,
    format_controller_prompt,
)
from .phase3_typed_effect_induction import TYPED_EFFECTS


EXACT_SHADOW = "EXACT_SHADOW"
STRUCTURAL_ONLY = "STRUCTURAL_ONLY"


class ControllerGenerator(Protocol):
    """Minimal inference boundary; implementations may use local or API models."""

    artifact_sha256: str

    def generate(self, prompt: str) -> str:
        """Return one JSON-only controller completion."""


@dataclass(frozen=True)
class NeuralControllerReceipt:
    status: str
    verification_mode: str
    shadow_only: bool
    objective: str
    source_program_sha256: str
    controller_artifact_sha256: str
    controller_input_sha256: str
    prompt_sha256: str
    generated_text_sha256: str
    parsed_output: Mapping[str, Any] | None
    symbolic_verifier_output_sha256: str | None
    structural_contract_valid: bool
    exact_symbolic_match: bool | None
    source_operator_authorized: bool
    authorized_operator_id: str | None
    authorized_binding: Mapping[str, Any] | None
    target_outcome_read: bool
    target_action_emitted: bool
    reason: str
    receipt_sha256: str

    @classmethod
    def create(cls, **values: Any) -> "NeuralControllerReceipt":
        core = dict(values)
        core["receipt_sha256"] = stable_hash(core)
        return cls(**core)

    def validate(self) -> None:
        body = asdict(self)
        claimed = body.pop("receipt_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("neural controller receipt hash mismatch")
        if self.target_outcome_read or self.target_action_emitted:
            raise ValueError("neural controller crossed its authority boundary")
        if self.verification_mode not in {EXACT_SHADOW, STRUCTURAL_ONLY}:
            raise ValueError("unknown neural controller verification mode")
        if self.shadow_only != (self.verification_mode == EXACT_SHADOW):
            raise ValueError("shadow receipt/mode mismatch")
        if self.source_operator_authorized:
            if (
                self.shadow_only
                or not self.structural_contract_valid
                or self.authorized_operator_id is None
                or self.authorized_binding is None
            ):
                raise ValueError("unverified source operator was authorized")
        elif self.authorized_operator_id is not None or self.authorized_binding is not None:
            raise ValueError("rejected neural output retained execution authority")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strict_json(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _checked_state(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {
        "controller_state", "active_candidate", "attempted_candidates", "trials_used",
    }:
        raise ValueError("controller state schema mismatch")
    controller_state = str(value["controller_state"])
    if controller_state not in {ENTRY, ACTIVE, TERMINAL, ABSTAIN}:
        raise ValueError("unknown controller state")
    attempted = list(map(str, value["attempted_candidates"]))
    if len(attempted) != len(set(attempted)):
        raise ValueError("attempt ledger contains duplicates")
    trials = int(value["trials_used"])
    if trials != len(attempted) or trials < 0:
        raise ValueError("attempt ledger/trial count mismatch")
    active = value["active_candidate"]
    active = None if active is None else str(active)
    if controller_state == ACTIVE and active not in attempted:
        raise ValueError("active candidate is absent from attempt ledger")
    return {
        "controller_state": controller_state,
        "active_candidate": active,
        "attempted_candidates": attempted,
        "trials_used": trials,
    }


def _candidate_ids(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    ids = [str(candidate.get("candidate_id")) for candidate in candidates]
    if (
        len(ids) < 2 or len(ids) != len(set(ids))
        or any(candidate_id in {"", "None"} for candidate_id in ids)
    ):
        raise ValueError("candidate identities are not executable")
    for candidate in candidates:
        effects = candidate.get("effects")
        if not isinstance(effects, Mapping) or set(effects) != set(TYPED_EFFECTS):
            raise ValueError("candidate effect schema mismatch")
        for effect in TYPED_EFFECTS:
            value = effects[effect]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("candidate effect is not numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("candidate effect is outside [0,1]")
    return ids


def structural_controller_output_valid(
    *, anonymous_program: Mapping[str, Any], state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]], observed_effect: str | None,
    output: Mapping[str, Any] | None,
) -> bool:
    """Check authority invariants without recomputing the candidate argmax."""

    if output is None or set(output) != {
        "decision", "operator_id", "binding", "next_symbolic_state", "reason",
    }:
        return False
    try:
        current = _checked_state(state)
        next_state = _checked_state(output["next_symbolic_state"])
    except (KeyError, TypeError, ValueError):
        return False
    decision = str(output["decision"])
    reason = output["reason"]
    if not isinstance(reason, str) or not reason:
        return False

    if decision == ABSTAIN:
        return (
            output["operator_id"] is None
            and output["binding"] is None
            and next_state == {
                **current, "controller_state": ABSTAIN, "active_candidate": None,
            }
        )

    if current["controller_state"] == ENTRY:
        if decision != "EXECUTE_OPERATOR" or observed_effect is not None:
            return False
        if anonymous_program.get("qualification_status") != "QUALIFIED":
            return False
        try:
            ids = _candidate_ids(candidates)
        except ValueError:
            return False
        operators = {
            str(operator.get("operator_id"))
            for operator in anonymous_program.get("operators") or ()
        }
        binding = output["binding"]
        if (
            str(output["operator_id"]) not in operators
            or not isinstance(binding, Mapping)
            or set(binding) != {"active_candidate"}
        ):
            return False
        selected = str(binding["active_candidate"])
        next_attempted = [*current["attempted_candidates"], selected]
        return (
            selected in ids
            and selected not in current["attempted_candidates"]
            and len(next_attempted) <= int(anonymous_program["maximum_trials"])
            and next_state == {
                "controller_state": ACTIVE,
                "active_candidate": selected,
                "attempted_candidates": next_attempted,
                "trials_used": len(next_attempted),
            }
        )

    if current["controller_state"] != ACTIVE:
        return False
    if output["operator_id"] is not None or output["binding"] is not None:
        return False
    allowed = {
        "OBSERVED_EFFECT_HIGH", "OBSERVED_EFFECT_LOW", "OBSERVED_EFFECT_UNKNOWN",
    }
    guard = str(observed_effect or "OBSERVED_EFFECT_UNKNOWN")
    if guard not in allowed:
        guard = "OBSERVED_EFFECT_UNKNOWN"
    edges = [
        edge for edge in anonymous_program["transition_graph"]["transitions"]
        if edge["from"] == ACTIVE and edge["guard"] == guard
    ]
    if len(edges) != 1:
        return False
    destination = str(edges[0]["to"])
    expected_decision = "TERMINATE" if destination == TERMINAL else "ADVANCE_STATE"
    if destination == ABSTAIN:
        expected_decision = ABSTAIN
    return (
        decision == expected_decision
        and reason == guard
        and next_state == {
            **current, "controller_state": destination, "active_candidate": None,
        }
    )


class NeuralHarnessController:
    """Neural controller with explicitly separated shadow and live authority."""

    def __init__(self, generator: ControllerGenerator, *, verification_mode: str):
        artifact = str(getattr(generator, "artifact_sha256", ""))
        if len(artifact) != 64:
            raise ValueError("controller generator requires a SHA-256 artifact identity")
        if verification_mode not in {EXACT_SHADOW, STRUCTURAL_ONLY}:
            raise ValueError("verification_mode must be EXACT_SHADOW or STRUCTURAL_ONLY")
        self.generator = generator
        self.artifact_sha256 = artifact
        self.verification_mode = verification_mode

    def decide(
        self, *, program: Mapping[str, Any], state: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        observed_effect: str | None = None,
    ) -> NeuralControllerReceipt:
        anonymous_program = anonymous_program_ir(program)
        objective = (
            "APPLY_TRANSITION"
            if str(state.get("controller_state")) == ACTIVE
            else "SELECT_OPERATOR"
        )
        input_payload = {
            "program": anonymous_program,
            "symbolic_state": dict(state),
            "candidate_effects": [dict(candidate) for candidate in candidates],
            "observed_effect": observed_effect,
        }
        prompt = format_controller_prompt(
            objective=objective, input_payload=input_payload,
        )
        inference_error = None
        try:
            generated = self.generator.generate(prompt)
            if not isinstance(generated, str):
                raise TypeError("controller generator returned a non-string value")
        except Exception as error:  # Fail closed at the model boundary.
            generated = ""
            inference_error = type(error).__name__
        parsed = _strict_json(generated)
        structural_valid = structural_controller_output_valid(
            anonymous_program=anonymous_program, state=state,
            candidates=candidates, observed_effect=observed_effect, output=parsed,
        )
        verifier_output = None
        exact = None
        if self.verification_mode == EXACT_SHADOW:
            verifier_output = execute_controller_step(
                program, state=state, candidates=candidates,
                observed_effect=observed_effect,
            )
            exact = parsed == verifier_output
        operator_authorized = bool(
            self.verification_mode == STRUCTURAL_ONLY
            and structural_valid and parsed is not None
            and parsed.get("decision") == "EXECUTE_OPERATOR"
        )
        if inference_error is not None:
            status = "NEURAL_CONTROLLER_INFERENCE_FAILED"
            reason = f"INFERENCE_ERROR:{inference_error}"
        elif parsed is None:
            status = "NEURAL_CONTROLLER_OUTPUT_REJECTED"
            reason = "INVALID_JSON_OBJECT"
        elif not structural_valid:
            status = "NEURAL_CONTROLLER_OUTPUT_REJECTED"
            reason = "STRUCTURAL_CONTRACT_MISMATCH"
        elif self.verification_mode == EXACT_SHADOW and not exact:
            status = "NEURAL_CONTROLLER_OUTPUT_REJECTED"
            reason = "SYMBOLIC_VERIFIER_MISMATCH"
        else:
            status = (
                "NEURAL_CONTROLLER_EXACT_SHADOW_VERIFIED"
                if self.verification_mode == EXACT_SHADOW
                else "NEURAL_CONTROLLER_STRUCTURALLY_VERIFIED"
            )
            reason = (
                "EXACT_FROZEN_SYMBOLIC_EXECUTOR_MATCH"
                if self.verification_mode == EXACT_SHADOW
                else "DECLARED_SCHEMA_STATE_AND_GRAPH_CONTRACT_VALID"
            )
        receipt = NeuralControllerReceipt.create(
            status=status,
            verification_mode=self.verification_mode,
            shadow_only=self.verification_mode == EXACT_SHADOW,
            objective=objective,
            source_program_sha256=str(program["program_sha256"]),
            controller_artifact_sha256=self.artifact_sha256,
            controller_input_sha256=stable_hash(input_payload),
            prompt_sha256=_sha256_text(prompt),
            generated_text_sha256=_sha256_text(generated),
            parsed_output=parsed,
            symbolic_verifier_output_sha256=(
                stable_hash(verifier_output) if verifier_output is not None else None
            ),
            structural_contract_valid=structural_valid,
            exact_symbolic_match=exact,
            source_operator_authorized=operator_authorized,
            authorized_operator_id=(
                str(parsed["operator_id"]) if operator_authorized else None
            ),
            authorized_binding=(
                dict(parsed["binding"]) if operator_authorized else None
            ),
            target_outcome_read=False,
            target_action_emitted=False,
            reason=reason,
        )
        receipt.validate()
        return receipt


__all__ = [
    "ControllerGenerator", "EXACT_SHADOW", "NeuralControllerReceipt",
    "NeuralHarnessController", "STRUCTURAL_ONLY",
    "structural_controller_output_valid",
]
