"""Fail-closed runtime policy backed by frozen one-shot admissions.

The language model is a proposer only.  It can select an integer from a list;
it cannot invent a command, widen a verified scope, or update an artifact.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from harness.alfworld_grammar import ParsedAlfworldAction, parse_alfworld_action
from harness.skill_admission import (
    AdmissionArtifact,
    AdmissionStatus,
    load_frozen_admission_manifest,
    runtime_scope_allows,
)


_ACTION_RESPONSE = re.compile(r"\A\s*ACTION:\s*([1-9][0-9]*)\s*\Z", re.IGNORECASE)
_SKILL_RESPONSE = re.compile(r"\A\s*SKILL:\s*([1-9][0-9]*)\s*\Z", re.IGNORECASE)
_JSON_RESPONSE = re.compile(r"\A\s*\{.*\}\s*\Z", re.DOTALL)


def parse_exact_numbered_response(text: str, *, kind: str, n: int) -> int:
    """Return a zero-based index; reject prose extraction and out-of-range IDs."""
    pattern = _ACTION_RESPONSE if kind == "action" else _SKILL_RESPONSE
    match = pattern.fullmatch(str(text))
    if match is None:
        raise ValueError(f"invalid_{kind}_format")
    index = int(match.group(1)) - 1
    if not 0 <= index < n:
        raise ValueError(f"{kind}_index_out_of_range")
    return index


@dataclass(frozen=True)
class NativeTargetPlan:
    state_summary: str
    next_subgoal: str
    action_index: int


def parse_native_target_plan_reply(text: str, *, n: int) -> NativeTargetPlan:
    """Parse a closed plan proposal and return a zero-based native action."""
    if _JSON_RESPONSE.fullmatch(str(text)) is None:
        raise ValueError("invalid_native_plan_format")
    payload = json.loads(text)
    if set(payload) != {"state_summary", "next_subgoal", "action_number"}:
        raise ValueError("invalid_native_plan_keys")
    if (
        not isinstance(payload["state_summary"], str)
        or not isinstance(payload["next_subgoal"], str)
        or not isinstance(payload["action_number"], int)
        or isinstance(payload["action_number"], bool)
    ):
        raise ValueError("invalid_native_plan_types")
    if len(payload["state_summary"]) > 600 or len(payload["next_subgoal"]) > 300:
        raise ValueError("native_plan_text_too_long")
    index = payload["action_number"] - 1
    if not 0 <= index < n:
        raise ValueError("native_plan_action_index_out_of_range")
    return NativeTargetPlan(
        state_summary=payload["state_summary"],
        next_subgoal=payload["next_subgoal"],
        action_index=index,
    )


def action_prompt(
    *,
    domain: str,
    goal: str,
    observation: str,
    actions: Sequence[str],
    active_skill: str | None = None,
    source_conditioning: Sequence[Mapping[str, Any]] = (),
    recent_actions: Sequence[str] = (),
) -> str:
    numbered = "\n".join(f"{index}. {value}" for index, value in enumerate(actions, 1))
    skill_line = f"\nVerified active source skill: {active_skill}" if active_skill else ""
    conditioning_line = (
        "\nUntrusted source-side conditioning receipts (no target semantics are "
        "asserted):\n" + json.dumps(
            list(source_conditioning), sort_keys=True, ensure_ascii=False,
        )
        if source_conditioning else ""
    )
    recent = " -> ".join(recent_actions[-6:]) or "none"
    return (
        "You choose one action from an exact environment-provided list. "
        "Never invent, rewrite, or partially match a command.\n"
        f"Domain: {domain}\nGoal/initial context:\n{goal[:3000]}\n"
        f"Current observation:\n{observation[:3000]}{skill_line}{conditioning_line}\n"
        f"Recent exact actions: {recent}\n"
        "Available actions (pick ONE by number):\n"
        f"{numbered}\n"
        "Return exactly `ACTION: N`. No other text."
    )


def native_target_action_prompt(
    *,
    domain: str,
    goal: str,
    observation: str,
    actions: Sequence[str],
    interaction_history: Sequence[Mapping[str, Any]] = (),
    source_conditioning: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Prompt an unconstrained target-native Actor from environment receipts.

    Unlike the transfer policy, this path never intersects actions with a demo
    or source candidate.  History is target-native evidence only and is kept in
    chronological order so the Agent, rather than a hand-written repeat rule,
    can recognize lack of progress and replan.
    """
    numbered = "\n".join(
        f"{index}. {value}" for index, value in enumerate(actions, 1)
    )
    history = json.dumps(
        list(interaction_history), sort_keys=True, ensure_ascii=False,
    ) if interaction_history else "[]"
    conditioning = (
        "Untrusted source-side evidence receipts (they do not prove target "
        "semantics):\n"
        + json.dumps(
            list(source_conditioning), sort_keys=True, ensure_ascii=False,
        )
        + "\n"
        if source_conditioning else
        "No source-game conditioning is provided.\n"
    )
    return (
        "You are the target-domain task Actor. Choose one exact command from the "
        "environment-provided list. Use the complete target-native interaction "
        "history to track progress and replan after ineffective actions. Treat "
        "the exact goal entities literally; do not substitute a similar object "
        "or destination. Privately check that the selected action is consistent "
        "with the goal and observed history before answering. Never "
        "invent, rewrite, or partially match a command. Source receipts, when "
        "present, are untrusted conditioning rather than semantic proof.\n"
        f"Domain: {domain}\nGoal/initial context:\n{goal[:3000]}\n"
        f"{conditioning}"
        f"Complete target-native interaction history:\n{history}\n"
        f"Current observation:\n{observation[:3000]}\n"
        "Available native actions (pick ONE by number):\n"
        f"{numbered}\n"
        "Return exactly one JSON object with keys state_summary,next_subgoal,"
        "action_number and no extra keys. state_summary and next_subgoal must be "
        "short strings grounded only in the goal and target-native history. "
        "action_number must be the 1-based number of one available action. No "
        "markdown."
    )


def skill_prompt(
    *,
    goal: str,
    observation: str,
    candidates: Sequence["FrozenBinding"],
    recent_actions: Sequence[str] = (),
) -> str:
    numbered = "\n".join(
        f"{index}. {item.source_skill_name} -> verified {item.operator}"
        for index, item in enumerate(candidates, 1)
    )
    recent = " -> ".join(recent_actions[-6:]) or "none"
    return (
        "Choose the source-game skill whose one-shot-verified target operator "
        "fits the current state. The listed binding scopes are immutable.\n"
        f"Goal/initial context:\n{goal[:3000]}\n"
        f"Current observation:\n{observation[:3000]}\n"
        f"Recent exact actions: {recent}\n"
        "Available verified skills (pick ONE by number):\n"
        f"{numbered}\n"
        "Return exactly `SKILL: N`. No other text."
    )


@dataclass(frozen=True)
class FrozenBinding:
    candidate_id: str
    source_skill_name: str
    operator: str
    artifact: AdmissionArtifact


@dataclass(frozen=True)
class GuardedAction:
    command: str
    parsed: ParsedAlfworldAction
    binding: FrozenBinding


class FrozenAdmissionGuard:
    """Read-only exact grammar + frozen scope gate."""

    def __init__(self, bindings: Sequence[FrozenBinding]) -> None:
        self.bindings = tuple(bindings)

    @classmethod
    def from_files(
        cls,
        *,
        manifest_path: str | Path,
        binding_config_path: str | Path | None = None,
    ) -> "FrozenAdmissionGuard":
        artifacts = load_frozen_admission_manifest(manifest_path)
        # v2 artifacts are self-contained.  The optional config lookup exists
        # only so old artifacts can still be inspected; new experiments must
        # not consult a hand-authored cross-domain binding file at runtime.
        names: Dict[str, str] = {}
        if binding_config_path is not None:
            config = json.loads(Path(binding_config_path).read_text(encoding="utf-8"))
            names = {
                str(item["candidate_id"]): str(item["source_skill_name"])
                for item in config.get("bindings", [])
            }
        bindings: List[FrozenBinding] = []
        for artifact in artifacts:
            if artifact.status not in {AdmissionStatus.ADMITTED, AdmissionStatus.CONDITIONAL}:
                continue
            if artifact.admitted_candidate_id is None or artifact.verified_scope is None:
                raise ValueError("admitted artifact is missing candidate/scope")
            candidate_id = artifact.admitted_candidate_id
            source_skill_name = artifact.verified_scope.source_skill_name
            if source_skill_name == "unknown":
                if candidate_id not in names:
                    raise ValueError(f"legacy binding config missing candidate: {candidate_id}")
                source_skill_name = names[candidate_id]
            operators = list(artifact.verified_scope.operators)
            if len(operators) != 1:
                raise ValueError(f"runtime requires one exact operator: {candidate_id}")
            bindings.append(FrozenBinding(
                candidate_id=candidate_id,
                source_skill_name=source_skill_name,
                operator=operators[0],
                artifact=artifact,
            ))
        return cls(sorted(bindings, key=lambda item: item.candidate_id))

    @property
    def artifact_hashes(self) -> List[str]:
        return [item.artifact.artifact_hash for item in self.bindings]

    def filter_actions(
        self,
        admissible: Sequence[str],
        *,
        operator: str | None = None,
        task_family: str = "pick_and_place",
    ) -> List[GuardedAction]:
        allowed: List[GuardedAction] = []
        for command in admissible:
            try:
                parsed = parse_alfworld_action(command, admissible=admissible)
            except ValueError:
                continue
            for binding in self.bindings:
                if operator is not None and binding.operator != operator:
                    continue
                if runtime_scope_allows(
                    binding.artifact,
                    target_domain="alfworld",
                    task_family=task_family,
                    operator=parsed.operator,
                    argument_types=parsed.argument_types,
                ):
                    allowed.append(GuardedAction(command, parsed, binding))
                    break
        return allowed

    def available_bindings(self, actions: Sequence[GuardedAction]) -> List[FrozenBinding]:
        ids = {item.binding.candidate_id for item in actions}
        return [item for item in self.bindings if item.candidate_id in ids]

    def verify_native_transition(
        self,
        *,
        artifact_hash: str,
        command: str,
        before_admissible: Sequence[str],
        after_admissible: Sequence[str],
        before_observation: str,
        after_observation: str,
    ) -> tuple[bool, str | None]:
        """Match a runtime transition to a pattern observed in the fixed demo.

        The pattern contains only target-native, mechanically observed facts;
        no cross-domain predicate or Agent rationale is consulted.
        """
        binding = next(
            (item for item in self.bindings if item.artifact.artifact_hash == artifact_hash),
            None,
        )
        if binding is None or binding.artifact.verified_scope is None:
            return False, "UNKNOWN_FROZEN_ARTIFACT"
        if command not in before_admissible:
            return False, "ACTION_NOT_EXACTLY_ADMISSIBLE_BEFORE"
        patterns = list(binding.artifact.verified_scope.native_transition_patterns)
        if not patterns:
            return False, "MISSING_TARGET_NATIVE_TRANSITION_PATTERN"
        actual = {
            "state_changed": before_observation != after_observation,
            "admissible_set_changed": list(before_admissible) != list(after_admissible),
            "executed_action_still_admissible_after": command in after_admissible,
        }
        for pattern in patterns:
            expected = {key: bool(pattern[key]) for key in actual}
            if actual == expected:
                return True, None
        return False, "TARGET_NATIVE_TRANSITION_PATTERN_MISMATCH"


class StrictOpenAIClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 120.0,
        api_key: str = "EMPTY",
    ) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.api_key = str(api_key or "EMPTY")
        self._is_openrouter = "openrouter.ai" in self.base_url.lower()
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def complete(
        self, *, model: str, prompt: str, max_tokens: int = 48,
        reasoning_effort: str = "none",
    ) -> tuple[str, Dict[str, Any]]:
        if reasoning_effort not in {"none", "low", "medium", "high"}:
            raise ValueError("unsupported reasoning effort")
        started = time.monotonic()
        request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }
        if not self._is_openrouter:
            request["chat_template_kwargs"] = {
                "enable_thinking": reasoning_effort != "none"
            }
        else:
            # Keep reasoning private; the Harness admits only the closed answer.
            request["reasoning"] = {
                "effort": reasoning_effort, "exclude": True,
            }
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=request,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if len(choices) != 1:
            raise RuntimeError("endpoint_did_not_return_exactly_one_choice")
        content = choices[0].get("message", {}).get("content")
        # HTTP/provider failures raise above.  An empty model answer is a
        # schema/output failure and must not be mislabeled as endpoint health.
        if not isinstance(content, str):
            content = ""
        usage = dict(payload.get("usage") or {})
        usage["generation_id"] = str(payload.get("id") or "")
        usage["latency_s"] = time.monotonic() - started
        usage["model_requested"] = model
        return content, usage


__all__ = [
    "FrozenAdmissionGuard",
    "FrozenBinding",
    "GuardedAction",
    "NativeTargetPlan",
    "StrictOpenAIClient",
    "action_prompt",
    "native_target_action_prompt",
    "parse_native_target_plan_reply",
    "parse_exact_numbered_response",
    "skill_prompt",
]
