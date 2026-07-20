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


_ACTION_RESPONSE = re.compile(
    r"\A\s*(?:REASONING:[^\n]*\n)?ACTION:\s*([1-9][0-9]*)\s*\Z",
    re.IGNORECASE,
)
_SKILL_RESPONSE = re.compile(
    r"\A\s*(?:REASONING:[^\n]*\n)?SKILL:\s*([1-9][0-9]*)\s*\Z",
    re.IGNORECASE,
)


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


def action_prompt(
    *,
    domain: str,
    goal: str,
    observation: str,
    actions: Sequence[str],
    active_skill: str | None = None,
    recent_actions: Sequence[str] = (),
) -> str:
    numbered = "\n".join(f"{index}. {value}" for index, value in enumerate(actions, 1))
    skill_line = f"\nVerified active source skill: {active_skill}" if active_skill else ""
    recent = " -> ".join(recent_actions[-6:]) or "none"
    return (
        "You choose one action from an exact environment-provided list. "
        "Never invent, rewrite, or partially match a command.\n"
        f"Domain: {domain}\nGoal/initial context:\n{goal[:3000]}\n"
        f"Current observation:\n{observation[:3000]}{skill_line}\n"
        f"Recent exact actions: {recent}\n"
        "Available actions (pick ONE by number):\n"
        f"{numbered}\n"
        "Return exactly `ACTION: N`. No other text."
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
        if not self.bindings:
            raise ValueError("no admitted bindings in frozen manifest")

    @classmethod
    def from_files(
        cls,
        *,
        manifest_path: str | Path,
        binding_config_path: str | Path,
    ) -> "FrozenAdmissionGuard":
        artifacts = load_frozen_admission_manifest(manifest_path)
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
            if candidate_id not in names:
                raise ValueError(f"binding config missing candidate: {candidate_id}")
            operators = list(artifact.verified_scope.operators)
            if len(operators) != 1:
                raise ValueError(f"runtime requires one exact operator: {candidate_id}")
            bindings.append(FrozenBinding(
                candidate_id=candidate_id,
                source_skill_name=names[candidate_id],
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


class StrictOpenAIClient:
    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def complete(self, *, model: str, prompt: str, max_tokens: int = 48) -> tuple[str, Dict[str, Any]]:
        started = time.monotonic()
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": "Bearer EMPTY"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if len(choices) != 1:
            raise RuntimeError("endpoint_did_not_return_exactly_one_choice")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("endpoint_returned_empty_content")
        usage = dict(payload.get("usage") or {})
        usage["latency_s"] = time.monotonic() - started
        usage["model_requested"] = model
        return content, usage


__all__ = [
    "FrozenAdmissionGuard",
    "FrozenBinding",
    "GuardedAction",
    "StrictOpenAIClient",
    "action_prompt",
    "parse_exact_numbered_response",
    "skill_prompt",
]
