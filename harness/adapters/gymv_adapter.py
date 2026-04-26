"""`gymv` adapter — game-env execution path.

This is a *minimal* but real adapter: it walks `skill.protocol` hop by
hop, treats each hop as `{action: <tool_name>, payload: {...}}`, and
emits a corresponding `SkillEpisodeStep`. In `dry_run=True` mode it
short-circuits to a deterministic deterministic outcome derived from the
seed state's stored facts (used by the gate's replay validator).

Real env wiring (gymnasium / gym-v) belongs in `gymv_wrapper.adapter`;
this adapter calls into that module via a late import so the harness
package remains importable in a unit-test environment.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from common.enums import SkillType
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.skill_record import SkillRecord
from harness.adapters._common import BudgetGuard, HopBindings, iter_hops, normalize_hop_action
from harness.skill_adapter import AdapterRunContext, AdapterRunResult, SkillAdapter


# Pluggable executor: a callable `(action_type, payload, ctx) -> dict`.
# The default executor is a deterministic stub useful for tests and
# the gate's dry-run path. Real env binding (gymnasium step()) is plugged
# in by `gymv_wrapper.adapter` at runtime via `set_executor`.
HopExecutor = Callable[[str, Dict[str, Any], AdapterRunContext], Dict[str, Any]]


def _deterministic_executor(
    action_type: str, payload: Dict[str, Any], ctx: AdapterRunContext
) -> Dict[str, Any]:
    facts = ctx.state.facts
    return {
        "ok": True,
        "observation": {
            "echo_action": action_type,
            "echo_payload": payload,
            "facts_seen": list(facts.keys()),
        },
        "evidence": [
            EvidenceRef(
                source=f"gymv:{action_type.lower()}",
                locator=f"step={ctx.state.inner_step}",
                role="GATHER",
                confidence=0.9,
            )
        ],
    }


class GymvAdapter(SkillAdapter):
    name = "gymv"
    supported_types = (SkillType.ACTION, SkillType.MIXED, SkillType.GROUNDING)

    def __init__(self, executor: Optional[HopExecutor] = None) -> None:
        self._executor: HopExecutor = executor or _deterministic_executor

    def set_executor(self, executor: HopExecutor) -> None:
        self._executor = executor

    def can_handle(self, skill: SkillRecord, state: StateSchema) -> bool:
        if state.domain != self.name:
            return False
        if not skill.protocol:
            return False
        return skill.skill_type in self.supported_types

    def run(self, skill: SkillRecord, ctx: AdapterRunContext) -> AdapterRunResult:
        bindings = HopBindings(
            bindings=dict(ctx.bindings),
            state_facts=dict(ctx.state.facts),
        )
        budget = BudgetGuard(
            max_hops=int(ctx.budget.get("hops", 8)),
            max_ms=float(ctx.budget.get("ms", 30_000.0)),
        )
        steps: List[Dict[str, Any]] = []
        evidence: List[EvidenceRef] = []
        executor = self._executor if not ctx.dry_run else _deterministic_executor

        for i, hop in iter_hops(skill):
            abort = budget.check(i)
            if abort:
                return AdapterRunResult(
                    success=False,
                    contract_satisfied=False,
                    abort_reason=abort,
                    steps=steps,
                    new_evidence=evidence,
                    cost={"hops": float(i), "ms": (time.time() - budget.started_at) * 1000},
                )
            action_type = normalize_hop_action(hop)
            payload = bindings.resolve_dict(hop.get("payload", {}))
            try:
                hop_result = executor(action_type, payload, ctx)
            except Exception as exc:                          # noqa: BLE001
                return AdapterRunResult(
                    success=False,
                    contract_satisfied=False,
                    abort_reason=f"adapter_exception: {exc!r}",
                    steps=steps,
                    new_evidence=evidence,
                )
            step_evidence: List[EvidenceRef] = list(hop_result.get("evidence", []))
            evidence.extend(step_evidence)
            steps.append(
                {
                    "action_type": action_type,
                    "payload": payload,
                    "pre_state": ctx.state.to_json() if i == 0 else None,
                    "post_state": None,
                    "evidence": step_evidence,
                    "notes": hop.get("notes", ""),
                }
            )
            ctx.state.inner_step += 1
            if not hop_result.get("ok", True):
                return AdapterRunResult(
                    success=False,
                    contract_satisfied=False,
                    abort_reason=str(hop_result.get("reason", "hop_failed")),
                    steps=steps,
                    new_evidence=evidence,
                )

        return AdapterRunResult(
            success=True,
            contract_satisfied=True,
            final_state=ctx.state,
            steps=steps,
            new_evidence=evidence,
            score=1.0,
            cost={
                "hops": float(len(steps)),
                "ms": (time.time() - budget.started_at) * 1000,
            },
        )


__all__ = ["GymvAdapter", "HopExecutor"]
