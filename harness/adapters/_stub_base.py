"""Shared boilerplate for transfer-target adapters.

PLAN-HARNESS §5.4 — adapters are the only place that talk to a concrete
env / tool surface. The four non-game adapters (`browser`, `osworld`,
`video`, `visual_reasoning`) all share the same hop-loop scaffolding:
walk `skill.protocol`, resolve slots, call a pluggable executor, honor
budget. We factor that loop here so each transfer-target adapter only
has to declare its `name`, `supported_types`, and (later) plug in its
real `HopExecutor` via `set_executor`.

The default executor is deterministic and evidence-emitting; it is
sufficient for the gate's dry-run path and for stub testing under
PLAN-COMPONENTS-IMPLEMENTATION Phase A.5 / Phase D, where target-domain
binding is exercised before the real env wiring is wired in.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from common.enums import SkillType
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.skill_record import SkillRecord
from harness.adapters._common import (
    BudgetGuard,
    HopBindings,
    iter_hops,
    normalize_hop_action,
)
from harness.skill_adapter import AdapterRunContext, AdapterRunResult, SkillAdapter

HopExecutor = Callable[[str, Dict[str, Any], AdapterRunContext], Dict[str, Any]]


def make_deterministic_executor(domain: str, *, confidence: float = 0.8) -> HopExecutor:
    """Return a deterministic stub executor tagged for `domain`.

    The stub emits a single `GATHER` evidence ref per hop so the
    evidence-driven invariant (PLAN-SKILL-BANK §0.3 Clause A) is
    trivially satisfied in dry-run / smoke contexts.
    """

    def _exec(action_type: str, payload: Dict[str, Any], ctx: AdapterRunContext) -> Dict[str, Any]:
        return {
            "ok": True,
            "observation": {
                "domain": domain,
                "echo_action": action_type,
                "echo_payload": payload,
            },
            "evidence": [
                EvidenceRef(
                    source=f"{domain}:{action_type.lower()}",
                    locator=f"step={ctx.state.inner_step}",
                    role="GATHER",
                    confidence=confidence,
                )
            ],
        }

    return _exec


class StubTransferTargetAdapter(SkillAdapter):
    """Hop-loop scaffolding shared by all transfer-target adapters.

    Concrete subclasses set `name` and may override `supported_types`
    or `_default_executor()`. Real env binding is plugged in by
    `vlm_wrapper/<domain>_adapter.py` via `set_executor()`.
    """

    name: str = ""
    supported_types: tuple[SkillType, ...] = (
        SkillType.ACTION,
        SkillType.MIXED,
        SkillType.GROUNDING,
        SkillType.REASONING,
    )

    def __init__(self, executor: Optional[HopExecutor] = None) -> None:
        self._executor: HopExecutor = executor or self._default_executor()

    def _default_executor(self) -> HopExecutor:
        return make_deterministic_executor(self.name)

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
        executor = self._executor if not ctx.dry_run else self._default_executor()

        for i, hop in iter_hops(skill):
            abort = budget.check(i)
            if abort:
                return AdapterRunResult(
                    success=False,
                    contract_satisfied=False,
                    abort_reason=abort,
                    steps=steps,
                    new_evidence=evidence,
                    cost={
                        "hops": float(i),
                        "ms": (time.time() - budget.started_at) * 1000,
                    },
                )
            action_type = normalize_hop_action(hop)
            payload = bindings.resolve_dict(hop.get("payload", {}))
            try:
                hop_result = executor(action_type, payload, ctx)
            except Exception as exc:  # noqa: BLE001
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


__all__ = [
    "HopExecutor",
    "StubTransferTargetAdapter",
    "make_deterministic_executor",
]
