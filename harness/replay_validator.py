"""`ReplayValidator` — gate Stage 1 (PLAN-UNIFIED-SKILL-GATE §7.1).

Re-runs a SkillRecord on a fixed set of stored `SkillEpisode`s ("replay
seeds") and checks two things:

  (a) the proposed skill produces non-worse outcomes on the seed set;
  (b) it does not introduce evidence-driven invariant violations (G0).

Validator is *deterministic* — it never calls the live env. It executes
adapters in `dry_run=True` mode, which adapters MUST honor by returning
their cached / stored response for the seed state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.enums import GateStage, GateVerdict
from common.state_schema import StateSchema
from data_structure.extensions.gate_verdict import StageVerdict
from data_structure.extensions.skill_episode import SkillEpisode
from data_structure.extensions.skill_record import SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.skill_adapter import AdapterRunContext


@dataclass
class ReplayResult:
    n_seeds: int
    n_pass: int
    n_fail: int
    n_invariant_violations: int = 0
    per_seed_outcomes: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return (self.n_pass / self.n_seeds) if self.n_seeds else 0.0

    def as_stage_verdict(self, *, threshold: float = 0.8) -> StageVerdict:
        verdict = GateVerdict.PASS if self.pass_rate >= threshold and self.n_invariant_violations == 0 else GateVerdict.FAIL
        return StageVerdict(
            stage=GateStage.REPLAY,
            verdict=verdict,
            metrics={
                "pass_rate": self.pass_rate,
                "n_seeds": float(self.n_seeds),
                "n_invariant_violations": float(self.n_invariant_violations),
            },
            failures=[
                f"seed={o['episode_id']} reason={o.get('abort_reason')}"
                for o in self.per_seed_outcomes
                if not o.get("success")
            ],
        )


class ReplayValidator:
    def __init__(self, registry: AdapterRegistry) -> None:
        self._registry = registry

    def validate(
        self,
        *,
        skill: SkillRecord,
        seeds: List[SkillEpisode],
        budget: Optional[Dict[str, float]] = None,
    ) -> ReplayResult:
        n_pass = 0
        n_violations = 0
        outcomes: List[Dict[str, Any]] = []
        for seed in seeds:
            adapter = self._registry.get(seed.domain, skill.skill_type)
            if adapter is None:
                outcomes.append(
                    {
                        "episode_id": seed.episode_id,
                        "success": False,
                        "abort_reason": f"no_adapter_for({seed.domain},{skill.skill_type.value})",
                    }
                )
                continue
            state = _state_from_seed(seed, fallback_domain=seed.domain)
            ctx = AdapterRunContext(
                state=state,
                parent_run_id=seed.parent_run_id,
                budget=dict(budget or {}),
                dry_run=True,
            )
            try:
                result = adapter.run(skill, ctx)
            except Exception as exc:                        # noqa: BLE001
                outcomes.append(
                    {
                        "episode_id": seed.episode_id,
                        "success": False,
                        "abort_reason": f"adapter_error: {exc!r}",
                    }
                )
                continue
            success = result.success and result.contract_satisfied
            if success and not result.new_evidence and skill.skill_type.value != "action":
                # G0: no evidence on a non-ACTION success.
                n_violations += 1
                success = False
            if success:
                n_pass += 1
            outcomes.append(
                {
                    "episode_id": seed.episode_id,
                    "success": success,
                    "abort_reason": result.abort_reason,
                    "score": result.score,
                }
            )
        return ReplayResult(
            n_seeds=len(seeds),
            n_pass=n_pass,
            n_fail=len(seeds) - n_pass,
            n_invariant_violations=n_violations,
            per_seed_outcomes=outcomes,
        )


def _state_from_seed(seed: SkillEpisode, *, fallback_domain: str) -> StateSchema:
    """Reconstruct a minimal StateSchema from a seed episode's stored
    initial state."""
    init = seed.initial_state or {}
    return StateSchema(
        task=str(init.get("task", "")),
        domain=str(init.get("domain", fallback_domain)),
        elements=list(init.get("elements", [])),
        facts=dict(init.get("facts", {})),
        open_questions=list(init.get("open_questions", [])),
        inner_step=int(init.get("inner_step", 0)),
        outer_step=int(init.get("outer_step", 0)),
        extra=dict(init.get("extra", {})),
    )


__all__ = ["ReplayResult", "ReplayValidator"]
