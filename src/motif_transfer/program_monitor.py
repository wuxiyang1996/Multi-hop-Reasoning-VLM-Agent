"""Fail-closed symbolic runtime for neural operational predicates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import Observation, TransitionReceipt, stable_hash
from .neural_event_probes import (
    NeuralProbeBackend,
    ProbeEvaluationReceipt,
    before_probe_payload,
    evaluate_probe,
    transition_probe_payload,
)
from .neurosymbolic_ir import (
    ControlRoute,
    NeuroSymbolicProgram,
    ProbeVerdict,
    RouteKind,
)


class MonitorVerdict(str, Enum):
    ADMIT = "ADMIT"
    CONTINUE = "CONTINUE"
    REPLAN = "REPLAN"
    ABSTAIN = "ABSTAIN"
    TERMINATE = "TERMINATE"


@dataclass(frozen=True)
class MonitorDecision:
    verdict: MonitorVerdict
    node_id: str
    reason: str
    evaluation: ProbeEvaluationReceipt
    next_node_id: str | None = None


class NeuroSymbolicMonitor:
    """Executes only control flow; it never proposes or executes an action."""

    def __init__(
        self,
        program: NeuroSymbolicProgram,
        backend: NeuralProbeBackend,
    ) -> None:
        program.validate()
        self.program = program
        self.backend = backend
        self.current_node_id = program.entry_node_id
        self._pending_before_hash: str | None = None
        self.terminated = False
        self.suspended = False

    @property
    def current_node(self):
        return self.program.node_by_id[self.current_node_id]

    def review_before_action(self, observation: Observation) -> MonitorDecision:
        if self.terminated:
            raise RuntimeError("neural-symbolic program already terminated")
        if self.suspended:
            raise RuntimeError("neural-symbolic program is suspended")
        if self._pending_before_hash is not None:
            raise RuntimeError("previous admitted action still lacks transition evidence")
        node = self.current_node
        probe = self.program.probe_by_id[node.before_guard_probe_id]
        evaluation = evaluate_probe(
            program_id=self.program.program_id,
            node_id=node.node_id,
            probe=probe,
            input_payload=before_probe_payload(observation),
            backend=self.backend,
        )
        if evaluation.verdict == ProbeVerdict.UNKNOWN:
            self.suspended = True
            return MonitorDecision(
                MonitorVerdict.ABSTAIN, node.node_id,
                "guard evidence is outside the calibrated probe scope", evaluation,
            )
        if evaluation.verdict == ProbeVerdict.REFUTED:
            if (
                node.on_guard_refuted.kind == RouteKind.TERMINATE
                and not observation.official_success
            ):
                self.suspended = True
                return MonitorDecision(
                    MonitorVerdict.ABSTAIN, node.node_id,
                    "symbolic route cannot substitute for official success",
                    evaluation,
                )
            return self._apply_route(
                node.on_guard_refuted, node.node_id, evaluation,
                "pre-action guard was refuted",
                next_verdict=MonitorVerdict.REPLAN,
            )
        self._pending_before_hash = stable_hash(observation.state)
        return MonitorDecision(
            MonitorVerdict.ADMIT, node.node_id,
            "guard supported; Decision Agent retains action authority", evaluation,
        )

    def observe_transition(
        self,
        *,
        before: Observation,
        action: str,
        after: Observation,
        receipt: TransitionReceipt,
    ) -> MonitorDecision:
        if self.terminated:
            raise RuntimeError("neural-symbolic program already terminated")
        if self.suspended:
            raise RuntimeError("neural-symbolic program is suspended")
        if self._pending_before_hash is None:
            raise RuntimeError("transition arrived without an admitted before-state")
        self._validate_live_transition(before, action, after, receipt)
        node = self.current_node
        probe = self.program.probe_by_id[node.transition_effect_probe_id]
        evaluation = evaluate_probe(
            program_id=self.program.program_id,
            node_id=node.node_id,
            probe=probe,
            input_payload=transition_probe_payload(before, action, after, receipt),
            backend=self.backend,
        )
        self._pending_before_hash = None
        if evaluation.verdict == ProbeVerdict.UNKNOWN:
            self.suspended = True
            return MonitorDecision(
                MonitorVerdict.ABSTAIN, node.node_id,
                "effect evidence is outside the calibrated probe scope", evaluation,
            )
        if evaluation.verdict == ProbeVerdict.REFUTED:
            if (
                node.on_effect_refuted.kind == RouteKind.TERMINATE
                and not after.official_success
            ):
                self.suspended = True
                return MonitorDecision(
                    MonitorVerdict.ABSTAIN, node.node_id,
                    "symbolic route cannot substitute for official success",
                    evaluation,
                )
            return self._apply_route(
                node.on_effect_refuted, node.node_id, evaluation,
                "post-action effect was refuted",
            )
        if (
            node.on_effect_supported.kind == RouteKind.TERMINATE
            and not after.official_success
        ):
            self.suspended = True
            return MonitorDecision(
                MonitorVerdict.ABSTAIN, node.node_id,
                "neural effect cannot substitute for official success", evaluation,
            )
        return self._apply_route(
            node.on_effect_supported, node.node_id, evaluation,
            "post-action effect was supported",
        )

    def _validate_live_transition(
        self,
        before: Observation,
        action: str,
        after: Observation,
        receipt: TransitionReceipt,
    ) -> None:
        if not receipt.validate():
            raise ValueError("transition receipt hash mismatch")
        if stable_hash(before.state) != self._pending_before_hash:
            raise ValueError("transition before-state differs from admitted state")
        if receipt.before_hash != stable_hash(before.state):
            raise ValueError("transition receipt before-state mismatch")
        if receipt.native_actions_hash != stable_hash(before.native_actions):
            raise ValueError("transition receipt native-action mismatch")
        if action not in before.native_actions:
            raise ValueError("executed action was not native-admissible")
        if receipt.action != action:
            raise ValueError("transition receipt action mismatch")
        if receipt.after_hash != stable_hash(after.state):
            raise ValueError("transition receipt after-state mismatch")
        if receipt.done != after.terminal:
            raise ValueError("transition receipt terminal mismatch")
        if receipt.official_success != after.official_success:
            raise ValueError("transition receipt official-success mismatch")

    def _apply_route(
        self,
        route: ControlRoute,
        node_id: str,
        evaluation: ProbeEvaluationReceipt,
        reason: str,
        next_verdict: MonitorVerdict = MonitorVerdict.CONTINUE,
    ) -> MonitorDecision:
        if route.kind == RouteKind.NEXT_NODE:
            self.current_node_id = str(route.target_node_id)
            return MonitorDecision(
                next_verdict, node_id, reason, evaluation,
                next_node_id=self.current_node_id,
            )
        verdict = {
            RouteKind.REPLAN: MonitorVerdict.REPLAN,
            RouteKind.ABSTAIN: MonitorVerdict.ABSTAIN,
            RouteKind.TERMINATE: MonitorVerdict.TERMINATE,
        }[route.kind]
        if verdict == MonitorVerdict.TERMINATE:
            self.terminated = True
        elif verdict == MonitorVerdict.ABSTAIN:
            self.suspended = True
        return MonitorDecision(verdict, node_id, reason, evaluation)


__all__ = ["MonitorDecision", "MonitorVerdict", "NeuroSymbolicMonitor"]
