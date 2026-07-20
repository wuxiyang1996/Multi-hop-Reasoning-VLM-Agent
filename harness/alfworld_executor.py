"""Live ALFWorld hop executor for ``AlfworldAdapter``.

Abstract reasoning hops observe the current text state without mutating the
environment.  Action hops are resolved against ALFWorld's current admissible
command list, which prevents the adapter from inventing invalid commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common.state_schema import EvidenceRef, StateSchema
from harness.adapters.alfworld_adapter import HopExecutor
from harness.skill_adapter import AdapterRunContext


OBSERVATIONAL_OPS = frozenset({
    "GROUND", "GATHER", "OBSERVE", "INSPECT", "READ", "RETRIEVE",
    "CHECK", "VERIFY", "COMPARE", "REASON", "INFER", "DEDUCE",
    "PLAN", "DECIDE", "RECALL", "TRACK",
})

ACTION_OPS = frozenset({
    "EXECUTE", "COMMIT", "ACT", "MOVE", "NAVIGATE", "GOTO",
    "TAKE", "PICKUP", "ACQUIRE", "PUT", "PLACE", "OPEN", "CLOSE",
    "HEAT", "COOL", "CLEAN", "TOGGLE", "USE", "EXAMINE",
})


@dataclass
class AlfworldExecutorState:
    last_observation: str = ""
    last_info: Dict[str, Any] = field(default_factory=dict)
    last_reward: float = 0.0
    cumulative_reward: float = 0.0
    completed: bool = False
    outer_step: int = 0
    terminated: bool = False
    truncated: bool = False
    last_post_state: Optional[StateSchema] = None
    active_context: Optional[Any] = None


def _normalise(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _current_commands(env: Any, holder: AlfworldExecutorState) -> List[str]:
    commands = getattr(env, "action_names", None)
    if commands is None:
        commands = holder.last_info.get("action_names") or holder.last_info.get(
            "admissible_actions"
        )
    return [str(command) for command in (commands or [])]


def _payload_commands(op: str, payload: Dict[str, Any]) -> Iterable[str]:
    for key in (
        "command", "action", "env_action", "chosen_action",
        "selected_action", "text",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            yield value.strip()

    target = payload.get("target") or payload.get("object") or payload.get("item")
    receptacle = (
        payload.get("receptacle") or payload.get("container")
        or payload.get("destination") or payload.get("location")
    )
    tool = payload.get("tool") or payload.get("appliance")
    verb = payload.get("verb") or payload.get("intent")
    if verb and target:
        yield f"{verb} {target}"
    elif verb:
        yield str(verb)
    if op in {"MOVE", "NAVIGATE", "GOTO"} and target:
        yield f"go to {target}"
    elif op in {"TAKE", "PICKUP", "ACQUIRE"} and target:
        if receptacle:
            yield f"take {target} from {receptacle}"
        yield f"take {target}"
    elif op in {"PUT", "PLACE"} and target and receptacle:
        yield f"put {target} in/on {receptacle}"
        yield f"put {target} in {receptacle}"
        yield f"put {target} on {receptacle}"
    elif op in {"OPEN", "CLOSE", "TOGGLE", "USE", "EXAMINE"} and target:
        yield f"{op.lower()} {target}"
    elif op in {"HEAT", "COOL", "CLEAN"} and target:
        if tool:
            yield f"{op.lower()} {target} with {tool}"
        yield f"{op.lower()} {target}"


def _resolve_command(
    op: str,
    payload: Dict[str, Any],
    admissible: List[str],
) -> Tuple[Optional[str], str]:
    if not admissible:
        return None, "no_admissible_commands"

    normalised = {_normalise(command): command for command in admissible}
    candidates = list(_payload_commands(op, payload))
    # A protocol may carry a literal ALFWorld command in its action field.
    candidates.append(op)
    for candidate in candidates:
        exact = normalised.get(_normalise(candidate))
        if exact is not None:
            return exact, "exact"

    # Fail closed: entity-name substrings are not action semantics. A caller
    # must supply a literal command or enough typed payload fields for one of
    # the constructors above to reproduce an admissible command exactly.
    return None, "no_exact_admissible_match"


def _post_state(
    ctx: AdapterRunContext,
    holder: AlfworldExecutorState,
    *,
    last_action: Optional[str],
) -> StateSchema:
    info = holder.last_info
    task_status = (
        "success" if holder.completed
        else "failed" if holder.terminated or holder.truncated
        else "in_progress"
    )
    facts = dict(ctx.state.facts)
    facts.update(
        cumulative_reward=holder.cumulative_reward,
        last_reward=holder.last_reward,
        task_status=task_status,
        terminated=holder.terminated,
        truncated=holder.truncated,
    )
    if last_action:
        facts["last_action"] = last_action
    return StateSchema(
        task=ctx.state.task,
        domain="alfworld",
        targets=ctx.state.targets,
        elements=list(ctx.state.elements),
        facts=facts,
        open_questions=list(ctx.state.open_questions),
        evidence=list(ctx.state.evidence),
        inner_step=ctx.state.inner_step,
        outer_step=holder.outer_step,
        extra={
            **dict(ctx.state.extra),
            "observation": holder.last_observation,
            "admissible_actions": list(
                info.get("admissible_actions") or info.get("action_names") or []
            ),
        },
    )


def make_alfworld_executor(
    *,
    env: Any,
    on_unresolved: str = "abort",
    reset_each_run: bool = True,
    state_holder: Optional[AlfworldExecutorState] = None,
) -> Tuple[HopExecutor, AlfworldExecutorState]:
    """Create a hop executor bound to an initialized ALFWorld wrapper."""
    if on_unresolved not in {"abort", "skip"}:
        raise ValueError("on_unresolved must be 'abort' or 'skip'")
    holder = state_holder or AlfworldExecutorState()

    def _start_run(ctx: AdapterRunContext) -> None:
        if holder.active_context is ctx:
            return
        holder.active_context = ctx
        holder.last_reward = 0.0
        holder.cumulative_reward = 0.0
        holder.completed = False
        holder.outer_step = 0
        holder.terminated = False
        holder.truncated = False
        holder.last_post_state = None
        if reset_each_run:
            observation, info = env.reset()
            holder.last_observation = str(observation)
            holder.last_info = dict(info or {})
        else:
            holder.last_observation = str(
                getattr(env, "last_observation", holder.last_observation)
            )
            holder.last_info = dict(
                getattr(env, "last_info", holder.last_info) or {}
            )

    def _evidence(op: str, role: str, payload: Dict[str, Any]) -> EvidenceRef:
        return EvidenceRef(
            source=f"alfworld:{op.lower()}",
            locator=f"outer_step={holder.outer_step}",
            role=role,
            confidence=1.0,
            payload=payload,
        )

    def executor(
        action_type: str,
        payload: Dict[str, Any],
        ctx: AdapterRunContext,
    ) -> Dict[str, Any]:
        _start_run(ctx)
        op = str(action_type or "OBSERVE").strip().upper()
        if op in OBSERVATIONAL_OPS:
            role = "VERIFY" if op == "VERIFY" else (
                "REASON" if op in {"CHECK", "COMPARE", "REASON", "INFER", "DEDUCE"}
                else "GATHER"
            )
            ev = _evidence(
                op,
                role,
                {
                    "observation": holder.last_observation[:500],
                    "n_admissible": len(_current_commands(env, holder)),
                },
            )
            post = _post_state(ctx, holder, last_action=None)
            holder.last_post_state = post
            return {
                "ok": True,
                "observation": holder.last_observation,
                "evidence": [ev],
                "evidence_in": list(ctx.state.evidence),
                "evidence_out": [ev],
                "post_state": post.to_json(),
                "_final_state": post,
                "score": holder.cumulative_reward,
            }

        if holder.terminated or holder.truncated:
            return {
                "ok": False,
                "reason": "alfworld_episode_already_finished",
                "evidence": [],
            }

        admissible = _current_commands(env, holder)
        command, match_kind = _resolve_command(op, payload, admissible)
        if command is None:
            reason = f"unresolved_alfworld_command:{match_kind}:op={op}"
            if on_unresolved == "abort" or op in ACTION_OPS:
                return {"ok": False, "reason": reason, "evidence": []}
            return {"ok": True, "observation": reason, "evidence": []}

        observation, reward, terminated, truncated, info = env.step(command)
        holder.outer_step += 1
        holder.last_observation = str(observation)
        holder.last_info = dict(info or {})
        holder.last_reward = float(reward)
        won = holder.last_info.get("won", False)
        if isinstance(won, (list, tuple)):
            won = won[0] if won else False
        holder.completed = bool(won) or holder.last_reward >= 1.0
        # ALFWorld/TextWorld reports an episode score (and a separate `won`
        # flag), not an incremental reward that should be summed.  Retain the
        # best score observed so shaped partial rewards cannot add up to a
        # false success across steps.
        holder.cumulative_reward = max(
            holder.cumulative_reward,
            holder.last_reward,
            1.0 if holder.completed else 0.0,
        )
        holder.terminated = bool(terminated)
        holder.truncated = bool(truncated)
        role = "COMMIT" if op in {"COMMIT", "EXECUTE", "ACT"} else "GATHER"
        ev = _evidence(
            op,
            role,
            {"command": command, "reward": float(reward), "match": match_kind},
        )
        post = _post_state(ctx, holder, last_action=command)
        holder.last_post_state = post
        return {
            "ok": True,
            "observation": observation,
            "evidence": [ev],
            "evidence_in": list(ctx.state.evidence),
            "evidence_out": [ev],
            "post_state": post.to_json(),
            "_final_state": post,
            "score": holder.cumulative_reward,
            "terminated": holder.terminated,
            "truncated": holder.truncated,
        }

    return executor, holder


__all__ = [
    "ACTION_OPS",
    "OBSERVATIONAL_OPS",
    "AlfworldExecutorState",
    "make_alfworld_executor",
]
