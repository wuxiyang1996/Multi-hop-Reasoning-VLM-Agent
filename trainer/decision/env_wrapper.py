"""
Environment wrapper that treats retrieval (QUERY_MEM, QUERY_SKILL, CALL_SKILL)
as first-class actions alongside primitives.

EnvWrapper.step(action) handles:
  - Primitive actions → forward to game env
  - QUERY_MEM(key) / QUERY_SKILL(key) → call tool, store returned cards
    in observation context, apply query cost
  - CALL_SKILL(skill_id, params) → set active_skill in wrapper state,
    apply call cost
Returns (obs, r_env, done, info) plus metadata for reward shaping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class WrapperState:
    """Internal state maintained by the environment wrapper."""

    active_skill_id: Optional[str] = None
    active_skill_contract: Any = None
    prev_skill_id: Optional[str] = None
    retrieved_cards: List[Dict[str, Any]] = field(default_factory=list)
    memory_cards: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    episode_id: str = ""
    context_buffer: str = ""


class EnvWrapper:
    """Wraps a game environment to unify primitive and retrieval actions.

    The wrapper intercepts QUERY_MEM, QUERY_SKILL, CALL_SKILL actions before
    they reach the game env, executes the retrieval/call logic internally,
    and exposes a consistent step() interface.
    """

    RETRIEVAL_ACTIONS = {"QUERY_MEM", "QUERY_SKILL", "CALL_SKILL"}
    _ACTION_PATTERN = re.compile(
        r"(QUERY_MEM|QUERY_SKILL|CALL_SKILL)\s*\(\s*(.+?)\s*\)", re.IGNORECASE
    )

    def __init__(
        self,
        env: Any,
        skill_bank: Any = None,
        memory: Any = None,
        query_engine: Any = None,
        reward_computer: Any = None,
    ):
        self.env = env
        self.skill_bank = skill_bank
        self.memory = memory
        self.query_engine = query_engine
        self.reward_computer = reward_computer
        self.state = WrapperState()

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and wrapper state."""
        self.state = WrapperState()

        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()

        observation = str(obs) if not isinstance(obs, str) else obs
        self.state.context_buffer = observation
        return observation, info

    def step(self, action: str) -> Dict[str, Any]:
        """Execute one action (primitive or retrieval).

        Returns a dict with:
            obs: next observation (str)
            r_env: environment reward (float)
            done: episode terminated (bool)
            info: env info dict
            action_type: "primitive" | "QUERY_MEM" | "QUERY_SKILL" | "CALL_SKILL"
            retrieved_cards: list of skill/memory cards (if retrieval)
            active_skill_id: currently active skill or None
            active_skill_contract: contract of active skill or None
            prev_skill_id: skill id from previous step
            step_count: current step number
        """
        self.state.step_count += 1
        self.state.prev_skill_id = self.state.active_skill_id

        parsed = self._parse_action(action)
        action_type = parsed["type"]

        if action_type in self.RETRIEVAL_ACTIONS:
            return self._handle_retrieval(parsed)
        else:
            return self._handle_primitive(action)

    def _parse_action(self, action: str) -> Dict[str, Any]:
        """Parse an action string into type and arguments."""
        action_upper = action.strip().upper() if action else ""

        match = self._ACTION_PATTERN.match(action.strip())
        if match:
            action_type = match.group(1).upper()
            raw_args = match.group(2).strip()
            if action_type == "CALL_SKILL":
                parts = [p.strip().strip("'\"") for p in raw_args.split(",", 1)]
                return {
                    "type": action_type,
                    "skill_id": parts[0],
                    "params": parts[1] if len(parts) > 1 else "",
                }
            else:
                return {"type": action_type, "key": raw_args.strip("'\""), }

        for prefix in self.RETRIEVAL_ACTIONS:
            if action_upper.startswith(prefix):
                remainder = action.strip()[len(prefix):].strip("() '\"")
                if prefix == "CALL_SKILL":
                    parts = [p.strip().strip("'\"") for p in remainder.split(",", 1)]
                    return {
                        "type": prefix,
                        "skill_id": parts[0],
                        "params": parts[1] if len(parts) > 1 else "",
                    }
                return {"type": prefix, "key": remainder}

        return {"type": "primitive", "action": action}

    def _handle_retrieval(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a retrieval action without stepping the environment."""
        action_type = parsed["type"]
        retrieved = []

        if action_type == "QUERY_SKILL" and self.skill_bank is not None:
            key = parsed.get("key", "")
            engine = self.query_engine
            if engine is None:
                try:
                    from skill_agents.query import SkillQueryEngine
                    engine = SkillQueryEngine(self.skill_bank)
                except ImportError:
                    engine = None

            if engine is not None:
                retrieved = engine.query(key, top_k=3)
            elif hasattr(self.skill_bank, "summary"):
                retrieved = [{"skill_id": sid} for sid in self.skill_bank.skill_ids[:3]]

            self.state.retrieved_cards = retrieved
            card_text = "\n".join(
                f"  [{c.get('skill_id', '?')}] score={c.get('score', 0):.3f}"
                for c in retrieved
            )
            self.state.context_buffer += f"\n[QUERY_SKILL result]\n{card_text}"

        elif action_type == "QUERY_MEM" and self.memory is not None:
            key = parsed.get("key", "")
            results = self.memory.query(key, k=3)
            self.state.memory_cards = results if isinstance(results, list) else []
            mem_text = str(results)[:500]
            self.state.context_buffer += f"\n[QUERY_MEM result]\n{mem_text}"

        elif action_type == "CALL_SKILL":
            skill_id = parsed.get("skill_id", "")
            self.state.active_skill_id = skill_id
            self.state.active_skill_contract = None
            if self.skill_bank and hasattr(self.skill_bank, "get_contract"):
                self.state.active_skill_contract = self.skill_bank.get_contract(skill_id)
            self.state.context_buffer += f"\n[CALL_SKILL activated: {skill_id}]"

        return {
            "obs": self.state.context_buffer,
            "r_env": 0.0,
            "done": False,
            "info": {},
            "action_type": action_type,
            "retrieved_cards": self.state.retrieved_cards,
            "active_skill_id": self.state.active_skill_id,
            "active_skill_contract": self.state.active_skill_contract,
            "prev_skill_id": self.state.prev_skill_id,
            "step_count": self.state.step_count,
            "query_key": parsed.get("key"),
        }

    def _handle_primitive(self, action: str) -> Dict[str, Any]:
        """Forward a primitive action to the game environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        r_env = float(reward) if not isinstance(reward, dict) else sum(reward.values())
        observation = str(obs) if not isinstance(obs, str) else obs
        self.state.context_buffer = observation

        return {
            "obs": observation,
            "r_env": r_env,
            "done": done,
            "info": info,
            "action_type": "primitive",
            "retrieved_cards": [],
            "active_skill_id": self.state.active_skill_id,
            "active_skill_contract": self.state.active_skill_contract,
            "prev_skill_id": self.state.prev_skill_id,
            "step_count": self.state.step_count,
            "query_key": None,
        }

    @property
    def active_skill_id(self) -> Optional[str]:
        return self.state.active_skill_id

    @property
    def active_contract(self) -> Any:
        return self.state.active_skill_contract

    def update_skill_bank(self, new_bank: Any) -> None:
        """Hot-swap the skill bank (called after co-evolution bank update)."""
        self.skill_bank = new_bank
        self.query_engine = None
        self.state.retrieved_cards = []
