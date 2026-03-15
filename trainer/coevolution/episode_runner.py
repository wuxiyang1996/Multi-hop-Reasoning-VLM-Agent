"""Async episode runner for the co-evolution loop.

Mirrors ``scripts/qwen3_decision_agent.run_episode()`` but replaces every
synchronous LLM call with an ``await`` on the shared :class:`AsyncVLLMClient`,
and runs ``env.step()`` in an executor to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Headless mode for retro/pyglet/SDL — must be set before any game env import
os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from trainer.coevolution.config import EMULATOR_GAMES
from trainer.coevolution.vllm_client import AsyncVLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — these pull in heavyweight packages that live in the project
# ---------------------------------------------------------------------------

def _lazy_imports():
    """Return project modules, imported once and cached."""
    global _IMPORTS_CACHE
    if "_IMPORTS_CACHE" not in globals():
        from evaluate_gamingagent.game_configs import GAME_CONFIGS
        from evaluate_gamingagent.gym_like import make_gaming_env
        from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
        try:
            from env_wrappers.sokoban_nl_wrapper import SokobanNLWrapper
        except ImportError:
            SokobanNLWrapper = None
        from decision_agents.agent_helper import (
            build_rag_summary,
            compact_text_observation,
            extract_game_facts,
            infer_intention,
            strip_think_tags,
            HARD_SUMMARY_CHAR_LIMIT,
            SUBGOAL_TAGS,
        )
        try:
            from decision_agents.agent_helper import _get_protocol_for_skill
        except ImportError:
            _get_protocol_for_skill = None

        _IMPORTS_CACHE = {
            "GAME_CONFIGS": GAME_CONFIGS,
            "make_gaming_env": make_gaming_env,
            "GamingAgentNLWrapper": GamingAgentNLWrapper,
            "SokobanNLWrapper": SokobanNLWrapper,
            "build_rag_summary": build_rag_summary,
            "compact_text_observation": compact_text_observation,
            "extract_game_facts": extract_game_facts,
            "infer_intention": infer_intention,
            "strip_think_tags": strip_think_tags,
            "HARD_SUMMARY_CHAR_LIMIT": HARD_SUMMARY_CHAR_LIMIT,
            "SUBGOAL_TAGS": SUBGOAL_TAGS,
            "_get_protocol_for_skill": _get_protocol_for_skill,
        }
    return _IMPORTS_CACHE


_IMPORTS_CACHE: Dict[str, Any] = {}

INTENTION_WORD_BUDGET = 15
MAX_REPEAT_ACTIONS = 2

SYSTEM_PROMPT = (
    "You are an expert game-playing agent. "
    "You receive a game state and must choose exactly one action by its NUMBER.\n\n"
    "Rules:\n"
    "- Study the state carefully before choosing.\n"
    "- Consider which action makes the most progress toward winning.\n"
    "- NEVER repeat the same action more than 2 times in a row — try something different.\n"
    "- If recent actions got zero reward, change strategy.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences>\n"
    "ACTION: <number>\n"
)

SKILL_SELECTION_SYSTEM_PROMPT = (
    "You are an expert game strategist. "
    "Given the current game state and a set of candidate strategies, "
    "choose the ONE strategy most likely to make progress.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences why this strategy fits the current state>\n"
    "SKILL: <number>\n"
)

_TAG_ALIASES: Dict[str, str] = {
    "PLACE": "SETUP", "DROP": "EXECUTE", "MOVE": "NAVIGATE",
    "SWAP": "EXECUTE", "PUSH": "NAVIGATE", "JUMP": "NAVIGATE",
    "MATCH": "CLEAR", "PLAN": "SETUP", "ARRANGE": "SETUP",
    "ROTATE": "SETUP", "ORGANIZE": "OPTIMIZE", "SCORE": "EXECUTE",
    "PROTECT": "DEFEND", "GRAB": "COLLECT", "FLEE": "SURVIVE",
    "RUN": "NAVIGATE", "CREATE": "BUILD", "FIND": "EXPLORE",
    "FIX": "OPTIMIZE", "ALIGN": "POSITION", "TARGET": "ATTACK",
    "SECURE": "DEFEND", "EXPAND": "ATTACK", "RETREAT": "DEFEND",
}
_TAG_RE = re.compile(r"\[(\w+)\]\s*")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GRPORecord:
    adapter: str  # "action_taking" or "skill_selection"
    game: str
    episode_id: str
    step: int
    prompt: str = ""
    completion: str = ""
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    game: str
    episode_id: str
    steps: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    skill_switches: int = 0
    grpo_records: List[GRPORecord] = field(default_factory=list)
    experiences: List[Dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Helpers (lightweight, no LLM calls)
# ---------------------------------------------------------------------------

def _generate_summary_state(
    state: str, game_name: str, step_idx: int, total_steps: int, reward: float,
) -> str:
    imp = _lazy_imports()
    return imp["build_rag_summary"](
        state, game_name, step_idx=step_idx, total_steps=total_steps, reward=reward,
    )


def _compute_state_delta(prev_ss: str, curr_ss: str) -> str:
    if not prev_ss or not curr_ss:
        return ""

    def _parse(ss: str) -> Dict[str, str]:
        d: Dict[str, str] = {}
        for seg in ss.split(" | "):
            if "=" in seg:
                k, v = seg.split("=", 1)
                d[k.strip()] = v.strip()
        return d

    skip = {"game", "step", "phase"}
    p, c = _parse(prev_ss), _parse(curr_ss)
    changes = [f"{k}:{p[k]}->{v}" for k, v in c.items()
               if k not in skip and k in p and p[k] != v]
    return ", ".join(changes[:5])


def _detect_urgency(summary_state: str, game_name: str) -> str:
    def _val(key: str) -> Optional[float]:
        for seg in summary_state.split(" | "):
            seg = seg.strip()
            if seg.startswith(f"{key}="):
                try:
                    return float(seg.split("=", 1)[1].split(",")[0])
                except (ValueError, IndexError):
                    return None
        return None

    gn = game_name.lower()
    warnings: List[str] = []
    if gn == "tetris":
        h = _val("holes")
        sh = _val("stack_h")
        if h is not None and h > 25:
            warnings.append("severe holes—prioritise CLEAR or SURVIVE")
        if sh is not None and sh > 14:
            warnings.append("stack near ceiling—SURVIVE")
    elif gn in ("2048", "twenty_forty_eight"):
        e = _val("empty")
        if e is not None and e < 3:
            warnings.append("board nearly full—must MERGE now")
    elif "candy" in gn:
        m = _val("moves")
        if m is not None and m < 5:
            warnings.append("very few moves left—maximise every action")
    elif "mario" in gn:
        t = _val("time")
        if t is not None and t < 50:
            warnings.append("time running out—NAVIGATE quickly")
    return "; ".join(warnings)


def _normalize_intention(raw: str) -> str:
    imp = _lazy_imports()
    _SUBGOAL_TAG_SET = frozenset(imp["SUBGOAL_TAGS"])
    raw = raw.split("\n")[0].strip().strip('"').strip("'")
    if not raw.startswith("["):
        return f"[EXECUTE] {raw}"
    m = _TAG_RE.match(raw)
    if not m:
        return f"[EXECUTE] {raw}"
    tag = m.group(1).upper()
    rest = raw[m.end():].strip()
    if tag not in _SUBGOAL_TAG_SET:
        tag = _TAG_ALIASES.get(tag, "EXECUTE")
    return f"[{tag}] {rest}" if rest else f"[{tag}]"


def _format_numbered_actions(action_names: List[str]) -> str:
    return "\n".join(f"  {i+1}. {a}" for i, a in enumerate(action_names))


def _build_recent_context(recent_actions: List[str], recent_rewards: List[float]) -> str:
    if not recent_actions:
        return ""
    lines = ["Recent actions and rewards:"]
    for a, r in zip(recent_actions[-5:], recent_rewards[-5:]):
        lines.append(f"  {a} -> reward {r:.1f}")
    total = sum(recent_rewards[-5:])
    if total == 0 and len(recent_actions) >= 3:
        lines.append("WARNING: Recent actions got 0 reward. Try a DIFFERENT action!")
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_skill_guidance_for_prompt(guidance: Optional[Dict[str, Any]], protocol_step_idx: int = 0) -> str:
    if guidance is None or not guidance.get("skill_id"):
        return ""
    parts = [f"\n--- Active Skill: {guidance.get('skill_name', guidance['skill_id'])} ---"]
    if guidance.get("execution_hint"):
        parts.append(f"  Strategy: {guidance['execution_hint'][:200]}")
    protocol = guidance.get("protocol", {})
    steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
    if steps:
        parts.append(f"  Plan ({len(steps)} steps):")
        for i, step in enumerate(steps[:7], 1):
            marker = ">>" if (i - 1) == protocol_step_idx else "  "
            parts.append(f"  {marker} {i}. {step}")
    preconditions = protocol.get("preconditions", []) if isinstance(protocol, dict) else []
    if preconditions:
        parts.append(f"  Preconditions: {'; '.join(preconditions[:3])}")
    success = protocol.get("success_criteria", []) if isinstance(protocol, dict) else []
    if success:
        parts.append(f"  Done when: {'; '.join(success[:2])}")
    abort = protocol.get("abort_criteria", []) if isinstance(protocol, dict) else []
    if abort:
        parts.append(f"  Abort if: {'; '.join(abort[:2])}")
    parts.append("--- end skill ---\n")
    return "\n".join(parts)


def _format_candidates_for_selection(candidates: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, c in enumerate(candidates, 1):
        name = c.get("skill_name") or c.get("skill_id", f"strategy_{i}")
        hint = c.get("execution_hint", "")
        protocol = c.get("protocol", {})
        steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
        lines.append(f"  {i}. {name}")
        if hint:
            lines.append(f"     Strategy: {hint[:150]}")
        if steps:
            step_text = " -> ".join(steps[:4])
            if len(steps) > 4:
                step_text += " -> ..."
            lines.append(f"     Plan: {step_text}")
        confidence = c.get("confidence")
        if confidence is not None:
            lines.append(f"     Confidence: {confidence:.2f}")
    return "\n".join(lines)


def _parse_skill_selection(reply: str, n_candidates: int, candidates: Optional[List[Dict[str, Any]]] = None) -> Tuple[int, Optional[str]]:
    imp = _lazy_imports()
    strip_think_tags = imp["strip_think_tags"]

    if not reply:
        return 0, None
    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply
    reasoning = None
    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nSKILL|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()
    skill_m = re.search(r"SKILL\s*:\s*(\d+)", cleaned, re.IGNORECASE)
    if skill_m:
        idx = int(skill_m.group(1)) - 1
        if 0 <= idx < n_candidates:
            return idx, reasoning
    tail = cleaned[-100:]
    nums = re.findall(r"\b(\d+)\b", tail)
    for n_str in reversed(nums):
        idx = int(n_str) - 1
        if 0 <= idx < n_candidates:
            return idx, reasoning
    if candidates:
        cleaned_lower = cleaned.lower()
        for i, c in enumerate(candidates):
            name = (c.get("skill_name") or "").lower()
            if name and len(name) >= 4 and name in cleaned_lower:
                return i, reasoning
    return 0, reasoning


def _parse_action_response(reply: str, valid_actions: List[str]) -> Tuple[str, Optional[str]]:
    imp = _lazy_imports()
    strip_think_tags = imp["strip_think_tags"]

    if not reply:
        return (valid_actions[0] if valid_actions else "stay"), None
    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply

    reasoning = None
    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nACTION|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    action_m = re.search(r"ACTION\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if action_m:
        raw = action_m.group(1).strip()
        matched = _fuzzy_match_action(raw, valid_actions)
        if matched:
            return matched, reasoning

    return (valid_actions[0] if valid_actions else "stay"), reasoning


def _fuzzy_match_action(raw: str, valid_actions: List[str]) -> Optional[str]:
    if not raw or not valid_actions:
        return None
    raw_lower = raw.lower().rstrip(".").strip()
    lower_map = {a.lower(): a for a in valid_actions}
    if raw_lower in lower_map:
        return lower_map[raw_lower]
    num_m = re.match(r"^(\d+)\.?\s*$", raw_lower)
    if num_m:
        idx = int(num_m.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx]
    num_m2 = re.search(r"(?:^|\s)(\d+)\s*[.:\-]", raw_lower)
    if num_m2:
        idx = int(num_m2.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx]
    for canon_lower, canon in lower_map.items():
        if len(canon_lower) < 3 and len(raw_lower) > 5:
            continue
        if canon_lower in raw_lower or raw_lower in canon_lower:
            return canon
    return None


def _apply_anti_repetition(
    action: str, valid_actions: List[str],
    recent_actions: List[str], recent_rewards: List[float],
) -> str:
    if len(recent_actions) < MAX_REPEAT_ACTIONS:
        return action
    tail = recent_actions[-MAX_REPEAT_ACTIONS:]
    tail_rewards = recent_rewards[-MAX_REPEAT_ACTIONS:]
    if all(a == action for a in tail) and sum(tail_rewards) <= 0:
        alternatives = [a for a in valid_actions if a != action]
        if alternatives:
            return random.choice(alternatives)
    return action


# ---------------------------------------------------------------------------
# Skill tracker (same logic as qwen3_decision_agent._SkillTracker)
# ---------------------------------------------------------------------------

class _SkillTracker:
    def __init__(self):
        self.active_skill_id: Optional[str] = None
        self.active_skill_name: str = ""
        self.steps_on_skill: int = 0
        self.reward_on_skill: float = 0.0
        self.max_skill_duration: int = 10
        self.skill_switches: int = 0
        self._protocol: Optional[Dict[str, Any]] = None
        self._protocol_step_idx: int = 0
        self._success_criteria: List[str] = []
        self._abort_criteria: List[str] = []
        self._reselect_reason: str = ""

    @property
    def protocol_step_idx(self) -> int:
        return self._protocol_step_idx

    @property
    def total_protocol_steps(self) -> int:
        if self._protocol and isinstance(self._protocol, dict):
            return len(self._protocol.get("steps", []))
        return 0

    def should_reselect(self, guidance: Optional[Dict[str, Any]], state_text: str = "") -> bool:
        self._reselect_reason = ""
        if guidance is None or not guidance.get("skill_id"):
            self._reselect_reason = "no_skill"
            return True
        new_id = guidance["skill_id"]
        if new_id != self.active_skill_id:
            return False
        if self.steps_on_skill >= self.max_skill_duration:
            self._reselect_reason = "duration_exceeded"
            return True
        if self.steps_on_skill >= 4 and self.reward_on_skill <= 0:
            self._reselect_reason = "zero_reward_stall"
            return True
        state_lower = state_text.lower() if state_text else ""
        if state_lower and self._abort_criteria:
            for crit in self._abort_criteria:
                tokens = [t for t in crit.lower().split() if len(t) >= 3]
                if tokens and all(tok in state_lower for tok in tokens[:3]):
                    self._reselect_reason = f"abort:{crit[:40]}"
                    return True
        if state_lower and self._success_criteria and self.steps_on_skill >= 2:
            for crit in self._success_criteria:
                tokens = [t for t in crit.lower().split() if len(t) >= 3]
                if tokens and all(tok in state_lower for tok in tokens[:3]):
                    self._reselect_reason = f"success:{crit[:40]}"
                    return True
        return False

    def update(self, skill_id: Optional[str], skill_name: str, reward: float):
        if skill_id != self.active_skill_id:
            self.active_skill_id = skill_id
            self.active_skill_name = skill_name
            self.steps_on_skill = 1
            self.reward_on_skill = reward
            self.skill_switches += 1
            self._protocol_step_idx = 0
        else:
            self.steps_on_skill += 1
            self.reward_on_skill += reward
            n_steps = self.total_protocol_steps
            if n_steps > 0:
                self._protocol_step_idx = min(self._protocol_step_idx + 1, n_steps - 1)

    def set_protocol(self, protocol: Optional[Dict[str, Any]]):
        self._protocol = protocol
        self._protocol_step_idx = 0
        self._success_criteria = []
        self._abort_criteria = []
        if protocol and isinstance(protocol, dict):
            dur = protocol.get("expected_duration", 0)
            if isinstance(dur, (int, float)) and dur > 0:
                self.max_skill_duration = max(int(dur) + 3, 5)
            else:
                self.max_skill_duration = 10
            self._success_criteria = protocol.get("success_criteria", []) or []
            self._abort_criteria = protocol.get("abort_criteria", []) or []
        else:
            self.max_skill_duration = 10


# ---------------------------------------------------------------------------
# Async episode runner
# ---------------------------------------------------------------------------

async def run_episode_async(
    game: str,
    max_steps: int,
    vllm_client: AsyncVLLMClient,
    *,
    skill_bank: Any = None,
    temperature: float = 0.3,
    executor: Optional[ThreadPoolExecutor] = None,
    process_executor: Optional[ProcessPoolExecutor] = None,
    stuck_window: int = 15,
    min_steps_before_stuck: int = 20,
) -> EpisodeResult:
    """Run one game episode asynchronously.

    All LLM calls go through *vllm_client* (``await``).
    ``env.step()`` runs in *executor* to avoid blocking the event loop.

    Parameters
    ----------
    skill_bank : object | None
        ``None`` triggers cold-start mode (no skill selection).
    """
    imp = _lazy_imports()
    GAME_CONFIGS = imp["GAME_CONFIGS"]
    make_gaming_env = imp["make_gaming_env"]
    GamingAgentNLWrapper = imp["GamingAgentNLWrapper"]
    SokobanNLWrapper = imp["SokobanNLWrapper"]
    HARD_SUMMARY_CHAR_LIMIT = imp["HARD_SUMMARY_CHAR_LIMIT"]
    extract_game_facts = imp["extract_game_facts"]
    compact_text_observation = imp["compact_text_observation"]
    strip_think_tags = imp["strip_think_tags"]

    loop = asyncio.get_running_loop()
    t0 = time.monotonic()

    game_cfg = GAME_CONFIGS.get(game)
    episode_id = f"{game}_{uuid.uuid4().hex[:8]}"

    # Create env (CPU-bound for emulators → executor)
    use_process = game in EMULATOR_GAMES and process_executor is not None
    exe = process_executor if use_process else executor

    if exe:
        base_env = await loop.run_in_executor(exe, make_gaming_env, game, max_steps)
    else:
        base_env = make_gaming_env(game=game, max_steps=max_steps)

    if game == "sokoban" and SokobanNLWrapper is not None:
        env = SokobanNLWrapper(base_env, reflect_every=3)
    else:
        env = GamingAgentNLWrapper(base_env)

    if exe:
        obs_nl, info = await loop.run_in_executor(exe, env.reset)
    else:
        obs_nl, info = env.reset()

    action_names = info.get("action_names", [])
    structured_state = info.get("structured_state")

    bank_available = skill_bank is not None and (
        hasattr(skill_bank, "__len__") and len(skill_bank) > 0
        or hasattr(skill_bank, "skill_ids") and len(list(skill_bank.skill_ids)) > 0
    )

    grpo_records: List[GRPORecord] = []
    experiences: List[Dict[str, Any]] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    current_intention = ""
    prev_summary_state = ""
    prev_intention = ""

    recent_actions: List[str] = []
    recent_rewards: List[float] = []
    skill_tracker = _SkillTracker()
    last_guidance: Optional[Dict[str, Any]] = None
    last_candidates: List[Dict[str, Any]] = []
    last_chosen_idx = 0
    last_skill_reasoning: Optional[str] = None

    while step_count < max_steps:
        step_actions = action_names if action_names else ["stay"]

        # ── 1. summary_state (deterministic, 0 LLM calls) ────────
        summary_state = _generate_summary_state(
            obs_nl, game_name=game,
            step_idx=step_count, total_steps=max_steps,
            reward=total_reward,
        )

        # ── 2. summary_prose (1 LLM call, cheap) ─────────────────
        compact = compact_text_observation(obs_nl, max_chars=200)
        state_text = compact if compact else obs_nl[:1000]
        game_label = game.replace("_", " ")
        delta = _compute_state_delta(prev_summary_state, summary_state)
        delta_line = f"Changed since last step: {delta}\n" if delta else ""
        summary_prompt = (
            f"{game_label}: {state_text}\n"
            f"{delta_line}"
            f"Key strategic note about the current threat or opportunity "
            f"(max 10 words, be specific to what changed).\nNote:"
        )
        summary_result = await vllm_client.generate(
            summary_prompt, adapter="base", temperature=0.2, max_tokens=25,
        )
        note = strip_think_tags(summary_result.text).strip() if summary_result.text else ""
        note = note.split("\n")[0].strip().strip('"').strip("'")[:80]
        current_summary = f"{summary_state} | note={note}" if note else summary_state
        current_summary = current_summary[:HARD_SUMMARY_CHAR_LIMIT]

        # ── 3. Skill selection (skill_selection LoRA) ─────────────
        need_reselect = skill_tracker.should_reselect(
            last_guidance, state_text=summary_state or obs_nl,
        )
        skill_select_prompt: Optional[str] = None

        if bank_available and (need_reselect or last_guidance is None):
            facts = extract_game_facts(obs_nl, game)
            step_structured = {k: v for k, v in facts.items() if v}

            from scripts.qwen3_decision_agent import get_top_k_skill_candidates
            candidates = get_top_k_skill_candidates(
                skill_bank,
                summary_state or obs_nl,
                game_name=game,
                intention=current_intention,
                structured_state=step_structured if step_structured else structured_state,
                top_k=3,
            )

            if candidates and len(candidates) >= 2:
                candidates_text = _format_candidates_for_selection(candidates)
                user_content = (
                    f"Game state:\n{(summary_state or obs_nl)[:3000]}\n\n"
                    f"Current intention: {current_intention[:500]}\n\n"
                    f"Available strategies (pick ONE by number):\n{candidates_text}\n\n"
                    f"Choose the best strategy. Output REASONING then SKILL number."
                )
                skill_select_prompt = SKILL_SELECTION_SYSTEM_PROMPT + "\n" + user_content
                sk_result = await vllm_client.generate(
                    skill_select_prompt, adapter="skill_selection",
                    temperature=temperature, max_tokens=256,
                )
                chosen_idx, skill_reasoning = _parse_skill_selection(
                    sk_result.text, len(candidates), candidates,
                )
                guidance = candidates[chosen_idx]
                if skill_reasoning:
                    guidance["why_selected"] = skill_reasoning
                last_candidates = candidates
                last_chosen_idx = chosen_idx
                last_skill_reasoning = skill_reasoning
                skill_tracker.set_protocol(guidance.get("protocol"))
            elif candidates:
                guidance = candidates[0]
                last_candidates = candidates
                last_chosen_idx = 0
                last_skill_reasoning = "only one candidate"
                skill_tracker.set_protocol(guidance.get("protocol"))
            else:
                guidance = None
                last_candidates = []
                last_chosen_idx = 0
                last_skill_reasoning = None

            last_guidance = guidance
        elif not bank_available:
            guidance = None
            last_guidance = None
        else:
            guidance = last_guidance

        # ── 4. Intention (1 LLM call) ────────────────────────────
        urgency = _detect_urgency(summary_state, game)
        urgency_line = f"URGENCY: {urgency}\n" if urgency else ""
        prev_line = f"Previous subgoal: {prev_intention}\n" if prev_intention else ""
        delta_intent = _compute_state_delta(prev_summary_state, summary_state)
        delta_intent_line = f"Changed: {delta_intent}\n" if delta_intent else ""
        skill_context = ""
        if guidance and guidance.get("skill_id"):
            sk_name = guidance.get("skill_name", guidance["skill_id"])
            sk_hint = guidance.get("execution_hint", "")
            skill_context = f"Active skill: {sk_name}"
            if sk_hint:
                skill_context += f" — {sk_hint[:100]}"
            skill_context += "\n"

        imp_tags = imp["SUBGOAL_TAGS"]
        tags_str = "|".join(imp_tags)
        facts_line = f"Facts: {summary_state}\n" if summary_state else ""
        intention_prompt = (
            f"{game_label}. Action: {recent_actions[-1] if recent_actions else 'start'}\n"
            f"State: {state_text}\n"
            f"{facts_line}{delta_intent_line}{urgency_line}{skill_context}{prev_line}"
            f"What subgoal? Reply ONLY: [TAG] phrase (max {INTENTION_WORD_BUDGET} words)\n"
            f"Tags: {tags_str}\nSubgoal:"
        )
        intention_result = await vllm_client.generate(
            intention_prompt, adapter="base", temperature=0.2, max_tokens=40,
        )
        intention_text = strip_think_tags(intention_result.text).strip() if intention_result.text else ""
        current_intention = _normalize_intention(intention_text)[:150] if intention_text else f"[EXECUTE] play"

        # ── 5. Action selection (action_taking LoRA) ──────────────
        recent_context = _build_recent_context(recent_actions, recent_rewards)
        summary_for_action = summary_state if summary_state else obs_nl[:4000]
        intention_line = f"Current intention: {current_intention}\n\n" if current_intention else ""
        skill_text = _format_skill_guidance_for_prompt(guidance, skill_tracker.protocol_step_idx)
        action_user = (
            f"Game state:\n\n{summary_for_action}\n\n"
            f"{intention_line}{recent_context}"
            f"Available actions (pick ONE by number):\n{_format_numbered_actions(step_actions)}\n\n"
            f"Choose the best action. Output REASONING then ACTION number."
        )
        action_prompt = SYSTEM_PROMPT + skill_text + "\n" + action_user

        action_result = await vllm_client.generate(
            action_prompt, adapter="action_taking",
            temperature=temperature, max_tokens=512,
        )
        action, reasoning = _parse_action_response(action_result.text, step_actions)
        action = _apply_anti_repetition(action, step_actions, recent_actions, recent_rewards)

        # ── 6. env.step() (in executor) ──────────────────────────
        try:
            if exe:
                next_obs_nl, reward, terminated, truncated, next_info = await loop.run_in_executor(
                    exe, env.step, action,
                )
            else:
                next_obs_nl, reward, terminated, truncated, next_info = env.step(action)
        except Exception as e:
            logger.warning("env.step failed at step %d: %s", step_count, e)
            break

        done = terminated or truncated
        total_reward += reward
        next_action_names = next_info.get("action_names", action_names)
        next_structured_state = next_info.get("structured_state")

        recent_actions.append(str(action))
        recent_rewards.append(float(reward))

        skill_id = guidance.get("skill_id") if guidance else None
        skill_name_val = guidance.get("skill_name", "") if guidance else ""
        skill_tracker.update(skill_id, skill_name_val, float(reward))

        # ── 7. Record GRPO I/O ───────────────────────────────────
        if action_prompt:
            try:
                action_num = step_actions.index(action) + 1
            except ValueError:
                action_num = 1
            action_completion = f"REASONING: {reasoning or 'Expert play.'}\nACTION: {action_num}"
            grpo_records.append(GRPORecord(
                adapter="action_taking", game=game, episode_id=episode_id, step=step_count,
                prompt=action_prompt, completion=action_completion, reward=float(reward),
                metadata={
                    "chosen_action": str(action),
                    "available_actions": list(step_actions),
                    "summary_state": summary_state,
                    "intention": current_intention,
                    "active_skill": skill_id,
                },
            ))

        if skill_select_prompt and last_candidates and len(last_candidates) >= 2:
            sk_completion = (
                f"REASONING: {last_skill_reasoning}\nSKILL: {last_chosen_idx + 1}"
                if last_skill_reasoning
                else f"SKILL: {last_chosen_idx + 1}"
            )
            grpo_records.append(GRPORecord(
                adapter="skill_selection", game=game, episode_id=episode_id, step=step_count,
                prompt=skill_select_prompt, completion=sk_completion, reward=float(reward),
                metadata={
                    "chosen_idx": last_chosen_idx,
                    "skill_candidates": [c.get("skill_id") for c in last_candidates],
                    "chosen_skill_id": (
                        last_candidates[last_chosen_idx].get("skill_id")
                        if last_chosen_idx < len(last_candidates) else None
                    ),
                    "summary_state": summary_state,
                    "intention": current_intention,
                },
            ))

        experiences.append({
            "step": step_count,
            "state": obs_nl,
            "action": str(action),
            "reward": float(reward),
            "next_state": next_obs_nl,
            "done": done,
            "intention": current_intention,
            "summary_state": summary_state,
            "skill_id": skill_id,
        })

        prev_summary_state = summary_state
        prev_intention = current_intention
        obs_nl = next_obs_nl
        action_names = next_action_names
        structured_state = next_structured_state
        step_count += 1

        if done:
            break

        # Early termination: stuck detection
        if (step_count >= min_steps_before_stuck
                and len(recent_rewards) >= stuck_window
                and sum(recent_rewards[-stuck_window:]) <= 0):
            logger.debug("Episode %s stuck at step %d, terminating early", episode_id, step_count)
            break

    try:
        env.close()
    except Exception:
        pass

    wall_time = time.monotonic() - t0
    return EpisodeResult(
        game=game,
        episode_id=episode_id,
        steps=step_count,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        skill_switches=skill_tracker.skill_switches,
        grpo_records=grpo_records,
        experiences=experiences,
        wall_time_s=wall_time,
    )
