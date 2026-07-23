"""Async episode runner for the co-evolution loop.

Mirrors ``scripts/qwen3_decision_agent.run_episode()`` but replaces every
synchronous LLM call with an ``await`` on the shared :class:`AsyncVLLMClient`,
and runs ``env.step()`` in an executor to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Headless mode for retro/pyglet/SDL — must be set before any game env import
os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

from trainer.coevolution.vllm_client import AsyncVLLMClient
from trainer.coevolution.skill_reward_shaping import (
    SkillChainTracker,
    PositionCollapseTracker,
    SkillDiversityTracker,
    exploration_bonus,
    premature_switch_penalty,
)

logger = logging.getLogger(__name__)


# T2.17b (2026-05-21): regex to rewrite the ``step=N`` line inside a
# cached ``<state>`` markup so reusing a previous step's markup looks
# fresh to downstream consumers.  Anchored to start-of-line and
# matches both ``step=12`` and ``step=12\r``.
_STATE_STEP_FIELD_RE = re.compile(r"^step=\d+", re.MULTILINE)


def _rewrite_step_field(markup: str, step: int) -> str:
    """Rewrite the ``step=<N>`` line in a cached state markup.

    Used by the vision-perception smart-fallback path: when the 35B
    judge fails for frame N we reuse the last successful markup (from
    frame N−1 or N−2), but the LoRA consumes ``step=`` as part of the
    state block, so we bump the field to keep the prompt coherent.
    Falls through to returning the markup unchanged if no
    ``step=...`` line is found (defensive — should not happen for
    35B output, which always includes the field).
    """
    if not markup:
        return markup
    new, n = _STATE_STEP_FIELD_RE.subn(f"step={int(step)}", markup, count=1)
    return new if n else markup


def _critical_actions_for(game: str, valid_actions: List[str]) -> List[str]:
    """Return the subset of :data:`GAME_CRITICAL_ACTIONS` that are
    actually exposed for *game* this step.  Imports lazily so callers
    that don't pass a real game string (eval / smoke tests) avoid
    pulling the whole config module into the hot path on every call.
    """
    try:
        from trainer.coevolution.config import GAME_CRITICAL_ACTIONS
    except Exception:                                        # pragma: no cover
        return []
    desired = GAME_CRITICAL_ACTIONS.get(game) or []
    if not desired:
        return []
    valid_set = set(valid_actions)
    return [a for a in desired if a in valid_set]


# ---------------------------------------------------------------------------
# Lazy imports — these pull in heavyweight packages that live in the project
# ---------------------------------------------------------------------------

_IMPORTS_CACHE: Dict[str, Any] = {}

# Games that use Orak env (env_wrappers.orak_nl_wrapper.make_orak_env)
ORAK_GAMES_SET = {"super_mario"}
# Orak games that MUST use SubprocessEnv (nes_py / NumPy 2.x incompatibility)
ORAK_SUBPROCESS_GAMES = {"super_mario"}
# Games that use GamingAgent make_gaming_env
GAMINGAGENT_GAMES = {
    "twenty_forty_eight", "candy_crush", "tetris",
}
# Games that use Gym-V Temporal/* (stable-retro Genesis) via
# env_wrappers.gymv_temporal_nl_wrapper.make_gymv_temporal_env. The set of
# wired slugs lives in that module's GYMV_TEMPORAL_GAMES dict (8 games for
# the default benchmark scope — the 4 Phase-1 source games plus the 4
# Phase-2 holdouts from
# training_notes/coevo-3phase-cross-game-ood-transfer-plan.md §4.1 / §7.1).
# We import the dict lazily inside _lazy_imports() so module-import time
# stays cheap when gym_v / stable-retro / ROMs aren't installed.
GYMV_TEMPORAL_GAMES_SET: set = set()  # populated by _lazy_imports()
# Gym-V games MUST use SubprocessEnv: stable-retro's Genesis emulator is
# a process-singleton ("Cannot create multiple emulator instances per
# process"), so any concurrent in-process episodes after the first one
# crash and return ``EpisodeResult(steps=0)`` — i.e. ``1/8 ok`` on the
# rollout collector log line.  Subprocess isolation gives each
# concurrent episode its own emulator and restores 8/8 GRPO rollouts.
# Set to the same membership as GYMV_TEMPORAL_GAMES_SET (populated lazily
# below) — keep them in sync.
GYMV_SUBPROCESS_GAMES: set = set()  # populated by _lazy_imports() (mirror)


def _lazy_imports():
    """Return project modules, imported once and cached."""
    global _IMPORTS_CACHE, GYMV_TEMPORAL_GAMES_SET, GYMV_SUBPROCESS_GAMES
    if not _IMPORTS_CACHE:
        from env_wrappers.game_configs import GAME_CONFIGS
        from env_wrappers.gym_like import make_gaming_env
        from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
        # Orak env (Super Mario, etc.)
        try:
            from env_wrappers.orak_nl_wrapper import make_orak_env
        except ImportError:
            make_orak_env = None

        from env_wrappers.subprocess_env import SubprocessEnv

        # Gym-V Temporal/* (stable-retro Genesis) — Phase-1 source +
        # Phase-2 holdout games. Import is best-effort: if gym_v /
        # stable-retro aren't installed, the slug set stays empty and
        # any --games gymv_<slug> request falls through to the
        # GAME_CONFIGS-not-found error instead of crashing the whole
        # runner at import time.
        try:
            from env_wrappers.gymv_temporal_nl_wrapper import (
                GYMV_TEMPORAL_GAMES,
                make_gymv_temporal_env,
            )
            GYMV_TEMPORAL_GAMES_SET.update(GYMV_TEMPORAL_GAMES.keys())
            # Mirror the slug set into the subprocess gate; we always
            # subprocess-isolate stable-retro envs (process-singleton
            # constraint, see GYMV_SUBPROCESS_GAMES docstring).
            GYMV_SUBPROCESS_GAMES.update(GYMV_TEMPORAL_GAMES.keys())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "env_wrappers.gymv_temporal_nl_wrapper unavailable (%s); "
                "Gym-V Temporal/* games will not be runnable in this process.",
                exc,
            )
            GYMV_TEMPORAL_GAMES = {}
            make_gymv_temporal_env = None

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
            "make_orak_env": make_orak_env,
            "make_gymv_temporal_env": make_gymv_temporal_env,
            "GYMV_TEMPORAL_GAMES": GYMV_TEMPORAL_GAMES,
            "SubprocessEnv": SubprocessEnv,
            "GamingAgentNLWrapper": GamingAgentNLWrapper,
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

INTENTION_WORD_BUDGET = 15
MAX_REPEAT_ACTIONS = 2

SYSTEM_PROMPT = (
    "You are an expert game-playing agent. "
    "You receive a game state and must choose exactly one action by its NUMBER.\n\n"
    "Rules:\n"
    "- Study the state carefully before choosing.\n"
    "- Consider which action makes the most progress toward winning.\n"
    "- NEVER repeat the same action more than 2 times in a row.\n"
    "- If recent actions got zero reward, change strategy.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences>\n"
    "ACTION: <number>\n\n"
)

SKILL_SELECTION_SYSTEM_PROMPT = (
    "You are a skill selector. Output exactly 3 lines:\n"
    "EFFECTS: <comma-separated effects achieved so far>\n"
    "DECISION: CONTINUE or SWITCH\n"
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

_TAG_KEYWORD_MAP: Dict[str, str] = {
    "surviv": "SURVIVE", "clear": "CLEAR", "merge": "MERGE",
    "setup": "SETUP", "position": "POSITION", "navigat": "NAVIGATE",
    "optimiz": "OPTIMIZE", "defend": "DEFEND", "attack": "ATTACK",
    "build": "BUILD", "explor": "EXPLORE", "collect": "COLLECT",
}


def _infer_tag_from_text(text: str) -> str:
    """Scan for keyword stems and return the best-matching tag."""
    low = text.lower()
    for stem, tag in _TAG_KEYWORD_MAP.items():
        if stem in low:
            return tag
    return "SETUP"


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
    episode_length: int = 1
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
    eval_only: bool = False

    # Multi-role metadata (populated when unified_role_rollouts=True)
    role: str = ""          # e.g. "Merlin", "FRANCE"
    side: str = ""          # e.g. "good", "evil", or power name
    role_index: int = -1    # player index (Avalon) or power ordinal

    # Per-skill runtime-discovered effects from StateEffectObserver.
    # {skill_id: {"eff_add": set_of_tags, "n_steps": int, "reward": float}}
    # Populated at each skill switch + episode end; used by
    # skillbank_pipeline to write-back contracts on seed mega-skills
    # whose Stage 3 segmentation labels as __NEW__.
    runtime_skill_effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Optional tamper-evident v3 source instrumentation. Empty on legacy
    # rollouts; this is evidence metadata and never changes GRPO reward.
    reasoning_event_log: Dict[str, Any] = field(default_factory=dict)
    # Same-snapshot policy contrasts used only by source qualification.
    # Kept outside the canonical event chain so the live policy stays singular.
    matched_policy_records: List[Dict[str, Any]] = field(default_factory=list)


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


def _reasoning_receipt_value(value: Any) -> Any:
    """Return a compact JSON-safe native-evidence value.

    Images and other large arrays are represented by shape/dtype metadata;
    textual observations and structured state remain fully inspectable.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _reasoning_receipt_value(item)
            for key, item in value.items()
            if str(key) not in {"image", "raw_obs"}
        }
    if isinstance(value, (list, tuple)):
        return [_reasoning_receipt_value(item) for item in value]
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {
            "array_omitted": True,
            "shape": [int(item) for item in shape],
            "dtype": str(dtype),
        }
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _reasoning_receipt_value(item())
        except Exception:
            pass
    return str(value)


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


def _build_rich_state_observation(
    info: Dict[str, Any],
    summary_state: str,
) -> str:
    """Build a rich spatial observation for games that provide grid data.
    Falls back to *summary_state* when grid is absent."""
    parts: List[str] = []
    grid_str = info.get("grid_string")
    if grid_str:
        parts.append(f"Board:\n{grid_str}")
    element_summary = info.get("element_summary")
    if element_summary:
        parts.append(f"Elements:\n{element_summary}")
    spatial = info.get("spatial_analysis")
    if spatial:
        parts.append(f"Analysis:\n{spatial}")
    deadlock_info = info.get("deadlock_info")
    if deadlock_info:
        parts.append(f"*** WARNING: {deadlock_info} — consider restart ***")
    if summary_state:
        parts.append(f"Status: {summary_state}")
    if parts:
        return "\n\n".join(parts)
    return summary_state or ""


def _normalize_intention(raw: str) -> str:
    imp = _lazy_imports()
    _SUBGOAL_TAG_SET = frozenset(imp["SUBGOAL_TAGS"])
    raw = raw.split("\n")[0].strip().strip('"').strip("'")
    if not raw.startswith("["):
        tag = _infer_tag_from_text(raw)
        return f"[{tag}] {raw}"
    m = _TAG_RE.match(raw)
    if not m:
        tag = _infer_tag_from_text(raw)
        return f"[{tag}] {raw}"
    tag = m.group(1).upper()
    rest = raw[m.end():].strip()
    if tag not in _SUBGOAL_TAG_SET:
        tag = _TAG_ALIASES.get(tag, _infer_tag_from_text(rest))
    return f"[{tag}] {rest}" if rest else f"[{tag}]"


async def _generate_intention(
    vllm_client: AsyncVLLMClient,
    state_text: str,
    game_name: str,
    summary_state: str,
    prev_intention: str,
    prev_summary_state: str,
    delta: str,
    urgency: str,
    skill_guidance: Optional[Dict[str, Any]],
    last_action: str,
    tag_history: Optional[List[str]] = None,
) -> str:
    """Generate a ``[TAG] subgoal`` via the **base model** (no LoRA).

    Ported from ``qwen3_decision_agent.generate_skill_aware_intention()``.
    Uses higher temperature (0.7) so the base model's SFT-trained tag
    diversity is preserved — unlike the action_taking LoRA which has
    collapsed to a single tag.
    """
    imp = _lazy_imports()
    SUBGOAL_TAGS = imp["SUBGOAL_TAGS"]
    tags_str = "|".join(SUBGOAL_TAGS)
    facts_line = f"Facts: {summary_state}\n" if summary_state else ""
    delta_line = f"Changed: {delta}\n" if delta else ""
    urgency_line = f"URGENCY: {urgency}\n" if urgency else ""
    prev_line = f"Previous subgoal: {prev_intention}\n" if prev_intention else ""
    shift_hint = (
        "IMPORTANT: If the situation changed significantly or urgency is high, "
        "pick a NEW tag that matches the new priority.\n"
        if delta or urgency else ""
    )

    diversity_hint = ""
    if tag_history and len(tag_history) >= 5:
        from collections import Counter
        window = tag_history[-10:]
        counts = Counter(window)
        top_tag, top_count = counts.most_common(1)[0]
        if top_count / len(window) > 0.5:
            others = [t for t in SUBGOAL_TAGS if t != top_tag][:4]
            diversity_hint = (
                f"DIVERSITY: You used [{top_tag}] {top_count}/{len(window)} "
                f"recent steps. Try a DIFFERENT tag like "
                f"{', '.join(others)}.\n"
            )

    skill_context = ""
    if skill_guidance and skill_guidance.get("skill_id"):
        sk_name = skill_guidance.get("skill_name", skill_guidance["skill_id"])
        sk_hint = skill_guidance.get("execution_hint", "")
        skill_context = f"Active skill: {sk_name}"
        if sk_hint:
            skill_context += f" — {sk_hint[:100]}"
        skill_context += "\n"

    game_label = game_name.replace("_", " ") if game_name else "game"

    examples = (
        "Examples:\n"
        "  tetris, stack_h=14, holes=8 → [SURVIVE] reduce stack height before game over\n"
        "  tetris, holes=2, stack_h=6 → [SETUP] position piece for future line clear\n"
        "  tetris, full row forming → [CLEAR] complete the line to score points\n"
        "  2048, empty=3, max=256 → [MERGE] combine tiles to free board space\n"
        "  2048, large tile in corner → [POSITION] keep max tile anchored in corner\n"
        "  2048, board nearly full → [SURVIVE] avoid game over by creating space\n"
        "  candy_crush, moves=4, target=500 → [CLEAR] maximize cascade combos now\n"
        "  candy_crush, special candy available → [EXECUTE] activate combo for big score\n"
        "  candy_crush, board cluttered → [OPTIMIZE] clear blockers to open matches\n"
    )

    prompt = (
        f"{game_label}. Action: {last_action}\n"
        f"State: {state_text}\n"
        f"{facts_line}"
        f"{delta_line}"
        f"{urgency_line}"
        f"{skill_context}"
        f"{prev_line}"
        f"{shift_hint}"
        f"{diversity_hint}"
        f"{examples}\n"
        f"What subgoal? Reply ONLY: [TAG] phrase "
        f"(max {INTENTION_WORD_BUDGET} words)\n"
        f"Tags: {tags_str}\n"
        f"Subgoal:"
    )

    try:
        result = await vllm_client.generate_chat(
            [{"role": "user", "content": prompt}],
            adapter="base", temperature=0.7, max_tokens=96,
        )
        text = result.text.strip() if result.text else ""
        if text:
            imp2 = _lazy_imports()
            text = imp2["strip_think_tags"](text).strip()
            first_line = text.split("\n")[0].strip()
            if first_line:
                return _normalize_intention(first_line)[:150]
    except Exception as exc:
        logger.debug("Intention generation failed: %s", exc)

    if prev_intention and prev_intention != "[EXECUTE] play":
        return prev_intention
    fallback_tag = _infer_tag_from_text(urgency or summary_state or "")
    return f"[{fallback_tag}] {game_label}"


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


def _format_skill_guidance_for_prompt(
    guidance: Optional[Dict[str, Any]],
    protocol_step_idx: int = 0,
    progress_summary: str = "",
) -> str:
    if guidance is None or not guidance.get("skill_id"):
        return ""
    parts = [f"\n--- Active Skill: {guidance.get('skill_name', guidance['skill_id'])} ---"]
    if guidance.get("execution_hint"):
        parts.append(f"  Strategy: {guidance['execution_hint'][:200]}")
    if progress_summary:
        parts.append(f"  Progress: {progress_summary}")
    protocol = guidance.get("protocol", {})
    req_effects = protocol.get("required_effects", []) if isinstance(protocol, dict) else []
    steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
    if req_effects:
        parts.append(f"  Required effects: {', '.join(req_effects[:8])}")
    if steps:
        parts.append(f"  Plan ({len(steps)} steps):")
        for i, step in enumerate(steps[:7], 1):
            parts.append(f"    {i}. {step}")

    # Paradigm C: render concrete reasoning exemplar from protocol_raw.
    proto_raw = guidance.get("protocol_raw")
    if isinstance(proto_raw, dict):
        raw_steps = proto_raw.get("steps", [])
    else:
        raw_steps = []
    exemplar_steps = guidance.get("exemplar_steps") or raw_steps
    if exemplar_steps:
        parts.append("  Example reasoning:")
        for es in exemplar_steps[:5]:
            parts.append(f"    - {str(es)[:150]}")
    failure_lesson = guidance.get("failure_lesson", "")
    if failure_lesson:
        parts.append(f"  Common mistake: {failure_lesson[:200]}")

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
    """Trainer-side mirror of
    :func:`scripts.qwen3_decision_agent._format_candidates_for_selection`.

    Renders ``_harness_adaptation_score`` (Refinement B) and the
    ``_harness_deboost`` recent-veto rate (Refinement A) when those
    fields are present on the candidate dict. Both fields are best-
    effort and omitted silently when the harness path didn't decorate
    the candidate. See ``harness/README.md`` §22.5 for the design.
    """
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
        adapt = c.get("_harness_adaptation_score")
        if isinstance(adapt, (int, float)):
            lines.append(f"     Adaptation: {float(adapt):.2f}")
        deboost = c.get("_harness_deboost")
        if isinstance(deboost, (int, float)) and float(deboost) < 0.95:
            lines.append(f"     Recent veto rate: {1.0 - float(deboost):.2f}")
    return "\n".join(lines)


def _parse_skill_selection(
    reply: str, n_candidates: int,
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, Optional[str], Optional[Tuple[int, int]]]:
    """Parse skill selection reply.

    Returns ``(chosen_idx, reasoning, step_progress)`` where
    ``step_progress`` is ``(completed, total)`` or ``None``.
    """
    imp = _lazy_imports()
    strip_think_tags = imp["strip_think_tags"]

    if not reply:
        return 0, None, None
    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply
    reasoning = None
    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nSTEP|\nSKILL|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    step_progress: Optional[Tuple[int, int]] = None
    step_m = re.search(r"STEP\s*:\s*(\d+)\s*/\s*(\d+)", cleaned, re.IGNORECASE)
    if step_m:
        completed = int(step_m.group(1))
        total = int(step_m.group(2))
        if 0 <= completed <= total <= 20 and total > 0:
            step_progress = (completed, total)

    skill_m = re.search(r"SKILL\s*:\s*(\d+)", cleaned, re.IGNORECASE)
    if skill_m:
        idx = int(skill_m.group(1)) - 1
        if 0 <= idx < n_candidates:
            return idx, reasoning, step_progress
    tail = cleaned[-100:]
    nums = re.findall(r"\b(\d+)\b", tail)
    for n_str in reversed(nums):
        idx = int(n_str) - 1
        if 0 <= idx < n_candidates:
            return idx, reasoning, step_progress
    if candidates:
        cleaned_lower = cleaned.lower()
        for i, c in enumerate(candidates):
            name = (c.get("skill_name") or "").lower()
            if name and len(name) >= 4 and name in cleaned_lower:
                return i, reasoning, step_progress
    return 0, reasoning, step_progress


def _parse_action_response(
    reply: str, valid_actions: List[str],
) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse action response (may contain optional SUBGOAL/REASONING lines).

    Returns (action, reasoning, intention).
    """
    imp = _lazy_imports()
    strip_think_tags = imp["strip_think_tags"]

    if not reply:
        return (valid_actions[0] if valid_actions else "stay"), None, None
    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply

    intention = None
    subgoal_m = re.search(
        r"SUBGOAL\s*:\s*(.+?)(?=\nREASONING|\nACTION|\Z)",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if subgoal_m:
        raw_sg = subgoal_m.group(1).strip().split("\n")[0].strip()
        intention = _normalize_intention(raw_sg)[:150] if raw_sg else None

    reasoning = None
    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nACTION|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    action_m = re.search(r"ACTION\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if action_m:
        raw = action_m.group(1).strip()
        matched = _fuzzy_match_action(raw, valid_actions)
        if matched:
            return matched, reasoning, intention

    # T2.19 (2026-05-18): trailing-number fallback (mirrors the
    # skill_selection parser).  v8 analysis (gymv_thunder_force_iii, 6
    # steps, 5237 samples) showed 6.2% of action_taking completions
    # never emit ``ACTION:`` and were stored as 150-char mid-word
    # fragments with reward=0.  Many of those completions DO end with a
    # bare digit ("...so I pick 3.") that maps to a valid action index
    # — recovering them as a soft "tail_number" match preserves the
    # rollout signal and is strictly better than the silent fallback.
    # We also accept any 1-based integer in the last 100 chars to align
    # with skill_decision_core.parse_skill_selection's recovery path.
    if valid_actions:
        tail = cleaned[-100:]
        nums = re.findall(r"\b(\d+)\b", tail)
        for n_str in reversed(nums):
            idx = int(n_str) - 1
            if 0 <= idx < len(valid_actions):
                return valid_actions[idx], reasoning, intention

    fallback = valid_actions[0] if valid_actions else "stay"
    return _ActionFallback(fallback), reasoning, intention


class _ActionFallback(str):
    """Marker subclass so callers can detect fuzzy-match failures."""
    pass


def _fuzzy_match_action(raw: str, valid_actions: List[str]) -> Optional[str]:
    if not raw or not valid_actions:
        return None
    raw_lower = raw.lower().rstrip(".").strip()
    lower_map = {a.lower(): a for a in valid_actions}
    if raw_lower in lower_map:
        return lower_map[raw_lower]

    raw_compact = re.sub(r"\s+", "", raw_lower)
    compact_map = {re.sub(r"\s+", "", a.lower()): a for a in valid_actions}
    if raw_compact in compact_map:
        return compact_map[raw_compact]
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


_CRITICAL_ACTION_DRY_SPELL = 8
"""Number of consecutive zero-reward decisions that haven't selected a
known-critical action before the anti-repetition shim force-substitutes
the critical action.  Tuned for gymv shooters where the natural episode
length is ~20 frames (frame_skip=8) and 8 consecutive non-firing decisions
≈ ~50% of an average episode without any scoring attempt."""


def _apply_anti_repetition(
    action: str, valid_actions: List[str],
    recent_actions: List[str], recent_rewards: List[float],
    game: str = "",
    rng: Optional[random.Random] = None,
) -> str:
    if len(recent_actions) < MAX_REPEAT_ACTIONS:
        return action
    tail = recent_actions[-MAX_REPEAT_ACTIONS:]
    tail_rewards = recent_rewards[-MAX_REPEAT_ACTIONS:]

    critical = _critical_actions_for(game, valid_actions)

    # Stuck on a single non-scoring action — break the loop, preferring
    # a critical action over a random alternative.
    if all(a == action for a in tail) and sum(tail_rewards) <= 0:
        alternatives = [a for a in valid_actions if a != action]
        if not alternatives:
            return action
        critical_alt = next((c for c in critical if c != action), None)
        if critical_alt is not None:
            return critical_alt
        return (rng or random).choice(alternatives)

    # Critical-action dry spell: the policy is exploring varied actions
    # but the env reward is zero AND no critical action has been picked
    # in the recent window.  Force-substitute the first critical action
    # so the learner at least observes the scoring action distribution.
    # Only fires for shooter-class games (those with critical actions).
    if (
        critical
        and len(recent_actions) >= _CRITICAL_ACTION_DRY_SPELL
        and len(recent_rewards) >= _CRITICAL_ACTION_DRY_SPELL
    ):
        window_actions = recent_actions[-_CRITICAL_ACTION_DRY_SPELL:]
        window_rewards = recent_rewards[-_CRITICAL_ACTION_DRY_SPELL:]
        no_critical_used = not any(a in critical for a in window_actions)
        no_reward = sum(r for r in window_rewards if r is not None) <= 0
        # Only override if the proposed action itself is non-critical;
        # never replace a critical action with a critical action.
        if no_critical_used and no_reward and action not in critical:
            return critical[0]

    return action


# ---------------------------------------------------------------------------
# Skill tracker (same logic as qwen3_decision_agent._SkillTracker)
# ---------------------------------------------------------------------------

from decision_agents.skill_decision_core import (
    StepTracker as _SkillTracker,
    DOMAIN_GAME,
    parse_skill_selection as _parse_skill_selection_unified,
    build_skill_selection_prompt as _build_skill_selection_prompt_unified,
    format_candidates_for_selection as _format_candidates_unified,
    SkillSelectionRecord,
)


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
    stuck_window: int = 15,
    min_steps_before_stuck: int = 20,
    vllm_base_urls: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    assigned_role: Optional[str] = None,
    assigned_role_index: Optional[int] = None,
    step_sync: Any = None,
    opponent_model: Optional[str] = None,
    opponent_api_base: Optional[str] = None,
    harness_hook: Any = None,
    reward_logger: Any = None,
    game_profile: Any = None,
    # Block B4 — intention trigger ablation:
    #   * "every-step"  (default): historical behaviour, intention LLM
    #     fires every inner step.
    #   * "sharp-shift": fires only when state delta or urgency
    #     indicates a meaningful shift; otherwise the prev intention
    #     is reused verbatim.
    #   * "disabled":   fires only at step 0; subsequent steps reuse
    #     the bootstrapped intention.
    intention_trigger: str = "every-step",
    # Block B5 — actor-side bank cap.  ``0`` (default) = no cap.
    # When >0, the SkillQueryEngine.select() restricts the candidate
    # pool to the top-K skills by relevance.
    actor_bank_cap_k: int = 0,
    # T2.17 (2026-05-05): per-step vision-grounded ``<state>`` markup.
    # When non-None and ``enabled``, the deterministic ``state_to_markup``
    # output is replaced (after env.reset and after each env.step) with a
    # 35B multimodal call grounded in the current frame.  All failures
    # silently fall back to the deterministic markup — see
    # :mod:`trainer.coevolution._vision_state_perception` for cache /
    # concurrency / timeout semantics.  ``None`` (default) preserves the
    # legacy text-only behaviour.
    #
    # Expected shape (set by ``rollout_collector`` from ``CoEvolutionConfig``)::
    #
    #   {
    #       "enabled": bool,
    #       "model":   str,    # 35B judge model name
    #       "concurrency": int,
    #       "timeout_s":   float,
    #       "max_tokens":  int,
    #       "temperature": float,
    #       "every_n_steps": int,  # 1 = every step (cold-start parity)
    #   }
    vision_perception_config: Optional[Dict[str, Any]] = None,
    # T2.18 (2026-05-05): early-death reward shaping (see implementation
    # at end of function).  Expected shape::
    #
    #   {
    #       "enabled": bool,
    #       "threshold_steps":  int,    # only penalise if steps < this
    #       "threshold_reward": float,  # only penalise if reward < this
    #       "base":             float,  # max penalty (at step 0)
    #   }
    #
    # Skipped silently when ``None`` or ``enabled=False``.  Truncated
    # episodes (timeout-based termination) are never penalised.
    early_death_config: Optional[Dict[str, Any]] = None,
    # T2.19: dense action reward shaping for sparse-reward envs.
    action_survival_bonus: float = 0.0,
    episode_return_redistribution_weight: float = 0.0,
    action_advance_bonus: float = 0.0,
    action_advance_actions: str = "RIGHT",
    # T2.19d (2026-05-20): RAM-watch driven hit / damage penalties.
    # Magnitudes are positive; the runtime applies the sign on a
    # negative-delta event in ``info["structured_state"]["ram_watch"]``.
    # 0.0 (default) → feature off.
    action_hit_penalty: float = 0.0,
    action_damage_penalty: float = 0.0,
    # T2.19e (2026-05-21): per-step attack / movement action bonuses.
    # ``action_attack_actions`` and ``action_movement_actions`` are
    # comma-separated lists of action tokens (uppercased) that trigger
    # the corresponding positive bonus when the actor's chosen action
    # is in the list.  Designed for shmups (AS) where teachers fire +
    # actively evade laterally — both at much higher rates than the
    # baseline agent — but ``action_advance_bonus`` (RIGHT-only) is the
    # wrong shape for a vertical-scrolling shmup.
    action_attack_bonus: float = 0.0,
    action_attack_actions: str = "B",
    action_movement_bonus: float = 0.0,
    action_movement_actions: str = "LEFT,RIGHT",
    episode_seed: Optional[int] = None,
    record_reasoning_events: bool = False,
    reasoning_backbone_harness: bool = False,
    matched_policy_skill_id: Optional[str] = None,
) -> EpisodeResult:
    """Run one game episode asynchronously.

    All LLM calls go through *vllm_client* (``await``).
    ``env.step()`` runs in *executor* to avoid blocking the event loop.

    Parameters
    ----------
    skill_bank : object | None
        ``None`` triggers cold-start mode (no skill selection).
    vllm_base_urls : list[str] | None
        Base URLs for vLLM instances (used for LLM opponent policies).
    model_name : str | None
        Model name for LLM opponent policy requests.
    assigned_role : str | None
        Reserved for multi-role games (unused for current game set).
    assigned_role_index : int | None
        Reserved for multi-role games (unused for current game set).
    opponent_model : str | None
        External API model for opponents (e.g. ``"gpt-5-mini"``).
        When set, non-controlled players use this model via API
        instead of vLLM self-play.
    opponent_api_base : str | None
        Base URL for the opponent API (default: OpenRouter).
    game_profile : ``trainer.coevolution._game_schema.GameProfile`` | None
        Per-phase static GameProfile (Path 1).  When supplied, the
        compact rendering (goal / win_signal / hazards / key_actions /
        failure_modes) is prepended to ``SYSTEM_PROMPT`` and
        ``SKILL_SELECTION_SYSTEM_PROMPT`` for every step of this
        episode.  Adds ~150 tokens to each actor / skill-selection
        prompt; no per-step LLM cost.  ``None`` (default) preserves
        the legacy hard-coded prompt.
    """
    imp = _lazy_imports()
    GAME_CONFIGS = imp["GAME_CONFIGS"]
    make_gaming_env = imp["make_gaming_env"]
    make_orak_env = imp["make_orak_env"]
    GamingAgentNLWrapper = imp["GamingAgentNLWrapper"]
    HARD_SUMMARY_CHAR_LIMIT = imp["HARD_SUMMARY_CHAR_LIMIT"]
    extract_game_facts = imp["extract_game_facts"]
    compact_text_observation = imp["compact_text_observation"]
    strip_think_tags = imp["strip_think_tags"]
    if reasoning_backbone_harness:
        if not record_reasoning_events:
            raise ValueError("reasoning_backbone_harness requires reasoning event recording")
        from harness.agent_reasoning_cycle import (  # noqa: WPS433
            parse_agent_action_proposal_set,
            parse_agent_post_transition_verdict,
        )

    # Path 1 wiring.  Imported at function entry so a refactor of
    # ``_state_to_markup`` / ``_game_schema`` cannot wedge episode
    # collection at module-import time, and so the ``GameProfile``
    # rendering is computed once per episode (not per step) for the
    # actor-prompt prefix.
    from trainer.coevolution._state_to_markup import state_to_markup

    # T2.17 (2026-05-05): vision-aware <state> markup wiring.  Resolved
    # once per episode so we can short-circuit cheaply on every step.
    _vision_cfg = vision_perception_config or {}
    _vision_on = bool(_vision_cfg.get("enabled", False))
    _vision_n = max(1, int(_vision_cfg.get("every_n_steps", 1)))
    _vision_model = str(_vision_cfg.get("model", "") or "")
    if _vision_on and not _vision_model:
        _vision_on = False
    if _vision_on:
        from trainer.coevolution._vision_state_perception import (
            vision_state_to_markup_async,
        )
    _last_vision_markup: Optional[str] = None

    async def _markup_for(
        *, obs_nl_v: str, info_v: Dict[str, Any], step_v: int,
    ) -> str:
        """Compute deterministic markup; optionally upgrade with vision.

        Always returns a non-empty string.  When vision is enabled and a
        frame is available we call the 35B judge with concurrency /
        timeout / fallback handled inside
        :func:`vision_state_to_markup_async`.  Throttling via
        ``every_n_steps`` reuses the previous frame's vision markup to
        cap judge spend at high frequencies.
        """
        nonlocal _last_vision_markup
        det = ""
        try:
            det = state_to_markup(
                obs_nl=obs_nl_v, info=info_v, game=game, step=step_v,
            )
        except Exception as _markup_exc:  # noqa: BLE001
            logger.debug(
                "state_to_markup failed at step %d: %s",
                step_v, _markup_exc,
            )
        if not _vision_on:
            return det
        if step_v > 0 and (step_v % _vision_n) != 0:
            if _last_vision_markup:
                return _last_vision_markup
            return det
        frame_url: Optional[str] = None
        try:
            renderer = getattr(env, "render", None)
            if callable(renderer):
                rendered = renderer()
                if isinstance(rendered, str) and rendered.startswith(
                    "data:image/"
                ):
                    frame_url = rendered
                elif rendered is not None:
                    from trainer.coevolution._game_schema import (
                        _encode_image_to_data_url,
                    )
                    frame_url = _encode_image_to_data_url(rendered)
        except Exception as _render_exc:  # noqa: BLE001
            logger.debug(
                "vision-perception render failed at step %d: %s",
                step_v, _render_exc,
            )
            frame_url = None
        try:
            markup_v = await vision_state_to_markup_async(
                obs_nl=obs_nl_v, info=info_v, game=game, step=step_v,
                frame_data_url=frame_url,
                fallback_markup=det,
                model=_vision_model,
                max_tokens=int(_vision_cfg.get("max_tokens", 768)),
                temperature=float(_vision_cfg.get("temperature", 0.1)),
                timeout_s=float(_vision_cfg.get("timeout_s", 6.0)),
                concurrency=int(_vision_cfg.get("concurrency", 12)),
                executor=executor,
            )
        except Exception as _vis_exc:  # noqa: BLE001
            logger.debug(
                "vision-perception fatal at step %d: %s",
                step_v, _vis_exc,
            )
            markup_v = det
        # T2.17b (2026-05-21): smart fallback on vision failure.
        # ``vision_state_to_markup_async`` returns ``fallback_markup``
        # (i.e. ``det``) on any failure path (timeout / parse_failure /
        # request_error / build_failure).  Previously we returned that
        # ``det`` directly, throwing away the last-good 35B markup —
        # an OOD shock for the action_taking LoRA that was trained on
        # entity-rich SFT markup, not the HUD-blind deterministic
        # output.  Now: when the call falls back AND we have a
        # previous successful 35B markup in this episode, reuse it
        # (with the step= field rewritten to the current step so
        # downstream consumers see a fresh-looking block).  Same-episode
        # frames are usually nearly identical (frame_skip=8), so a
        # 1-2 step old good markup is far more in-distribution than
        # det.  Only when no prior vision success exists (typical only
        # for step 0) do we genuinely return det.
        is_vision_success = bool(markup_v) and markup_v != det
        if is_vision_success:
            _last_vision_markup = markup_v
            return markup_v
        if _last_vision_markup:
            return _rewrite_step_field(_last_vision_markup, step_v)
        return markup_v or det

    if game_profile is not None:
        try:
            from trainer.coevolution._game_schema import render_for_actor_prompt
            _profile_prefix = render_for_actor_prompt(game_profile) + "\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GameProfile render failed for game=%s: %s — "
                "proceeding without profile prefix",
                game, exc,
            )
            _profile_prefix = ""
    else:
        _profile_prefix = ""

    loop = asyncio.get_running_loop()
    t0 = time.monotonic()

    game_cfg = GAME_CONFIGS.get(game)
    episode_id = f"{game}_{uuid.uuid4().hex[:8]}"
    exe = executor
    _reasoning_recorder = None
    _event_hash = None
    _policy_rng = random.Random(episode_seed)
    if record_reasoning_events:
        from harness.reasoning_event_log import (  # noqa: WPS433
            ReasoningEventKind,
            ReasoningEventRecorder,
        )
        import hashlib as _event_hashlib

        _reasoning_recorder = ReasoningEventRecorder(episode_id)
        def _event_hash(value: Any) -> str:
            encoded = json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, default=str,
            ).encode("utf-8")
            return _event_hashlib.sha256(encoded).hexdigest()
    _matched_policy_records: List[Dict[str, Any]] = []

    if game in ORAK_GAMES_SET:
        SubprocessEnv = imp["SubprocessEnv"]
        use_subprocess = (
            make_orak_env is None or game in ORAK_SUBPROCESS_GAMES
        )
        if use_subprocess:
            logger.info(
                "Using SubprocessEnv for %s (orak_import=%s, forced=%s)",
                game, make_orak_env is not None, game in ORAK_SUBPROCESS_GAMES,
            )
            if exe:
                env = await loop.run_in_executor(
                    exe, SubprocessEnv, game, max_steps,
                )
            else:
                env = SubprocessEnv(game=game, max_steps=max_steps)
        elif exe:
            env = await loop.run_in_executor(
                exe, make_orak_env, game, max_steps,
            )
        else:
            env = make_orak_env(game, max_steps=max_steps)

    elif game in GYMV_TEMPORAL_GAMES_SET:
        # Gym-V Temporal/* (stable-retro Genesis).  Always go through
        # SubprocessEnv: stable-retro's emulator is a process singleton
        # ("Cannot create multiple emulator instances per process"), so
        # in-process concurrent episodes drop 7/8 rollouts — see
        # GYMV_SUBPROCESS_GAMES docstring above.  The worker calls
        # ``make_gymv_temporal_env`` inside the child (frame_skip=8,
        # max_steps from caller) so the obs_nl / info contract is
        # preserved 1:1 with the in-process path.
        SubprocessEnv = imp["SubprocessEnv"]
        if game not in GYMV_SUBPROCESS_GAMES:
            raise RuntimeError(
                f"Gym-V Temporal/* env requested ({game!r}) but gym_v / "
                "stable-retro / Mega Drive ROMs are not installed; run "
                "install/install_gymv.sh + install/gymv_temporal_patch/"
                "apply_patch.sh and retry."
            )
        logger.info(
            "Using SubprocessEnv[gymv] for %s (emulator-singleton isolation)",
            game,
        )
        if exe:
            env = await loop.run_in_executor(
                exe,
                lambda: SubprocessEnv(
                    game=game, max_steps=max_steps, env_kind="gymv",
                ),
            )
        else:
            env = SubprocessEnv(game=game, max_steps=max_steps, env_kind="gymv")

    else:
        if exe:
            base_env = await loop.run_in_executor(
                exe, make_gaming_env, game, max_steps,
            )
        else:
            base_env = make_gaming_env(game=game, max_steps=max_steps)

        if game == "tetris":
            from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper
            env = TetrisMacroActionWrapper(GamingAgentNLWrapper(base_env))
        else:
            env = GamingAgentNLWrapper(base_env)

    # ── Resolve role / side metadata for multi-role games ───────
    _ep_role = ""
    _ep_side = ""
    _ep_role_idx = -1

    if _reasoning_recorder is not None:
        _reasoning_recorder.append(ReasoningEventKind.RESET, {
            "requested_seed": episode_seed,
            "seed_application_status": (
                "PASSED_TO_ENV_RESET_NOT_HIDDEN_STATE_VERIFIED"
                if episode_seed is not None else "NOT_REQUESTED"
            ),
            "environment_fingerprint": {
                "game": game,
                "wrapper_module": type(env).__module__,
                "wrapper_class": type(env).__qualname__,
                "max_steps": int(max_steps),
            },
        })
    if exe:
        if episode_seed is None:
            obs_nl, info = await loop.run_in_executor(exe, env.reset)
        else:
            from functools import partial
            obs_nl, info = await loop.run_in_executor(
                exe, partial(env.reset, seed=episode_seed),
            )
    else:
        obs_nl, info = env.reset(seed=episode_seed) if episode_seed is not None else env.reset()

    action_names = info.get("action_names", [])
    structured_state = info.get("structured_state")
    # Path 1: emit the unified <state> markup so it's available to any
    # downstream consumer (skill_selection, Crafter, Harness validator,
    # eval scorers).  Same call site is used at every env.step below so
    # train and eval see byte-identical schema for the same observation.
    # T2.17 (2026-05-05): when vision_perception is enabled, the
    # deterministic markup is replaced with a 35B-grounded one matching
    # the cold-start SFT distribution.  ``_markup_for`` falls back to
    # ``state_to_markup`` on every failure path.
    info["state_markup"] = await _markup_for(
        obs_nl_v=obs_nl, info_v=info, step_v=0,
    )
    current_info = info
    if _reasoning_recorder is not None:
        _reasoning_recorder.append(ReasoningEventKind.OBSERVATION, {
            "step": 0,
            "observable_state_sha256": _event_hash(obs_nl),
            "observable_state": str(obs_nl),
            "structured_state": _reasoning_receipt_value(structured_state),
            "simulator_state_sha256": None,
            "simulator_state_available": False,
            "native_actions_sha256": _event_hash(list(action_names)),
        })

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
    tag_history: List[str] = []
    skill_tracker = _SkillTracker(domain=DOMAIN_GAME, game_name=game)
    chain_tracker = SkillChainTracker()
    collapse_tracker = PositionCollapseTracker()
    diversity_tracker = SkillDiversityTracker()
    last_guidance: Optional[Dict[str, Any]] = None
    last_candidates: List[Dict[str, Any]] = []
    last_chosen_idx = 0
    last_skill_reasoning: Optional[str] = None
    last_sk_lora_text: Optional[str] = None
    _pending_markup: Optional[asyncio.Task] = None
    # T2.19d (2026-05-20): RAM-watch driven hit / damage penalties.
    # Tracks the previously-observed ``lives`` / ``health`` from
    # ``info["structured_state"]["ram_watch"]`` so we can apply a
    # per-step penalty when the agent gets hit.  ``None`` means
    # "no valid prior observation" → first step never fires a
    # penalty (no delta possible).  These persist across the entire
    # episode; once a value is observed it stays "stuck" at the
    # last non-None reading even if a later step's ram_watch is
    # missing the key (gym-v fallback dict).
    _prev_lives: Optional[int] = None
    _prev_health: Optional[int] = None

    def _safe_int_ram(v: Any) -> Optional[int]:
        """Coerce a ram_watch entry to ``int`` defensively.

        stable-retro returns numpy scalars (``np.uint16``) but
        downstream consumers occasionally insert raw ints or strings.
        Anything not convertible returns ``None`` (no delta computed).
        """
        if v is None:
            return None
        try:
            return int(v.item() if hasattr(v, "item") else v)
        except (TypeError, ValueError):
            return None

    while step_count < max_steps:
        step_actions = action_names if action_names else ["stay"]

        # Await pipelined 35B vision markup from the previous step.
        # Overlaps 35B generation with reward logging, GRPO record
        # assembly, and the loop-transition overhead of the prior step.
        if _pending_markup is not None:
            try:
                current_info["state_markup"] = await _pending_markup
            except Exception:
                pass
            _pending_markup = None

        # ── 1. summary_state (deterministic, 0 LLM calls) ────────
        summary_state = _generate_summary_state(
            obs_nl, game_name=game,
            step_idx=step_count, total_steps=max_steps,
            reward=total_reward,
        )

        compact = compact_text_observation(obs_nl, max_chars=200)
        state_text = compact if compact else obs_nl[:1000]
        game_label = game.replace("_", " ")
        delta = _compute_state_delta(prev_summary_state, summary_state)
        delta_line = f"Changed since last step: {delta}\n" if delta else ""

        # Pre-compute urgency (needed by both intention and action prompts)
        urgency = _detect_urgency(summary_state, game)

        # ── 2+3+4. intention, skill_selection (PARALLEL)
        # Summary-prose LLM call removed: the 10-word strategic note added
        # ~6s latency per step (competing for vLLM slots) while providing
        # negligible signal.  Use summary_state directly instead.
        summary_coro = None

        # Block B4 — intention trigger ablation.  When the trigger
        # fires, we generate a fresh intention via the LLM; when it
        # doesn't, we synthesise a coroutine that returns
        # ``prev_intention`` so the downstream ``asyncio.gather`` shape
        # stays unchanged.
        def _should_regen_intention() -> bool:
            if intention_trigger == "every-step":
                return True
            if intention_trigger == "disabled":
                return step_count == 0 or not prev_intention
            # "sharp-shift": fire on bootstrap, on detected urgency, or
            # on a non-trivial state delta.  Heuristic chosen so the
            # ablation matches the §4.1 definition without a separate
            # 35B classifier.
            if not prev_intention:
                return True
            if urgency:
                return True
            if delta and len(delta) >= 8:
                return True
            return False

        if _should_regen_intention():
            intention_coro = _generate_intention(
                vllm_client,
                state_text=state_text,
                game_name=game,
                summary_state=summary_state,
                prev_intention=prev_intention,
                prev_summary_state=prev_summary_state,
                delta=delta,
                urgency=urgency,
                skill_guidance=last_guidance,
                last_action=recent_actions[-1] if recent_actions else "start",
                tag_history=tag_history,
            )
        else:
            async def _passthrough_intention(_keep: str = prev_intention) -> str:
                return _keep
            intention_coro = _passthrough_intention()

        need_reselect = skill_tracker.should_reselect(
            last_guidance, state_text=summary_state or obs_nl,
        )
        skill_select_prompt: Optional[str] = None
        skill_coro = None

        # T2.15: pre-bind `harness_filter_diag` at the per-step scope.
        # Previously initialized only inside the `if bank_available …:`
        # block below, but read unconditionally at the experience-dict
        # assembly site (~L1384) — when the skill bank was empty (cold-
        # start step 0) or sticky-guidance kept us out of the inner
        # block, we hit `UnboundLocalError: harness_filter_diag`, which
        # collapsed every rollout in the wave (8/8 episodes ERR with 0
        # GRPO records).  Mirror the existing `harness_validate_diag`
        # outer init below.
        harness_filter_diag: Optional[Dict[str, Any]] = None

        if bank_available and (need_reselect or last_guidance is None):
            facts = extract_game_facts(obs_nl, game)
            _ss_for_facts = (current_info.get("structured_state") or {}).get("ram_watch") or {}
            for _fk, _fv in _ss_for_facts.items():
                if _fk not in facts and _fv is not None:
                    facts[_fk] = str(_fv.item() if hasattr(_fv, "item") else _fv)
            step_structured = {k: v for k, v in facts.items() if v}

            from scripts.qwen3_decision_agent import get_top_k_skill_candidates
            candidates = get_top_k_skill_candidates(
                skill_bank,
                summary_state or obs_nl,
                game_name=game,
                intention=current_intention,
                structured_state=step_structured if step_structured else structured_state,
                top_k=3,
                bank_cap_k=int(actor_bank_cap_k or 0),
            )

            # Pre-LLM harness eligibility filter (PLAN-HARNESS §5.2). When
            # `harness_hook` is supplied, candidates the harness vetoes
            # (status / domain / task / adapter / can_handle) are dropped
            # *before* the skill_selection LLM sees them, and their veto
            # reason is observed by the hook's RejectedSkillSink for the
            # Crafter to consume in Phase B′. See
            # `trainer/coevolution/_harness_hook.py` for the full
            # contract.
            if harness_hook is not None:
                try:
                    _hstate = harness_hook.state_for_step(
                        game=game,
                        summary_state=summary_state,
                        intention=current_intention,
                        inner_step=step_count,
                        outer_step=step_count,
                    )
                    candidates, harness_filter_diag = harness_hook.filter_candidates(
                        list(candidates), _hstate,
                        episode_id=episode_id,
                    )
                except Exception as _hexc:                       # noqa: BLE001
                    logger.debug(
                        "harness_hook.filter_candidates failed at step=%d: %s — "
                        "falling back to unfiltered candidates",
                        step_count, _hexc,
                    )
                    harness_filter_diag = {"harness_error": repr(_hexc)}

            if candidates and len(candidates) >= 2:
                _ss_rich = (
                    current_info.get("state_markup") if current_info else None
                )
                _ss_state_text = (
                    _ss_rich if _ss_rich
                    else (summary_state or obs_nl)
                )
                if _ss_state_text and "<state>" in _ss_state_text:
                    _ss_state_text = (
                        _ss_state_text
                        .replace("<state>\n", "", 1)
                        .replace("\n</state>", "", 1)
                        .replace("<state>", "", 1)
                        .replace("</state>", "", 1)
                    )
                skill_select_prompt = _build_skill_selection_prompt_unified(
                    state_text=_ss_state_text,
                    intention=current_intention,
                    candidates=candidates,
                    tracker=skill_tracker,
                    profile_prefix=_profile_prefix,
                    recent_actions=recent_actions,
                    recent_rewards=recent_rewards,
                )
                # NOTE (May-2026 fix, mirrors the same fix on action_taking
                # at line ~1550): the skill_selection LoRA was SFT-trained on
                # the ``/completions`` endpoint (raw prompt text, no chat
                # template wrapping).  Calling ``generate_chat`` here
                # produced an OOD distribution -- the chat-template prefix
                # (``<|im_start|>user``...) flipped Qwen3.5 into thinking-
                # mode (``Thinking Process: ...``) and the actual
                # ``EFFECTS:/DECISION:/SKILL:`` payload was either never
                # emitted or got chopped by the ``stop`` list before
                # reaching the SKILL line.  Direct probing of the deployed
                # adapter showed: via /completions it emits clean
                # ``SKILL: 2`` (idx=1) consistently; via /chat/completions
                # it collapses to ``EFFECTS:.. DECISION: CONTINUE`` and
                # truncates -- the parser then silently fell back to
                # candidate 0 every single step.  This was the dominant
                # silent-fallback in TF3 runs prior to commit 0f8f668.
                skill_coro = vllm_client.generate(
                    skill_select_prompt,
                    adapter="skill_selection",
                    temperature=temperature, max_tokens=128,
                    stop=["\n\nAvailable", "\n\nGame state", "\n\n---"],
                )

        # Sync with other episodes so LLM requests hit vLLM together
        # (batch-size-1 throughput is 10-20x worse than batched).
        if step_sync is not None:
            await step_sync.arrive()

        # Fire all LLM calls concurrently
        if skill_coro is not None:
            assigned_subgoal, sk_result = await asyncio.gather(
                intention_coro, skill_coro,
            )
        else:
            assigned_subgoal = await intention_coro
            sk_result = None

        current_summary = (summary_state or obs_nl[:500])[:HARD_SUMMARY_CHAR_LIMIT]

        # Post-LLM harness validate_invocation (PLAN-UNIFIED §3.4).
        # Wraps the LLM's chosen skill in a structured second-pass veto.
        # When vetoed we walk to the next candidate; when no eligible
        # candidate survives, we drop guidance and run unguided.
        harness_validate_diag: Optional[Dict[str, Any]] = None

        def _harness_validate(_idx: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
            """Validate `candidates[_idx]` via the harness.

            Returns ``(ok, diag)``. ``ok=True`` when no harness hook is
            configured or when validation admits.
            """
            if harness_hook is None:
                return True, None
            try:
                _sid = (candidates[_idx] or {}).get("skill_id")
                _hstate2 = harness_hook.state_for_step(
                    game=game,
                    summary_state=summary_state,
                    intention=current_intention,
                    inner_step=step_count,
                    outer_step=step_count,
                )
                # Path 4 — pass ``episode_id`` for per-episode LLM
                # validator caching (no-op when validator is off).
                # ``inner_step`` is plumbed for block A2 logging only.
                _ok, _d = harness_hook.validate_choice(
                    _sid, _hstate2,
                    episode_id=episode_id,
                    inner_step=step_count,
                )
                return bool(_ok), _d
            except Exception as _vexc:                          # noqa: BLE001
                logger.debug(
                    "harness_hook.validate_choice failed at step=%d "
                    "idx=%d: %s — admitting",
                    step_count, _idx, _vexc,
                )
                return True, {"status": "harness_error", "error": repr(_vexc)}

        # Process skill selection result
        if bank_available and (need_reselect or last_guidance is None):
            sk_parse_path: Optional[str] = None
            # Initialised up-front so the three branches below
            # (LoRA-fires / single-candidate-harness / no-candidate)
            # all share the same downstream metadata schema.  These
            # are populated by the LoRA branch and stay at their
            # defaults in the harness-only / empty branches (where
            # ``harness_override`` is meaningless — the LoRA never
            # picked anything to override).
            lora_chosen_idx: int = -1
            harness_override: bool = False
            if sk_result is not None and candidates and len(candidates) >= 2:
                # ``return_parse_path=True`` surfaces which parse strategy
                # produced the chosen_idx (skill_tag = clean LoRA emission,
                # tail_number / name_substring = heuristic recovery,
                # fallback_zero / empty_reply = LoRA failed entirely).
                # Used downstream to (1) include in GRPO metadata so
                # reward analysis can separate intelligent selections
                # from silent fallbacks, and (2) penalise unparseable
                # LoRA output in the skill_selection reward so the
                # adapter learns the SFT format.
                _parsed = _parse_skill_selection_unified(
                    sk_result.text, len(candidates), candidates,
                    strip_think_tags=strip_think_tags,
                    return_parse_path=True,
                )
                chosen_idx, _lora_effects, _decision, sk_parse_path = _parsed  # type: ignore[misc]
                last_sk_lora_text = sk_result.text

                if _lora_effects:
                    skill_tracker.receive_lora_effects(_lora_effects)

                # ``lora_chosen_idx`` is the index the skill_selection
                # LoRA explicitly emitted (or that the parse heuristics
                # recovered).  ``chosen_idx`` may diverge below if the
                # harness vetoes the LoRA's pick and we fall through to
                # another candidate — we track the divergence so the
                # GRPO metadata can correlate "LoRA selected X but
                # harness overrode to Y" (an instructive learning
                # signal that's invisible in the legacy reward log).
                lora_chosen_idx = chosen_idx
                if _decision == "CONTINUE" and last_guidance is not None:
                    guidance = last_guidance
                    last_candidates = candidates
                    last_chosen_idx = chosen_idx
                    last_skill_reasoning = None
                else:
                    # SWITCH: validate and adopt the new skill
                    _scan_order = [chosen_idx] + [
                        i for i in range(len(candidates)) if i != chosen_idx
                    ]
                    _picked: Optional[int] = None
                    _last_v: Optional[Dict[str, Any]] = None
                    for _i in _scan_order:
                        _ok, _d = _harness_validate(_i)
                        if _ok:
                            _picked = _i
                            _last_v = _d
                            break
                        _last_v = _d
                    harness_validate_diag = _last_v
                    if _picked is None:
                        # Harness rejected EVERY candidate.  Episode
                        # runs without skill guidance this step but
                        # ``lora_chosen_idx`` is preserved so the GRPO
                        # reward can still penalise the LoRA's
                        # selection (it picked a skill the harness
                        # would have rejected — that IS a learning
                        # signal worth backpropagating).
                        harness_override = True
                        guidance = None
                        last_candidates = candidates
                        last_chosen_idx = chosen_idx
                        last_skill_reasoning = None
                    else:
                        if _picked != lora_chosen_idx:
                            # Harness silently switched the LoRA's
                            # selection.  Log + flag so the metadata
                            # captures the divergence; the reward
                            # shaping below applies a small penalty
                            # so the LoRA learns the harness's
                            # eligibility rules.
                            harness_override = True
                            logger.info(
                                "skill_selection: harness override "
                                "for episode=%s step=%d — LoRA picked "
                                "idx=%d (%s) but harness vetoed; "
                                "fell through to idx=%d (%s)",
                                episode_id, step_count,
                                lora_chosen_idx,
                                candidates[lora_chosen_idx].get("skill_id", "?"),
                                _picked,
                                candidates[_picked].get("skill_id", "?"),
                            )
                        chosen_idx = _picked
                        guidance = candidates[chosen_idx]
                        last_candidates = candidates
                        last_chosen_idx = chosen_idx
                        last_skill_reasoning = None
                        skill_tracker.set_protocol(guidance.get("protocol"))
                        _chosen_sid = guidance.get("skill_id")
                        if _chosen_sid and hasattr(skill_bank, "selection_tracker"):
                            skill_bank.selection_tracker.increment(_chosen_sid)
            elif candidates:
                _picked2: Optional[int] = None
                _last_v2: Optional[Dict[str, Any]] = None
                for _i in range(len(candidates)):
                    _ok, _d = _harness_validate(_i)
                    if _ok:
                        _picked2 = _i
                        _last_v2 = _d
                        break
                    _last_v2 = _d
                harness_validate_diag = _last_v2
                if _picked2 is None:
                    guidance = None
                    last_candidates = candidates
                    last_chosen_idx = 0
                    last_skill_reasoning = None
                else:
                    guidance = candidates[_picked2]
                    last_candidates = candidates
                    last_chosen_idx = _picked2
                    last_skill_reasoning = None
                    skill_tracker.set_protocol(guidance.get("protocol"))
                    _chosen_sid = guidance.get("skill_id")
                    if _chosen_sid and hasattr(skill_bank, "selection_tracker"):
                        skill_bank.selection_tracker.increment(_chosen_sid)
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

        # ── 5. Action selection (action_taking LoRA) ────────────
        # The intention tag comes from the base model (step 4 above).
        # We inject it as "Assigned subgoal" so the LoRA follows it.
        urgency_line = f"URGENCY: {urgency}\n" if urgency else ""

        recent_context = _build_recent_context(recent_actions, recent_rewards)
        _is_macro = getattr(env, "_is_macro_action", False)

        _rich_markup = (
            current_info.get("state_markup") if current_info else None
        )
        _use_rich_obs = getattr(env, "_has_rich_observation", False)
        if _rich_markup and "<state>" in _rich_markup:
            summary_for_action = _rich_markup
        elif _use_rich_obs and current_info:
            summary_for_action = _build_rich_state_observation(
                current_info, summary_state,
            )
        elif current_summary:
            summary_for_action = current_summary
        else:
            summary_for_action = obs_nl[:4000]

        if _is_macro and "<actions>" in summary_for_action:
            summary_for_action = re.sub(
                r"\n?<actions>\n.*?(?=\n<|\Z)",
                "",
                summary_for_action,
                flags=re.DOTALL,
            )

        _backbone_plan = None
        _backbone_plan_result = None
        _backbone_plan_error = None
        _backbone_plan_prompt = ""
        if reasoning_backbone_harness:
            _backbone_plan_prompt = (
                "You are an untrusted planning Agent. Propose 1-3 candidate native "
                "actions and predict only their observable state deltas. Do not name "
                "cross-domain skills or invent predicates. ACTION_NUMBER is 1-based "
                "and must refer to the exact list below. Return exactly one JSON object "
                "with keys proposals,selected_proposal_id,decision. decision is EXECUTE "
                "or ABSTAIN; ABSTAIN requires selected_proposal_id=null. Each proposal "
                "has exactly proposal_id,action_number,predicted_observable_delta,rationale. "
                "proposal_id and selected_proposal_id are JSON strings (for example "
                "\"p0\"); action_number is a JSON integer; "
                "predicted_observable_delta and rationale are JSON strings, not "
                "objects or arrays.\n"
                f"GAME={game}\nSTEP={step_count}\n"
                f"OBSERVATION={summary_for_action[:6000]}\n"
                f"NATIVE_ACTIONS={_format_numbered_actions(step_actions)}"
            )
            try:
                _backbone_plan_result = await vllm_client.generate(
                    _backbone_plan_prompt, adapter=None, temperature=0.0,
                    max_tokens=512,
                )
                _backbone_plan = parse_agent_action_proposal_set(
                    _backbone_plan_result.text,
                    n_native_actions=len(step_actions),
                )
            except Exception as exc:  # fail closed; baseline actor still advances env
                _backbone_plan_error = f"{type(exc).__name__}:{exc}"

        _backbone_context = ""
        if _backbone_plan is not None:
            _backbone_context = (
                "Untrusted planning-Agent candidates (use only if useful; all native "
                "actions remain available):\n"
                + json.dumps(_backbone_plan.to_dict(), ensure_ascii=False)
                + "\n"
            )

        imp_tags = imp["SUBGOAL_TAGS"]
        tags_str = "|".join(imp_tags)
        # Surface critical actions (e.g. B = fire on shooter games) as
        # a single-line in-context prior so the LLM doesn't have to
        # rediscover the action vocabulary via GRPO alone.  Schema
        # source: trainer.coevolution.config.GAME_CRITICAL_ACTIONS.
        _critical_hint = ""
        # Human-authored game advice is excluded from backbone collection.
        _critical_for_game = ()
        if _critical_for_game:
            _critical_hint = (
                f"Critical actions for this game (use frequently when scoring): "
                f"{', '.join(_critical_for_game)}.\n"
            )
        _quality_sort_hint = ""
        # Keep macro actions as native interface, but remove best-first advice.
        if False and _is_macro:
            _quality_sort_hint = (
                "Actions are sorted best-first (fewest holes, most line clears). "
                "Prefer ACTION 1 unless you have a strong reason to pick another.\n"
            )

        def _render_action_prompt(_guidance: Optional[Dict[str, Any]]) -> str:
            _skill_context = ""
            if _guidance and _guidance.get("skill_id"):
                _sk_name = _guidance.get("skill_name", _guidance["skill_id"])
                _sk_hint = _guidance.get("execution_hint", "")
                _skill_context = f"Active skill: {_sk_name}"
                if _sk_hint:
                    _skill_context += f" — {_sk_hint[:100]}"
                _skill_context += "\n"
            _skill_text = _format_skill_guidance_for_prompt(
                _guidance, skill_tracker.protocol_step_idx,
                progress_summary=skill_tracker.get_progress_summary(summary_state),
            )
            _action_user = (
                f"Game state:\n\n{summary_for_action}\n\n"
                f"Subgoal: {assigned_subgoal}\n"
                f"{urgency_line}{_skill_context}{recent_context}{_critical_hint}"
                f"{_quality_sort_hint}{_backbone_context}"
                f"Available actions (pick ONE by number):\n{_format_numbered_actions(step_actions)}\n\n"
                f"Brief REASONING (1 sentence max) then ACTION: <number>."
            )
            return _profile_prefix + SYSTEM_PROMPT + _skill_text + "\n" + _action_user

        action_prompt = _render_action_prompt(guidance)
        # Retain the exact authentic context payload for the pre-existing
        # tamper-evident AGENT_PROPOSAL_SET receipt below.
        skill_text = _format_skill_guidance_for_prompt(
            guidance, skill_tracker.protocol_step_idx,
            progress_summary=skill_tracker.get_progress_summary(summary_state),
        )

        if step_sync is not None:
            await step_sync.arrive()

        _selected_backbone_plan = (
            _backbone_plan.selected()
            if _backbone_plan is not None and _backbone_plan.decision == "EXECUTE"
            else None
        )
        if _selected_backbone_plan is not None:
            # The planning Agent is the policy whose reasoning backbone is
            # being observed. Harness admission already constrained its
            # 1-based action number to the exact native list.
            action_result = _backbone_plan_result
            action = step_actions[_selected_backbone_plan.action_number - 1]
            reasoning = _selected_backbone_plan.rationale
            parsed_intention = None
            _parse_fallback = False
            _parsed_agent_action = str(action)
            _executed_action = str(action)
            _decision_origin = "AGENT"
            _transform_kind = "IDENTITY"
            _action_event_prompt = _backbone_plan_prompt
            # This is source evidence collection, not an action_taking GRPO
            # sample. Avoid training the LoRA on the planner's JSON protocol.
            action_prompt = ""
        else:
            # Invalid/abstained planning never authorizes an action. Fall back
            # to the existing native action Agent and mark the v2 cycle as
            # unusable for reasoning induction.
            _stop = ["\n\nGame state:", "\n\nAvailable actions", "<think", "<thinking"]
            _matched_here = bool(
                matched_policy_skill_id
                and guidance
                and str(guidance.get("skill_id") or "") == matched_policy_skill_id
            )
            if _matched_here:
                _other_candidates = [
                    item for item in last_candidates
                    if str(item.get("skill_id") or "") != matched_policy_skill_id
                ]
                if not _other_candidates:
                    raise RuntimeError("matched_policy_random_control_unavailable")
                _random_key = f"{episode_seed}:{step_count}:{matched_policy_skill_id}"
                _random_idx = int(hashlib.sha256(_random_key.encode()).hexdigest()[:16], 16)
                _random_guidance = _other_candidates[_random_idx % len(_other_candidates)]
                _treatment_specs = (
                    ("B", None, None),
                    ("G_MINUS_S", None, "action_taking"),
                    ("G_PLUS_S", guidance, "action_taking"),
                    ("G_PLUS_RANDOM", _random_guidance, "action_taking"),
                )
                _treatment_prompts = {
                    _name: _render_action_prompt(_guidance)
                    for _name, _guidance, _adapter in _treatment_specs
                }
                # Preserve the live policy's request shape and authority:
                # authentic is generated alone exactly as in an unobserved run.
                # Only after it is frozen do the three shadow calls execute.
                action_result = await vllm_client.generate(
                    _treatment_prompts["G_PLUS_S"], adapter="action_taking",
                    temperature=temperature, max_tokens=128, stop=_stop,
                )
                _shadow_specs = tuple(
                    item for item in _treatment_specs if item[0] != "G_PLUS_S"
                )
                _shadow_results = await asyncio.gather(*[
                    vllm_client.generate(
                        _treatment_prompts[_name], adapter=_adapter,
                        temperature=temperature, max_tokens=128, stop=_stop,
                    )
                    for _name, _guidance, _adapter in _shadow_specs
                ])
                _result_by_treatment = {
                    "G_PLUS_S": action_result,
                    **{
                        _spec[0]: _result
                        for _spec, _result in zip(_shadow_specs, _shadow_results)
                    },
                }
                for _name, _guidance, _adapter in _treatment_specs:
                    _result = _result_by_treatment[_name]
                    _parsed, _why, _subgoal = _parse_action_response(
                        _result.text, step_actions,
                    )
                    _matched_policy_records.append({
                        "schema_version": 1,
                        "episode_id": episode_id,
                        "episode_seed": episode_seed,
                        "step": step_count,
                        "treatment": _name,
                        "sampling_order": "AUTHENTIC_FIRST_SHADOW_AFTER_V1",
                        "requested_adapter": _adapter,
                        "used_adapter": getattr(_result, "adapter", None),
                        "source_skill_id": matched_policy_skill_id,
                        "context_skill_id": (
                            str(_guidance.get("skill_id") or "") if _guidance else None
                        ),
                        "prefix_actions": list(recent_actions),
                        "before_observable_sha256": _event_hash(obs_nl),
                        "native_actions": list(step_actions),
                        "native_actions_sha256": _event_hash(list(step_actions)),
                        "prompt": _treatment_prompts[_name],
                        "prompt_sha256": _event_hash(_treatment_prompts[_name]),
                        "raw_response": str(_result.text),
                        "raw_response_sha256": _event_hash(_result.text),
                        "parsed_action": str(_parsed),
                        "parser_fallback": isinstance(_parsed, _ActionFallback),
                        "reasoning": _why,
                        "parsed_subgoal": _subgoal,
                        "prompt_tokens": int(getattr(_result, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(_result, "completion_tokens", 0) or 0),
                    })
            else:
                action_result = await vllm_client.generate(
                    action_prompt, adapter="action_taking",
                    temperature=temperature, max_tokens=128, stop=_stop,
                )
            action, reasoning, parsed_intention = _parse_action_response(
                action_result.text, step_actions,
            )
            _parse_fallback = isinstance(action, _ActionFallback)
            _parsed_agent_action = str(action)
            action = _apply_anti_repetition(
                action, step_actions, recent_actions, recent_rewards,
                game=game, rng=_policy_rng,
            )
            _executed_action = str(action)
            _action_event_prompt = action_prompt
            if _parse_fallback:
                _decision_origin = "FALLBACK"
                _transform_kind = "PARSER_FALLBACK"
            elif _executed_action != _parsed_agent_action:
                _decision_origin = "POLICY_POSTPROCESSOR"
                _transform_kind = "ANTI_REPETITION_OVERRIDE"
            else:
                _decision_origin = "AGENT"
                _transform_kind = "IDENTITY"
        current_intention = (
            parsed_intention or assigned_subgoal or prev_intention or f"[SETUP] {game}"
        )

        if _reasoning_recorder is not None:
            if reasoning_backbone_harness:
                _reasoning_recorder.append(
                    ReasoningEventKind.AGENT_ACTION_PROPOSAL_SET, {
                        "step": step_count,
                        "schema_valid": _backbone_plan is not None,
                        "parse_error": _backbone_plan_error,
                        "proposal_set": (
                            _backbone_plan.to_dict() if _backbone_plan is not None else None
                        ),
                        "proposal_set_sha256": (
                            _backbone_plan.content_hash()
                            if _backbone_plan is not None else None
                        ),
                        "raw_response": (
                            str(_backbone_plan_result.text)
                            if _backbone_plan_result is not None else ""
                        ),
                        "prompt_sha256": _event_hash(_backbone_plan_prompt),
                        "prompt_tokens": int(getattr(
                            _backbone_plan_result, "prompt_tokens", 0,
                        ) or 0),
                        "completion_tokens": int(getattr(
                            _backbone_plan_result, "completion_tokens", 0,
                        ) or 0),
                        "provider_usage": _reasoning_receipt_value(getattr(
                            _backbone_plan_result, "provider_usage", {},
                        )),
                        "claim_status": "UNTRUSTED_AGENT_CLAIM",
                    },
                )
            _reasoning_recorder.append(ReasoningEventKind.AGENT_PROPOSAL_SET, {
                "step": step_count,
                "selected_skill_id": str(guidance.get("skill_id", "")) if guidance else None,
                "selected_skill_sha256": _event_hash(_reasoning_receipt_value(guidance)) if guidance else None,
                "selected_skill_context_sha256": _event_hash(skill_text) if guidance else None,
                "action_policy_adapter_expected": "action_taking",
                "claim_boundary": "SKILL_CANDIDATES_NOT_ACTION_PROPOSALS",
                "skill_candidates": [{
                    "skill_id": str(item.get("skill_id") or ""),
                    "candidate_sha256": _event_hash(_reasoning_receipt_value(item)),
                } for item in last_candidates],
                "harness_filter_diagnostic": _reasoning_receipt_value(harness_filter_diag),
                "harness_validate_diagnostic": _reasoning_receipt_value(harness_validate_diag),
            })
            _reasoning_recorder.append(ReasoningEventKind.AGENT_RESPONSE, {
                "step": step_count,
                "raw_response": str(action_result.text),
                "raw_response_sha256": _event_hash(action_result.text),
                "prompt_sha256": _event_hash(_action_event_prompt),
                "adapter": getattr(action_result, "adapter", None),
                "prompt_tokens": int(getattr(action_result, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(action_result, "completion_tokens", 0) or 0),
                "provider_usage": _reasoning_receipt_value(
                    getattr(action_result, "provider_usage", {})
                ),
            })
            _reasoning_recorder.append(ReasoningEventKind.PARSED_DECISION, {
                "step": step_count,
                "parsed_agent_action": _parsed_agent_action,
                "parser_fallback": _parse_fallback,
                "reasoning": reasoning,
                "parsed_intention": parsed_intention,
                "agent_protocol_supports_replan_abstain": reasoning_backbone_harness,
            })
            _reasoning_recorder.append(ReasoningEventKind.POLICY_TRANSFORM, {
                "step": step_count,
                "input_action": _parsed_agent_action,
                "output_action": _executed_action,
                "transform_kind": _transform_kind,
                "changed_action": _parsed_agent_action != _executed_action,
            })
            _reasoning_recorder.append(ReasoningEventKind.NATIVE_ADMISSIBILITY, {
                "step": step_count,
                "native_actions": [str(item) for item in step_actions],
                "native_actions_sha256": _event_hash(list(step_actions)),
                "parsed_action_exact_member": _parsed_agent_action in step_actions,
                "executed_action_exact_member": _executed_action in step_actions,
            })
            _reasoning_recorder.append(ReasoningEventKind.AGENT_DECISION, {
                "step": step_count,
                "decision_type": "EXECUTE" if _decision_origin == "AGENT" else "NO_VALID_AGENT_EXECUTION",
                "decision_origin": _decision_origin,
                "parsed_agent_action": _parsed_agent_action,
                "executed_action": _executed_action,
                "can_support_agent_reasoning_induction": _decision_origin == "AGENT",
                "selected_action_matches_reasoning_proposal": bool(
                    _backbone_plan is not None
                    and _backbone_plan.decision == "EXECUTE"
                    and _backbone_plan.selected() is not None
                    and step_actions[_backbone_plan.selected().action_number - 1]
                    == _executed_action
                ) if reasoning_backbone_harness else None,
            })

        # Block A4: stream the per-step intention update.  ``switched``
        # = textual inequality;  ``sharp_shift`` = tag-prefix change OR
        # high urgency (a working definition of §4.1's "sharp shift" we
        # can sharpen post-hoc).  Drives intention-trigger ablation +
        # the §4.1 method-section sharp-shift threshold definition.
        try:
            from trainer.coevolution._run_loggers import (  # noqa: WPS433
                log_intention_switch,
            )
            _prev_tag_m = _TAG_RE.match(prev_intention) if prev_intention else None
            _curr_tag_m = _TAG_RE.match(current_intention) if current_intention else None
            _prev_tag = _prev_tag_m.group(1).upper() if _prev_tag_m else ""
            _curr_tag = _curr_tag_m.group(1).upper() if _curr_tag_m else ""
            _switched = bool(prev_intention) and (current_intention != prev_intention)
            _sharp = (
                bool(_prev_tag)
                and bool(_curr_tag)
                and _prev_tag != _curr_tag
            ) or bool(urgency)
            # Trainer outer step is best-effort: read from the harness
            # hook when wired (orchestrator sets it via for_game).  When
            # there is no hook (e.g. cold-start mode), emit step=-1 and
            # post-hoc joiners can correlate via timestamp.
            _outer = -1
            if harness_hook is not None and hasattr(harness_hook, "_trainer_step"):
                try:
                    _outer = int(getattr(harness_hook, "_trainer_step", -1))
                except Exception:
                    _outer = -1
            log_intention_switch(
                step=_outer,
                episode_id=episode_id,
                game=game,
                inner_step=step_count,
                prev_intention=prev_intention,
                new_intention=current_intention,
                switched=_switched,
                sharp_shift=_sharp,
                summary_state_delta=delta or "",
                urgency=urgency or "",
            )
        except Exception:  # noqa: BLE001
            pass

        # ── 6. env.step() (in executor) ─────────────────────────
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
        raw_env_reward = next_info.get("raw_env_reward", float(reward))
        total_reward += reward
        chain_tracker.observe_step(total_reward)
        next_action_names = next_info.get("action_names", action_names)
        next_structured_state = next_info.get("structured_state")
        if _reasoning_recorder is not None:
            _reasoning_recorder.append(ReasoningEventKind.ENVIRONMENT_STEP, {
                "step": step_count,
                "executed_action": str(action),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            _reasoning_recorder.append(ReasoningEventKind.NATIVE_DELTA, {
                "step": step_count,
                "before_observable_sha256": _event_hash(obs_nl),
                "after_observable_sha256": _event_hash(next_obs_nl),
                "before_native_actions_sha256": _event_hash(list(step_actions)),
                "after_native_actions_sha256": _event_hash(list(next_action_names)),
                "simulator_state_sha256": None,
            })
            _reasoning_recorder.append(ReasoningEventKind.OBSERVATION, {
                "step": step_count + 1,
                "observable_state_sha256": _event_hash(next_obs_nl),
                "observable_state": str(next_obs_nl),
                "structured_state": _reasoning_receipt_value(next_structured_state),
                "simulator_state_sha256": None,
                "simulator_state_available": False,
                "native_actions_sha256": _event_hash(list(next_action_names)),
            })
            if reasoning_backbone_harness:
                _selected_plan = (
                    _backbone_plan.selected()
                    if _backbone_plan is not None and _backbone_plan.decision == "EXECUTE"
                    else None
                )
                _executed_proposal_id = (
                    _selected_plan.proposal_id
                    if _selected_plan is not None
                    and step_actions[_selected_plan.action_number - 1] == str(action)
                    else None
                )
                _verdict_prompt = (
                    "You are an untrusted post-transition verification Agent. Compare "
                    "the predicted observable delta with the real before/after evidence. "
                    "Return exactly one JSON object with keys proposal_id,verdict,decision,"
                    "evidence_claim. verdict is SUPPORTED, REFUTED, or INCONCLUSIVE; "
                    "decision is CONTINUE, REPLAN, or ABSTAIN. proposal_id must equal "
                    "EXPECTED_PROPOSAL_ID, including null. evidence_claim must be a "
                    "JSON string, not an object or array. Do not invent hidden state.\n"
                    f"EXPECTED_PROPOSAL_ID={json.dumps(_executed_proposal_id)}\n"
                    f"PROPOSAL={json.dumps(asdict(_selected_plan) if _selected_plan else None, ensure_ascii=False)}\n"
                    f"BEFORE={summary_for_action[:5000]}\n"
                    f"EXECUTED_ACTION={str(action)}\nREWARD={float(reward)}\n"
                    f"AFTER={str(next_obs_nl)[:5000]}"
                )
                _verdict_result = None
                _verdict = None
                _verdict_error = None
                try:
                    _verdict_result = await vllm_client.generate(
                        _verdict_prompt, adapter=None, temperature=0.0,
                        max_tokens=384,
                    )
                    _verdict = parse_agent_post_transition_verdict(
                        _verdict_result.text,
                        expected_proposal_id=_executed_proposal_id,
                    )
                except Exception as exc:
                    _verdict_error = f"{type(exc).__name__}:{exc}"
                _closed_loop_valid = bool(
                    _executed_proposal_id is not None and _verdict is not None
                )
                _reasoning_recorder.append(
                    ReasoningEventKind.AGENT_POST_TRANSITION_VERDICT, {
                        "step": step_count,
                        "schema_valid": _verdict is not None,
                        "parse_error": _verdict_error,
                        "executed_proposal_id": _executed_proposal_id,
                        "verdict": _verdict.to_dict() if _verdict is not None else None,
                        "verdict_sha256": (
                            _verdict.content_hash() if _verdict is not None else None
                        ),
                        "raw_response": (
                            str(_verdict_result.text) if _verdict_result is not None else ""
                        ),
                        "prompt_sha256": _event_hash(_verdict_prompt),
                        "prompt_tokens": int(getattr(
                            _verdict_result, "prompt_tokens", 0,
                        ) or 0),
                        "completion_tokens": int(getattr(
                            _verdict_result, "completion_tokens", 0,
                        ) or 0),
                        "provider_usage": _reasoning_receipt_value(getattr(
                            _verdict_result, "provider_usage", {},
                        )),
                        "can_support_closed_loop_reasoning_induction": _closed_loop_valid,
                        "claim_status": "UNTRUSTED_AGENT_CLAIM",
                    },
                )
        # Pipeline the 35B vision markup: fire async now, await at the
        # start of the NEXT iteration.  This overlaps 35B generation
        # with all the reward logging, GRPO record assembly, experience
        # dict construction, and loop-transition work below — saving
        # ~3-5s per wave (8 episodes × 69 steps).
        _pending_markup = asyncio.ensure_future(_markup_for(
            obs_nl_v=next_obs_nl, info_v=next_info,
            step_v=step_count + 1,
        ))

        recent_actions.append(str(action))
        recent_rewards.append(float(reward))

        next_facts = extract_game_facts(next_obs_nl, game)
        _ss = next_info.get("structured_state") or {}
        _rw = _ss.get("ram_watch") or {}
        for _rk, _rv in _rw.items():
            if _rk not in next_facts and _rv is not None:
                next_facts[_rk] = str(
                    _rv.item() if hasattr(_rv, "item") else _rv
                )

        # T2.19d: per-step hit / damage delta from the stable-retro RAM
        # watch.  Only fires when (a) the game exposes the key (7/8
        # gymv games for ``lives``; 3/8 for ``health``), and (b) the
        # value decreased vs the previous step.  Positive deltas
        # (1UP pickup, mid-episode health refill) are ignored — we
        # never *reward* the agent for these, only *penalise* the
        # complement.  See ``CoEvolutionConfig.action_hit_penalty``
        # for magnitude calibration.
        _curr_lives = _safe_int_ram(_rw.get("lives"))
        _curr_health = _safe_int_ram(_rw.get("health"))
        _lives_delta = (
            _curr_lives - _prev_lives
            if (_curr_lives is not None and _prev_lives is not None)
            else 0
        )
        _health_delta = (
            _curr_health - _prev_health
            if (_curr_health is not None and _prev_health is not None)
            else 0
        )
        _hit_pen = (
            -float(action_hit_penalty) * abs(_lives_delta)
            if (action_hit_penalty > 0.0 and _lives_delta < 0)
            else 0.0
        )
        _damage_pen = (
            -float(action_damage_penalty) * abs(_health_delta)
            if (action_damage_penalty > 0.0 and _health_delta < 0)
            else 0.0
        )
        skill_tracker.observe_state_effects(
            next_facts, reward=float(reward), action=str(action),
        )

        skill_id = guidance.get("skill_id") if guidance else None
        skill_name_val = guidance.get("skill_name", "") if guidance else ""
        skill_tracker.update(skill_id, skill_name_val, float(reward),
                             state_text=summary_state)

        try:
            from trainer.coevolution._run_loggers import log_step_progress
            _itag_m = _TAG_RE.match(current_intention) if current_intention else None
            log_step_progress(
                episode_id=episode_id, game=game, inner_step=step_count,
                protocol_step_idx=skill_tracker.protocol_step_idx,
                total_steps=skill_tracker.total_protocol_steps,
                intention_tag=_itag_m.group(1).upper() if _itag_m else "",
                source="effect_tags",
                active_skill_id=skill_id or "",
            )
        except Exception:
            pass

        # ── 7. Record GRPO I/O ───────────────────────────────────
        if action_prompt:
            _format_failed = isinstance(action, _ActionFallback)
            try:
                action_num = step_actions.index(action) + 1
            except ValueError:
                action_num = 1
            # T2.19n (v8): REASONING+ACTION restored with quality gate.
            #
            # Three-tier reward:
            #   1. Format failed (no ACTION: N) → -0.05 penalty
            #   2. Format OK but reasoning too short (<50 chars) → 0.0
            #      (prevents "Expert play." degenerate shortcut)
            #   3. Format OK + real reasoning (≥50 chars) →
            #      raw_env_reward + 0.05 format bonus
            #
            # Combined with halved GRPO LR (2.5e-5→1e-5) this keeps
            # format stable while still learning from game reward.
            _FORMAT_BONUS = 0.05
            _FORMAT_FAIL_PENALTY = -0.05
            _MIN_REASONING_LEN = 50
            _has_reasoning = bool(reasoning and len(reasoning.strip()) >= _MIN_REASONING_LEN)
            if _format_failed:
                action_completion = action_result.text.strip()
                _action_reward = _FORMAT_FAIL_PENALTY
            elif not _has_reasoning:
                action_completion = (
                    action_result.text.strip()[:200]
                    or f"ACTION: {action_num}"
                )
                _action_reward = 0.0
            else:
                action_completion = f"REASONING: {(reasoning or 'Best move.')[:80]}\nACTION: {action_num}"
                _critical = _critical_actions_for(game, step_actions)
                _is_critical = (str(action) in _critical) if _critical else False
                _passive_penalty = 0.0
                _action_str = str(action).upper()
                _passive_set = {"NOOP", "STAY", "PASS"}
                if (
                    _action_str in _passive_set
                    and not _is_critical
                    and float(raw_env_reward) <= 0.0
                ):
                    _passive_penalty = -0.05
                _action_reward = (
                    float(reward)
                    + skill_tracker._intrinsic_bonus
                    + _FORMAT_BONUS
                    + _passive_penalty
                )
                try:
                    from trainer.coevolution._run_loggers import (  # noqa: WPS433
                        record_shaping_signal,
                    )
                    record_shaping_signal(
                        game=game,
                        raw_env=float(raw_env_reward),
                        intrinsic=0.0,
                        constant_offset=0.0,
                    )
                except Exception:                                # pragma: no cover
                    pass
            _action_meta = {
                "chosen_action": str(action),
                "available_actions": list(step_actions),
                "summary_state": summary_state,
                "intention": current_intention,
                "assigned_intention": assigned_subgoal,
                "intention_source": "base_model",
                "active_skill": skill_id,
                "intrinsic_bonus": skill_tracker._intrinsic_bonus,
                "raw_env_reward": raw_env_reward,
                "placement_metrics": next_info.get("placement_metrics"),
                "board_stats": next_info.get("board_stats"),
            }
            grpo_records.append(GRPORecord(
                adapter="action_taking", game=game, episode_id=episode_id, step=step_count,
                prompt=action_prompt, completion=action_completion, reward=_action_reward,
                metadata=_action_meta,
            ))
            # T2.4 single-sink: mirror the same scalar + metadata into
            # ``RewardLogger`` so eval and training read from one source.
            # Logger errors are non-fatal — we never let a logging hiccup
            # break a rollout. Metadata is kept *minimal* (chosen_action +
            # raw_env_reward) to avoid bloating the JSONL; full per-step
            # metadata stays on the in-memory ``GRPORecord``.
            if reward_logger is not None:
                try:
                    reward_logger.log_grpo_record(
                        episode_id=episode_id,
                        adapter="action_taking",
                        step=step_count,
                        reward=_action_reward,
                        game=game,
                        metadata={
                            "chosen_action": _action_meta["chosen_action"],
                            "raw_env_reward": _action_meta["raw_env_reward"],
                            "active_skill": _action_meta["active_skill"],
                            "intrinsic_bonus": _action_meta["intrinsic_bonus"],
                            "action_num": action_num,
                            "num_actions": len(step_actions),
                            "format_failed": _format_failed,
                            "raw_completion": action_result.text.strip()[:300],
                        },
                    )
                except Exception:  # pragma: no cover  (defensive)
                    logger.exception("reward_logger.log_grpo_record(action_taking) failed")

        if skill_select_prompt and last_candidates and len(last_candidates) >= 2:
            # Prefer actual LoRA output; fall back to reconstruction
            # only when the LLM call failed.
            if last_sk_lora_text:
                sk_completion = last_sk_lora_text
            else:
                _achieved_set = skill_tracker.achieved_effects
                _effects_str = ", ".join(sorted(_achieved_set)) if _achieved_set else "none"
                _decision_str = "CONTINUE" if not skill_tracker._just_switched else "SWITCH"
                sk_completion = (
                    f"EFFECTS: {_effects_str}\n"
                    f"DECISION: {_decision_str}\n"
                    f"SKILL: {last_chosen_idx + 1}"
                )
            if skill_tracker._just_switched and skill_tracker._prev_steps_on_skill > 0:
                # SWITCH reward: evaluate the previous skill's quality.
                # Uses deterministic-only effects for progress to prevent
                # the LoRA from inflating reward via hallucinated tags.
                from skill_agents.grpo.rewards import skill_selection_reward
                _reason = skill_tracker._reselect_reason
                _progress = skill_tracker._prev_deterministic_ratio
                sk_reward = skill_selection_reward(
                    reward_on_skill=skill_tracker._prev_reward_on_skill,
                    steps_on_skill=skill_tracker._prev_steps_on_skill,
                    max_skill_duration=skill_tracker.max_skill_duration,
                    success_met=_reason.startswith("success:") if _reason else False,
                    abort_triggered=_reason.startswith("abort:") if _reason else False,
                    confidence=0.5,
                    step_progress_ratio=_progress,
                )
            else:
                # CONTINUE reward: was staying on the current skill
                # justified?  Uses deterministic effect progress and
                # env reward — not LoRA-reported effects.
                #
                # May-2026 fix: the old formula clamped reward to [0,1]
                # which destroyed all signal for games like Candy Crush
                # where per-step reward is always positive (0.85–70+).
                # Every skill got ~1.0 → GRPO advantage ≈ 0 → no
                # learning.  Use log1p scaling instead: preserves
                # ordering, compresses heavy tails, and gives zero
                # reward a clearly distinct value from positive reward.
                import math as _math_mod
                _has_progress = float(
                    skill_tracker._new_effects_this_step or reward > 0
                )
                _log_r = _math_mod.log1p(max(0.0, float(reward)))
                sk_reward = (
                    0.2
                    + 0.3 * _has_progress
                    + 0.5 * min(3.0, _log_r)
                )
            # Penalty for unparseable LoRA output so the adapter learns
            # to emit the SFT-canonical ``EFFECTS/DECISION/SKILL:N``
            # format.  Without this term, the legacy 4-layer silent
            # fallback in ``parse_skill_selection`` defaulted to
            # candidate 0 and the GRPO reward couldn't distinguish
            # "LoRA chose intelligently" from "LoRA produced garbage,
            # parser fell back".  Penalties are intentionally small
            # (the dominant signal is still progress / env reward) but
            # consistent so the adapter feels the pull toward
            # parseable output.
            if sk_parse_path == "fallback_zero" or sk_parse_path == "empty_reply":
                sk_reward = sk_reward * 0.5 - 0.15
            elif sk_parse_path in ("tail_number", "name_substring"):
                sk_reward = sk_reward * 0.8 - 0.05

            # Harness-override penalty: when the LoRA's pick is
            # vetoed by ``_harness_validate`` and we fall through to
            # a different candidate (or to ``guidance=None``), the
            # LoRA's selection was effectively wrong from a harness-
            # eligibility standpoint.  We add a small negative term
            # so the adapter learns the harness's per-domain rules
            # (e.g. "Crafter rejects skills tagged for cross-domain
            # board games").  Penalty magnitude matches the
            # heuristic-recovery tier so it never dominates the
            # progress / env-reward signal but is consistently felt.
            if harness_override:
                sk_reward = sk_reward - 0.05

            # Exploration bonus: break SFT positional bias toward SKILL: 1
            _expl_bonus = exploration_bonus(
                chosen_idx=last_chosen_idx,
                env_reward=float(reward),
                n_candidates=len(last_candidates),
            )
            sk_reward += _expl_bonus

            # Anti-collapse: penalise consecutive same-position picks
            collapse_tracker.record(last_chosen_idx)
            _collapse_pen = collapse_tracker.penalty()
            sk_reward += _collapse_pen

            # Skill-ID diversity: penalise monopoly by a single skill
            _chosen_sid_for_div = (
                last_candidates[last_chosen_idx].get("skill_id", "")
                if last_chosen_idx < len(last_candidates) else ""
            )
            _diversity_bonus = diversity_tracker.record_and_shape(_chosen_sid_for_div)
            sk_reward += _diversity_bonus

            # Premature switch penalty: don't abandon a skill too early.
            # Use _prev_deterministic_ratio (saved before set_protocol
            # reset) so we measure the OLD skill's completion, not the
            # new skill's (which is always 0 right after reset).
            if skill_tracker._just_switched and skill_tracker._prev_steps_on_skill > 0:
                _completion_ratio = skill_tracker._prev_deterministic_ratio
                _premature_pen = premature_switch_penalty(
                    protocol_completion_ratio=_completion_ratio,
                    reselect_reason=skill_tracker._reselect_reason,
                )
                sk_reward += _premature_pen
            else:
                _premature_pen = 0.0

            _sk_meta = {
                "chosen_idx": last_chosen_idx,
                "lora_chosen_idx": lora_chosen_idx,
                "harness_override": harness_override,
                "exploration_bonus": round(_expl_bonus, 4),
                "collapse_penalty": round(_collapse_pen, 4),
                "diversity_bonus": round(_diversity_bonus, 4),
                "premature_switch_penalty": round(_premature_pen, 4),
                "skill_candidates": [c.get("skill_id") for c in last_candidates],
                "chosen_skill_id": (
                    last_candidates[last_chosen_idx].get("skill_id")
                    if last_chosen_idx < len(last_candidates) else None
                ),
                "lora_chosen_skill_id": (
                    last_candidates[lora_chosen_idx].get("skill_id")
                    if lora_chosen_idx >= 0 and lora_chosen_idx < len(last_candidates)
                    else None
                ),
                "rag_source": (
                    last_candidates[last_chosen_idx].get("_rag_source")
                    if 0 <= last_chosen_idx < len(last_candidates) else None
                ),
                "summary_state": summary_state,
                "intention": current_intention,
                "reselect_reason": skill_tracker._reselect_reason,
                "parse_path": sk_parse_path or "no_lora_call",
            }
            grpo_records.append(GRPORecord(
                adapter="skill_selection", game=game, episode_id=episode_id, step=step_count,
                prompt=skill_select_prompt, completion=sk_completion, reward=sk_reward,
                metadata=_sk_meta,
            ))
            chain_tracker.register(
                grpo_idx=len(grpo_records) - 1,
                step=step_count,
                current_score=total_reward,
            )
            # T2.4 single-sink mirror — see the action_taking branch.
            if reward_logger is not None:
                try:
                    reward_logger.log_grpo_record(
                        episode_id=episode_id,
                        adapter="skill_selection",
                        step=step_count,
                        reward=sk_reward,
                        game=game,
                        metadata={
                            "chosen_skill_id": _sk_meta["chosen_skill_id"],
                            "lora_chosen_skill_id": _sk_meta["lora_chosen_skill_id"],
                            "harness_override": _sk_meta["harness_override"],
                            "reselect_reason": _sk_meta["reselect_reason"],
                            "n_candidates": len(_sk_meta["skill_candidates"]),
                            "parse_path": _sk_meta["parse_path"],
                            "rag_source": _sk_meta["rag_source"],
                        },
                    )
                except Exception:  # pragma: no cover  (defensive)
                    logger.exception("reward_logger.log_grpo_record(skill_selection) failed")

            # Cross-game skill transfer logging (PLAN-SKILL-BANK §22 + Phase
            # 1→2 monitoring).  When a skill is committed (i.e. survived the
            # harness validate), we record its provenance so post-hoc
            # analysis can compute (a) cross-game-translated skill usage
            # rate, (b) re-grounding success rate, (c) crafter-v2 skill
            # uptake.  We resolve full skill metadata via skill_bank
            # because ``last_candidates`` only carries the runtime
            # SkillSelectionResult fields, not the ``confidence_tag`` /
            # ``derived_from`` provenance.
            try:
                _chosen_sid = _sk_meta.get("chosen_skill_id")
                if _chosen_sid and skill_bank is not None and hasattr(skill_bank, "get_skill"):
                    _full = skill_bank.get_skill(_chosen_sid)
                    if _full is not None:
                        from trainer.coevolution._run_loggers import (
                            log_transfer_usage,  # noqa: WPS433
                        )
                        log_transfer_usage(
                            step=step_count,
                            episode_id=episode_id,
                            game=game,
                            inner_step=step_count,
                            skill_id=_chosen_sid,
                            skill_name=getattr(_full, "name", "") or "",
                            confidence_tag=getattr(_full, "confidence_tag", "stable") or "stable",
                            derived_from=getattr(_full, "derived_from", None),
                            feasible_tasks=list(getattr(_full, "feasible_tasks", []) or []),
                            verified_tasks=list(getattr(_full, "verified_tasks", []) or []),
                            n_candidates=len(_sk_meta.get("skill_candidates") or []),
                            chosen_idx=int(_sk_meta.get("chosen_idx", 0) or 0),
                            harness_verdict=str(_sk_meta.get("reselect_reason") or ""),
                            raw_env_reward=float(raw_env_reward or 0.0),
                        )
            except Exception:  # pragma: no cover (defensive)
                logger.debug("log_transfer_usage failed", exc_info=True)

        _exp_dict: Dict[str, Any] = {
            "step": step_count,
            "state": obs_nl,
            "action": str(action),
            "reward": float(reward),
            "raw_env_reward": raw_env_reward,
            "next_state": next_obs_nl,
            "done": done,
            "intention": current_intention,
            "summary_state": summary_state,
            "skill_id": skill_id,
        }
        if next_info.get("board_stats"):
            _exp_dict["board_stats"] = next_info["board_stats"]
        if _ep_role:
            _exp_dict["role"] = _ep_role
            _exp_dict["side"] = _ep_side
        # Harness diagnostics — surfaces eligibility filter / validate_invocation
        # output for the Crafter hook (Phase B′) to drain into
        # `SkillRecord.false_binding_patterns` via `RejectedSkillSink`.
        # Empty dict when the harness wasn't enabled this step.
        if harness_filter_diag is not None or harness_validate_diag is not None:
            _exp_dict["harness"] = {
                "filter": harness_filter_diag,
                "validate": harness_validate_diag,
            }
        experiences.append(_exp_dict)

        prev_summary_state = summary_state
        prev_intention = current_intention
        m = _TAG_RE.match(current_intention) if current_intention else None
        if m:
            tag_history.append(m.group(1).upper())
        obs_nl = next_obs_nl
        current_info = next_info
        action_names = next_action_names
        structured_state = next_structured_state
        # T2.19d: only overwrite the prev-state cache when the RAM
        # watch actually exposed the key this step.  Otherwise keep
        # the last known value so a transient miss doesn't reset
        # the delta tracker to "no penalty fires next step either".
        if _curr_lives is not None:
            _prev_lives = _curr_lives
        if _curr_health is not None:
            _prev_health = _curr_health
        step_count += 1

        if done:
            break

        # Early termination: stuck detection
        if (step_count >= min_steps_before_stuck
                and len(recent_rewards) >= stuck_window
                and sum(recent_rewards[-stuck_window:]) <= 0):
            logger.debug("Episode %s stuck at step %d, terminating early", episode_id, step_count)
            break

    # Cancel any in-flight 35B vision task from the pipelined markup.
    if _pending_markup is not None and not _pending_markup.done():
        _pending_markup.cancel()
        try:
            await _pending_markup
        except (asyncio.CancelledError, Exception):
            pass

    if step_sync is not None:
        step_sync.depart()

    try:
        env.close()
    except Exception:
        pass

    for rec in grpo_records:
        rec.episode_length = max(step_count, 1)

    # T2.18 (2026-05-05): early-death reward shaping.
    # ----------------------------------------------------------------
    # Rollout analysis on TF3 showed top-10 episodes mean ≈830 vs
    # bottom-10 mean ≈50 — a long tail of early-death runs (RIGHT-heavy
    # exploration crashing into the cave wall) drags the overall mean
    # below Gemini's (~725).  GRPO sees these as "low advantage but
    # terminal" and corrects them slowly because the symmetric adv_clip
    # bounds the negative-side gradient.  Asymmetric clipping (in
    # grpo_training) fixes the gradient side; this reward shaping fixes
    # the SIGNAL side by directly penalising terminal death before the
    # cap is hit.  Smooth-scaled so dying at step 0 incurs the full
    # ``base`` penalty while dying at ``threshold_steps`` incurs ~0.
    # Truncated episodes (max_steps reached → survived) are NOT
    # penalised regardless of total_reward.
    edc = early_death_config or {}
    if (
        bool(edc.get("enabled", False))
        and terminated and not truncated
        and step_count < int(edc.get("threshold_steps", 40))
        and total_reward < float(edc.get("threshold_reward", 100.0))
    ):
        threshold_steps = int(edc.get("threshold_steps", 40))
        scale = (threshold_steps - step_count) / max(1, threshold_steps)
        penalty = -float(edc.get("base", 2.0)) * float(scale)
        new_total = float(total_reward) + penalty
        logger.debug(
            "early_death_penalty %s/%s steps=%d term=%s trunc=%s "
            "reward=%.3f penalty=%.3f → final=%.3f",
            game, episode_id, step_count, terminated, truncated,
            total_reward, penalty, new_total,
        )
        total_reward = new_total

    # T2.19: episode return redistribution — spread a fraction of the
    # episode score across all action_taking GRPO records so that good
    # positioning/approach actions get credit even when they didn't
    # directly coincide with a score event.
    if episode_return_redistribution_weight > 0.0 and total_reward > 0.0:
        at_records = [r for r in grpo_records if r.adapter == "action_taking"]
        if at_records:
            per_action_bonus = (
                total_reward * episode_return_redistribution_weight
                / len(at_records)
            )
            for rec in at_records:
                rec.reward += per_action_bonus
                if rec.metadata is None:
                    rec.metadata = {}
                rec.metadata["return_redistribution"] = round(per_action_bonus, 4)

    chain_tracker.finalize(grpo_records, current_score=total_reward)

    runtime_effects = {}
    if hasattr(skill_tracker, "snapshot_runtime_effects"):
        try:
            runtime_effects = skill_tracker.snapshot_runtime_effects()
        except Exception:  # noqa: BLE001
            pass

    wall_time = time.monotonic() - t0
    reasoning_event_log: Dict[str, Any] = {}
    if _reasoning_recorder is not None:
        _official_success = None
        _official_success_key = None
        for _key in ("won", "success", "is_success"):
            if _key in (current_info or {}):
                _value = (current_info or {}).get(_key)
                if isinstance(_value, (list, tuple)) and len(_value) == 1:
                    _value = _value[0]
                _official_success = bool(_value)
                _official_success_key = _key
                break
        _reasoning_recorder.append(ReasoningEventKind.OFFICIAL_STOP, {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "steps": int(step_count),
            "total_reward": float(total_reward),
            "official_success_evaluator_available": _official_success_key is not None,
            "official_success": _official_success,
            "official_success_source_key": _official_success_key,
            "native_final_info": _reasoning_receipt_value(current_info),
        })
        reasoning_event_log = _reasoning_recorder.to_dict()
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
        role=_ep_role,
        side=_ep_side,
        role_index=_ep_role_idx,
        runtime_skill_effects=runtime_effects,
        reasoning_event_log=reasoning_event_log,
        matched_policy_records=_matched_policy_records,
    )
