"""Post-update skill enrichment for the co-evolution loop.

Ports key techniques from ``labeling/extract_skillbank_gpt54.py`` into the
online co-evolution pipeline.  These run after Stage 3+4, enriching skills
with protocols, execution hints, expected durations, and outcome tracking —
data that the decision agent consumes during rollouts.

Deterministic enrichments (protocols, hints, durations, sub-episode refs,
protocol_raw from 9B intentions) run CPU-only with no LLM calls.

Optional **35B LLM enrichment** (``llm_enrichment_enabled=True`` in config)
runs between GRPO iterations as an offline batch.  It reuses the
``API_func.ask_model`` → ``VLLM_BASE_URL_MAP`` routing from
``_llm_crafter.py`` but with a fundamentally different purpose:

    Old Crafter (failed)          New 35B Enricher
    ─────────────────────         ─────────────────
    Input:  abstract failure ctx  Input:  real 9B rollout traces
    Output: new skills (struct)   Output: better NL text on existing skills
    Timing: online, per-step      Timing: offline, iteration-between batch
    Touches: effects/contract     Touches: ONLY prompt text fields
    Risk:   high (hallucinate)    Risk:   low (NL only, no structural edits)

Three enrichment passes (each fail-soft, each optional):

1. **Step description grounding** — rewrite ``protocol_raw.steps`` for
   clarity given the skill's archetype context.
2. **Exemplar curation** — from N success traces, select the clearest
   one (by LLM ranking) as the canonical exemplar.
3. **Failure diagnosis** — from failed traces stuck at the same step,
   produce a ``failure_lesson`` string that becomes the prompt's
   "Common mistake:" block.

Cross-refs:
    * ``trainer/coevolution/_llm_crafter.py`` — same LLM call pattern
    * ``trainer/coevolution/config.py`` — ``llm_enrichment_*`` flags
    * ``frontier_data/SKILL_PARADIGM_COMPARISON.md`` — Paradigm C spec
    * ``frontier_data/PLAN_FEW_SHOT_SKILL_BANK.md`` — 35B enricher proposal
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)


# ── Tag constants (mirrors decision_agents.agent_helper.SUBGOAL_TAGS) ──

_SUBGOAL_TAGS = (
    "SETUP", "CLEAR", "MERGE", "ATTACK", "DEFEND",
    "NAVIGATE", "POSITION", "COLLECT", "BUILD", "SURVIVE",
    "OPTIMIZE", "EXPLORE", "EXECUTE",
)
_SUBGOAL_TAG_SET = frozenset(_SUBGOAL_TAGS)

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

_TAG_RE = re.compile(r"\[(\w+)\]")


# ── Tag-specific protocol / hint templates ──
# Ported from extract_skillbank_gpt54.py generate_skill_protocol() and
# _populate_execution_hints().

_TAG_PROTOCOL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "SETUP": {
        "preconditions": ["Board/state allows preparatory placement"],
        "steps": [
            "Assess current layout for setup opportunities",
            "Place elements to prepare for future gains",
            "Verify arrangement is stable",
        ],
        "success_criteria": ["Target arrangement achieved"],
        "abort_criteria": ["Setup impossible given current constraints"],
        "failure_modes": ["Structure broken — anchor dislodged or ordering disrupted"],
    },
    "CLEAR": {
        "preconditions": ["Clearable groups or lines exist"],
        "steps": [
            "Identify best clearing opportunity",
            "Execute clearing move",
            "Assess board state after clear",
        ],
        "success_criteria": ["Target elements cleared"],
        "abort_criteria": ["No clearing moves available"],
        "failure_modes": ["Clearing move creates worse congestion than before"],
    },
    "MERGE": {
        "preconditions": ["Merge-eligible pairs or groups present"],
        "steps": [
            "Locate highest-value merge opportunity",
            "Execute merge sequence",
            "Reposition for next merge",
        ],
        "success_criteria": ["Merge completed, value increased"],
        "abort_criteria": ["No merge opportunities on any legal move"],
        "failure_modes": ["No merge opportunities available on any legal move"],
    },
    "ATTACK": {
        "preconditions": ["Offensive opportunity identified"],
        "steps": [
            "Evaluate target priority",
            "Execute attack action",
            "Confirm damage or progress",
        ],
        "success_criteria": ["Target defeated or objective advanced"],
        "abort_criteria": ["Health critical or target unreachable"],
        "failure_modes": ["Overcommitted to attack while defense deteriorated"],
    },
    "DEFEND": {
        "preconditions": ["Threat detected requiring defensive response"],
        "steps": [
            "Identify primary threat",
            "Take defensive position or action",
            "Hold until threat passes",
        ],
        "success_criteria": ["Threat neutralized, state stabilized"],
        "abort_criteria": ["Defense untenable, must change strategy"],
        "failure_modes": ["Board state deteriorates despite defensive moves"],
    },
    "NAVIGATE": {
        "preconditions": ["Movement toward target is possible"],
        "steps": [
            "Determine path to destination",
            "Move toward target avoiding hazards",
            "Confirm arrival or approach",
        ],
        "success_criteria": ["Reached target location"],
        "abort_criteria": ["Path blocked or environment changed"],
        "failure_modes": ["Stuck in loop or path is blocked"],
    },
    "POSITION": {
        "preconditions": ["Positioning adjustment needed"],
        "steps": [
            "Assess optimal target position",
            "Move elements into alignment",
            "Verify position is stable",
        ],
        "success_criteria": ["Elements in desired positions"],
        "abort_criteria": ["Repositioning would worsen state"],
        "failure_modes": ["Structure broken — anchor tile dislodged or ordering disrupted"],
    },
    "SURVIVE": {
        "preconditions": ["State is critical, survival priority"],
        "steps": [
            "Identify most dangerous constraint",
            "Take action to relieve pressure",
            "Stabilize to avoid game-over",
        ],
        "success_criteria": ["Danger reduced, stable state restored"],
        "abort_criteria": ["Recovery impossible"],
        "failure_modes": ["Board state deteriorates despite defensive moves"],
    },
    "OPTIMIZE": {
        "preconditions": ["Improvement opportunity exists in current layout"],
        "steps": [
            "Analyze current inefficiencies",
            "Make targeted improvement move",
            "Verify improvement achieved",
        ],
        "success_criteria": ["Measurable state improvement"],
        "abort_criteria": ["Optimization would sacrifice critical position"],
        "failure_modes": ["Optimization broke a more important structure"],
    },
    "EXPLORE": {
        "preconditions": ["Unknown territory or options available"],
        "steps": [
            "Choose unexplored direction or option",
            "Investigate and gather information",
            "Update strategy based on findings",
        ],
        "success_criteria": ["New information or area discovered"],
        "abort_criteria": ["Exploration too risky given current state"],
        "failure_modes": ["Exploration consumed resources with no useful discovery"],
    },
    "COLLECT": {
        "preconditions": ["Collectible resources in range"],
        "steps": [
            "Identify nearest valuable collectible",
            "Navigate to collectible",
            "Acquire and confirm collection",
        ],
        "success_criteria": ["Target resource collected"],
        "abort_criteria": ["Collection path too dangerous"],
        "failure_modes": ["Detour to collect cost more than the resource is worth"],
    },
    "BUILD": {
        "preconditions": ["Resources available for construction"],
        "steps": [
            "Select build target",
            "Place or construct elements",
            "Verify build is functional",
        ],
        "success_criteria": ["Construction completed"],
        "abort_criteria": ["Resources insufficient or location blocked"],
        "failure_modes": ["Build placed suboptimally, blocking future moves"],
    },
    "EXECUTE": {
        "preconditions": ["Action opportunity present"],
        "steps": [
            "Evaluate best available action",
            "Execute chosen action",
            "Observe result",
        ],
        "success_criteria": ["Action completed with positive effect"],
        "abort_criteria": ["No productive action available"],
        "failure_modes": ["No progress toward skill objective after several moves"],
    },
}


def _extract_tag_from_skill_id(skill_id: str) -> str:
    """Extract the subgoal tag from a compound skill ID like 'midgame:CLEAR'."""
    if ":" in skill_id:
        tag = skill_id.split(":", 1)[1].upper()
    else:
        tag = skill_id.upper()
    if tag in _SUBGOAL_TAG_SET:
        return tag
    return _TAG_ALIASES.get(tag, "EXECUTE")


def _extract_phase_from_skill_id(skill_id: str) -> str:
    """Extract the phase from a compound skill ID like 'midgame:CLEAR'."""
    if ":" in skill_id:
        return skill_id.split(":", 1)[0]
    return ""


def enrich_skill_protocols(
    agent: Any,
    segment_durations: Optional[Dict[str, List[int]]] = None,
) -> int:
    """Fill empty protocols on skills using tag-based templates.

    Mirrors ``populate_skill_protocols()`` in extract_skillbank_gpt54.py
    but uses deterministic templates instead of LLM calls for speed.

    Returns the number of skills updated.
    """
    from skill_agents.stage3_mvp.schemas import Protocol

    bank = agent.bank
    updated = 0

    for sid in list(bank.skill_ids):
        skill = bank.get_skill(sid)
        if skill is None or getattr(skill, "retired", False):
            continue
        if skill.protocol.steps and getattr(skill.protocol, "source", "template") == "llm":
            continue

        tag = _extract_tag_from_skill_id(sid)
        phase = _extract_phase_from_skill_id(sid)
        template = _TAG_PROTOCOL_TEMPLATES.get(tag, _TAG_PROTOCOL_TEMPLATES["EXECUTE"])

        preconditions = list(template["preconditions"])
        steps = list(template["steps"])
        success_criteria = list(template["success_criteria"])
        abort_criteria = list(template["abort_criteria"])

        contract = skill.contract
        if contract is not None:
            eff_add = getattr(contract, "eff_add", None) or set()
            eff_del = getattr(contract, "eff_del", None) or set()
            if eff_add:
                steps.append(f"Achieve: {', '.join(sorted(eff_add)[:3])}")
                success_criteria = [
                    f"{lit} achieved" for lit in sorted(eff_add)[:2]
                ] + success_criteria[:1]
            if eff_del:
                steps.append(f"Remove: {', '.join(sorted(eff_del)[:3])}")

        if phase:
            preconditions.insert(0, f"Game is in {phase} phase")

        durations = (segment_durations or {}).get(sid, [])
        if durations:
            avg_dur = max(1, sum(durations) // len(durations))
        else:
            avg_dur = 10

        protocol = Protocol(
            preconditions=preconditions,
            steps=steps[:7],
            success_criteria=success_criteria[:3],
            abort_criteria=abort_criteria[:3],
            expected_duration=avg_dur,
        )

        skill.protocol = protocol
        bank.add_or_update_skill(skill)
        updated += 1

    if updated:
        logger.info("Enriched %d skill(s) with tag-based protocols", updated)
    return updated


def enrich_execution_hints(agent: Any) -> int:
    """Generate ExecutionHint for skills that lack one.

    Mirrors ``_populate_execution_hints()`` in extract_skillbank_gpt54.py.
    """
    from skill_agents.stage3_mvp.schemas import ExecutionHint

    bank = agent.bank
    updated = 0

    for sid in list(bank.skill_ids):
        skill = bank.get_skill(sid)
        if skill is None or getattr(skill, "retired", False):
            continue
        if skill.execution_hint is not None:
            continue

        tag = _extract_tag_from_skill_id(sid)
        phase = _extract_phase_from_skill_id(sid)
        template = _TAG_PROTOCOL_TEMPLATES.get(tag, _TAG_PROTOCOL_TEMPLATES["EXECUTE"])

        name = skill.name or sid
        desc = skill.strategic_description or ""
        if not desc and skill.contract:
            eff_add = getattr(skill.contract, "eff_add", None) or set()
            eff_del = getattr(skill.contract, "eff_del", None) or set()
            parts = []
            if eff_add:
                parts.append("causes: " + ", ".join(sorted(eff_add)[:4]))
            if eff_del:
                parts.append("ends: " + ", ".join(sorted(eff_del)[:4]))
            desc = "; ".join(parts) if parts else name

        preconditions = skill.protocol.preconditions[:2] if skill.protocol.preconditions else []
        success_crit = skill.protocol.success_criteria[:2] if skill.protocol.success_criteria else []

        termination_cues = list(success_crit) if success_crit else []
        if not termination_cues and skill.contract:
            eff_add = getattr(skill.contract, "eff_add", None) or set()
            if eff_add:
                termination_cues = [f"{lit} achieved" for lit in sorted(eff_add)[:2]]
        if not termination_cues:
            termination_cues = [f"{name} objective met"]

        failure_modes = [template.get("failure_modes", ["No progress"])[0]]

        n_refs = len(skill.sub_episodes) if skill.sub_episodes else 0
        transition = f"[{tag}] {desc[:80]}" if tag else desc[:80]

        hint = ExecutionHint(
            common_preconditions=preconditions,
            common_target_objects=[],
            state_transition_pattern=transition,
            termination_cues=termination_cues,
            common_failure_modes=failure_modes,
            execution_description=desc[:150],
            n_source_segments=n_refs,
        )

        skill.execution_hint = hint
        bank.add_or_update_skill(skill)
        updated += 1

    if updated:
        logger.info("Generated %d execution hint(s)", updated)
    return updated


def compute_segment_durations(agent: Any) -> Dict[str, List[int]]:
    """Compute per-skill segment durations from accumulated segments."""
    durations: Dict[str, List[int]] = {}
    for seg in getattr(agent, "_all_segments", []):
        sid = getattr(seg, "skill_label", None)
        if not sid or sid in ("__NEW__", "NEW"):
            continue
        t_start = getattr(seg, "t_start", 0)
        t_end = getattr(seg, "t_end", 0)
        dur = max(1, t_end - t_start)
        durations.setdefault(sid, []).append(dur)
    return durations


def update_expected_durations(
    agent: Any,
    segment_durations: Dict[str, List[int]],
) -> int:
    """Update skill protocol expected_duration from actual segment data."""
    bank = agent.bank
    updated = 0
    for sid, durs in segment_durations.items():
        if not durs:
            continue
        skill = bank.get_skill(sid)
        if skill is None:
            continue
        avg = max(1, sum(durs) // len(durs))
        if skill.protocol.expected_duration != avg:
            skill.protocol.expected_duration = avg
            bank.add_or_update_skill(skill)
            updated += 1
    if updated:
        logger.info("Updated expected_duration for %d skill(s)", updated)
    return updated


def link_sub_episode_outcomes(
    agent: Any,
    episodes: list,
) -> int:
    """Create SubEpisodeRef entries and track success/failure outcomes.

    Mirrors ``_link_sub_episodes_to_skills()`` in extract_skillbank_gpt54.py.
    """
    from skill_agents.stage3_mvp.schemas import SubEpisodeRef

    bank = agent.bank
    linked = 0

    for seg in getattr(agent, "_all_segments", []):
        sid = getattr(seg, "skill_label", None)
        if not sid or sid in ("__NEW__", "NEW"):
            continue
        skill = bank.get_skill(sid)
        if skill is None:
            continue

        t_start = getattr(seg, "t_start", 0)
        t_end = getattr(seg, "t_end", 0)
        traj_id = getattr(seg, "traj_id", "")

        cum_reward = 0.0
        n_steps = max(1, t_end - t_start)
        obs_by_traj = getattr(agent, "_observations_by_traj", {})
        if traj_id in obs_by_traj:
            pass

        intent_tags = []
        for ep in episodes:
            exps = getattr(ep, "experiences", [])
            for t in range(t_start, min(t_end, len(exps))):
                exp = exps[t]
                r = getattr(exp, "reward", 0.0) or 0.0
                cum_reward += r
                intent = getattr(exp, "intentions", None) or ""
                m = _TAG_RE.match(str(intent).strip())
                if m:
                    intent_tags.append(m.group(1).upper())
            if cum_reward != 0.0:
                break

        outcome = "success" if cum_reward > 0 else "partial"

        ref = SubEpisodeRef(
            episode_id=getattr(seg, "episode_id", "") or traj_id,
            seg_start=t_start,
            seg_end=t_end,
            rollout_source=traj_id,
            summary=f"{sid}: {n_steps} steps, r={cum_reward:.1f}",
            intention_tags=intent_tags[:10],
            outcome=outcome,
            cumulative_reward=cum_reward,
        )

        if skill.sub_episodes is None:
            skill.sub_episodes = []
        skill.sub_episodes.append(ref)
        skill.n_instances = max(skill.n_instances, len(skill.sub_episodes))
        bank.add_or_update_skill(skill)
        linked += 1

    if linked:
        logger.info("Linked %d sub-episode ref(s) to skills", linked)
    return linked


def _enrich_role_side_stage_tags(
    agent: Any,
    episodes: Optional[list] = None,
) -> int:
    """Augment skill ``tags`` with role / side / stage from episode metadata.

    Scans sub-episode references and the episodes themselves to find
    role, side, and stage labels.  Adds them to ``skill.tags`` using
    canonical prefixes (``role:<name>``, ``side:<name>``, ``stage:<name>``)
    so the skill bank can segment and query skills along these dimensions.

    Only fires when episode metadata actually contains role info
    (i.e. ``unified_role_rollouts=True`` was used during rollout
    collection).  Otherwise this is a no-op.
    """
    bank = agent.bank
    updated = 0

    seg_to_meta: Dict[str, Dict[str, str]] = {}
    if episodes:
        for ep in episodes:
            ep_meta = getattr(ep, "metadata", {}) or {}
            ep_role = ep_meta.get("role", "")
            ep_side = ep_meta.get("side", "")
            if not ep_role:
                continue
            eid = getattr(ep, "episode_id", "")
            for exp in getattr(ep, "experiences", []):
                iface = getattr(exp, "interface", {}) or {}
                idx = getattr(exp, "idx", 0)
                key = f"{eid}:{idx}"
                seg_to_meta[key] = {
                    "role": iface.get("role", ep_role),
                    "side": iface.get("side", ep_side),
                    "stage": iface.get("stage", ""),
                }

    if not seg_to_meta:
        return 0

    for seg in getattr(agent, "_all_segments", []):
        sid = getattr(seg, "skill_label", None)
        if not sid or sid in ("__NEW__", "NEW"):
            continue
        skill = bank.get_skill(sid)
        if skill is None:
            continue

        traj_id = getattr(seg, "traj_id", "")
        t_start = getattr(seg, "t_start", 0)
        t_end = getattr(seg, "t_end", 0)

        roles_seen: Set[str] = set()
        sides_seen: Set[str] = set()
        stages_seen: Set[str] = set()
        for t in range(t_start, t_end):
            key = f"{traj_id}:{t}"
            meta = seg_to_meta.get(key, {})
            if meta.get("role"):
                roles_seen.add(meta["role"])
            if meta.get("side"):
                sides_seen.add(meta["side"])
            if meta.get("stage"):
                stages_seen.add(meta["stage"])

        new_tags: List[str] = []
        existing = set(skill.tags or [])
        for r in sorted(roles_seen):
            tag = f"role:{r}"
            if tag not in existing:
                new_tags.append(tag)
        for s in sorted(sides_seen):
            tag = f"side:{s}"
            if tag not in existing:
                new_tags.append(tag)
        for st in sorted(stages_seen):
            tag = f"stage:{st}"
            if tag not in existing:
                new_tags.append(tag)

        if new_tags:
            skill.tags = list(existing | set(new_tags))
            bank.add_or_update_skill(skill)
            updated += 1

    if updated:
        logger.info(
            "Enriched %d skill(s) with role/side/stage tags", updated,
        )
    return updated


def enrich_protocol_raw(
    agent: Any,
    episodes: list,
) -> int:
    """Extract protocol_raw.steps from the best sub-episode's intentions.

    For each skill, finds the highest-reward segment and packages its
    intention sequence as ``protocol_raw.steps`` — concrete reasoning
    exemplars for Paradigm C prompt rendering.

    Only updates skills that don't already have protocol_raw, or replaces
    when a higher-reward exemplar is found.
    """
    bank = agent.bank
    updated = 0

    best_per_skill: Dict[str, Dict[str, Any]] = {}

    for seg in getattr(agent, "_all_segments", []):
        sid = getattr(seg, "skill_label", None)
        if not sid or sid in ("__NEW__", "NEW"):
            continue
        skill = bank.get_skill(sid)
        if skill is None or getattr(skill, "retired", False):
            continue

        t_start = getattr(seg, "t_start", 0)
        t_end = getattr(seg, "t_end", 0)

        intentions: List[str] = []
        cum_reward = 0.0

        for ep in episodes:
            exps = getattr(ep, "experiences", [])
            if t_start >= len(exps):
                continue
            for t in range(t_start, min(t_end, len(exps))):
                exp = exps[t]
                r = getattr(exp, "reward", 0.0) or 0.0
                cum_reward += r
                intent = getattr(exp, "intentions", None) or ""
                intent_str = str(intent).strip()
                if intent_str:
                    tag_m = _TAG_RE.match(intent_str)
                    if tag_m:
                        after_tag = intent_str[tag_m.end():].strip()
                        if after_tag:
                            intent_str = after_tag
                    intentions.append(intent_str)
            if intentions:
                break

        if not intentions:
            continue

        prev = best_per_skill.get(sid)
        if prev is None or cum_reward > prev["reward"]:
            best_per_skill[sid] = {
                "steps": intentions[:7],
                "reward": cum_reward,
                "source": "self_rollout",
            }

    for sid, raw_data in best_per_skill.items():
        skill = bank.get_skill(sid)
        if skill is None:
            continue

        existing_raw = getattr(skill, "protocol_raw", None)
        if existing_raw and isinstance(existing_raw, dict):
            existing_reward = existing_raw.get("reward", -float("inf"))
            if raw_data["reward"] <= existing_reward:
                continue

        skill.protocol_raw = {
            "steps": raw_data["steps"],
            "source": raw_data["source"],
            "reward": raw_data["reward"],
        }
        bank.add_or_update_skill(skill)
        updated += 1

    if updated:
        logger.info(
            "Enriched %d skill(s) with protocol_raw exemplars (Paradigm C)",
            updated,
        )
    return updated


# ── 35B LLM enrichment (offline batch, fail-soft) ─────────────────────
#
# Reuses ``_llm_crafter.py``'s async+thread pattern for fail-soft LLM
# calls via ``API_func.ask_model``.  Each call is capped by a hard
# timeout; failures degrade to no-op so a flaky 35B never blocks the
# training loop.

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_LLM_ENRICH_MAX_TOKENS: int = 1024
_LLM_ENRICH_TEMPERATURE: float = 0.2
_LLM_ENRICH_TIMEOUT_S: float = 45.0
_LLM_ENRICH_MAX_SKILLS: int = 50


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from a potentially noisy LLM response."""
    if not raw:
        return None
    text = raw.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if m is not None:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _build_step_grounding_prompt(
    skill_name: str,
    archetype: str,
    raw_steps: List[str],
    strategic_desc: str,
) -> str:
    """Prompt for rewriting protocol_raw.steps with grounded NL."""
    steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(raw_steps[:7]))
    return (
        "You are an expert analyst reviewing an AI agent's reasoning "
        "traces.  The agent executed a skill and produced the reasoning "
        "steps below.  Rewrite the steps to be clearer and more specific "
        "while preserving the original meaning and order.\n\n"
        f"Skill: {skill_name}\n"
        f"Archetype: {archetype}\n"
        f"Description: {strategic_desc[:300]}\n\n"
        f"Original steps:\n{steps_text}\n\n"
        "Respond with EXACTLY one JSON object (no fences, no preamble):\n"
        "{\n"
        '  "steps": ["<rewritten step 1>", "<rewritten step 2>", ...],\n'
        '  "summary": "<one-sentence summary of the procedure>"\n'
        "}\n\n"
        "Constraints:\n"
        "- Keep the same number of steps (or fewer if some are redundant)\n"
        "- Each step should be a concrete, actionable instruction\n"
        "- Use domain-specific vocabulary from the description\n"
        "- Do NOT invent new steps that weren't in the original\n"
        "- Maximum 7 steps\n"
    )


def _build_exemplar_selection_prompt(
    skill_name: str,
    traces: List[Dict[str, Any]],
) -> str:
    """Prompt for selecting the clearest exemplar from N traces."""
    trace_blocks = []
    for i, t in enumerate(traces[:10]):
        steps = t.get("steps", [])
        reward = t.get("reward", 0.0)
        steps_text = " → ".join(str(s)[:100] for s in steps[:5])
        trace_blocks.append(
            f"  Trace {i+1} (reward={reward:.1f}): {steps_text}"
        )
    traces_text = "\n".join(trace_blocks)
    return (
        "You are selecting the best exemplar trace for an AI agent's "
        "skill prompt.  Given multiple successful reasoning traces for "
        f"the skill '{skill_name}', pick the ONE trace that:\n"
        "1. Has the clearest step-by-step reasoning\n"
        "2. Best represents the skill's procedure\n"
        "3. Would be most helpful as a concrete example\n\n"
        f"Traces:\n{traces_text}\n\n"
        "Respond with EXACTLY one JSON object:\n"
        "{\n"
        '  "selected_index": <1-based index of best trace>,\n'
        '  "reason": "<why this trace is the clearest>"\n'
        "}\n"
    )


def _build_failure_diagnosis_prompt(
    skill_name: str,
    failure_traces: List[Dict[str, Any]],
    stuck_step_index: Optional[int],
) -> str:
    """Prompt for diagnosing failure patterns from 9B traces."""
    trace_blocks = []
    for i, t in enumerate(failure_traces[:8]):
        steps = t.get("steps", [])
        steps_text = " → ".join(str(s)[:100] for s in steps[:5])
        trace_blocks.append(f"  Failure {i+1}: {steps_text}")
    traces_text = "\n".join(trace_blocks)
    stuck_info = (
        f"\nMost failures get stuck at step {stuck_step_index + 1}."
        if stuck_step_index is not None else ""
    )
    return (
        "You are diagnosing why an AI agent repeatedly fails at a "
        f"specific skill ('{skill_name}').  Analyze the failure traces "
        "below and identify the common mistake pattern.\n\n"
        f"Failed traces:{stuck_info}\n{traces_text}\n\n"
        "Respond with EXACTLY one JSON object:\n"
        "{\n"
        '  "failure_pattern": "<one-sentence diagnosis of the common failure>",\n'
        '  "lesson": "<one-sentence advice to avoid this mistake>",\n'
        '  "bottleneck_step": <0-based step index where failures cluster, or null>\n'
        "}\n"
    )


async def _llm_call_one(
    prompt: str,
    *,
    model: str,
    max_tokens: int = _LLM_ENRICH_MAX_TOKENS,
    temperature: float = _LLM_ENRICH_TEMPERATURE,
    timeout_s: float = _LLM_ENRICH_TIMEOUT_S,
    executor: Optional[ThreadPoolExecutor] = None,
) -> Optional[Dict[str, Any]]:
    """Single fail-soft LLM call → parsed JSON dict or None."""
    try:
        from API_func import ask_model
    except ImportError:
        logger.debug("API_func not available; skipping LLM enrichment call")
        return None

    loop = asyncio.get_running_loop()

    def _call() -> str:
        t0 = time.monotonic()
        try:
            return ask_model(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ) or ""
        finally:
            try:
                from trainer.coevolution._run_loggers import record_component_call
                record_component_call(
                    "enricher.llm",
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                )
            except Exception:
                pass

    try:
        raw = await asyncio.wait_for(
            loop.run_in_executor(executor, _call),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.debug("LLM enrichment call timed out after %.0fs", timeout_s)
        return None
    except Exception as exc:
        logger.debug("LLM enrichment call error: %s", exc)
        return None

    return _parse_json_response(raw)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine, handling nested event loops."""
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if not in_loop:
        return asyncio.run(coro)

    import threading
    result_holder: List[Any] = []
    exc_holder: List[BaseException] = []

    def _worker() -> None:
        try:
            result_holder.append(asyncio.run(coro))
        except BaseException as exc:
            exc_holder.append(exc)

    t = threading.Thread(target=_worker, name="llm_enricher_runner")
    t.start()
    t.join()
    if exc_holder:
        raise exc_holder[0]
    return result_holder[0]


async def _enrich_steps_llm_async(
    bank: Any,
    model: str,
    executor: Optional[ThreadPoolExecutor] = None,
) -> int:
    """Rewrite protocol_raw.steps via 35B for clarity (pass 1)."""
    updated = 0
    skills_to_enrich = []

    for sid in list(bank.skill_ids):
        skill = bank.get_skill(sid)
        if skill is None or getattr(skill, "retired", False):
            continue
        proto_raw = getattr(skill, "protocol_raw", None)
        if not isinstance(proto_raw, dict):
            continue
        raw_steps = proto_raw.get("steps", [])
        if not raw_steps or proto_raw.get("llm_grounded"):
            continue
        skills_to_enrich.append(skill)

    if not skills_to_enrich:
        return 0

    for skill in skills_to_enrich[:_LLM_ENRICH_MAX_SKILLS]:
        proto_raw = skill.protocol_raw
        raw_steps = proto_raw.get("steps", [])
        archetype = ""
        if "." in (skill.skill_id or ""):
            parts = skill.skill_id.split(".")
            archetype = parts[-1] if len(parts) > 1 else ""

        prompt = _build_step_grounding_prompt(
            skill_name=skill.name or skill.skill_id,
            archetype=archetype,
            raw_steps=raw_steps,
            strategic_desc=skill.strategic_description or "",
        )

        parsed = await _llm_call_one(prompt, model=model, executor=executor)
        if parsed is None:
            continue

        new_steps = parsed.get("steps")
        if not isinstance(new_steps, list) or not new_steps:
            continue

        new_steps = [str(s).strip() for s in new_steps[:7] if s]
        if not new_steps:
            continue

        skill.protocol_raw = {
            **proto_raw,
            "steps": new_steps,
            "llm_grounded": True,
            "grounded_by": model,
        }
        summary = parsed.get("summary", "")
        if summary and isinstance(summary, str):
            skill.protocol_raw["summary"] = summary[:300]
        bank.add_or_update_skill(skill)
        updated += 1

    if updated:
        logger.info(
            "LLM enricher: grounded %d skill step description(s) via %s",
            updated, model,
        )
    return updated


async def _enrich_exemplar_selection_llm_async(
    bank: Any,
    model: str,
    all_traces: Dict[str, List[Dict[str, Any]]],
    executor: Optional[ThreadPoolExecutor] = None,
) -> int:
    """From N traces per skill, ask 35B to pick the clearest (pass 2)."""
    updated = 0

    for sid, traces in list(all_traces.items())[:_LLM_ENRICH_MAX_SKILLS]:
        if len(traces) < 2:
            continue
        skill = bank.get_skill(sid)
        if skill is None or getattr(skill, "retired", False):
            continue

        prompt = _build_exemplar_selection_prompt(
            skill_name=skill.name or skill.skill_id,
            traces=traces,
        )

        parsed = await _llm_call_one(prompt, model=model, executor=executor)
        if parsed is None:
            continue

        idx = parsed.get("selected_index")
        if not isinstance(idx, (int, float)):
            continue
        idx = int(idx) - 1
        if idx < 0 or idx >= len(traces):
            continue

        best_trace = traces[idx]
        best_steps = best_trace.get("steps", [])
        if not best_steps:
            continue

        proto_raw = getattr(skill, "protocol_raw", {}) or {}
        skill.protocol_raw = {
            **proto_raw,
            "steps": [str(s) for s in best_steps[:7]],
            "source": "llm_curated",
            "reward": best_trace.get("reward", proto_raw.get("reward", 0.0)),
            "curator_reason": str(parsed.get("reason", ""))[:200],
        }
        bank.add_or_update_skill(skill)
        updated += 1

    if updated:
        logger.info(
            "LLM enricher: curated %d exemplar(s) via %s", updated, model,
        )
    return updated


async def _enrich_failure_diagnosis_llm_async(
    bank: Any,
    model: str,
    failure_traces: Dict[str, List[Dict[str, Any]]],
    stuck_steps: Optional[Dict[str, int]] = None,
    executor: Optional[ThreadPoolExecutor] = None,
) -> int:
    """Diagnose failure patterns from 9B traces and store as failure_lesson (pass 3)."""
    updated = 0

    for sid, traces in list(failure_traces.items())[:_LLM_ENRICH_MAX_SKILLS]:
        if len(traces) < 2:
            continue
        skill = bank.get_skill(sid)
        if skill is None or getattr(skill, "retired", False):
            continue

        stuck_idx = (stuck_steps or {}).get(sid)
        prompt = _build_failure_diagnosis_prompt(
            skill_name=skill.name or skill.skill_id,
            failure_traces=traces,
            stuck_step_index=stuck_idx,
        )

        parsed = await _llm_call_one(prompt, model=model, executor=executor)
        if parsed is None:
            continue

        lesson = parsed.get("lesson", "")
        pattern = parsed.get("failure_pattern", "")
        if not lesson and not pattern:
            continue

        proto_raw = getattr(skill, "protocol_raw", {}) or {}
        failure_text = lesson or pattern
        skill.protocol_raw = {
            **proto_raw,
            "failure_lesson": str(failure_text)[:300],
            "failure_pattern": str(pattern)[:300] if pattern else "",
        }
        bneck = parsed.get("bottleneck_step")
        if isinstance(bneck, (int, float)):
            skill.protocol_raw["bottleneck_step"] = int(bneck)
        bank.add_or_update_skill(skill)
        updated += 1

    if updated:
        logger.info(
            "LLM enricher: diagnosed %d failure pattern(s) via %s",
            updated, model,
        )
    return updated


def _collect_traces_per_skill(
    agent: Any,
    episodes: list,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Collect success and failure trace dicts keyed by skill_id.

    Returns ``{"success": {sid: [trace, ...]}, "failure": {sid: [trace, ...]},
    "stuck_steps": {sid: step_index}}``.
    """
    success: Dict[str, List[Dict[str, Any]]] = {}
    failure: Dict[str, List[Dict[str, Any]]] = {}
    stuck_counts: Dict[str, Dict[int, int]] = {}

    for seg in getattr(agent, "_all_segments", []):
        sid = getattr(seg, "skill_label", None)
        if not sid or sid in ("__NEW__", "NEW"):
            continue

        t_start = getattr(seg, "t_start", 0)
        t_end = getattr(seg, "t_end", 0)
        intentions: List[str] = []
        cum_reward = 0.0

        for ep in episodes:
            exps = getattr(ep, "experiences", [])
            if t_start >= len(exps):
                continue
            for t in range(t_start, min(t_end, len(exps))):
                exp = exps[t]
                r = getattr(exp, "reward", 0.0) or 0.0
                cum_reward += r
                intent = getattr(exp, "intentions", None) or ""
                intent_str = str(intent).strip()
                if intent_str:
                    tag_m = _TAG_RE.match(intent_str)
                    if tag_m:
                        after = intent_str[tag_m.end():].strip()
                        if after:
                            intent_str = after
                    intentions.append(intent_str)
            if intentions:
                break

        if not intentions:
            continue

        trace = {
            "steps": intentions[:7],
            "reward": cum_reward,
            "n_steps": max(1, t_end - t_start),
        }

        if cum_reward > 0:
            success.setdefault(sid, []).append(trace)
        else:
            failure.setdefault(sid, []).append(trace)
            last_step = len(intentions) - 1
            stuck_counts.setdefault(sid, {})
            stuck_counts[sid][last_step] = stuck_counts[sid].get(last_step, 0) + 1

    stuck_steps: Dict[str, int] = {}
    for sid, counts in stuck_counts.items():
        if counts:
            stuck_steps[sid] = max(counts, key=counts.get)  # type: ignore[arg-type]

    return {
        "success": success,
        "failure": failure,
        "stuck_steps": stuck_steps,
    }


def run_llm_enrichment(
    agent: Any,
    episodes: list,
    *,
    model: str = "",
    enable_step_grounding: bool = True,
    enable_exemplar_curation: bool = True,
    enable_failure_diagnosis: bool = True,
) -> Dict[str, int]:
    """Run 35B LLM enrichment passes on the skill bank (offline batch).

    Designed to run BETWEEN GRPO iterations, not during rollouts.
    All calls are fail-soft: LLM errors degrade to no-op.

    Parameters
    ----------
    agent
        The skill bank agent with a ``.bank`` attribute.
    episodes
        Recent rollout episodes (same as passed to ``enrich_bank_after_update``).
    model
        Model identifier for ``API_func.ask_model``.  Empty string defers
        to ``BACKBONE_JUDGE_MODEL`` via ``VLLM_BASE_URL_MAP``.
    enable_step_grounding
        Pass 1: rewrite protocol_raw.steps for clarity.
    enable_exemplar_curation
        Pass 2: from N traces, pick the clearest as exemplar.
    enable_failure_diagnosis
        Pass 3: diagnose failure patterns → failure_lesson.

    Returns
    -------
    Dict mapping enrichment type → count of skills updated.
    """
    results: Dict[str, int] = {}
    t0 = time.monotonic()

    bank = agent.bank
    trace_data = _collect_traces_per_skill(agent, episodes)

    async def _run_all() -> None:
        executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm_enrich")
        try:
            if enable_step_grounding:
                results["llm_step_grounding"] = await _enrich_steps_llm_async(
                    bank, model=model, executor=executor,
                )
            if enable_exemplar_curation:
                results["llm_exemplar_curation"] = await _enrich_exemplar_selection_llm_async(
                    bank, model=model,
                    all_traces=trace_data["success"],
                    executor=executor,
                )
            if enable_failure_diagnosis:
                results["llm_failure_diagnosis"] = await _enrich_failure_diagnosis_llm_async(
                    bank, model=model,
                    failure_traces=trace_data["failure"],
                    stuck_steps=trace_data.get("stuck_steps"),
                    executor=executor,
                )
        finally:
            executor.shutdown(wait=False)

    try:
        _run_async(_run_all())
    except Exception as exc:
        logger.warning("LLM enrichment batch failed (non-fatal): %s", exc)
        return results

    elapsed = time.monotonic() - t0
    total = sum(results.values())
    if total:
        logger.info(
            "LLM enrichment complete: %d update(s) in %.1fs "
            "(grounding=%d, curation=%d, diagnosis=%d)",
            total, elapsed,
            results.get("llm_step_grounding", 0),
            results.get("llm_exemplar_curation", 0),
            results.get("llm_failure_diagnosis", 0),
        )
    return results


def enrich_bank_after_update(
    agent: Any,
    episodes: Optional[list] = None,
    *,
    llm_enrichment_enabled: bool = False,
    llm_enrichment_model: str = "",
    llm_enrichment_step_grounding: bool = True,
    llm_enrichment_exemplar_curation: bool = True,
    llm_enrichment_failure_diagnosis: bool = True,
) -> Dict[str, int]:
    """Run all enrichment steps after a bank update.

    Call this after Stage 3+4 in the co-evolution pipeline.

    Deterministic enrichments (protocols, hints, durations, refs,
    protocol_raw) always run.  35B LLM enrichment runs only when
    ``llm_enrichment_enabled=True`` — this is the offline batch pass
    proposed in PLAN_FEW_SHOT_SKILL_BANK.md.

    Returns a dict of counts for each enrichment type.
    """
    results: Dict[str, int] = {}

    durations = compute_segment_durations(agent)
    results["protocols"] = enrich_skill_protocols(agent, segment_durations=durations)
    results["execution_hints"] = enrich_execution_hints(agent)
    results["durations_updated"] = update_expected_durations(agent, durations)

    if episodes:
        results["sub_episode_refs"] = link_sub_episode_outcomes(agent, episodes)
        results["role_side_stage_tags"] = _enrich_role_side_stage_tags(
            agent, episodes,
        )
        results["protocol_raw"] = enrich_protocol_raw(agent, episodes)

        if llm_enrichment_enabled:
            llm_results = run_llm_enrichment(
                agent, episodes,
                model=llm_enrichment_model,
                enable_step_grounding=llm_enrichment_step_grounding,
                enable_exemplar_curation=llm_enrichment_exemplar_curation,
                enable_failure_diagnosis=llm_enrichment_failure_diagnosis,
            )
            results.update(llm_results)

    total = sum(results.values())
    if total:
        logger.info(
            "Skill enrichment: %d protocol(s), %d hint(s), %d duration(s), "
            "%d ref(s), %d role/side/stage tag(s), %d protocol_raw(s)"
            + (", %d llm update(s)" if llm_enrichment_enabled else ""),
            results.get("protocols", 0),
            results.get("execution_hints", 0),
            results.get("durations_updated", 0),
            results.get("sub_episode_refs", 0),
            results.get("role_side_stage_tags", 0),
            results.get("protocol_raw", 0),
            *([sum(v for k, v in results.items() if k.startswith("llm_"))]
              if llm_enrichment_enabled else []),
        )
    return results
