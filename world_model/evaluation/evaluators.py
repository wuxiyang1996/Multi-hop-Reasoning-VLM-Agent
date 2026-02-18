"""
LLM-as-a-Judge evaluators for synthetic experience quality.

Five dimension evaluators + one holistic synthesis pass.  Each evaluator
builds a structured prompt from the trajectory, its synthesis plan, and
optional context (reference experiences, environment description), then
calls the LLM to produce a JSON judgement.

All quality reasoning is delegated to the LLM — no hardcoded heuristic
thresholds.  Follows the same ``ask_model`` pattern used elsewhere in
the codebase (``API_func.ask_model``).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from world_model.evaluation.config import LLMJudgeConfig
from world_model.evaluation.schemas import (
    DimensionScore,
    ExperienceRecord,
    QualityDimension,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# LLM call helpers
# ═════════════════════════════════════════════════════════════════════


def _get_ask_model(config: LLMJudgeConfig) -> Callable:
    """Return the LLM call function, preferring user-supplied, then API_func."""
    if config.ask_model_fn is not None:
        return config.ask_model_fn
    from API_func import ask_model
    return ask_model


def _call_llm(prompt: str, config: LLMJudgeConfig) -> str:
    ask = _get_ask_model(config)
    kwargs: dict = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.model is not None:
        kwargs["model"] = config.model
    return ask(prompt, **kwargs)


def _parse_json_from_response(response: str) -> Optional[dict]:
    """Extract JSON object from LLM response (handles markdown fences)."""
    text = re.sub(r"```(?:json)?\s*", "", response)
    text = text.strip().rstrip("`")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidate = match.group(0)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


def _parse_dimension_response(
    response: str,
    dimension: QualityDimension,
) -> DimensionScore:
    """Parse LLM JSON response into a DimensionScore."""
    parsed = _parse_json_from_response(response)
    if parsed is None:
        logger.warning("Failed to parse LLM response for %s", dimension.value)
        return DimensionScore(
            dimension=dimension,
            score=0.0,
            evidence=["LLM response could not be parsed"],
            details={"raw_response": response[:500]},
        )

    raw_score = parsed.get("score", 5)
    if isinstance(raw_score, (int, float)):
        score = max(0.0, min(float(raw_score) / 10.0, 1.0))
    else:
        score = 0.5

    evidence = parsed.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]

    return DimensionScore(
        dimension=dimension,
        score=score,
        evidence=evidence,
        details={
            k: v for k, v in parsed.items()
            if k not in ("score", "evidence")
        },
    )


# ═════════════════════════════════════════════════════════════════════
# Trajectory formatting helpers
# ═════════════════════════════════════════════════════════════════════


def _format_trajectory(
    record: ExperienceRecord,
    max_steps: int,
    max_chars: int,
) -> str:
    """Format a trajectory for prompt injection."""
    steps = record.steps[:max_steps]
    lines = [f"Trajectory ID: {record.record_id}  (total steps: {record.n_steps})"]
    if record.env_hint:
        lines.append(f"Environment: {record.env_hint}")
    if record.skill_id:
        lines.append(f"Skill ID: {record.skill_id}")
    lines.append("")
    for i, s in enumerate(steps):
        st = s.state[:max_chars]
        ns = s.next_state[:max_chars]
        act = s.action or "(inferred)"
        rew = f", reward={s.reward}" if s.reward is not None else ""
        lines.append(f"  t={i}: state=[{st}]")
        lines.append(f"        instruction=[{s.instruction}]")
        lines.append(f"        action=[{act}]{rew}")
        lines.append(f"        next_state=[{ns}]")
        if s.state_visual_description:
            lines.append(f"        visual_before=[{s.state_visual_description[:max_chars]}]")
        if s.next_state_visual_description:
            lines.append(f"        visual_after=[{s.next_state_visual_description[:max_chars]}]")
    if record.n_steps > max_steps:
        lines.append(f"  ... ({record.n_steps - max_steps} more steps omitted)")
    return "\n".join(lines)


def _format_plan(record: ExperienceRecord) -> str:
    """Format the synthesis plan / instructions."""
    if record.plan_steps:
        lines = [f"  {i+1}. {step}" for i, step in enumerate(record.plan_steps)]
        return "\n".join(lines)
    if record.synthesis_instructions:
        return record.synthesis_instructions
    return "(no synthesis plan available)"


def _format_reference_set(
    reference_records: List[ExperienceRecord],
    max_records: int,
    max_chars: int,
) -> str:
    """Summarise a reference set of existing trajectories for diversity eval."""
    if not reference_records:
        return "(no reference trajectories provided)"
    sample = reference_records[:max_records]
    lines = []
    for rec in sample:
        first_state = rec.steps[0].state[:max_chars] if rec.steps else "?"
        last_state = rec.steps[-1].next_state[:max_chars] if rec.steps else "?"
        instrs = rec.synthesis_instructions[:max_chars] if rec.synthesis_instructions else "?"
        lines.append(
            f"  {rec.record_id}: {rec.n_steps} steps, "
            f"start=[{first_state}], end=[{last_state}], plan=[{instrs}]"
        )
    if len(reference_records) > max_records:
        lines.append(f"  ... ({len(reference_records) - max_records} more omitted)")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Shared JSON response schema
# ═════════════════════════════════════════════════════════════════════

_RESPONSE_SCHEMA = """\
Return ONLY a JSON object (no extra text) with this structure:
{
  "score": <integer 1-10, where 10 is best>,
  "evidence": ["reason 1", "reason 2", ...],
  "issues": ["issue 1", ...],
  "verdict": "ACCEPT" | "REFINE" | "REGENERATE" | "DISCARD"
}"""


# ═════════════════════════════════════════════════════════════════════
# 1. FIDELITY
# ═════════════════════════════════════════════════════════════════════

_FIDELITY_PROMPT = """\
You are an expert evaluator for synthetic game experiences generated by a world model.

Your task: evaluate the **FIDELITY** of the following synthetic trajectory — are the predicted states physically / logically plausible given the environment dynamics?

=== Synthesis Plan ===
{plan}

=== Synthetic Trajectory ===
{trajectory}

Evaluate fidelity by reasoning about:
1. Does each predicted next_state follow logically from the current state and action?  Could this transition actually happen in the environment?
2. Are environmental constraints respected (e.g., an agent cannot walk through walls, a pot cannot hold more items than its capacity, an eliminated player cannot take actions)?
3. Are quantities, positions, and resource counts updated correctly?
4. Would a domain expert looking at this trajectory consider it a plausible episode in the given environment?

A score of 10 means every transition is physically / logically plausible.
A score of 1 means most transitions violate basic environment rules.

{response_schema}"""


def evaluate_fidelity(
    record: ExperienceRecord,
    config: LLMJudgeConfig,
) -> DimensionScore:
    """LLM-judged fidelity: are state transitions physically plausible?"""
    prompt = _FIDELITY_PROMPT.format(
        plan=_format_plan(record),
        trajectory=_format_trajectory(
            record, config.max_steps_in_prompt, config.max_state_chars,
        ),
        response_schema=_RESPONSE_SCHEMA,
    )
    response = _call_llm(prompt, config)
    return _parse_dimension_response(response, QualityDimension.FIDELITY)


# ═════════════════════════════════════════════════════════════════════
# 2. CONSISTENCY
# ═════════════════════════════════════════════════════════════════════

_CONSISTENCY_PROMPT = """\
You are an expert evaluator for synthetic game experiences generated by a world model.

Your task: evaluate the **CONSISTENCY** of the following synthetic trajectory — are consecutive state transitions internally coherent (no contradictions, no teleportation, no information loss)?

=== Synthetic Trajectory ===
{trajectory}

Evaluate consistency by reasoning about:
1. Does the next_state at step t match the state at step t+1?  Any contradictions (e.g., an item appearing then disappearing without explanation)?
2. Is there temporal continuity — do events unfold in a logical order without jumps or resets?
3. Are entities (agents, objects, locations) tracked consistently throughout the trajectory?
4. Are there any "hallucinated" state changes that appear without a corresponding action?
5. If rewards are present, are they consistent with the state changes?

A score of 10 means the trajectory is perfectly self-consistent.
A score of 1 means the trajectory is riddled with internal contradictions.

{response_schema}"""


def evaluate_consistency(
    record: ExperienceRecord,
    config: LLMJudgeConfig,
) -> DimensionScore:
    """LLM-judged consistency: are consecutive transitions internally coherent?"""
    prompt = _CONSISTENCY_PROMPT.format(
        trajectory=_format_trajectory(
            record, config.max_steps_in_prompt, config.max_state_chars,
        ),
        response_schema=_RESPONSE_SCHEMA,
    )
    response = _call_llm(prompt, config)
    return _parse_dimension_response(response, QualityDimension.CONSISTENCY)


# ═════════════════════════════════════════════════════════════════════
# 3. INSTRUCTION ADHERENCE
# ═════════════════════════════════════════════════════════════════════

_INSTRUCTION_ADHERENCE_PROMPT = """\
You are an expert evaluator for synthetic game experiences generated by a world model.

Your task: evaluate the **INSTRUCTION ADHERENCE** of the following synthetic trajectory — does the generated trajectory faithfully follow the synthesis plan / instructions that were provided to the world model?

=== Synthesis Plan ===
{plan}

=== Synthetic Trajectory ===
{trajectory}

Evaluate instruction adherence by reasoning about:
1. For each plan step / instruction, is there a corresponding transition in the trajectory that executes it?
2. Are the instructions followed in the correct order?
3. Are there extraneous transitions not covered by the plan (over-generation)?
4. Are there plan steps that were skipped or only partially executed (under-generation)?
5. Does the overall trajectory achieve the intended goal described in the synthesis instructions?

A score of 10 means every instruction is faithfully and completely executed.
A score of 1 means the trajectory ignores or contradicts the instructions.

{response_schema}"""


def evaluate_instruction_adherence(
    record: ExperienceRecord,
    config: LLMJudgeConfig,
) -> DimensionScore:
    """LLM-judged instruction adherence: does the trajectory follow the plan?"""
    prompt = _INSTRUCTION_ADHERENCE_PROMPT.format(
        plan=_format_plan(record),
        trajectory=_format_trajectory(
            record, config.max_steps_in_prompt, config.max_state_chars,
        ),
        response_schema=_RESPONSE_SCHEMA,
    )
    response = _call_llm(prompt, config)
    return _parse_dimension_response(response, QualityDimension.INSTRUCTION_ADHERENCE)


# ═════════════════════════════════════════════════════════════════════
# 4. DIVERSITY (batch-level, requires reference set)
# ═════════════════════════════════════════════════════════════════════

_DIVERSITY_PROMPT = """\
You are an expert evaluator for synthetic game experiences generated by a world model.

Your task: evaluate the **DIVERSITY** of the following synthetic trajectory relative to a reference set of existing trajectories — does it cover a novel scenario, or is it essentially a duplicate of existing data?

=== Target Trajectory ===
{trajectory}

=== Reference Set (existing trajectories) ===
{reference_set}

Evaluate diversity by reasoning about:
1. Does this trajectory visit states or situations not covered by the reference set?
2. Does it follow a different action sequence / strategy compared to existing trajectories?
3. Is the outcome (final state) meaningfully different from existing endpoints?
4. Would adding this trajectory to a training buffer provide new information, or is it redundant?
5. Does it explore edge cases, failure modes, or uncommon paths?

A score of 10 means the trajectory is highly novel and adds significant coverage.
A score of 1 means it is essentially a duplicate of an existing trajectory.

If no reference set is provided, evaluate diversity based on internal variety (does the trajectory visit diverse states, or does it loop / stagnate?).

{response_schema}"""


def evaluate_diversity(
    record: ExperienceRecord,
    reference_records: Optional[List[ExperienceRecord]],
    config: LLMJudgeConfig,
) -> DimensionScore:
    """LLM-judged diversity: does this trajectory add novel coverage?"""
    prompt = _DIVERSITY_PROMPT.format(
        trajectory=_format_trajectory(
            record, config.max_steps_in_prompt, config.max_state_chars,
        ),
        reference_set=_format_reference_set(
            reference_records or [], max_records=8, max_chars=config.max_state_chars,
        ),
        response_schema=_RESPONSE_SCHEMA,
    )
    response = _call_llm(prompt, config)
    return _parse_dimension_response(response, QualityDimension.DIVERSITY)


# ═════════════════════════════════════════════════════════════════════
# 5. INFORMATIVENESS
# ═════════════════════════════════════════════════════════════════════

_INFORMATIVENESS_PROMPT = """\
You are an expert evaluator for synthetic game experiences generated by a world model.

Your task: evaluate the **INFORMATIVENESS** of the following synthetic trajectory — does it provide useful learning signal for training a downstream decision-making agent?

=== Synthesis Plan ===
{plan}

=== Synthetic Trajectory ===
{trajectory}

Evaluate informativeness by reasoning about:
1. Does the trajectory contain non-trivial state changes that an agent can learn from?  A trajectory where nothing changes is uninformative.
2. Is there a meaningful reward signal (positive or negative) that correlates with good or bad actions?
3. Does the trajectory demonstrate a clear cause-effect relationship between actions and outcomes?
4. Does it illustrate important decision points, trade-offs, or failure modes?
5. Would an RL or imitation-learning agent benefit from training on this trajectory, or is it too noisy / trivial / degenerate?

A score of 10 means the trajectory is packed with useful learning signal.
A score of 1 means the trajectory is trivial, degenerate, or uninformative.

{response_schema}"""


def evaluate_informativeness(
    record: ExperienceRecord,
    config: LLMJudgeConfig,
) -> DimensionScore:
    """LLM-judged informativeness: does this trajectory provide learning signal?"""
    prompt = _INFORMATIVENESS_PROMPT.format(
        plan=_format_plan(record),
        trajectory=_format_trajectory(
            record, config.max_steps_in_prompt, config.max_state_chars,
        ),
        response_schema=_RESPONSE_SCHEMA,
    )
    response = _call_llm(prompt, config)
    return _parse_dimension_response(response, QualityDimension.INFORMATIVENESS)


# ═════════════════════════════════════════════════════════════════════
# HOLISTIC — final synthesis pass
# ═════════════════════════════════════════════════════════════════════

_HOLISTIC_PROMPT = """\
You are an expert evaluator for synthetic game experiences generated by a world model.

You have already evaluated the trajectory "{record_id}" across five quality dimensions.  Now synthesise these assessments into a **holistic quality judgement** and a final **verdict**.

=== Trajectory Summary ===
{trajectory_summary}

=== Dimension Scores ===
{dimension_scores}

Based on the dimension-level evaluations, provide your overall assessment:

1. What is the overall quality of this synthetic trajectory (1-10)?
2. What is the recommended verdict?
   - ACCEPT: the trajectory is high quality and ready for downstream use (training buffer, planning, etc.)
   - REFINE: the trajectory has potential but specific steps need re-synthesis or correction
   - REGENERATE: the trajectory has fundamental issues and should be regenerated from scratch with the same or modified plan
   - DISCARD: the trajectory is too low quality or harmful to be useful at all
3. What specific issues should be fixed (for REFINE) or what plan modifications would help (for REGENERATE)?

Return ONLY a JSON object:
{{
  "score": <integer 1-10>,
  "evidence": ["holistic reason 1", "holistic reason 2", ...],
  "verdict": "ACCEPT" | "REFINE" | "REGENERATE" | "DISCARD",
  "issues": ["specific issue 1", ...],
  "suggestions": ["improvement suggestion 1", ...],
  "reasoning": "one-paragraph synthesis of overall quality"
}}"""


def evaluate_holistic(
    record: ExperienceRecord,
    dimension_scores: Dict[str, DimensionScore],
    config: LLMJudgeConfig,
) -> dict:
    """Final LLM pass that synthesises dimension scores into overall judgement.

    Returns a dict with keys: score, evidence, verdict, issues, suggestions, reasoning.
    """
    # Trajectory summary (compact)
    summary_lines = [
        f"ID: {record.record_id}, {record.n_steps} steps",
    ]
    if record.env_hint:
        summary_lines.append(f"Env: {record.env_hint}")
    if record.skill_id:
        summary_lines.append(f"Skill: {record.skill_id}")
    if record.synthesis_instructions:
        summary_lines.append(
            f"Plan: {record.synthesis_instructions[:300]}"
        )

    dim_lines = []
    for dim_name, ds in dimension_scores.items():
        dim_lines.append(f"  {dim_name}: {ds.score:.2f}/1.0 ({ds.grade.value})")
        for ev in ds.evidence[:3]:
            dim_lines.append(f"    - {ev}")

    prompt = _HOLISTIC_PROMPT.format(
        record_id=record.record_id,
        trajectory_summary="\n".join(summary_lines),
        dimension_scores="\n".join(dim_lines),
    )
    response = _call_llm(prompt, config)
    parsed = _parse_json_from_response(response)

    if parsed is None:
        logger.warning(
            "Failed to parse holistic LLM response for %s", record.record_id,
        )
        return {
            "score": 5,
            "evidence": ["Holistic LLM response could not be parsed"],
            "verdict": "ACCEPT",
            "issues": [],
            "suggestions": [],
            "reasoning": "",
        }

    return {
        "score": parsed.get("score", 5),
        "evidence": parsed.get("evidence", []),
        "verdict": parsed.get("verdict", "ACCEPT"),
        "issues": parsed.get("issues", []),
        "suggestions": parsed.get("suggestions", []),
        "reasoning": parsed.get("reasoning", ""),
    }
