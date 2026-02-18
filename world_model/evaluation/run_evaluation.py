"""
Synthetic Experience Evaluation — Main Orchestrator.

Runs LLM-as-a-judge on every synthetic trajectory across five quality
dimensions, optionally followed by a holistic synthesis pass.
Assigns per-trajectory verdicts (ACCEPT / REFINE / REGENERATE / DISCARD)
and produces a batch-wide summary.

Integration points:
  - Consumes ``ExperienceRecord`` wrappers around world-model outputs
    (both textual and multi-modal).
  - Produces ``BatchEvaluationSummary`` that can feed back into:
      * World model re-synthesis (regenerate low-quality trajectories).
      * Experience buffer curation (accept/discard filtering).
      * Downstream agents (weight experiences by quality score).

Convenience converters are provided to build ``ExperienceRecord`` objects
from ``TextSyntheticExperienceSequence`` and ``SyntheticExperienceSequence``.

Typical call site::

    from world_model.evaluation import (
        evaluate_experiences,
        records_from_text_sequences,
    )
    records = records_from_text_sequences(sequences, plans)
    summary = evaluate_experiences(records)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from world_model.evaluation.config import ExperienceEvaluationConfig
from world_model.evaluation.schemas import (
    BatchEvaluationSummary,
    DimensionScore,
    ExperienceQualityReport,
    ExperienceRecord,
    ExperienceStep,
    ExperienceVerdict,
    QualityDimension,
    QualityGrade,
)
from world_model.evaluation.evaluators import (
    evaluate_consistency,
    evaluate_diversity,
    evaluate_fidelity,
    evaluate_holistic,
    evaluate_informativeness,
    evaluate_instruction_adherence,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# Converters: world-model outputs -> ExperienceRecord
# ═════════════════════════════════════════════════════════════════════


def records_from_text_sequences(
    sequences: list,
    plans: Optional[list] = None,
    env_hint: Optional[str] = None,
    initial_states: Optional[List[str]] = None,
) -> List[ExperienceRecord]:
    """Convert ``TextSyntheticExperienceSequence`` objects into evaluation records.

    Parameters
    ----------
    sequences : list[TextSyntheticExperienceSequence]
    plans : list[SynthesisPlan], optional
        Corresponding plans (from experience_planning).
    env_hint : str, optional
        Environment name (e.g. "overcooked").
    initial_states : list[str], optional
        Starting state text for each sequence.
    """
    records: List[ExperienceRecord] = []
    for idx, seq in enumerate(sequences):
        steps: List[ExperienceStep] = []
        plan_steps_list: Optional[List[str]] = None

        if plans and idx < len(plans):
            plan = plans[idx]
            if hasattr(plan, "to_instructions"):
                plan_steps_list = plan.to_instructions()

        prev_state = (
            initial_states[idx] if initial_states and idx < len(initial_states)
            else ""
        )
        for step in seq.steps:
            steps.append(ExperienceStep(
                state=prev_state or getattr(step, "prompt", ""),
                next_state=step.next_state,
                instruction=getattr(step, "action", "") or "",
                action=getattr(step, "action", None),
                reward=getattr(step, "reward", None),
                metadata=getattr(step, "metadata", None),
            ))
            prev_state = step.next_state

        records.append(ExperienceRecord(
            record_id=f"text_{idx}",
            steps=steps,
            synthesis_instructions=seq.sequence_instructions,
            plan_steps=plan_steps_list,
            env_hint=env_hint,
        ))
    return records


def records_from_multimodal_sequences(
    sequences: list,
    visual_descriptions: Optional[List[List[Tuple[str, str]]]] = None,
    plans: Optional[list] = None,
    env_hint: Optional[str] = None,
) -> List[ExperienceRecord]:
    """Convert ``SyntheticExperienceSequence`` objects into evaluation records.

    Since multi-modal sequences use images, callers should provide
    ``visual_descriptions`` — a list of (before_desc, after_desc) per step,
    obtained via a VLM captioner or environment renderer.

    Parameters
    ----------
    sequences : list[SyntheticExperienceSequence]
    visual_descriptions : list[list[tuple[str, str]]], optional
        Per-sequence, per-step (before_caption, after_caption).
    plans : list[SynthesisPlan], optional
    env_hint : str, optional
    """
    records: List[ExperienceRecord] = []
    for idx, seq in enumerate(sequences):
        steps: List[ExperienceStep] = []
        plan_steps_list: Optional[List[str]] = None
        if plans and idx < len(plans):
            plan = plans[idx]
            if hasattr(plan, "to_instructions"):
                plan_steps_list = plan.to_instructions()

        vis = (
            visual_descriptions[idx]
            if visual_descriptions and idx < len(visual_descriptions)
            else None
        )

        for si, step in enumerate(seq.steps):
            before_cap = vis[si][0] if vis and si < len(vis) else ""
            after_cap = vis[si][1] if vis and si < len(vis) else ""

            steps.append(ExperienceStep(
                state=before_cap,
                next_state=after_cap,
                instruction=step.edit_prompt,
                state_visual_description=before_cap,
                next_state_visual_description=after_cap,
                metadata=getattr(step, "metadata", None),
            ))

        records.append(ExperienceRecord(
            record_id=f"mm_{idx}",
            steps=steps,
            synthesis_instructions=seq.sequence_instructions,
            plan_steps=plan_steps_list,
            env_hint=env_hint,
        ))
    return records


# ═════════════════════════════════════════════════════════════════════
# Verdict assignment
# ═════════════════════════════════════════════════════════════════════


def _assign_verdict(
    score: float,
    config: ExperienceEvaluationConfig,
) -> ExperienceVerdict:
    """Map overall score to a verdict using config thresholds."""
    if score >= config.accept_threshold:
        return ExperienceVerdict.ACCEPT
    if score >= config.refine_threshold:
        return ExperienceVerdict.REFINE
    if score >= config.discard_threshold:
        return ExperienceVerdict.REGENERATE
    return ExperienceVerdict.DISCARD


# ═════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════


def evaluate_experiences(
    records: List[ExperienceRecord],
    config: Optional[ExperienceEvaluationConfig] = None,
    reference_records: Optional[List[ExperienceRecord]] = None,
    report_path: Optional[str] = None,
) -> BatchEvaluationSummary:
    """Run the full LLM-agentic experience evaluation pipeline.

    Parameters
    ----------
    records : list[ExperienceRecord]
        Synthetic trajectories to evaluate.
    config : ExperienceEvaluationConfig, optional
    reference_records : list[ExperienceRecord], optional
        Existing trajectories for diversity comparison.  If ``None``,
        diversity is evaluated based on internal variety only.
    report_path : str, optional
        Path to write JSON evaluation report.

    Returns
    -------
    BatchEvaluationSummary
        Batch-wide evaluation summary with per-trajectory reports.
    """
    cfg = config or ExperienceEvaluationConfig()
    llm_cfg = cfg.llm
    summary = BatchEvaluationSummary()
    enabled = set(cfg.enabled_dimensions)

    logger.info(
        "Experience Evaluation (LLM-agentic): evaluating %d trajectories "
        "across %d dimensions",
        len(records), len(enabled),
    )

    # Build a growing reference set for diversity (previously evaluated
    # trajectories in this batch serve as references for later ones).
    diversity_refs: List[ExperienceRecord] = list(reference_records or [])

    for record in records:
        report = ExperienceQualityReport(record_id=record.record_id)

        logger.info(
            "  Evaluating trajectory: %s (%d steps)",
            record.record_id, record.n_steps,
        )

        # ── Dimension evaluations (each is an LLM call) ──────────

        if "fidelity" in enabled:
            logger.debug("    [%s] fidelity", record.record_id)
            report.dimensions[QualityDimension.FIDELITY.value] = (
                evaluate_fidelity(record, llm_cfg)
            )

        if "consistency" in enabled:
            logger.debug("    [%s] consistency", record.record_id)
            report.dimensions[QualityDimension.CONSISTENCY.value] = (
                evaluate_consistency(record, llm_cfg)
            )

        if "instruction_adherence" in enabled:
            logger.debug("    [%s] instruction_adherence", record.record_id)
            report.dimensions[QualityDimension.INSTRUCTION_ADHERENCE.value] = (
                evaluate_instruction_adherence(record, llm_cfg)
            )

        if "diversity" in enabled:
            logger.debug("    [%s] diversity", record.record_id)
            report.dimensions[QualityDimension.DIVERSITY.value] = (
                evaluate_diversity(record, diversity_refs or None, llm_cfg)
            )

        if "informativeness" in enabled:
            logger.debug("    [%s] informativeness", record.record_id)
            report.dimensions[QualityDimension.INFORMATIVENESS.value] = (
                evaluate_informativeness(record, llm_cfg)
            )

        # ── Weighted overall ─────────────────────────────────────
        report.compute_overall(cfg.dimension_weights)

        # ── Threshold-based verdict ──────────────────────────────
        report.verdict = _assign_verdict(report.overall_score, cfg)

        # ── Holistic synthesis pass (optional) ───────────────────
        if cfg.run_holistic_pass:
            logger.debug("    [%s] holistic synthesis", record.record_id)
            holistic = evaluate_holistic(record, report.dimensions, llm_cfg)
            _apply_holistic(report, holistic, cfg)

        summary.reports[record.record_id] = report

        # Add to diversity reference set for subsequent records
        diversity_refs.append(record)

        logger.info(
            "  %s: %.3f (%s) -> %s",
            record.record_id,
            report.overall_score,
            report.overall_grade.value,
            report.verdict.value,
        )

    # ── Batch-level summary ──────────────────────────────────────
    summary.compute_summary()

    path = report_path or cfg.report_path
    if path:
        _save_report(summary, path)

    logger.info(
        "Experience Evaluation complete: %d trajectories, mean=%.3f, "
        "acceptance_rate=%.1f%%, %d excellent, %d good, %d fair, %d poor, "
        "%d failing",
        len(records), summary.mean_overall,
        summary.acceptance_rate() * 100,
        summary.n_excellent, summary.n_good,
        summary.n_fair, summary.n_poor, summary.n_failing,
    )

    return summary


# ═════════════════════════════════════════════════════════════════════
# Convenience: evaluate a single trajectory
# ═════════════════════════════════════════════════════════════════════


def evaluate_single(
    record: ExperienceRecord,
    config: Optional[ExperienceEvaluationConfig] = None,
    reference_records: Optional[List[ExperienceRecord]] = None,
) -> ExperienceQualityReport:
    """Evaluate a single trajectory and return its quality report.

    Thin wrapper around ``evaluate_experiences`` for single-trajectory use.
    """
    summary = evaluate_experiences(
        [record], config=config, reference_records=reference_records,
    )
    return summary.reports[record.record_id]


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════


def _apply_holistic(
    report: ExperienceQualityReport,
    holistic: dict,
    config: ExperienceEvaluationConfig,
) -> None:
    """Apply holistic LLM synthesis to override threshold-based verdict."""
    raw_score = holistic.get("score", 5)
    if isinstance(raw_score, (int, float)):
        report.overall_score = max(0.0, min(float(raw_score) / 10.0, 1.0))
    report.overall_grade = QualityGrade.from_score(report.overall_score)

    verdict_str = holistic.get("verdict", "").upper()
    verdict_map = {
        "ACCEPT": ExperienceVerdict.ACCEPT,
        "REFINE": ExperienceVerdict.REFINE,
        "REGENERATE": ExperienceVerdict.REGENERATE,
        "DISCARD": ExperienceVerdict.DISCARD,
    }
    if verdict_str in verdict_map:
        report.verdict = verdict_map[verdict_str]
    else:
        report.verdict = _assign_verdict(report.overall_score, config)

    issues = holistic.get("issues", [])
    if isinstance(issues, list):
        report.issues.extend(issues)

    suggestions = holistic.get("suggestions", [])
    if isinstance(suggestions, list):
        report.suggestions.extend(suggestions)

    reasoning = holistic.get("reasoning", "")
    if reasoning:
        report.issues.append(f"Holistic: {reasoning}")


def _save_report(summary: BatchEvaluationSummary, filepath: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, default=str)
    logger.info("Evaluation report written to %s", filepath)
