"""
Data schemas for Synthetic Experience Evaluation.

Provides LLM-as-a-judge quality assessment of synthetic experiences, episodes,
and skill trajectories produced by the world model.  Evaluates across five
dimensions designed for agentic pipelines:

  1. **Fidelity**            — physical / logical plausibility of predicted states.
  2. **Consistency**         — internal coherence across consecutive transitions.
  3. **Instruction Adherence** — how well the trajectory follows the synthesis plan.
  4. **Diversity**           — novelty of the experience relative to a reference set (batch-level).
  5. **Informativeness**     — learning-signal density for downstream agent training.

Core types:
  - ``ExperienceRecord``     : unified wrapper for one synthetic trajectory.
  - ``DimensionScore``       : score + evidence for one quality dimension.
  - ``ExperienceQualityReport`` : per-trajectory report aggregating all dimensions.
  - ``BatchEvaluationSummary``  : batch-wide evaluation summary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# ─── Quality dimensions ──────────────────────────────────────────────

class QualityDimension(str, Enum):
    FIDELITY = "fidelity"
    CONSISTENCY = "consistency"
    INSTRUCTION_ADHERENCE = "instruction_adherence"
    DIVERSITY = "diversity"
    INFORMATIVENESS = "informativeness"


class QualityGrade(str, Enum):
    """Ordinal grade derived from a numeric score."""
    EXCELLENT = "excellent"    # >= 0.8
    GOOD = "good"              # >= 0.6
    FAIR = "fair"              # >= 0.4
    POOR = "poor"              # >= 0.2
    FAILING = "failing"        # < 0.2

    @classmethod
    def from_score(cls, score: float) -> QualityGrade:
        if score >= 0.8:
            return cls.EXCELLENT
        if score >= 0.6:
            return cls.GOOD
        if score >= 0.4:
            return cls.FAIR
        if score >= 0.2:
            return cls.POOR
        return cls.FAILING


# ─── Unified experience record ───────────────────────────────────────

@dataclass
class ExperienceStep:
    """One step in a synthetic trajectory (modality-agnostic)."""

    state: str
    next_state: str
    instruction: str
    action: Optional[str] = None
    reward: Optional[float] = None
    metadata: Optional[dict] = None

    # For multi-modal: paths or descriptors (the evaluator works on text;
    # if the caller has image captions / VLM descriptions, put them here).
    state_visual_description: Optional[str] = None
    next_state_visual_description: Optional[str] = None


@dataclass
class ExperienceRecord:
    """Unified representation of one synthetic trajectory for evaluation.

    Works with both ``TextSyntheticExperienceSequence`` (textual world model)
    and ``SyntheticExperienceSequence`` (multi-modal world model) outputs.
    Callers convert world-model outputs into this format before evaluation.
    """

    record_id: str
    steps: List[ExperienceStep]

    # Plan / instructions that drove this trajectory
    synthesis_instructions: str = ""
    plan_steps: Optional[List[str]] = None

    # Context
    env_hint: Optional[str] = None  # e.g. "overcooked", "avalon"
    skill_id: Optional[str] = None
    episode_id: Optional[str] = None
    initial_state: Optional[str] = None
    metadata: Optional[dict] = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def states_text(self, max_chars: int = 400) -> str:
        """Render state sequence as compact text for prompt injection."""
        lines = []
        for i, s in enumerate(self.steps):
            st = s.state[:max_chars]
            ns = s.next_state[:max_chars]
            act = s.action or "(inferred)"
            lines.append(f"  t={i}: state={st} -> action={act} -> next_state={ns}")
        return "\n".join(lines)


# ─── Dimension score ─────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """Score and supporting evidence for one quality dimension."""

    dimension: QualityDimension
    score: float = 0.0  # normalised to [0, 1]
    grade: QualityGrade = QualityGrade.FAILING
    evidence: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.grade = QualityGrade.from_score(self.score)

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 4),
            "grade": self.grade.value,
            "evidence": self.evidence,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DimensionScore:
        return cls(
            dimension=QualityDimension(d["dimension"]),
            score=d["score"],
            evidence=d.get("evidence", []),
            details=d.get("details", {}),
        )


# ─── Per-trajectory quality report ───────────────────────────────────

class ExperienceVerdict(str, Enum):
    """Actionable recommendation for a single synthetic trajectory."""
    ACCEPT = "accept"
    REFINE = "refine"
    REGENERATE = "regenerate"
    DISCARD = "discard"


@dataclass
class ExperienceQualityReport:
    """Holistic quality report for one synthetic trajectory."""

    record_id: str
    dimensions: Dict[str, DimensionScore] = field(default_factory=dict)

    overall_score: float = 0.0
    overall_grade: QualityGrade = QualityGrade.FAILING
    verdict: ExperienceVerdict = ExperienceVerdict.ACCEPT

    # Actionable feedback for the synthesis pipeline
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    evaluated_at: float = field(default_factory=time.time)

    def compute_overall(self, weights: Optional[Dict[str, float]] = None) -> None:
        """Compute weighted overall score from dimension scores."""
        if not self.dimensions:
            return
        default_weights = {d.value: 1.0 for d in QualityDimension}
        w = weights or default_weights
        total_weight = 0.0
        weighted_sum = 0.0
        for dim_name, ds in self.dimensions.items():
            wt = w.get(dim_name, 1.0)
            weighted_sum += ds.score * wt
            total_weight += wt
        self.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        self.overall_grade = QualityGrade.from_score(self.overall_score)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "overall_score": round(self.overall_score, 4),
            "overall_grade": self.overall_grade.value,
            "verdict": self.verdict.value,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExperienceQualityReport:
        report = cls(
            record_id=d["record_id"],
            overall_score=d.get("overall_score", 0.0),
            verdict=ExperienceVerdict(d.get("verdict", "accept")),
            issues=d.get("issues", []),
            suggestions=d.get("suggestions", []),
            evaluated_at=d.get("evaluated_at", 0.0),
        )
        for k, v in d.get("dimensions", {}).items():
            report.dimensions[k] = DimensionScore.from_dict(v)
        report.overall_grade = QualityGrade.from_score(report.overall_score)
        return report

    def format_for_llm(self) -> str:
        """Render as compact text for LLM context injection."""
        lines = [f"=== Experience Quality: {self.record_id} ==="]
        lines.append(
            f"  Overall: {self.overall_score:.2f} ({self.overall_grade.value}) "
            f"-> {self.verdict.value}"
        )
        for dim_name, ds in self.dimensions.items():
            lines.append(f"  {dim_name}: {ds.score:.2f} ({ds.grade.value})")
            for ev in ds.evidence[:3]:
                lines.append(f"    - {ev}")
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(f"    ! {issue}")
        if self.suggestions:
            lines.append("  Suggestions:")
            for sug in self.suggestions:
                lines.append(f"    > {sug}")
        return "\n".join(lines)


# ─── Batch-level summary ─────────────────────────────────────────────

@dataclass
class BatchEvaluationSummary:
    """Evaluation summary across a batch of synthetic trajectories."""

    reports: Dict[str, ExperienceQualityReport] = field(default_factory=dict)

    mean_overall: float = 0.0
    n_excellent: int = 0
    n_good: int = 0
    n_fair: int = 0
    n_poor: int = 0
    n_failing: int = 0

    accept_ids: List[str] = field(default_factory=list)
    refine_ids: List[str] = field(default_factory=list)
    regenerate_ids: List[str] = field(default_factory=list)
    discard_ids: List[str] = field(default_factory=list)

    evaluated_at: float = field(default_factory=time.time)

    def compute_summary(self) -> None:
        if not self.reports:
            return
        scores = [r.overall_score for r in self.reports.values()]
        self.mean_overall = sum(scores) / len(scores)
        self.n_excellent = sum(1 for s in scores if s >= 0.8)
        self.n_good = sum(1 for s in scores if 0.6 <= s < 0.8)
        self.n_fair = sum(1 for s in scores if 0.4 <= s < 0.6)
        self.n_poor = sum(1 for s in scores if 0.2 <= s < 0.4)
        self.n_failing = sum(1 for s in scores if s < 0.2)

        self.accept_ids = [
            rid for rid, r in self.reports.items()
            if r.verdict == ExperienceVerdict.ACCEPT
        ]
        self.refine_ids = [
            rid for rid, r in self.reports.items()
            if r.verdict == ExperienceVerdict.REFINE
        ]
        self.regenerate_ids = [
            rid for rid, r in self.reports.items()
            if r.verdict == ExperienceVerdict.REGENERATE
        ]
        self.discard_ids = [
            rid for rid, r in self.reports.items()
            if r.verdict == ExperienceVerdict.DISCARD
        ]

    def acceptance_rate(self) -> float:
        if not self.reports:
            return 0.0
        return len(self.accept_ids) / len(self.reports)

    def to_dict(self) -> dict:
        return {
            "n_trajectories": len(self.reports),
            "mean_overall": round(self.mean_overall, 4),
            "grade_distribution": {
                "excellent": self.n_excellent,
                "good": self.n_good,
                "fair": self.n_fair,
                "poor": self.n_poor,
                "failing": self.n_failing,
            },
            "acceptance_rate": round(self.acceptance_rate(), 4),
            "verdicts": {
                "accept": self.accept_ids,
                "refine": self.refine_ids,
                "regenerate": self.regenerate_ids,
                "discard": self.discard_ids,
            },
            "per_trajectory": {
                rid: r.to_dict() for rid, r in self.reports.items()
            },
            "evaluated_at": self.evaluated_at,
        }

    def format_for_llm(self) -> str:
        lines = ["=== Synthetic Experience Batch Evaluation ==="]
        lines.append(f"  Trajectories evaluated: {len(self.reports)}")
        lines.append(f"  Mean quality: {self.mean_overall:.2f}")
        lines.append(f"  Acceptance rate: {self.acceptance_rate():.1%}")
        lines.append(
            f"  Distribution: {self.n_excellent} excellent, {self.n_good} good, "
            f"{self.n_fair} fair, {self.n_poor} poor, {self.n_failing} failing"
        )
        if self.discard_ids:
            lines.append(f"  Discard: {', '.join(self.discard_ids)}")
        if self.regenerate_ids:
            lines.append(f"  Regenerate: {', '.join(self.regenerate_ids)}")
        if self.refine_ids:
            lines.append(f"  Refine: {', '.join(self.refine_ids)}")
        return "\n".join(lines)
