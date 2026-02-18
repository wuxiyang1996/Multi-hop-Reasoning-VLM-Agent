"""
Synthetic Experience Evaluation — LLM-as-a-Judge quality assessment for
world-model outputs (experiences, episodes, skill trajectories).

Quick start::

    from world_model.evaluation import (
        evaluate_experiences,
        evaluate_single,
        records_from_text_sequences,
        ExperienceRecord,
        ExperienceStep,
    )

    records = records_from_text_sequences(sequences, plans)
    summary = evaluate_experiences(records)

    for rid, report in summary.reports.items():
        print(f"{rid}: {report.overall_score:.2f} -> {report.verdict.value}")
"""

from world_model.evaluation.run_evaluation import (
    evaluate_experiences,
    evaluate_single,
    records_from_text_sequences,
    records_from_multimodal_sequences,
)
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
from world_model.evaluation.config import (
    ExperienceEvaluationConfig,
    LLMJudgeConfig,
)

__all__ = [
    # Orchestrators
    "evaluate_experiences",
    "evaluate_single",
    # Converters
    "records_from_text_sequences",
    "records_from_multimodal_sequences",
    # Schemas
    "BatchEvaluationSummary",
    "DimensionScore",
    "ExperienceQualityReport",
    "ExperienceRecord",
    "ExperienceStep",
    "ExperienceVerdict",
    "QualityDimension",
    "QualityGrade",
    # Config
    "ExperienceEvaluationConfig",
    "LLMJudgeConfig",
]
