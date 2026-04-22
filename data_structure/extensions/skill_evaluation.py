"""`SkillEvaluationRecord` — the harness/gate's signed report on a skill.

Spec: PLAN-UNIFIED-SKILL-GATE §3.2 — owned by the harness/gate,
NOT by the bank. The bank persists *references* to evaluation records
(via `SkillRecord.last_evaluation_id`) but never mutates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data_structure.extensions.gate_verdict import GateVerdictPayload


@dataclass
class SkillEvaluationRecord:
    evaluation_id: str
    proposal_id: str
    skill_id: str
    skill_content_hash: str
    episode_ids: List[str] = field(default_factory=list)   # SkillEpisode IDs evaluated
    verdict: Optional[GateVerdictPayload] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    failure_class_distribution: Dict[str, int] = field(default_factory=dict)
    transfer_labels: Dict[str, int] = field(default_factory=dict)  # PLAN-HARNESS §6.4 labels
    judge_model: Optional[str] = None
    seed: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "proposal_id": self.proposal_id,
            "skill_id": self.skill_id,
            "skill_content_hash": self.skill_content_hash,
            "episode_ids": list(self.episode_ids),
            "verdict": self.verdict.to_json() if self.verdict else None,
            "metrics": dict(self.metrics),
            "failure_class_distribution": dict(self.failure_class_distribution),
            "transfer_labels": dict(self.transfer_labels),
            "judge_model": self.judge_model,
            "seed": self.seed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


__all__ = ["SkillEvaluationRecord"]
