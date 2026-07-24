from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .contracts import (
    SkillRankingReceipt,
    SourcePolicyStepRecord,
    SourceSegmentReceipt,
    stable_hash,
)


@dataclass(frozen=True)
class SourceSkillCandidate:
    skill_id: str
    description: str


class NativePromptBackend(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def complete_prompt(self, role: str, prompt: str) -> str: ...


def load_source_skill_bank(path: str | Path) -> tuple[SourceSkillCandidate, ...]:
    candidates: list[SourceSkillCandidate] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            skill = (json.loads(line).get("skill") or {})
            skill_id = str(skill.get("skill_id", "")).strip()
            if not skill_id or skill_id in seen:
                continue
            description = str(
                skill.get("strategic_description")
                or skill.get("description")
                or (skill.get("contract") or {}).get("description")
                or ""
            )
            candidates.append(SourceSkillCandidate(skill_id, description))
            seen.add(skill_id)
    if not candidates:
        raise ValueError(f"skill bank has no candidates: {path}")
    return tuple(candidates)


def segment_native_policy(
    records: Sequence[SourcePolicyStepRecord],
) -> tuple[tuple[SourceSegmentReceipt, tuple[SourcePolicyStepRecord, ...]], ...]:
    """Use only recorded skill-hash changes; no semantic boundary heuristic."""

    if not records:
        return ()
    ordered = tuple(sorted(records, key=lambda row: row.step))
    result: list[tuple[SourceSegmentReceipt, tuple[SourcePolicyStepRecord, ...]]] = []
    start = 0
    for index in range(1, len(ordered) + 1):
        boundary = (
            index == len(ordered)
            or ordered[index].selected_skill_id != ordered[index - 1].selected_skill_id
            or ordered[index].step != ordered[index - 1].step + 1
        )
        if boundary:
            span = ordered[start:index]
            receipt = SourceSegmentReceipt.create(span[0].episode_id, span)
            result.append((receipt, span))
            start = index
    return tuple(result)


def _format_candidates(candidates: Sequence[SourceSkillCandidate]) -> str:
    lines = []
    for candidate in candidates:
        if candidate.description:
            lines.append(f'  - "{candidate.skill_id}": {candidate.description[:150]}')
        else:
            lines.append(f'  - "{candidate.skill_id}"')
    return "\n".join(lines)


def build_native_segment_ranking_prompt(
    segment: SourceSegmentReceipt,
    records: Sequence[SourcePolicyStepRecord],
    candidates: Sequence[SourceSkillCandidate],
) -> str:
    """Mirror the old co-evolution segment teacher prompt exactly in structure."""

    observations = [row.before.state for row in records]
    actions = [row.action for row in records]
    start_state = records[0].before.state.get("structured_state")
    end_state = records[-1].after.state.get("structured_state")
    predicates_start = str(start_state) if start_state else "N/A"
    predicates_end = str(end_state) if end_state else "N/A"
    return (
        "You are an expert at recognizing skills in agent trajectories.\n\n"
        f"A trajectory segment spans timesteps {segment.start_step} to "
        f"{segment.end_step} (length {len(records)}).\n\n"
        f"Observations:\n{str(observations)}\n\n"
        f"Actions:\n{str(actions)}\n\n"
        f"State at segment start: {predicates_start}\n"
        f"State at segment end:   {predicates_end}\n\n"
        f"Candidate skills:\n{_format_candidates(candidates)}\n\n"
        "Rank ALL candidate skills from best fit to worst fit for this segment.  Consider:\n"
        "  - Do the actions match what this skill would produce?\n"
        "  - Is the segment length reasonable for this skill?\n"
        "  - Is the state change consistent with this skill's purpose?\n\n"
        "Return ONLY a JSON object (no extra text):\n"
        '{"ranking": ["best_skill", "second_best", ...], '
        '"reasoning": "brief explanation"}\n'
    )


class SourceSkillRanker:
    """Runs the old segment head only on its native all-skills ranking task."""

    def __init__(self, backend: NativePromptBackend) -> None:
        self.backend = backend

    def rank(
        self,
        segment: SourceSegmentReceipt,
        records: Sequence[SourcePolicyStepRecord],
        candidates: Sequence[SourceSkillCandidate],
    ) -> SkillRankingReceipt:
        prompt = build_native_segment_ranking_prompt(segment, records, candidates)
        raw = self.backend.complete_prompt("segment", prompt)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("ranking output must be one JSON object")
        ranking_raw = parsed.get("ranking")
        if not isinstance(ranking_raw, list) or not all(
            isinstance(value, str) for value in ranking_raw
        ):
            raise ValueError("ranking must be a string list")
        ranking = tuple(value.strip() for value in ranking_raw)
        expected = tuple(candidate.skill_id for candidate in candidates)
        missing = sorted(set(expected) - set(ranking))
        unknown = sorted(set(ranking) - set(expected))
        duplicates = sorted({value for value in ranking if ranking.count(value) > 1})
        if len(ranking) != len(expected) or missing or unknown or duplicates:
            raise ValueError(
                "ranking must contain every candidate exactly once "
                f"(missing={missing[:8]}, unknown={unknown[:8]}, duplicates={duplicates[:8]})"
            )
        candidate_bank_hash = stable_hash([
            {"skill_id": row.skill_id, "description": row.description}
            for row in candidates
        ])
        return SkillRankingReceipt.create(
            segment_receipt_id=segment.receipt_id,
            candidate_bank_hash=candidate_bank_hash,
            model_identity_hash=stable_hash(dict(self.backend.identity)),
            ranking=ranking,
            reasoning=str(parsed.get("reasoning", "")),
            raw_response=raw,
        )
