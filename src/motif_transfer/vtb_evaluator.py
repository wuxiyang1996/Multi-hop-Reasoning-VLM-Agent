from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Sequence


OFFICIAL_REPOSITORY = "https://github.com/xi1ngang/VisualToolBench"
OFFICIAL_COMMIT = "d4f200a0a44790349667bed09334ac88623074b2"


@dataclass(frozen=True)
class VTBRubric:
    rubric_id: str
    description: str
    weight: int

    @property
    def critical(self) -> bool:
        # The official judge implementation ignores the dataset's `critical`
        # string and defines APR-critical rubrics mechanically by weight >= 4.
        return self.weight >= 4


@dataclass(frozen=True)
class VTBRubricVerdict:
    rubric_id: str
    met: bool
    explanation: str
    prompt_sha256: str = ""
    response_sha256: str = ""


@dataclass(frozen=True)
class VTBTurnScore:
    turn_index: int
    met_weight: int
    total_weight: int
    rubric_score: float
    critical_pass: bool
    verdicts: tuple[VTBRubricVerdict, ...]


@dataclass(frozen=True)
class VTBTaskScore:
    task_id: str
    met_weight: int
    total_weight: int
    ars: float
    apr_pass: bool
    turns: tuple[VTBTurnScore, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def parse_rubric_blob(blob: str | Mapping[str, Any]) -> tuple[VTBRubric, ...]:
    value = json.loads(blob) if isinstance(blob, str) else dict(blob)
    if not isinstance(value, dict) or not value:
        raise ValueError("VTB rubric turn must be one non-empty object")
    rubrics = []
    for rubric_id, raw in value.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"rubric {rubric_id} is not an object")
        weight = int(raw["weight"])
        if weight < 1 or weight > 5:
            raise ValueError(f"rubric {rubric_id} has invalid weight {weight}")
        description = str(raw.get("description") or "").strip()
        if not description:
            raise ValueError(f"rubric {rubric_id} has no description")
        rubrics.append(VTBRubric(str(rubric_id), description, weight))
    return tuple(rubrics)


def official_judge_prompt(
    question: str,
    golden_answer: str,
    rubric: VTBRubric,
    model_answer: str,
) -> str:
    """Exact single-rubric prompt from official commit d4f200a."""
    hard_break = "  "
    return f"""
        You are an expert evaluator tasked with judging whether a model's answer meets a specific rubric criterion.{hard_break}
        You will be provided with:{hard_break}
        - a question{hard_break}
        - a golden (reference) answer{hard_break}
        - a rubric criterion{hard_break}
        - the model's answer{hard_break}

        Your task is to decide if the model's answer **meets** or **does not meet** the given rubric criterion, referencing the golden answer only as needed.

        ### Inputs:
        **Question:** {question}{hard_break}

        **Golden Answer:** {golden_answer}{hard_break}

        **Rubric Criterion:** {rubric.description}{hard_break}

        **Model Answer:** {model_answer}{hard_break}

        ### Important Notes:
        - The model's answer does not need to be correct to meet the criterion if correctness is not required.{hard_break}
        *Example:* If the rubric is "The model should show its reasoning process to answer the question," the answer can be incorrect but still meet the rubric if model's reasoning process is present.{hard_break}
        - For writing style or presentation rubrics, apply leniency.{hard_break}
        *Example:* If the rubric asks for conciseness, answers that are slightly longer than the golden answer but still reasonably length should be considered as meeting the rubric.
        - The model's answer may satisfy the rubric implicitly without explicitly mentioning the exact term. This should still be considered as meeting the criterion if model's answer is reasonable and makes sense.{hard_break}
        *Example:* If the rubric is "The model should demonstrate understanding of photosynthesis," and the model states "Plants make their own food using sunlight," without explicitly mentioning the term "photosynthesis," it still meets the criterion.

        ### Output Format:
        Return your judgement in the following JSON format:
        {{
            "explanation": "Brief explanation of your judgement",
            "judge_result": "Met" or "Not Met"
        }}
        """


def parse_official_judge_response(
    rubric_id: str,
    response: str | Mapping[str, Any],
    *,
    prompt_sha256: str = "",
    response_sha256: str = "",
) -> VTBRubricVerdict:
    value = json.loads(response) if isinstance(response, str) else dict(response)
    result = str(value.get("judge_result") or "").strip().lower().replace("_", " ")
    if result not in {"met", "not met"}:
        raise ValueError(f"invalid VTB judge_result {value.get('judge_result')!r}")
    explanation = str(value.get("explanation") or "").strip()
    if not explanation:
        raise ValueError("VTB judge response has no explanation")
    return VTBRubricVerdict(
        rubric_id=str(rubric_id),
        met=result == "met",
        explanation=explanation,
        prompt_sha256=prompt_sha256,
        response_sha256=response_sha256,
    )


def score_vtb_task(
    task_id: str,
    rubric_turns: Sequence[Sequence[VTBRubric]],
    verdict_turns: Sequence[Sequence[VTBRubricVerdict]],
) -> VTBTaskScore:
    if len(rubric_turns) != len(verdict_turns) or not rubric_turns:
        raise ValueError("VTB judge must receive exactly one verdict set for every task turn")
    turn_scores = []
    task_met = 0
    task_total = 0
    task_pass = True
    for turn_index, (rubrics, verdicts) in enumerate(zip(rubric_turns, verdict_turns)):
        by_id = {row.rubric_id: row for row in verdicts}
        expected = {row.rubric_id for row in rubrics}
        if len(by_id) != len(verdicts) or set(by_id) != expected:
            raise ValueError(f"turn {turn_index} verdict ids do not exactly match rubric ids")
        total_weight = sum(row.weight for row in rubrics)
        met_weight = sum(row.weight for row in rubrics if by_id[row.rubric_id].met)
        critical_pass = all(by_id[row.rubric_id].met for row in rubrics if row.critical)
        turn_scores.append(VTBTurnScore(
            turn_index=turn_index,
            met_weight=met_weight,
            total_weight=total_weight,
            rubric_score=met_weight / total_weight,
            critical_pass=critical_pass,
            verdicts=tuple(verdicts),
        ))
        task_met += met_weight
        task_total += total_weight
        task_pass = task_pass and critical_pass
    return VTBTaskScore(
        task_id=str(task_id),
        met_weight=task_met,
        total_weight=task_total,
        ars=task_met / task_total,
        apr_pass=task_pass,
        turns=tuple(turn_scores),
    )
