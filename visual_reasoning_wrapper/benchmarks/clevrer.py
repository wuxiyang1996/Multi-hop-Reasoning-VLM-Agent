"""CLEVRER causal-choice loader for intervention-grounded video reasoning.

The official CLEVRER annotations represent explanatory, predictive, and
counterfactual questions as a *set of candidate events*.  Every candidate is
labelled ``correct`` or ``wrong``; this is not necessarily a single-answer
MCQ.  The loader therefore exposes one binary sample per candidate.  That
preserves the official per-choice supervision and gives the transfer harness
an explicit hypothesis to test without leaking the functional program or the
gold event graph into the model prompt.

Expected disk layout::

    data/CLEVRER/
        questions/train.json
        questions/validation.json
        questions/test.json
        videos/train/video_00000.mp4
        videos/validation/video_10000.mp4
        videos/test/video_15000.mp4

Question JSON files are available in the authors' official code repository;
videos are downloaded from the official CLEVRER project page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


_SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}
_CAUSAL_TYPES = ("explanatory", "predictive", "counterfactual")
_ANSWER_SLOTS = ("A", "B")


@dataclass(frozen=True)
class CLEVRERChoiceSample:
    """One labelled candidate event for a CLEVRER causal question."""

    video_id: str
    scene_index: int
    question_id: int
    choice_id: int
    question_type: str
    question: str
    candidate: str
    answer: str | None
    split: str
    video_path: Path | None
    question_subtype: str | None = None
    question_program: tuple[str, ...] = ()
    choice_program: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def sample_id(self) -> str:
        return f"{self.video_id}.Q{self.question_id}.C{self.choice_id}"

    @property
    def options(self) -> dict[str, str]:
        return {
            "A": "The candidate statement is correct.",
            "B": "The candidate statement is incorrect.",
        }

    @property
    def answer_slots(self) -> tuple[str, str]:
        return _ANSWER_SLOTS

    def format_question(self) -> str:
        return (
            f"{self.question.strip()}\n\n"
            f"Candidate statement: {self.candidate.strip()}\n\n"
            "Decide whether this candidate is correct according to the visible "
            "video dynamics.\nOptions:\n"
            "A. The candidate statement is correct.\n"
            "B. The candidate statement is incorrect.\n\n"
            "First inspect a question-conditioned temporal window, identify "
            "and track the relevant objects, and distinguish observed events "
            "from predicted or counterfactual events. Return only A or B."
        )

    def to_dict(self, *, include_oracle_programs: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "scene_index": self.scene_index,
            "question_id": self.question_id,
            "choice_id": self.choice_id,
            "question_type": self.question_type,
            "question_subtype": self.question_subtype,
            "question": self.question,
            "candidate": self.candidate,
            "options": self.options,
            "answer_slots": list(self.answer_slots),
            "answer": self.answer,
            "split": self.split,
            "video_path": str(self.video_path) if self.video_path else None,
        }
        if include_oracle_programs:
            payload["question_program"] = list(self.question_program)
            payload["choice_program"] = list(self.choice_program)
        return payload


def default_clevrer_root(workspace_root: str | Path | None = None) -> Path:
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[2]
    return Path(workspace_root) / "data" / "CLEVRER"


def _canonical_split(split: str) -> str:
    try:
        return _SPLIT_ALIASES[split.lower()]
    except KeyError as exc:
        raise ValueError(
            f"split must be one of {sorted(_SPLIT_ALIASES)}, got {split!r}"
        ) from exc


def _question_path(root: Path, split: str) -> Path:
    return root / "questions" / f"{split}.json"


def _video_path(root: Path, split: str, filename: str) -> Path:
    candidates = (
        root / "videos" / split / filename,
        root / f"{split}_video" / filename,
        root / split / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_clevrer_scenes(
    split: str = "validation",
    *,
    clevrer_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load official scene-level CLEVRER question annotations."""

    canonical = _canonical_split(split)
    root = Path(clevrer_root) if clevrer_root else default_clevrer_root()
    path = _question_path(root, canonical)
    if not path.is_file():
        raise FileNotFoundError(
            f"CLEVRER questions not found at {path}. Download the official "
            "annotations before constructing a split."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"CLEVRER question file must contain a list: {path}")
    return payload


def iter_clevrer_choice_samples(
    split: str = "validation",
    *,
    clevrer_root: str | Path | None = None,
    question_types: Iterable[str] | None = None,
    scene_indices: Iterable[int] | None = None,
    sample_ids: Iterable[str] | None = None,
    require_video: bool = False,
    limit: int | None = None,
) -> Iterator[CLEVRERChoiceSample]:
    """Yield binary causal candidates without exposing oracle programs.

    ``question_types`` defaults to explanatory, predictive, and
    counterfactual. Descriptive questions have scalar/open-ended answers and
    are intentionally excluded from this per-choice transfer interface.
    """

    canonical = _canonical_split(split)
    root = Path(clevrer_root) if clevrer_root else default_clevrer_root()
    allowed_types = set(question_types or _CAUSAL_TYPES)
    unknown = allowed_types - set(_CAUSAL_TYPES)
    if unknown:
        raise ValueError(
            "choice samples support only causal question types; unknown: "
            + ", ".join(sorted(unknown))
        )
    allowed_scenes = set(map(int, scene_indices)) if scene_indices else None
    allowed_ids = set(map(str, sample_ids)) if sample_ids else None

    emitted = 0
    for scene in load_clevrer_scenes(canonical, clevrer_root=root):
        scene_index = int(scene["scene_index"])
        if allowed_scenes is not None and scene_index not in allowed_scenes:
            continue
        filename = str(scene.get("video_filename") or f"video_{scene_index:05d}.mp4")
        video_path = _video_path(root, canonical, filename)
        for question in scene.get("questions") or ():
            question_type = str(question.get("question_type") or "")
            if question_type not in allowed_types:
                continue
            choices = question.get("choices") or ()
            for choice in choices:
                choice_id = int(choice["choice_id"])
                question_id = int(question["question_id"])
                sample_id = f"{filename}.Q{question_id}.C{choice_id}"
                if allowed_ids is not None and sample_id not in allowed_ids:
                    continue
                if require_video and not video_path.is_file():
                    continue
                label = choice.get("answer")
                answer = None
                if label is not None:
                    normalized = str(label).strip().lower()
                    if normalized not in {"correct", "wrong"}:
                        raise ValueError(
                            f"unexpected CLEVRER choice label {label!r} in {sample_id}"
                        )
                    answer = "A" if normalized == "correct" else "B"
                sample = CLEVRERChoiceSample(
                    video_id=filename,
                    scene_index=scene_index,
                    question_id=question_id,
                    choice_id=choice_id,
                    question_type=question_type,
                    question_subtype=question.get("question_subtype"),
                    question=str(question["question"]),
                    candidate=str(choice["choice"]),
                    answer=answer,
                    split=canonical,
                    video_path=video_path if video_path.is_file() else None,
                    question_program=tuple(map(str, question.get("program") or ())),
                    choice_program=tuple(map(str, choice.get("program") or ())),
                    raw={"question": question, "choice": choice},
                )
                yield sample
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def parse_clevrer_choice_sample(
    sample: CLEVRERChoiceSample,
    *,
    frames: Sequence[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the shared video tool loop on a binary CLEVRER candidate."""

    from .video_holmes import parse_video_holmes_sample

    result = parse_video_holmes_sample(sample, frames=frames, **kwargs)  # type: ignore[arg-type]
    result["benchmark"] = "clevrer"
    return result


__all__ = [
    "CLEVRERChoiceSample",
    "default_clevrer_root",
    "iter_clevrer_choice_samples",
    "load_clevrer_scenes",
    "parse_clevrer_choice_sample",
]
