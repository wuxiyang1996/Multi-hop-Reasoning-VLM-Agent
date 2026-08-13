"""Official STAR situated-video benchmark loader.

Runtime samples expose questions, choices, clip boundaries, and native answer
slots.  Situation hypergraphs and functional programs are retained only as
explicit oracle diagnostics and are excluded from the default model payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


QUESTION_TYPES = ("Interaction", "Sequence", "Prediction", "Feasibility")


@dataclass(frozen=True)
class STARSample:
    question_id: str
    video_id: str
    start_sec: float
    end_sec: float
    question_type: str
    question: str
    options: dict[str, str]
    answer: str
    split: str
    video_path: Path | None
    question_program: tuple[Any, ...] = ()
    choice_programs: tuple[tuple[Any, ...], ...] = ()
    situations: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def sample_id(self) -> str:
        return self.question_id

    @property
    def answer_slots(self) -> tuple[str, ...]:
        return tuple(self.options)

    def format_question(self) -> str:
        options = "\n".join(f"{slot}. {text}" for slot, text in self.options.items())
        return f"{self.question.strip()}\nOptions:\n{options}\nReturn one option letter."

    def to_dict(self, *, include_oracle_graph: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "question_type": self.question_type,
            "question": self.question,
            "options": dict(self.options),
            "answer_slots": list(self.answer_slots),
            "answer": self.answer,
            "split": self.split,
            "video_path": str(self.video_path) if self.video_path else None,
        }
        if include_oracle_graph:
            payload["question_program"] = list(self.question_program)
            payload["choice_programs"] = [list(value) for value in self.choice_programs]
            payload["situations"] = self.situations
        return payload


def default_star_root(workspace_root: str | Path | None = None) -> Path:
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[3]
    return Path(workspace_root) / "datasets" / "STAR-official"


def _annotation_path(root: Path, split: str) -> Path:
    canonical = {"valid": "val", "validation": "val"}.get(split.lower(), split.lower())
    return root / "annotations" / f"STAR_{canonical}.json"


def _video_path(root: Path, video_id: str) -> Path:
    candidates = (
        root / "videos" / "charades" / f"{video_id}.mp4",
        root / "videos" / "Charades_v1_480" / f"{video_id}.mp4",
        root / "videos" / f"{video_id}.mp4",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def load_star_questions(
    split: str = "val", *, star_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(star_root) if star_root else default_star_root()
    path = _annotation_path(root, split)
    if not path.is_file():
        raise FileNotFoundError(f"STAR annotations not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("STAR annotation file must contain a list")
    return payload


def iter_star_samples(
    split: str = "val",
    *,
    star_root: str | Path | None = None,
    question_types: Iterable[str] | None = None,
    sample_ids: Iterable[str] | None = None,
    require_video: bool = False,
    limit: int | None = None,
) -> Iterator[STARSample]:
    root = Path(star_root) if star_root else default_star_root()
    allowed_types = set(question_types or QUESTION_TYPES)
    if allowed_types - set(QUESTION_TYPES):
        raise ValueError("unknown STAR question type")
    allowed_ids = set(map(str, sample_ids)) if sample_ids else None
    emitted = 0
    for row in load_star_questions(split, star_root=root):
        question_id = str(row["question_id"])
        question_type = question_id.split("_", 1)[0]
        if question_type not in allowed_types:
            continue
        if allowed_ids is not None and question_id not in allowed_ids:
            continue
        video_id = str(row["video_id"])
        path = _video_path(root, video_id)
        if require_video and not path.is_file():
            continue
        choices = tuple(sorted(row.get("choices") or (), key=lambda value: int(value["choice_id"])))
        if len(choices) < 2:
            raise ValueError(f"STAR question needs multiple choices: {question_id}")
        options = {chr(65 + index): str(choice["choice"]) for index, choice in enumerate(choices)}
        matches = [slot for slot, value in options.items() if value == str(row["answer"])]
        if len(matches) != 1:
            raise ValueError(f"STAR answer does not uniquely match choices: {question_id}")
        yield STARSample(
            question_id=question_id,
            video_id=video_id,
            start_sec=float(row["start"]),
            end_sec=float(row["end"]),
            question_type=question_type,
            question=str(row["question"]),
            options=options,
            answer=matches[0],
            split={"valid": "val", "validation": "val"}.get(split.lower(), split.lower()),
            video_path=path if path.is_file() else None,
            question_program=tuple(row.get("question_program") or ()),
            choice_programs=tuple(tuple(choice.get("choice_program") or ()) for choice in choices),
            situations=dict(row.get("situations") or {}),
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def parse_star_sample(
    sample: STARSample, *, frames: Sequence[Any] | None = None, **kwargs: Any,
) -> dict[str, Any]:
    from .video_holmes import parse_video_holmes_sample

    result = parse_video_holmes_sample(sample, frames=frames, **kwargs)  # type: ignore[arg-type]
    result["benchmark"] = "star"
    return result


__all__ = [
    "QUESTION_TYPES",
    "STARSample",
    "default_star_root",
    "iter_star_samples",
    "load_star_questions",
    "parse_star_sample",
]
