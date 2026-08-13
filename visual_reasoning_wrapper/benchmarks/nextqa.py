"""Official NExT-QA multiple-choice video benchmark loader."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


TYPE_FAMILIES = {"C": "Causal", "T": "Temporal", "D": "Descriptive"}


@dataclass(frozen=True)
class NExTQASample:
    video_id: str
    qid: int
    question_type: str
    question_family: str
    question: str
    options: dict[str, str]
    answer: str
    split: str
    frame_count: int
    width: int
    height: int
    video_path: Path | None
    vidor_path: str

    @property
    def sample_id(self) -> str:
        return f"{self.video_id}.Q{self.qid}"

    @property
    def answer_slots(self) -> tuple[str, ...]:
        return tuple(self.options)

    @property
    def question_type_family(self) -> str:
        return self.question_family

    @property
    def start_sec(self) -> float:
        return 0.0

    @property
    def end_sec(self) -> None:
        return None

    def format_question(self) -> str:
        options = "\n".join(f"{slot}. {text}" for slot, text in self.options.items())
        return f"{self.question.strip()}\nOptions:\n{options}\nReturn one option letter."

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "qid": self.qid,
            "question_type": self.question_type,
            "question_family": self.question_family,
            "question": self.question,
            "options": dict(self.options),
            "answer_slots": list(self.answer_slots),
            "answer": self.answer,
            "split": self.split,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "vidor_path": self.vidor_path,
            "video_path": str(self.video_path) if self.video_path else None,
        }


def default_nextqa_root(workspace_root: str | Path | None = None) -> Path:
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[3]
    return Path(workspace_root) / "datasets" / "NExT-QA-official"


def _canonical_split(split: str) -> str:
    value = {"valid": "val", "validation": "val"}.get(split.lower(), split.lower())
    if value not in {"train", "val", "test"}:
        raise ValueError("NExT-QA split must be train, val, or test")
    return value


def _annotations(root: Path, split: str) -> Path:
    return root / "dataset" / "nextqa" / f"{split}.csv"


def _mapping(root: Path) -> dict[str, str]:
    path = root / "dataset" / "nextqa" / "map_vid_vidorID.json"
    if not path.is_file():
        raise FileNotFoundError(f"NExT-QA video mapping not found: {path}")
    return {str(key): str(value) for key, value in json.loads(path.read_text()).items()}


def _video_path(root: Path, video_id: str, vidor_path: str) -> Path:
    relative = Path(vidor_path).with_suffix(".mp4")
    candidates = (
        root / "videos" / relative,
        root / "videos" / "NExTVideo" / relative,
        root / "NExTVideo" / relative,
        root / "videos" / f"{video_id}.mp4",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def load_nextqa_rows(
    split: str = "val", *, nextqa_root: str | Path | None = None,
) -> list[dict[str, str]]:
    root = Path(nextqa_root) if nextqa_root else default_nextqa_root()
    canonical = _canonical_split(split)
    path = _annotations(root, canonical)
    if not path.is_file():
        raise FileNotFoundError(f"NExT-QA annotations not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_nextqa_samples(
    split: str = "val",
    *,
    nextqa_root: str | Path | None = None,
    question_families: Iterable[str] | None = None,
    sample_ids: Iterable[str] | None = None,
    require_video: bool = False,
    limit: int | None = None,
) -> Iterator[NExTQASample]:
    root = Path(nextqa_root) if nextqa_root else default_nextqa_root()
    canonical = _canonical_split(split)
    mapping = _mapping(root)
    allowed = set(question_families or TYPE_FAMILIES.values())
    if allowed - set(TYPE_FAMILIES.values()):
        raise ValueError("unknown NExT-QA question family")
    allowed_ids = set(map(str, sample_ids)) if sample_ids else None
    emitted = 0
    for row in load_nextqa_rows(canonical, nextqa_root=root):
        video_id = str(row["video"])
        qid = int(row["qid"])
        sample_id = f"{video_id}.Q{qid}"
        if allowed_ids is not None and sample_id not in allowed_ids:
            continue
        question_type = str(row["type"])
        family = TYPE_FAMILIES.get(question_type[:1])
        if family is None:
            raise ValueError(f"unknown NExT-QA type: {question_type}")
        if family not in allowed:
            continue
        vidor_path = mapping[video_id]
        path = _video_path(root, video_id, vidor_path)
        if require_video and not path.is_file():
            continue
        options = {chr(65 + index): str(row[f"a{index}"]) for index in range(5)}
        answer_index = int(row["answer"])
        if not 0 <= answer_index < len(options):
            raise ValueError(f"answer index out of range: {sample_id}")
        yield NExTQASample(
            video_id=video_id,
            qid=qid,
            question_type=question_type,
            question_family=family,
            question=str(row["question"]),
            options=options,
            answer=chr(65 + answer_index),
            split=canonical,
            frame_count=int(row["frame_count"]),
            width=int(row["width"]),
            height=int(row["height"]),
            video_path=path if path.is_file() else None,
            vidor_path=vidor_path,
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def parse_nextqa_sample(
    sample: NExTQASample, *, frames: Sequence[Any] | None = None, **kwargs: Any,
) -> dict[str, Any]:
    from .video_holmes import parse_video_holmes_sample

    result = parse_video_holmes_sample(sample, frames=frames, **kwargs)  # type: ignore[arg-type]
    result["benchmark"] = "nextqa"
    return result


__all__ = [
    "NExTQASample",
    "TYPE_FAMILIES",
    "default_nextqa_root",
    "iter_nextqa_samples",
    "load_nextqa_rows",
    "parse_nextqa_sample",
]
