"""CG-Bench loader + parser.

CG-Bench (https://cg-bench.github.io/leaderboard/) is a 1,219-video /
12,129-QA clue-grounded long-video benchmark (ICLR 2025).  Each question
comes with ``clue_intervals`` — gold temporal windows that contain the
evidence needed to answer — making it the primary benchmark for
evidence-attribution evaluation.

Three task modes:

* ``long_acc``  — standard MCQ accuracy over long videos
* ``clue_acc`` — MCQ accuracy on clue-grounded questions
* ``miou``     — mean IoU between predicted and gold clue intervals

Disk layout expected at ``default_cg_bench_root()``::

    <root>/
        cgbench_mini.json           # master annotation (12,129 QA pairs)
        cg_annotations/<qid>.json   # per-question JSON (from save_as_jsons.py)
        cg_videos/<video_uid>.mp4   # video files (or extracted frames in cg_images/)
        cg_subtitles/               # subtitle files (optional)
        video_meta_info.json        # video duration / fps metadata

Per-question JSON structure::

    {
      "qid": "q_001",
      "video_uid": "abc123",
      "question": "What caused ...?",
      "answer": "The man dropped ...",
      "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "right_answer": "B",
      "clue_intervals": [[10.5, 25.3], [42.1, 55.8]],
      "version": 0,
      "results": {}
    }

Usage::

    from visual_reasoning_wrapper.benchmarks.cg_bench import (
        iter_cg_bench_samples, parse_cg_bench_sample,
    )

    for sample in iter_cg_bench_samples(limit=3):
        out = parse_cg_bench_sample(
            sample, num_frames=16, model="gpt-4o", api_key=KEY,
        )
        print(out["answer"], "gt:", sample.right_answer)

Download (gated, 411 GB)::

    pip install -U "huggingface_hub[cli]"
    huggingface-cli login
    huggingface-cli download CG-Bench/CG-Bench \\
        --repo-type dataset --local-dir <root>
    cd <root> && python unzip_hf_zip.py
    python run/save_as_jsons.py
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from PIL import Image

from vlm_wrapper.ground import GroundingRequest, cascaded_ground
from ..question_router import classify_question
from .video_holmes import sample_video_frames

logger = logging.getLogger(__name__)

_ANSWER_LETTERS = ("A", "B", "C", "D")


@dataclass
class CGBenchSample:
    """One CG-Bench question with clue-interval gold annotations."""

    qid: str
    video_uid: str
    question: str
    choices: list[str]
    options: dict[str, str]
    right_answer: str | None = None
    answer_text: str | None = None
    clue_intervals: list[list[float]] = field(default_factory=list)
    video_path: Path | None = None
    task_mode: str = "long_acc"

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "video_uid": self.video_uid,
            "question": self.question,
            "choices": list(self.choices),
            "options": dict(self.options),
            "right_answer": self.right_answer,
            "answer_text": self.answer_text,
            "clue_intervals": [list(ci) for ci in self.clue_intervals],
            "video_path": str(self.video_path) if self.video_path else None,
            "task_mode": self.task_mode,
        }

    def format_question(self) -> str:
        """Render the MCQ prompt for CG-Bench evaluation."""
        lines = [self.question.strip(), "", "Options:"]
        for letter in _ANSWER_LETTERS:
            opt = self.options.get(letter)
            if opt is None:
                continue
            lines.append(f"{letter}. {opt}")
        lines.append("")
        lines.append(
            "Procedure (follow strictly):\n"
            "1. FIRST hop must be temporal — call `find_moment` or "
            "`detect_scene_changes` to localise the relevant window(s) "
            "that contain evidence for the answer.\n"
            "2. Then call `sample_frames` and `detect_objects_at_frame` "
            "on each localised window.\n"
            "3. This is a long-video question.  Focus on finding the "
            "specific clue windows rather than scanning the whole video.\n"
            "4. Each <evidence> hop must declare `abstract_op=` "
            "(GROUND / CHECK / RETRIEVE / CONCLUDE) and `tool=` of the "
            "ACTUAL tool you called.  Include `timestamp=<s>` for every "
            "temporal hop.\n"
            "5. Inside <answer>, put a SINGLE letter (A-D) on the line "
            "beginning with 'answer=' — for example 'answer=C'."
        )
        return "\n".join(lines)


# ======================================================================
# Disk layout helpers
# ======================================================================

def default_cg_bench_root(
    workspace_root: str | Path | None = None,
) -> Path:
    """Return the canonical CG-Bench root on this workspace."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[2]
    local = Path(workspace_root) / "data" / "CG-Bench"
    if local.exists():
        return local
    gamma = Path("/fs/gamma-projects/vlm-robot/datasets/CG-Bench")
    if gamma.exists():
        return gamma
    return local


def _master_json_path(root: Path) -> Path:
    return root / "cgbench_mini.json"


def _annotations_dir(root: Path) -> Path:
    return root / "cg_annotations"


def _resolve_video_path(root: Path, video_uid: str) -> Path | None:
    """Find the video for ``video_uid``, checking multiple directories."""
    for subdir in ("cg_videos", "videos", ""):
        candidate = root / subdir / f"{video_uid}.mp4" if subdir else root / f"{video_uid}.mp4"
        if candidate.exists():
            return candidate
    return None


# ======================================================================
# Loaders
# ======================================================================

def _parse_choices_to_options(choices: list[str]) -> dict[str, str]:
    """Convert CG-Bench choices list to letter->text dict.

    CG-Bench choices come as ``["A. text", "B. text", ...]`` or just
    ``["text1", "text2", ...]``.
    """
    options: dict[str, str] = {}
    for i, choice in enumerate(choices):
        choice = str(choice).strip()
        if len(choice) >= 3 and choice[0].upper() in _ANSWER_LETTERS and choice[1] in ".):":
            letter = choice[0].upper()
            text = choice[2:].strip()
        elif i < len(_ANSWER_LETTERS):
            letter = _ANSWER_LETTERS[i]
            text = choice
        else:
            continue
        options[letter] = text
    return options


def load_cg_bench_questions(
    *,
    cg_bench_root: str | Path | None = None,
    limit: int | None = None,
    task_mode: str = "long_acc",
) -> list[dict[str, Any]]:
    """Load CG-Bench questions from the master JSON or per-question JSONs."""
    root = Path(cg_bench_root) if cg_bench_root else default_cg_bench_root()

    master = _master_json_path(root)
    if master.exists():
        with master.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if limit is not None:
            data = data[:limit]
        return data

    anno_dir = _annotations_dir(root)
    if anno_dir.exists():
        records = []
        for fp in sorted(anno_dir.glob("*.json")):
            with fp.open("r", encoding="utf-8") as f:
                records.append(json.load(f))
            if limit is not None and len(records) >= limit:
                break
        return records

    raise FileNotFoundError(
        f"CG-Bench data not found at {root}. Download via:\n"
        f"  huggingface-cli download CG-Bench/CG-Bench "
        f"--repo-type dataset --local-dir {root}\n"
        f"Then run: python unzip_hf_zip.py && python run/save_as_jsons.py"
    )


def iter_cg_bench_samples(
    *,
    cg_bench_root: str | Path | None = None,
    limit: int | None = None,
    task_mode: str = "long_acc",
    video_uids: Iterable[str] | None = None,
    require_video: bool = False,
) -> Iterator[CGBenchSample]:
    """Yield ``CGBenchSample`` objects.

    Parameters
    ----------
    cg_bench_root : path, optional
        Override auto-detected root.
    limit : int, optional
        Stop after this many samples.
    task_mode : str
        ``long_acc``, ``clue_acc``, ``miou``, or ``open``.
    video_uids : iterable of str, optional
        Restrict to a whitelist of video UIDs.
    require_video : bool
        If True, skip samples whose video file is not on disk.
    """
    root = Path(cg_bench_root) if cg_bench_root else default_cg_bench_root()
    records = load_cg_bench_questions(
        cg_bench_root=root, limit=None, task_mode=task_mode,
    )

    uid_set = set(video_uids) if video_uids else None

    count = 0
    for rec in records:
        video_uid = rec.get("video_uid", "")
        if uid_set is not None and video_uid not in uid_set:
            continue

        vpath = _resolve_video_path(root, video_uid)
        if require_video and vpath is None:
            continue

        choices = rec.get("choices", [])
        options = _parse_choices_to_options(choices)
        clue_intervals = rec.get("clue_intervals", [])
        if isinstance(clue_intervals, str):
            try:
                clue_intervals = json.loads(clue_intervals)
            except json.JSONDecodeError:
                clue_intervals = []

        sample = CGBenchSample(
            qid=str(rec.get("qid", f"q_{count}")),
            video_uid=video_uid,
            question=rec.get("question", ""),
            choices=choices,
            options=options,
            right_answer=rec.get("right_answer"),
            answer_text=rec.get("answer"),
            clue_intervals=clue_intervals,
            video_path=vpath,
            task_mode=task_mode,
        )
        yield sample
        count += 1
        if limit is not None and count >= limit:
            return


# ======================================================================
# GPT-4o parser
# ======================================================================

def _normalise_answer_letter(text: str | None) -> str | None:
    """Extract a single A-D letter from the model's free-form answer."""
    if not text:
        return None
    stripped = text.strip().strip(".").strip()
    if stripped and stripped[0].upper() in _ANSWER_LETTERS:
        candidate = stripped[0].upper()
        if len(stripped) == 1 or not stripped[1].isalpha():
            return candidate
    for letter in _ANSWER_LETTERS:
        for token in (f"{letter}.", f"({letter})", f"[{letter}]",
                      f"Option {letter}", f"option {letter}"):
            if token in stripped:
                return letter
    upper = stripped.upper()
    for letter in _ANSWER_LETTERS:
        if upper.startswith(letter + " ") or upper.startswith(letter + ","):
            return letter
    return None


def parse_cg_bench_sample(
    sample: CGBenchSample,
    *,
    frames: Sequence[Image.Image] | None = None,
    num_frames: int = 32,
    fps_override: float | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_entities: int = 20,
    max_rounds: int = 6,
    max_side: int = 640,
) -> dict[str, Any]:
    """Run the VLM parser on one CG-Bench question.

    Default ``num_frames=32`` matches CG-Bench's official eval (32 segments).
    """
    if frames is None:
        if sample.video_path is None or not sample.video_path.exists():
            raise FileNotFoundError(
                f"Video file missing for {sample.video_uid} "
                f"(expected at {sample.video_path})."
            )
        frames, fps, video_meta = sample_video_frames(
            sample.video_path, num_frames=num_frames, max_side=max_side,
        )
    else:
        frames = list(frames)
        fps = fps_override or 1.0
        video_meta = {
            "backend": "preloaded",
            "total_frames": len(frames),
            "num_frames": len(frames),
        }

    if fps_override is not None:
        fps = fps_override

    if not frames:
        return {
            "schema": None,
            "answer": None,
            "answer_raw": None,
            "ground_truth": sample.right_answer,
            "correct": None,
            "tool_trace": [],
            "rounds": 0,
            "num_frames": 0,
            "video_meta": video_meta,
            "model": model,
            "warnings": ["no frames decoded"],
            "sample": sample.to_dict(),
        }

    current_index = len(frames) // 2
    task_id = f"cg_bench.{sample.task_mode}.{sample.qid}"
    routing = classify_question(sample.question, modality="video")
    routing_block = routing.to_prompt_block()
    question_prompt = sample.format_question()
    if routing_block:
        question_prompt = (
            f"{question_prompt}\n\n"
            "Reasoning tools available: count_value, compute_ratio, "
            "compare_values, verify_claim — use them to RECORD any "
            "counts / ratios / comparisons you derive across frames, "
            "and cite the resulting `derivation_id` (d1, d2, …) inside "
            "<derivations> and <answer>.\n"
            f"{routing_block}"
        )

    duration_s = video_meta.get("duration_s") or 0.0
    if duration_s and len(frames) > 0:
        effective_fps = len(frames) / duration_s
    else:
        effective_fps = fps or 1.0
    video_meta["effective_fps"] = round(effective_fps, 6)

    req = GroundingRequest(
        images=frames,
        goal=question_prompt,
        domain="video_qa",
        output_mode="answer",
        task_id=task_id,
        step=0,
        context={
            "fps": effective_fps,
            "native_fps": fps,
            "duration_s": duration_s,
            "sample_timestamps": video_meta.get("sample_timestamps") or [],
            "current_index": current_index,
            "task_mode": sample.task_mode,
            "clue_intervals": sample.clue_intervals,
            "options": dict(sample.options),
            "question_classes": routing.classes,
            "required_reasoning_tools": routing.required_tools,
            "derivation_kinds": routing.derivation_kinds,
        },
        max_entities=max_entities,
        max_rounds=max_rounds,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )

    primary_size = frames[current_index].size
    result = cascaded_ground(req, image_size=primary_size)

    answer_raw = result.answer
    answer_letter = _normalise_answer_letter(answer_raw)
    gt = sample.right_answer
    correct: bool | None = None
    if gt and answer_letter:
        correct = answer_letter.upper() == gt.strip().upper()

    return {
        "schema": result.schema,
        "answer": answer_letter,
        "answer_raw": answer_raw,
        "ground_truth": gt,
        "correct": correct,
        "tool_trace": result.tool_trace,
        "rounds": result.rounds,
        "num_frames": len(frames),
        "video_meta": video_meta,
        "model": result.model,
        "raw": result.raw,
        "warnings": result.warnings,
        "validation": result.validation.as_dict() if result.validation else None,
        "head_used": result.head_used,
        "escalation_trace": result.escalation_trace,
        "sample": sample.to_dict(),
    }


# ======================================================================
# Batch helper
# ======================================================================

def parse_cg_bench_batch(
    samples: Iterable[CGBenchSample],
    *,
    output_jsonl: str | Path | None = None,
    num_frames: int = 32,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_entities: int = 20,
    max_rounds: int = 6,
    temperature: float | None = None,
    max_side: int = 640,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Run ``parse_cg_bench_sample`` over a stream of samples."""
    results: list[dict[str, Any]] = []
    fh = None
    if output_jsonl is not None:
        output_path = Path(output_jsonl)
        if output_path.parent and not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(output_path, "a", encoding="utf-8")

    try:
        for i, sample in enumerate(samples, 1):
            try:
                out = parse_cg_bench_sample(
                    sample,
                    num_frames=num_frames,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    max_entities=max_entities,
                    max_rounds=max_rounds,
                    temperature=temperature,
                    max_side=max_side,
                )
            except Exception as exc:
                logger.warning(
                    "CG-Bench sample %s failed: %s",
                    sample.qid, exc,
                )
                out = {
                    "error": str(exc),
                    "sample": sample.to_dict(),
                }

            results.append(out)
            if fh is not None:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            if progress:
                correct = out.get("correct")
                tag = (
                    "OK" if correct is True
                    else "NO" if correct is False
                    else "??"
                )
                logger.info(
                    "[CG %s] %d: %s mode=%s pred=%s gt=%s",
                    tag, i, sample.qid, sample.task_mode,
                    out.get("answer"),
                    out.get("ground_truth"),
                )
    finally:
        if fh is not None:
            fh.close()

    return results
