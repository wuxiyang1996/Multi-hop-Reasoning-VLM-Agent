"""VRBench loader + parser.

VRBench (https://huggingface.co/datasets/Uni-MoE/VRBench) is a 960-video
multi-step narrative reasoning benchmark.  Each video comes with up to 7
MCQ questions, each including a gold ``reasoning_process`` with timestamped
intermediate steps — making it the strongest benchmark for evaluating
multi-hop evidence-chain retrieval quality on long videos.

Disk layout expected at ``default_vrbench_root()``::

    <root>/
        VRBench_eval.jsonl          # 960 lines, one JSON per video
        v001_360p/<video_id>.mp4    # 360p video files

JSONL record structure (per video)::

    {
      "video_id": "TZk_p-q8Fzo",
      "video_path": "VRBench/videos/v001/TZk_p-q8Fzo.mp4",
      "video_summary": "...",
      "mcq": {
        "qa1": {
          "question": "...",
          "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
          "answer": "D",
          "original_question": "...",
          "original_answer": "...",
          "reasoning_process": {"1": "step [ts->ts]", "2": "..."},
          "reasoning_type": "Implicit Inference"
        },
        ...  # up to qa7
      },
      "video_read_type": "av"
    }

Usage::

    from visual_reasoning_wrapper.benchmarks.vrbench import (
        iter_vrbench_samples, parse_vrbench_sample,
    )

    for sample in iter_vrbench_samples(limit=3):
        out = parse_vrbench_sample(
            sample, num_frames=8, model="gpt-4o", api_key=KEY,
        )
        print(sample.reasoning_type, out["answer"], "gt:", sample.answer)
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
class VRBenchSample:
    """One VRBench question paired with a long video on disk."""

    video_id: str
    qa_key: str
    question: str
    options: dict[str, str]
    answer: str | None = None
    original_question: str | None = None
    original_answer: str | None = None
    reasoning_process: dict[str, str] = field(default_factory=dict)
    reasoning_type: str | None = None
    video_summary: str | None = None
    video_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "qa_key": self.qa_key,
            "question": self.question,
            "options": dict(self.options),
            "answer": self.answer,
            "original_question": self.original_question,
            "original_answer": self.original_answer,
            "reasoning_process": dict(self.reasoning_process),
            "reasoning_type": self.reasoning_type,
            "video_path": str(self.video_path) if self.video_path else None,
        }

    def format_question(self) -> str:
        """Render the MCQ prompt for VRBench evaluation."""
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
            "in the video before reasoning about content.\n"
            "2. Then call `sample_frames` and `detect_objects_at_frame` "
            "on the localised window.\n"
            "3. This question may require multi-step reasoning across "
            "distant parts of the video.  Chain your evidence hops to "
            "cover all relevant moments.\n"
            "4. Each <evidence> hop must declare `abstract_op=` "
            "(GROUND / CHECK / RETRIEVE / CONCLUDE) and `tool=` of the "
            "ACTUAL tool you called.\n"
            "5. Inside <answer>, put a SINGLE letter (A-D) on the line "
            "beginning with 'answer=' — for example 'answer=C'."
        )
        return "\n".join(lines)


# ======================================================================
# Disk layout helpers
# ======================================================================

def default_vrbench_root(
    workspace_root: str | Path | None = None,
) -> Path:
    """Return the canonical VRBench root on this workspace.

    Checks two locations in order:
    1. ``<workspace>/data/VRBench/``
    2. ``<gamma-projects>/datasets/VRBench/``
    """
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[2]
    local = Path(workspace_root) / "data" / "VRBench"
    if local.exists():
        return local
    gamma = Path("/fs/gamma-projects/vlm-robot/datasets/VRBench")
    if gamma.exists():
        return gamma
    return local


def _eval_jsonl_path(root: Path) -> Path:
    return root / "VRBench_eval.jsonl"


def _resolve_video_path(root: Path, video_id: str) -> Path | None:
    """Find the .mp4 for ``video_id`` under v001_360p/."""
    candidate = root / "v001_360p" / f"{video_id}.mp4"
    if candidate.exists():
        return candidate
    return None


# ======================================================================
# Loaders
# ======================================================================

def load_vrbench_records(
    *,
    vrbench_root: str | Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load the raw VRBench JSONL records (one per video)."""
    root = Path(vrbench_root) if vrbench_root else default_vrbench_root()
    qpath = _eval_jsonl_path(root)
    if not qpath.exists():
        raise FileNotFoundError(
            f"VRBench JSONL not found at {qpath}. Download via "
            f"`huggingface-cli download Uni-MoE/VRBench "
            f"--repo-type dataset --local-dir {root}`."
        )

    records: list[dict[str, Any]] = []
    with qpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def iter_vrbench_samples(
    *,
    vrbench_root: str | Path | None = None,
    limit: int | None = None,
    video_ids: Iterable[str] | None = None,
    reasoning_types: Iterable[str] | None = None,
    require_video: bool = False,
) -> Iterator[VRBenchSample]:
    """Yield ``VRBenchSample`` objects — one per (video, qa_key) pair.

    Parameters
    ----------
    vrbench_root : path, optional
        Override auto-detected root.
    limit : int, optional
        Stop after this many samples (across all videos).
    video_ids : iterable of str, optional
        Restrict to a whitelist of video IDs.
    reasoning_types : iterable of str, optional
        Restrict to reasoning types such as ``Implicit Inference``,
        ``Causal Reasoning``, etc.
    require_video : bool
        If True, skip samples whose video file is not on disk.
    """
    root = Path(vrbench_root) if vrbench_root else default_vrbench_root()
    records = load_vrbench_records(vrbench_root=root, limit=None)

    vid_set = set(video_ids) if video_ids else None
    type_set = set(reasoning_types) if reasoning_types else None

    count = 0
    for rec in records:
        video_id = rec.get("video_id", "")
        if vid_set is not None and video_id not in vid_set:
            continue

        vpath = _resolve_video_path(root, video_id)
        if require_video and vpath is None:
            continue

        mcq = rec.get("mcq", {})
        summary = rec.get("video_summary", "")

        for qa_key in sorted(mcq.keys()):
            qa = mcq[qa_key]
            if not isinstance(qa, dict):
                continue

            rtype = qa.get("reasoning_type")
            if type_set is not None and rtype not in type_set:
                continue

            sample = VRBenchSample(
                video_id=video_id,
                qa_key=qa_key,
                question=qa.get("question", ""),
                options=dict(qa.get("options", {})),
                answer=qa.get("answer"),
                original_question=qa.get("original_question"),
                original_answer=qa.get("original_answer"),
                reasoning_process=dict(qa.get("reasoning_process", {})),
                reasoning_type=rtype,
                video_summary=summary,
                video_path=vpath,
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


def parse_vrbench_sample(
    sample: VRBenchSample,
    *,
    frames: Sequence[Image.Image] | None = None,
    num_frames: int = 16,
    fps_override: float | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_entities: int = 20,
    max_rounds: int = 6,
    max_side: int = 640,
) -> dict[str, Any]:
    """Run the VLM parser on one VRBench question.

    Default ``num_frames=16`` (higher than Video-Holmes' 8) because
    VRBench videos are longer and reasoning spans distant segments.
    """
    if frames is None:
        if sample.video_path is None or not sample.video_path.exists():
            raise FileNotFoundError(
                f"Video file missing for {sample.video_id} "
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
            "ground_truth": sample.answer,
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
    task_id = f"vrbench.{sample.video_id}.{sample.qa_key}"
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
            "reasoning_type": sample.reasoning_type,
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
    gt = sample.answer
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

def parse_vrbench_batch(
    samples: Iterable[VRBenchSample],
    *,
    output_jsonl: str | Path | None = None,
    num_frames: int = 16,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_entities: int = 20,
    max_rounds: int = 6,
    temperature: float | None = None,
    max_side: int = 640,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Run ``parse_vrbench_sample`` over a stream of samples."""
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
                out = parse_vrbench_sample(
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
                    "VRBench sample %s %s failed: %s",
                    sample.video_id, sample.qa_key, exc,
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
                    "[VR %s] %d: %s %s type=%s pred=%s gt=%s",
                    tag, i, sample.video_id, sample.qa_key,
                    sample.reasoning_type,
                    out.get("answer"),
                    out.get("ground_truth"),
                )
    finally:
        if fh is not None:
            fh.close()

    return results
