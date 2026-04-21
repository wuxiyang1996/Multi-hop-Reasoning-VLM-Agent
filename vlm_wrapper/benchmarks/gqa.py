"""GQA loader + GPT-4o parser.

GQA (https://cs.stanford.edu/people/dorarad/gqa/) is a real-image
visual-reasoning benchmark with ~22M questions over ~113k images
sourced from Visual Genome.  Every question is paired with a symbolic
``semanticStr`` reasoning program *and* a ground-truth scene graph,
which makes GQA an ideal cross-validation target for the multi-hop
schema (entities + relations + attributes are all spelled out by hand).

Disk layout expected at ``default_gqa_root()`` — i.e. ``data/GQA/``
relative to the repo root::

    <root>/
        questions/
            train_balanced_questions.json
            val_balanced_questions.json
            testdev_balanced_questions.json     # optional
            test_balanced_questions.json        # optional
        sceneGraphs/
            train_sceneGraphs.json
            val_sceneGraphs.json
        images/                                  # flat folder
            <imageId>.jpg

Usage::

    from vlm_wrapper.benchmarks.gqa import (
        iter_gqa_samples, parse_gqa_sample,
    )

    for sample in iter_gqa_samples(split="val", limit=5):
        out = parse_gqa_sample(sample, model="gpt-4o", api_key=KEY)
        print(sample.question, "→", out["answer"], "(gt:", sample.answer, ")")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image

from ..ground import GroundingRequest, cascaded_ground

logger = logging.getLogger(__name__)

# GQA ships the questions in two flavours: ``balanced`` (downsampled to
# remove answer-distribution bias, 1.7M Q) and ``all`` (~22M Q, only
# released as 10-shard packs).  We default to balanced — the same split
# the official leaderboard scores against.
_QUESTION_FILES: dict[str, str] = {
    "train":   "train_balanced_questions.json",
    "val":     "val_balanced_questions.json",
    "testdev": "testdev_balanced_questions.json",
    "test":    "test_balanced_questions.json",
}

_SCENE_GRAPH_FILES: dict[str, str] = {
    "train": "train_sceneGraphs.json",
    "val":   "val_sceneGraphs.json",
}


@dataclass
class GQASample:
    """One GQA question paired with its image path + (optional) scene graph.

    ``scene_graph`` is populated only when a matching ``sceneGraphs/``
    file is present on disk; ``test`` and ``testdev`` splits ship without
    one.  The scene graph is what the IoU / scene-graph eval metrics in
    ``vlm_wrapper/eval/metrics.py`` score against.
    """

    split: str
    question_id: str
    image_id: str
    image_path: Path
    question: str
    answer: str | None = None
    full_answer: str | None = None
    semantic: list[dict[str, Any]] | None = field(default=None, repr=False)
    semantic_str: str | None = None
    types: dict[str, Any] | None = field(default=None, repr=False)
    question_family: str | None = None
    is_balanced: bool = True
    scene_graph: dict[str, Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "question_id": self.question_id,
            "image_id": self.image_id,
            "image_path": str(self.image_path),
            "question": self.question,
            "answer": self.answer,
            "full_answer": self.full_answer,
            "semantic_str": self.semantic_str,
            "types": dict(self.types) if self.types else None,
            "question_family": self.question_family,
            "is_balanced": self.is_balanced,
        }


# ======================================================================
# Disk layout helpers
# ======================================================================

def default_gqa_root(workspace_root: str | Path | None = None) -> Path:
    """Return the canonical GQA root on this workspace (``data/GQA``)."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[2]
    return Path(workspace_root) / "data" / "GQA"


def _questions_path(gqa_root: Path, split: str) -> Path:
    fname = _QUESTION_FILES.get(split)
    if fname is None:
        raise ValueError(
            f"split must be one of {list(_QUESTION_FILES)}, got {split!r}"
        )
    return gqa_root / "questions" / fname


def _scene_graph_path(gqa_root: Path, split: str) -> Path | None:
    fname = _SCENE_GRAPH_FILES.get(split)
    if fname is None:
        return None
    p = gqa_root / "sceneGraphs" / fname
    return p if p.exists() else None


def _image_path(gqa_root: Path, image_id: str) -> Path:
    return gqa_root / "images" / f"{image_id}.jpg"


# ======================================================================
# Loaders
# ======================================================================

_SCENE_GRAPH_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _load_scene_graphs(
    gqa_root: Path, split: str,
) -> dict[str, Any]:
    """Load (and cache) the per-split sceneGraphs JSON.

    Each scene-graph file is ~250-700 MB.  We keep one copy per (root,
    split) in the module-level cache so repeated ``iter_gqa_samples``
    calls don't re-parse it.
    """
    key = (str(gqa_root), split)
    cached = _SCENE_GRAPH_CACHE.get(key)
    if cached is not None:
        return cached
    sg_path = _scene_graph_path(gqa_root, split)
    if sg_path is None:
        _SCENE_GRAPH_CACHE[key] = {}
        return {}
    with sg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    _SCENE_GRAPH_CACHE[key] = data
    return data


def load_gqa_questions(
    split: str = "val",
    *,
    gqa_root: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the raw GQA question dict for a split.

    GQA questions are stored as ``{question_id: question_dict}`` JSON
    objects, not arrays — we keep that shape so callers can look up by
    id without scanning.
    """
    root = Path(gqa_root) if gqa_root else default_gqa_root()
    qpath = _questions_path(root, split)
    if not qpath.exists():
        raise FileNotFoundError(
            f"GQA questions not found at {qpath}. Download per "
            f"install/INSTALL_BENCHMARKS.md §4 (GQA section): "
            f"questions1.2.zip + sceneGraphs.zip + images.zip."
        )
    with qpath.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expected GQA questions to be a dict, got {type(payload).__name__}"
        )
    if limit is not None:
        # dict insertion order is preserved (CPython 3.7+); take the
        # first ``limit`` entries.
        items = list(payload.items())[:limit]
        return dict(items)
    return payload


def iter_gqa_samples(
    split: str = "val",
    *,
    gqa_root: str | Path | None = None,
    limit: int | None = None,
    question_types: Iterable[str] | None = None,
    attach_scene_graph: bool = True,
) -> Iterator[GQASample]:
    """Yield ``GQASample`` objects for a split.

    Parameters
    ----------
    split : str
        ``train`` / ``val`` / ``testdev`` / ``test``.
    gqa_root : path, optional
        Override the default ``data/GQA`` path.
    limit : int, optional
        Cap the iteration count for smoke tests.
    question_types : iterable of str, optional
        Restrict to questions whose ``types["semantic"]`` is in this set
        (e.g. ``{"obj", "rel", "attr", "global", "cat"}``).  See the GQA
        paper §3.3.
    attach_scene_graph : bool
        When True (default), look up the matching scene graph for each
        sample.  Only affects ``train`` / ``val`` since the test splits
        do not ship scene graphs.
    """
    root = Path(gqa_root) if gqa_root else default_gqa_root()
    type_set = set(question_types) if question_types else None

    questions = load_gqa_questions(split, gqa_root=root, limit=None)
    scene_graphs = (
        _load_scene_graphs(root, split) if attach_scene_graph else {}
    )

    count = 0
    for qid, q in questions.items():
        types = q.get("types") or {}
        sem_type = types.get("semantic") if isinstance(types, dict) else None
        if type_set is not None and sem_type not in type_set:
            continue

        image_id = str(q.get("imageId", ""))
        sample = GQASample(
            split=split,
            question_id=str(qid),
            image_id=image_id,
            image_path=_image_path(root, image_id),
            question=q.get("question", ""),
            answer=q.get("answer"),
            full_answer=q.get("fullAnswer"),
            semantic=q.get("semantic"),
            semantic_str=q.get("semanticStr"),
            types=types if isinstance(types, dict) else None,
            question_family=q.get("questionFamily"),
            is_balanced=bool(q.get("isBalanced", True)),
            scene_graph=scene_graphs.get(image_id),
        )
        yield sample
        count += 1
        if limit is not None and count >= limit:
            return


def load_gqa_image(sample: GQASample) -> Image.Image:
    """Load the JPEG for a ``GQASample`` as an RGB ``PIL.Image``."""
    if not sample.image_path.exists():
        raise FileNotFoundError(
            f"GQA image missing: {sample.image_path}.  Make sure "
            f"images.zip was unpacked into data/GQA/images/."
        )
    return Image.open(sample.image_path).convert("RGB")


# ======================================================================
# GPT-4o parser
# ======================================================================

def parse_gqa_sample(
    sample: GQASample,
    *,
    image: Image.Image | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_entities: int = 20,
    max_rounds: int = 4,
) -> dict[str, Any]:
    """Run the GPT-4o multi-hop parser on one GQA question.

    Mirrors ``parse_clevr_sample`` — delegates to ``cascaded_ground`` with
    ``domain="image_qa"`` so the schema includes ``<answer>`` and the
    VLM has access to GroundingDINO + the visual tool registry
    (``grounded_detect`` for open-vocabulary entity grounding,
    ``spatial_query`` for relations, ``describe_region`` for attributes).

    Returns a dict whose top-level keys mirror the CLEVR parser so the
    eval harness can treat both benchmarks uniformly.
    """
    if image is None:
        image = load_gqa_image(sample)

    task_id = f"gqa.{sample.split}.{sample.question_id}"
    # GQA answers are short (one token, usually a noun / yes-no /
    # attribute / count digit).  Mirror the CLEVR guidance and add the
    # GQA-specific scene-graph constraints so the schema is comparable
    # to the gold ``sceneGraphs/`` annotation that ``eval/metrics.py``
    # IoU-scores against.
    sem_type = (sample.types or {}).get("semantic") if sample.types else None
    structural = (sample.types or {}).get("structural") if sample.types else None
    type_hint = ""
    if sem_type or structural:
        type_hint = (
            f"\nThis question is GQA semantic-type={sem_type or '?'}, "
            f"structural-type={structural or '?'}."
        )

    goal_with_guidance = (
        f"{sample.question}{type_hint}\n"
        "Inside <targets>, set `target=` to the entity ID of the "
        "object the question is asking ABOUT (the answer referent), and "
        "`candidate_set=[e1,e2,...]` to the entity IDs you considered "
        "(NEVER put raw nouns / attributes / yes-no tokens there).  Use "
        "`grounded_detect` to ground each named object in the question "
        "with a real bbox; record those calls in <evidence>.  Each "
        "<evidence> hop must declare `abstract_op=` (GROUND/CHECK/"
        "RETRIEVE/CONCLUDE) and `tool=` of the actual tool you called.  "
        "In <answer>, `answer=` must be a single GQA token (an object "
        "name, attribute, colour, count digit, or yes/no) — NOT a full "
        "sentence; ``fullAnswer`` style sentences belong outside the "
        "schema.  In <state_flags>, set `scene_type=image_qa`, leave "
        "`progress=null`, `phase=null`, `dialog_open=false`, "
        "`input_pending=false`.  `pos=` must be `x,y,w,h` pixel ints or "
        "`null` — no parens / brackets.  Pixel coordinates use the "
        "image's NATIVE resolution."
    )

    req = GroundingRequest(
        images=image,
        goal=goal_with_guidance,
        domain="image_qa",
        output_mode="answer",
        task_id=task_id,
        step=0,
        context={
            "question": sample.question,
            "image_id": sample.image_id,
            "semantic_type": sem_type,
            "structural_type": structural,
            "semantic_str": sample.semantic_str,
        },
        max_entities=max_entities,
        max_rounds=max_rounds,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )

    result = cascaded_ground(req, image_size=image.size)

    predicted = (result.answer or "").strip()
    gt = (sample.answer or "").strip() if sample.answer is not None else None
    correct: bool | None = None
    if gt is not None and predicted:
        correct = predicted.lower() == gt.lower()

    return {
        "schema": result.schema,
        "answer": predicted or None,
        "ground_truth": gt,
        "correct": correct,
        "tool_trace": result.tool_trace,
        "rounds": result.rounds,
        "model": result.model,
        "warnings": result.warnings,
        "validation": result.validation.as_dict() if result.validation else None,
        "head_used": result.head_used,
        "escalation_trace": result.escalation_trace,
        "scene_graph": sample.scene_graph,
        "sample": sample.to_dict(),
    }


# ======================================================================
# Batch helper
# ======================================================================

def parse_gqa_batch(
    samples: Iterable[GQASample],
    *,
    output_jsonl: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_entities: int = 20,
    max_rounds: int = 4,
    temperature: float | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Run ``parse_gqa_sample`` over a stream of samples.

    Mirrors ``parse_clevr_batch`` — appends to ``output_jsonl`` after
    every sample so long runs survive interruption.
    """
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
                out = parse_gqa_sample(
                    sample,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    max_entities=max_entities,
                    max_rounds=max_rounds,
                    temperature=temperature,
                )
            except Exception as exc:
                logger.warning("GQA sample %s failed: %s", sample.question_id, exc)
                out = {
                    "error": str(exc),
                    "sample": sample.to_dict(),
                }

            results.append(out)
            if fh is not None:
                # Drop the (very large) scene_graph before serialising —
                # it's already on disk and would inflate the JSONL by
                # ~10×.  Callers that need it can re-load via
                # ``iter_gqa_samples``.
                serial = {k: v for k, v in out.items() if k != "scene_graph"}
                fh.write(json.dumps(serial, ensure_ascii=False) + "\n")
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
                    "[GQA %s] %d: q=%s pred=%r gt=%r",
                    tag, i, sample.question_id,
                    out.get("answer"), out.get("ground_truth"),
                )
    finally:
        if fh is not None:
            fh.close()

    return results
