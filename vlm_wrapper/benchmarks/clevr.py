"""CLEVR v1.0 loader + GPT-4o parser.

CLEVR is a synthetic visual reasoning benchmark: simple 3-D scenes of
rubber / metal cubes, spheres, and cylinders, with questions that
require multi-step reasoning over colour, size, shape, material, and
spatial relations. Because the scenes are synthetic the ground truth
is exact (scene graphs in ``scenes/``), which makes CLEVR an ideal
Phase-0 cross-validation target for GPT-4o labels.

Layout expected on disk (relative to ``default_clevr_root``):

```
<root>/CLEVR_v1.0/
    images/{train,val,test}/CLEVR_*_NNNNNN.png
    questions/CLEVR_{train,val,test}_questions.json
    scenes/CLEVR_{train,val}_scenes.json
```

Usage::

    from vlm_wrapper.benchmarks.clevr import iter_clevr_samples, parse_clevr_sample

    for sample in iter_clevr_samples(split="val", limit=5):
        out = parse_clevr_sample(sample, model="gpt-4o", api_key=KEY)
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

from ..ground import GroundingRequest, ground

logger = logging.getLogger(__name__)

_QUESTION_FILES: dict[str, str] = {
    "train": "CLEVR_train_questions.json",
    "val": "CLEVR_val_questions.json",
    "test": "CLEVR_test_questions.json",
}


@dataclass
class CLEVRSample:
    """One CLEVR question paired with its image metadata.

    ``answer`` is present for train/val, ``None`` for the held-out
    ``test`` split. ``program`` contains the symbolic reasoning program
    used to generate the question (also absent on test).
    """

    split: str
    image_index: int
    image_filename: str
    image_path: Path
    question: str
    answer: str | None = None
    question_family_index: int | None = None
    program: list[dict[str, Any]] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "image_index": self.image_index,
            "image_filename": self.image_filename,
            "image_path": str(self.image_path),
            "question": self.question,
            "answer": self.answer,
            "question_family_index": self.question_family_index,
        }


# ======================================================================
# Disk layout helpers
# ======================================================================

def default_clevr_root(
    workspace_root: str | Path | None = None,
) -> Path:
    """Return the canonical CLEVR root on this workspace.

    We expect ``data/CLEVR/CLEVR_v1.0`` relative to the repository root.
    ``workspace_root`` may be overridden to support alternative layouts
    (useful for tests or when running from outside the repo).
    """
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[2]
    return Path(workspace_root) / "data" / "CLEVR" / "CLEVR_v1.0"


def _questions_path(clevr_root: Path, split: str) -> Path:
    fname = _QUESTION_FILES.get(split)
    if fname is None:
        raise ValueError(
            f"split must be one of {list(_QUESTION_FILES)}, got {split!r}"
        )
    return clevr_root / "questions" / fname


def _image_path(clevr_root: Path, split: str, image_filename: str) -> Path:
    return clevr_root / "images" / split / image_filename


# ======================================================================
# Loaders
# ======================================================================

def load_clevr_questions(
    split: str = "val",
    *,
    clevr_root: str | Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load raw CLEVR question dicts for a split.

    The train file is ~700 MB and contains 699 989 questions — prefer
    ``iter_clevr_samples`` for streaming. ``limit`` lets you cap the
    load for smoke-testing.
    """
    root = Path(clevr_root) if clevr_root else default_clevr_root()
    qpath = _questions_path(root, split)
    if not qpath.exists():
        raise FileNotFoundError(
            f"CLEVR questions not found at {qpath}. Download the dataset "
            f"per install/INSTALL_BENCHMARKS.md §4."
        )
    with qpath.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    questions = payload.get("questions", [])
    if limit is not None:
        questions = questions[:limit]
    return questions


def iter_clevr_samples(
    split: str = "val",
    *,
    clevr_root: str | Path | None = None,
    limit: int | None = None,
    question_family_filter: Iterable[int] | None = None,
) -> Iterator[CLEVRSample]:
    """Yield ``CLEVRSample`` objects one at a time.

    Loads the whole question JSON once (stdlib ``json`` has no streaming
    parser) but yields ``CLEVRSample`` dataclasses so downstream code can
    stop iterating at any point without paying for materialising every
    row into RAM.

    Parameters
    ----------
    split : str
        ``"train"`` / ``"val"`` / ``"test"``.
    clevr_root : path, optional
        Override the default ``data/CLEVR/CLEVR_v1.0`` path.
    limit : int, optional
        Stop after this many samples.
    question_family_filter : iterable of int, optional
        Only yield questions whose ``question_family_index`` is in this
        set. Useful for drilling into a specific reasoning category.
    """
    root = Path(clevr_root) if clevr_root else default_clevr_root()
    family_set = set(question_family_filter) if question_family_filter else None

    questions = load_clevr_questions(split, clevr_root=root, limit=None)

    count = 0
    for q in questions:
        fam = q.get("question_family_index")
        if family_set is not None and fam not in family_set:
            continue
        image_filename = q["image_filename"]
        sample = CLEVRSample(
            split=q.get("split", split),
            image_index=q["image_index"],
            image_filename=image_filename,
            image_path=_image_path(root, split, image_filename),
            question=q["question"],
            answer=q.get("answer"),
            question_family_index=fam,
            program=q.get("program"),
        )
        yield sample
        count += 1
        if limit is not None and count >= limit:
            return


def load_clevr_image(sample: CLEVRSample) -> Image.Image:
    """Load the image for a ``CLEVRSample`` as an RGB ``PIL.Image``."""
    if not sample.image_path.exists():
        raise FileNotFoundError(
            f"CLEVR image missing: {sample.image_path}. The val split has "
            f"15 000 images named CLEVR_val_NNNNNN.png."
        )
    return Image.open(sample.image_path).convert("RGB")


# ======================================================================
# GPT-4o parser
# ======================================================================

def parse_clevr_sample(
    sample: CLEVRSample,
    *,
    image: Image.Image | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_entities: int = 15,
    max_rounds: int = 4,
) -> dict[str, Any]:
    """Run the GPT-4o multi-hop parser on one CLEVR question.

    Delegates to ``vlm_wrapper.ground.ground`` with
    ``domain="image_qa"`` so the schema includes ``<answer>`` and the
    VLM has access to the visual tool registry (detect_objects,
    spatial_query, count_objects, zoom_region, …).

    Parameters
    ----------
    sample : CLEVRSample
        One row from ``iter_clevr_samples``.
    image : PIL.Image, optional
        Pre-loaded image. Avoids re-opening the PNG when running in a
        loop. If omitted, the image is loaded lazily.
    model : str, optional
        Override the default ``$VLM_LABEL_MODEL`` / ``gpt-4o``.
    api_key, base_url, temperature :
        Pass-through overrides for the OpenAI client.
    max_entities : int
        Entity cap for ``<entities>`` — CLEVR scenes have 3–10 objects.
    max_rounds : int
        Max tool-calling rounds.

    Returns
    -------
    dict with keys:
        ``schema``             – raw ``<state>…</state>`` text
        ``answer``             – GPT-4o's predicted answer (or ``None``)
        ``ground_truth``       – ``sample.answer``
        ``correct``            – bool (case-insensitive, stripped)
        ``tool_trace``         – list of tool calls from the loop
        ``rounds``             – number of VLM rounds consumed
        ``model``              – model string used
        ``sample``             – ``sample.to_dict()``
    """
    if image is None:
        image = load_clevr_image(sample)

    task_id = f"clevr.{sample.split}.{sample.image_index}"
    # Append target-semantics guidance so the VLM doesn't pick a random
    # OCR fragment as `target=`.  CLEVR questions always reference one or
    # more shapes by colour/material/size — that's what `target` should be.
    goal_with_guidance = (
        f"{sample.question}\n"
        "Inside <targets>, set `target=` to the entity ID of the shape "
        "the question is asking ABOUT (the answer referent), and "
        "`candidate_set=[...]` to all entities you considered.  Each "
        "<evidence> hop must declare `abstract_op=` (GROUND/CHECK/"
        "RETRIEVE/CONCLUDE) and the actual `tool=` you called."
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
            "image_filename": sample.image_filename,
        },
        max_entities=max_entities,
        max_rounds=max_rounds,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )

    result = ground(req)

    predicted = (result.answer or "").strip()
    gt = (sample.answer or "").strip() if sample.answer is not None else None
    correct = None
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
        "sample": sample.to_dict(),
    }


# ======================================================================
# Batch helper
# ======================================================================

def parse_clevr_batch(
    samples: Iterable[CLEVRSample],
    *,
    output_jsonl: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_entities: int = 15,
    max_rounds: int = 4,
    temperature: float | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Run ``parse_clevr_sample`` over a stream of samples.

    If ``output_jsonl`` is given, each result is appended as a JSON
    line; this lets long runs survive interruption since the file is
    valid after every row.
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
                out = parse_clevr_sample(
                    sample,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    max_entities=max_entities,
                    max_rounds=max_rounds,
                    temperature=temperature,
                )
            except Exception as exc:  # keep the batch going
                logger.warning("CLEVR sample %s failed: %s",
                               sample.image_filename, exc)
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
                    "[CLEVR %s] %d: %s  pred=%r  gt=%r",
                    tag, i, sample.image_filename,
                    out.get("answer"), out.get("ground_truth"),
                )
    finally:
        if fh is not None:
            fh.close()

    return results
