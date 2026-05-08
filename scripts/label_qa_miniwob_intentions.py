"""Dual-axis intention labeling for QA + miniwob + webshop benchmarks.

Mirrors the pipeline used for gymv (``labeling/label_intentions_gpt54.py``)
but adapted for:

  * QA benchmarks (video_holmes, siv_bench, tir_bench, visual_toolbench)
    -- one ``samples.jsonl`` per (bench, model) combination.  Only
    ``correct=True`` samples get labelled.
  * miniwob (browsergym) -- one ``rollouts.jsonl`` per task containing
    ``experiences[]`` with the cold-start LLM's natural-language
    ``intentions`` already attached.  Each step gets re-tagged.
  * webshop (browsergym) -- same ``rollouts.jsonl`` shape as miniwob
    but ``intentions`` is usually ``null`` in cold-start; the per-step
    context (state/action/scene) is enough for the LLM to assign a
    dual-axis tag.  Tasks live as ``webshop.<idx>/`` subdirs of each
    ``webshop_50task_<tag>/`` model-specific root.

For every sample/step we call gpt-5.4 (via OpenRouter) with a tight
system prompt + the question/state context and obtain a JSON object
``{"operator": ..., "subgoal": ..., "note": ...}``.  The fields are
written back onto the original sample dict so subsequent stages
(skill-bank build, skill_query) can index them directly.

Reuses the operator/subgoal vocabulary + few-shot calibration from
``labeling/label_intentions_gpt54.py`` to keep the dual-axis taxonomy
unified across all SFT corpora.

Usage::

    python scripts/label_qa_miniwob_intentions.py \\
        --source video_holmes \\
        --inputs Cold-start-out-visual-reasoning-video/video_holmes/samples.jsonl \\
                 openrouter-transfer-baselines-out/2026-05-01_08-06-44/{claude,gemini,qwen}/vr_video/video_holmes/samples.jsonl \\
        --output-dir labeling/qa_miniwob_labeled/run_<ts>/video_holmes \\
        --workers 16

WebShop example::

    # --inputs is one webshop_50task_<tag>/ directory per model-tag.  Each
    # directory holds ``webshop.<idx>/rollouts.jsonl``; the script writes
    # one labeled file per task under
    # ``<output-dir>/webshop.<idx>/rollouts.jsonl``.
    python scripts/label_qa_miniwob_intentions.py \\
        --source webshop \\
        --inputs Cold-start-out-browsergym/webshop_50task_low \\
        --output-dir labeling/qa_miniwob_labeled/run_<ts>/webshop/gpt-5.4 \\
        --source-model-tag gpt-5.4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
for p in [str(WORKSPACE_ROOT), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Pull in API keys (mirrors the pattern used by labeling/label_intentions_gpt54.py)
try:
    import api_keys as _ak  # type: ignore
    if getattr(_ak, "openrouter_api_key", "") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = _ak.openrouter_api_key  # type: ignore
    if getattr(_ak, "openai_api_key", "") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = _ak.openai_api_key  # type: ignore
except Exception:  # pragma: no cover
    pass

# Reuse the calibrated vocab + normalisation from the gymv labeler.
from labeling.label_intentions_gpt54 import (  # noqa: E402
    _OPERATOR_DEFINITIONS,
    _SUBGOAL_DEFINITIONS,
    _build_vocab_block,
    _extract_json_object,
    _normalize_dual_tag,
    _trim_note,
)

logger = logging.getLogger("label_qa_miniwob")

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_WORKERS = 16
NOTE_BUDGET = 32  # words


# ---------------------------------------------------------------------------
# Source kind detection
# ---------------------------------------------------------------------------
_QA_SOURCES = ("video_holmes", "siv_bench", "tir_bench", "visual_toolbench")
_MINIWOB_SOURCE = "miniwob"
_WEBSHOP_SOURCE = "webshop"
# Browser sources share the same per-step rollout schema and use the same
# ``_process_miniwob_file`` driver below.  They differ only in the task
# directory glob (``miniwob.*`` vs ``webshop.*``) and the few-shot block
# we tilt the prompt with.
_BROWSER_SOURCES = (_MINIWOB_SOURCE, _WEBSHOP_SOURCE)
_BROWSER_TASK_GLOB: Dict[str, str] = {
    _MINIWOB_SOURCE: "miniwob.*",
    _WEBSHOP_SOURCE: "webshop.*",
}


# ---------------------------------------------------------------------------
# OpenRouter / OpenAI client
# ---------------------------------------------------------------------------
def _get_openai_client():
    """Lazy-init an OpenAI-compatible client routed through OpenRouter."""
    from openai import OpenAI  # type: ignore

    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You categorise ONE reasoning trace along TWO independent axes:\n"
    "  - operator (cognitive mode, 6 choices)\n"
    "  - subgoal  (achievement, 14 choices)\n"
    "The operator says HOW the agent is thinking; the subgoal says WHAT "
    "is being achieved. They are orthogonal — the same answer can come "
    "from very different (operator, subgoal) pairs depending on context.\n"
    "Reply ONLY a JSON object with three keys: operator, subgoal, note. "
    "The note should be one sentence (<= 30 words) that summarises the "
    "concrete reasoning step."
)

_QA_FEWSHOT: List[Dict[str, str]] = [
    {  # social/intent inference (siv_bench)
        "context": "MCQ; question='What does e2 intend by raising her hand?'; "
                   "options='A:greet B:stop C:vote'; answer=B; "
                   "reasoning='e2's hand is up at chest height with palm out, "
                   "matching a stop gesture rather than a wave.'",
        "operator": "INSPECT",
        "subgoal":  "EXPLORE",
        "note": "Read the gesture's shape and palm orientation to infer a stop intent.",
    },
    {  # spatial counting (tir_bench)
        "context": "MCQ; question='How many red apples are visible?'; "
                   "options='A:2 B:3 C:4'; answer=B; "
                   "reasoning='Three red apples visible: top-left, center, bottom-right.'",
        "operator": "COMMIT",
        "subgoal":  "EXECUTE",
        "note": "Count red apples by enumerating their positions in the scene.",
    },
    {  # cause inference (video_holmes)
        "context": "open-ended; question='Why does the man complain?'; answer="
                   "'food is cold'; reasoning='The man pushes the plate away after "
                   "touching it; the steam visible earlier is gone.'",
        "operator": "COMMIT",
        "subgoal":  "EXPLORE",
        "note": "Trace the temporal cue (steam fading) to identify the cause of complaint.",
    },
    {  # rotation task (tir_bench)
        "context": "MCQ; question='Rotate the train image to upright'; "
                   "options='A:0 B:90cw C:180 D:270cw'; answer=B; "
                   "reasoning='Tracks at right side; rotating 90 cw makes them lie horizontally.'",
        "operator": "COMPARE",
        "subgoal":  "POSITION",
        "note": "Compare each rotation option against the target upright orientation.",
    },
    {  # OCR + arithmetic (visual_toolbench)
        "context": "open-ended; question='How many rebars total?'; answer='6'; "
                   "reasoning='Text reads 3 top + 3 bottom rebars in the image.'",
        "operator": "COMMIT",
        "subgoal":  "EXECUTE",
        "note": "Read the OCR text and sum the rebar counts.",
    },
    {  # miniwob click target
        "context": "miniwob/click-button; goal='Click the OK button.'; "
                   "state='button bid=42 text=OK'; "
                   "intentions='Click bid 42 to dismiss.'",
        "operator": "COMMIT",
        "subgoal":  "EXECUTE",
        "note": "Issue a click on bid 42 to satisfy the goal.",
    },
    {  # miniwob form filling
        "context": "miniwob/form; goal='Enter john@example.com in email field.'; "
                   "state='input bid=11 placeholder=email'; "
                   "intentions='Type the email into bid 11.'",
        "operator": "COMMIT",
        "subgoal":  "BUILD",
        "note": "Type the literal email value into the email input field.",
    },
    {  # miniwob navigation / tab
        "context": "miniwob/nav; goal='Open settings tab.'; "
                   "state='tab bid=8 label=Settings'; "
                   "intentions='Switch to Settings tab.'",
        "operator": "COMMIT",
        "subgoal":  "NAVIGATE",
        "note": "Click the Settings tab to bring its panel into view.",
    },
    {  # webshop search query (compose+execute)
        "context": "webshop.0; goal='blue toothbrushes under $60'; "
                   "action='fill(\"18\", \"blue toothbrush\")'; "
                   "state='search input bid=18 focused; Search button bid=20'",
        "operator": "COMMIT",
        "subgoal":  "BUILD",
        "note": "Compose the search query that captures color+item from the goal.",
    },
    {  # webshop result inspection / compare
        "context": "webshop.0; goal='blue toothbrushes under $60'; "
                   "action='click(\"47\")'; "
                   "state='results page; B09JT3Z6JV blue dinosaur $10.95'",
        "operator": "COMPARE",
        "subgoal":  "EXPLORE",
        "note": "Pick the result that matches the color and price constraints.",
    },
    {  # webshop final commit (Buy Now)
        "context": "webshop.0; goal='blue toothbrushes under $60'; "
                   "action='click(\"55\")'; "
                   "state='item page Buy Now bid=55 price $10.95'",
        "operator": "COMMIT",
        "subgoal":  "EXECUTE",
        "note": "Press Buy Now once the item satisfies the constraints.",
    },
]


def _build_user_prompt(context_block: str) -> str:
    """User prompt: vocab + few-shots + the actual context to label."""
    vocab = _build_vocab_block()
    shots = []
    for ex in _QA_FEWSHOT:
        shots.append(
            "CONTEXT:\n"
            + ex["context"]
            + "\nLABEL: "
            + json.dumps({
                "operator": ex["operator"],
                "subgoal":  ex["subgoal"],
                "note":     ex["note"],
            })
        )
    fewshot_block = "\n\n".join(shots)
    return (
        f"{vocab}\n\n"
        f"Few-shot examples:\n{fewshot_block}\n\n"
        f"--- LABEL THIS ---\n"
        f"CONTEXT:\n{context_block}\n"
        f"LABEL (JSON only):"
    )


# ---------------------------------------------------------------------------
# Per-source context builders
# ---------------------------------------------------------------------------
def _qa_context(sample: Dict[str, Any]) -> str:
    """Compact, prompt-friendly context block for a QA sample."""
    is_mcq = bool(sample.get("is_mcq"))
    q = (sample.get("question") or "").strip()
    opts = (sample.get("options_block") or "").strip()
    ans = (sample.get("gold_answer") or sample.get("answer") or "").strip()
    reasoning = (sample.get("answer_reasoning") or "").strip()
    bench = (sample.get("benchmark") or "").strip()
    dim = (
        sample.get("dimension")
        or sample.get("question_type")
        or sample.get("task_type")
        or sample.get("category")
        or ""
    )
    schema = (sample.get("schema") or "").strip()

    parts = [
        f"benchmark={bench}",
        f"kind={'MCQ' if is_mcq else 'open-ended'}",
    ]
    if dim:
        parts.append(f"dimension={dim}")
    parts.append(f"question='{q[:250]}'")
    if is_mcq and opts:
        parts.append(f"options='{opts[:300].replace(chr(10), ' / ')}'")
    if ans:
        parts.append(f"answer='{ans[:120]}'")
    if reasoning:
        parts.append(f"reasoning='{reasoning[:400]}'")
    if schema:
        parts.append(f"scene='{schema[:300]}'")
    return "; ".join(parts)


def _miniwob_context(step: Dict[str, Any], *, goal: str = "") -> str:
    """Compact context block for a browsergym step (miniwob or webshop).

    Both sources share the same on-disk schema (Cold-start /
    BrowserGym ``Experience`` shape).  The few-shot block leans on
    ``goal`` + ``action`` + ``state`` + ``scene`` so the LLM can pick a
    dual-axis tag even when ``intentions`` is empty (the common case
    for webshop, where cold-start did not record per-step natural-language
    intentions).
    """
    state = (step.get("state") or "").strip()
    sched = (step.get("metadata") or {}).get("schema") or ""
    schema = sched.strip()
    action = (step.get("action") or "").strip()
    intent = (step.get("intentions") or step.get("subgoal") or "").strip()
    iface = step.get("interface") or {}
    if isinstance(iface, str):
        try:
            iface = json.loads(iface)
        except Exception:
            iface = {}
    task = iface.get("game_name") or iface.get("env_name") or ""

    parts = [
        f"task={task}",
        f"goal='{goal[:200]}'",
        f"action='{action[:200]}'",
        f"state='{state[:200]}'",
    ]
    if schema:
        # Strip XML wrappers and trim
        s = re.sub(r"<state>|</state>", "", schema).strip()
        parts.append(f"scene='{s[:400]}'")
    if intent:
        parts.append(f"intentions='{intent[:280]}'")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# LLM call + normalisation
# ---------------------------------------------------------------------------
def _call_llm(client: Any, *, model: str, context: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
    user = _build_user_prompt(context)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user},
                ],
                temperature=0.0 if attempt == 0 else 0.2,
                max_tokens=200,
            )
            text = resp.choices[0].message.content or ""
            obj = _extract_json_object(text)
            if obj is not None:
                return obj
        except Exception as exc:  # pragma: no cover
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    if last_err is not None:
        logger.debug("LLM call failed: %s", last_err)
    return None


def _label_one(client: Any, model: str, context: str) -> Tuple[str, str, str, str]:
    """Return (operator, subgoal, note, source_tag).

    ``source_tag`` is one of {"llm", "fallback"}.
    """
    obj = _call_llm(client, model=model, context=context)
    if obj is None:
        return ("COMMIT", "EXECUTE", "", "fallback")
    op, sg = _normalize_dual_tag(obj)
    note = _trim_note(str(obj.get("note") or ""))
    return (op, sg, note, "llm")


# ---------------------------------------------------------------------------
# Stage drivers
# ---------------------------------------------------------------------------
def _process_qa_file(
    in_path: Path,
    out_path: Path,
    *,
    model: str,
    workers: int,
    max_samples: Optional[int],
) -> Dict[str, Any]:
    """Label one QA samples.jsonl.  Only correct samples receive intentions."""
    if not in_path.is_file():
        return {"input": str(in_path), "n_in": 0, "skipped": "missing"}

    samples: List[Dict[str, Any]] = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception:
                continue

    n_in = len(samples)
    targets: List[Tuple[int, str]] = []  # (idx, context)
    for i, s in enumerate(samples):
        if not s.get("correct"):
            continue
        targets.append((i, _qa_context(s)))
        if max_samples is not None and len(targets) >= max_samples:
            break

    n_targets = len(targets)
    if n_targets == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        return {"input": str(in_path), "n_in": n_in, "n_correct": 0,
                "n_labeled": 0, "elapsed_s": 0.0}

    client = _get_openai_client()
    started = time.time()
    n_llm = n_fb = 0
    op_counter: Dict[str, int] = {}
    sg_counter: Dict[str, int] = {}

    def task(item):
        idx, ctx = item
        return idx, _label_one(client, model, ctx)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, (op, sg, note, src) in pool.map(task, targets):
            samples[idx]["intention_operator"] = op
            samples[idx]["intention_subgoal"]  = sg
            samples[idx]["intention_note"]     = note
            display = f"[{op}/{sg}] {note}".strip()
            samples[idx]["intentions"] = display
            samples[idx]["intention_source"] = src
            if src == "llm":
                n_llm += 1
            else:
                n_fb += 1
            op_counter[op] = op_counter.get(op, 0) + 1
            sg_counter[sg] = sg_counter.get(sg, 0) + 1

    elapsed = time.time() - started

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    return {
        "input": str(in_path),
        "output": str(out_path),
        "n_in": n_in,
        "n_correct": n_targets,
        "n_labeled": n_targets,
        "n_llm": n_llm,
        "n_fallback": n_fb,
        "elapsed_s": round(elapsed, 1),
        "op_top": dict(sorted(op_counter.items(), key=lambda kv: -kv[1])[:5]),
        "sg_top": dict(sorted(sg_counter.items(), key=lambda kv: -kv[1])[:5]),
    }


def _process_miniwob_file(
    in_path: Path,
    out_path: Path,
    *,
    model: str,
    workers: int,
    max_steps: Optional[int],
) -> Dict[str, Any]:
    """Label one miniwob rollouts.jsonl (one episode per line).

    Each step's existing ``intentions`` text becomes context for the
    dual-axis tagger.
    """
    if not in_path.is_file():
        return {"input": str(in_path), "skipped": "missing"}

    episodes: List[Dict[str, Any]] = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except Exception:
                continue

    n_in_steps = sum(len(ep.get("experiences") or []) for ep in episodes)
    targets: List[Tuple[int, int, str]] = []  # (ep_idx, step_idx, context)
    for ei, ep in enumerate(episodes):
        goal = (ep.get("query") or ep.get("task") or "")
        if isinstance(goal, dict):
            goal = goal.get("goal") or json.dumps(goal)
        goal = str(goal or "")
        for si, step in enumerate(ep.get("experiences") or []):
            targets.append((ei, si, _miniwob_context(step, goal=goal)))
            if max_steps is not None and len(targets) >= max_steps:
                break
        if max_steps is not None and len(targets) >= max_steps:
            break

    n_targets = len(targets)
    if n_targets == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")
        return {"input": str(in_path), "n_episodes": len(episodes),
                "n_in_steps": n_in_steps, "n_labeled": 0, "elapsed_s": 0.0}

    client = _get_openai_client()
    started = time.time()
    n_llm = n_fb = 0
    op_counter: Dict[str, int] = {}
    sg_counter: Dict[str, int] = {}

    def task(item):
        ei, si, ctx = item
        return ei, si, _label_one(client, model, ctx)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ei, si, (op, sg, note, src) in pool.map(task, targets):
            step = episodes[ei]["experiences"][si]
            step["intention_operator"] = op
            step["intention_subgoal"]  = sg
            step["intention_note"]     = note
            display = f"[{op}/{sg}] {note}".strip()
            # Preserve original intentions for fall-through but also
            # replace the field with the canonical dual-axis form.
            if step.get("intentions") and not step.get("intentions_raw"):
                step["intentions_raw"] = step["intentions"]
            step["intentions"] = display
            step["intention_source"] = src
            if src == "llm":
                n_llm += 1
            else:
                n_fb += 1
            op_counter[op] = op_counter.get(op, 0) + 1
            sg_counter[sg] = sg_counter.get(sg, 0) + 1

    elapsed = time.time() - started

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    return {
        "input": str(in_path),
        "output": str(out_path),
        "n_episodes": len(episodes),
        "n_in_steps": n_in_steps,
        "n_labeled": n_targets,
        "n_llm": n_llm,
        "n_fallback": n_fb,
        "elapsed_s": round(elapsed, 1),
        "op_top": dict(sorted(op_counter.items(), key=lambda kv: -kv[1])[:5]),
        "sg_top": dict(sorted(sg_counter.items(), key=lambda kv: -kv[1])[:5]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    choices=list(_QA_SOURCES) + list(_BROWSER_SOURCES),
                    help="Source kind: which input layout to expect.")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more input paths.  For QA: samples.jsonl files. "
                         "For miniwob: directories containing per-task "
                         "``miniwob.*/rollouts.jsonl`` or specific rollouts.jsonl. "
                         "For webshop: directories containing per-task "
                         "``webshop.*/rollouts.jsonl`` (typically one "
                         "``webshop_50task_<tag>/`` per model) or specific "
                         "rollouts.jsonl files.")
    ap.add_argument("--output-dir", required=True,
                    help="Output dir; one labeled file per input.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"OpenRouter / OpenAI model (default {DEFAULT_MODEL}).")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"Concurrent LLM calls per file (default {DEFAULT_WORKERS}).")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap labeled correct samples per QA file.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Cap labeled steps per browsergym (miniwob/webshop) file.")
    ap.add_argument("--source-model-tag", default=None,
                    help="Optional source_model label written onto each row "
                         "for downstream SFT row attribution.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect (input_path, output_path) pairs.
    work: List[Tuple[Path, Path]] = []
    for raw in args.inputs:
        p = Path(raw)
        if not p.exists():
            logger.warning("input missing: %s", p)
            continue
        if args.source in _BROWSER_SOURCES:
            # Either a dir of <task_glob> subfolders (each with rollouts.jsonl)
            # or a specific rollouts.jsonl.
            task_glob = _BROWSER_TASK_GLOB[args.source]
            if p.is_dir():
                for ep_root in sorted(p.glob(task_glob)):
                    rj = ep_root / "rollouts.jsonl"
                    if rj.is_file():
                        work.append((rj, out_root / ep_root.name / "rollouts.jsonl"))
            elif p.suffix == ".jsonl":
                # Place under a subdir derived from the parent task folder.
                task_name = p.parent.name
                work.append((p, out_root / task_name / "rollouts.jsonl"))
        else:
            # QA: input is a samples.jsonl; preserve the model-source folder
            # structure so multiple teachers can sit side-by-side.
            tag = args.source_model_tag or _detect_model_tag(p)
            work.append((p, out_root / tag / "samples.jsonl"))

    if not work:
        logger.error("No matching inputs.")
        return 2

    summary: List[Dict[str, Any]] = []
    started_all = time.time()
    for in_path, out_path in work:
        logger.info(">> labeling %s -> %s", in_path, out_path)
        if args.source in _BROWSER_SOURCES:
            res = _process_miniwob_file(
                in_path, out_path,
                model=args.model, workers=args.workers,
                max_steps=args.max_steps,
            )
        else:
            res = _process_qa_file(
                in_path, out_path,
                model=args.model, workers=args.workers,
                max_samples=args.max_samples,
            )
        summary.append(res)
        logger.info("   done: %s", json.dumps({k: v for k, v in res.items()
                                              if k not in ("op_top", "sg_top")}))
        logger.info("   op_top=%s sg_top=%s",
                    res.get("op_top"), res.get("sg_top"))

    elapsed_all = time.time() - started_all
    summary_path = out_root / "_intentions_summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "source": args.source,
            "model": args.model,
            "workers": args.workers,
            "n_files": len(work),
            "elapsed_s": round(elapsed_all, 1),
            "files": summary,
        }, f, indent=2)
    logger.info("=== summary -> %s (elapsed %.1fs) ===", summary_path, elapsed_all)
    return 0


def _detect_model_tag(p: Path) -> str:
    """Heuristic: pick a short label from the input path components."""
    parts = [s.lower() for s in p.parts]
    for s in parts:
        if "claude" in s:
            return "claude"
        if "gemini" in s:
            return "gemini"
        if "qwen" in s:
            return "qwen"
        if "gpt-5.4" in s or "gpt5p4" in s:
            return "gpt-5.4"
    return "default"


if __name__ == "__main__":
    raise SystemExit(main())
