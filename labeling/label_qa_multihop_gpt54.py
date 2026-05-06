#!/usr/bin/env python
"""Multihop CoT decomposition + dual-axis intention labeling for QA / VR.

Companion to :mod:`labeling.label_intentions_gpt54` (which labels
*action-game step level* intentions).  This driver targets the four
visual-reasoning / QA benchmarks where each "sample" is a single Q&A
shot with a free-form ``answer_reasoning`` chain-of-thought:

* ``video_holmes``       — long-form video detective QA (multimodal CoT)
* ``siv_bench``          — short-video instruction-following QA
* ``tir_bench``          — tool-augmented reasoning (reasoning + tool calls)
* ``visual_toolbench``   — visual tool use (frame-pick, OCR, search)

For each sample we ask gpt-5.4 to:

1. Read the question + answer_reasoning (and gold_answer when correct).
2. Decompose the reasoning into 1-N **hops** (≈ atomic reasoning moves).
3. Tag each hop with ``(operator, subgoal, note)`` from the canonical
   QA-extended vocabulary in :mod:`labeling.qa_vocab`:

   - ``operator``  ∈ INTENT_OPERATORS_QA (canonical 6 + REASON, TOOL_USE)
   - ``subgoal``   ∈ UNIFIED_SUBGOALS_QA (canonical 14 + EVIDENCE,
     IDENTIFY, TIMELINE, COUNT, MEASURE, LOOKUP, DEDUCE, RULE_OUT,
     ANSWER, FORM_FILL, SUBMIT)
   - ``note``      ≤ 25 words, concrete reference to evidence/tool/action
4. Mark each hop's ``evidence`` source (frame/text/option/tool/external)
   and (for tir/vtb) detect ``tool_call`` invocations.

Writes per (source, model) pair:

    labeling/qa_multihop_out/run_<ts>/<source>/<model>/samples_with_hops.jsonl

Each output line preserves all original sample fields and adds:

    "hops":   [ {step, operator, subgoal, note, evidence, tool_call?}, ... ]
    "n_hops": int  (length of hops; clamped 1..MAX_HOPS)
    "intentions_summary": "[OP/SG] note ; [OP/SG] note ; ..." for downstream

Also writes ``labeling/qa_multihop_out/run_<ts>/_run_summary.json`` with
per-pair counts, error rates, and operator/subgoal histograms.

Usage
~~~~~

    python -m labeling.label_qa_multihop_gpt54 \\
        --input-run labeling/qa_miniwob_labeled/run_20260506_070722 \\
        --output-dir labeling/qa_multihop_out/run_<ts> \\
        --sources video_holmes siv_bench tir_bench visual_toolbench \\
        --models gpt-5.4 claude gemini qwen \\
        --workers 16

For miniwob (which already carries step-level intentions on
``Experience.intentions``) use the companion driver
``labeling/build_skillbank_qa_gpt54.py`` directly — there is no CoT
decomposition step there.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path / API key bootstrap (mirrors labeling/label_intentions_gpt54.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for p in (CODEBASE_ROOT, WORKSPACE_ROOT):
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

try:
    import api_keys as _ak  # type: ignore
    if getattr(_ak, "openrouter_api_key", "") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = _ak.openrouter_api_key
    if getattr(_ak, "openai_api_key", "") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = _ak.openai_api_key
except Exception:
    pass

from labeling.qa_vocab import (
    INTENT_OPERATORS_QA,
    UNIFIED_SUBGOALS_QA,
    OPERATOR_TO_SUBGOAL_QA,
    normalize_dual_tag_qa,
    normalize_operator_qa,
    normalize_subgoal_qa,
)

try:
    from API_func import ask_model  # type: ignore
except ImportError:
    ask_model = None

logger = logging.getLogger("labeling.label_qa_multihop")

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_WORKERS = 16
LLM_MAX_TOKENS = 1400
LLM_TEMPERATURE = 0.1
NOTE_WORD_BUDGET = 25
MAX_HOPS = 8           # safety upper bound
MIN_HOPS = 1
QUESTION_CHAR_BUDGET = 800
REASONING_CHAR_BUDGET = 4000


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_OPS_LIST = ", ".join(INTENT_OPERATORS_QA)
_SGS_LIST = ", ".join(UNIFIED_SUBGOALS_QA)

_SYSTEM_PROMPT = (
    "You are an expert reasoning-trace analyst.  Given a QA sample and a "
    "candidate model's chain-of-thought, decompose the chain into atomic "
    "REASONING HOPS and tag each hop with a canonical (operator, subgoal) "
    "pair.  You must output strictly valid JSON.  Do not include prose."
)


_FEWSHOT_VIDEO = """\
Example (video_holmes):
  Question: Who was at the diner before the murder?
  Reasoning: "Looking at the timestamps, frame 12 shows Alice at 8pm. \
Frame 19 shows Bob entering at 9pm. The murder happened at 10pm based on \
the clock visible in frame 25, so both were present beforehand."
  Output:
  {"hops":[
    {"step":0,"operator":"INSPECT","subgoal":"EVIDENCE",
     "note":"frame 12 shows Alice at 8pm (timestamp visible)",
     "evidence":"frame_12"},
    {"step":1,"operator":"INSPECT","subgoal":"EVIDENCE",
     "note":"frame 19 shows Bob entering diner at 9pm",
     "evidence":"frame_19"},
    {"step":2,"operator":"INSPECT","subgoal":"TIMELINE",
     "note":"frame 25 clock confirms murder at 10pm",
     "evidence":"frame_25"},
    {"step":3,"operator":"REASON","subgoal":"DEDUCE",
     "note":"both Alice and Bob present before 10pm murder time",
     "evidence":"derived"},
    {"step":4,"operator":"COMMIT","subgoal":"ANSWER",
     "note":"answer: Alice and Bob were at the diner",
     "evidence":"final"}
  ]}
"""

_FEWSHOT_TIR = """\
Example (tir_bench, with tool calls):
  Question: How many seconds between sunrise and sunset on 2024-06-21 in Paris?
  Reasoning: "Need exact times. Calling weather_lookup(date=2024-06-21, \
city=Paris) returns sunrise=05:48, sunset=21:58. Compute 21:58 - 05:48 = \
16h10m = 58200 seconds."
  Output:
  {"hops":[
    {"step":0,"operator":"INSPECT","subgoal":"IDENTIFY",
     "note":"identify needed quantities: sunrise and sunset times",
     "evidence":"question"},
    {"step":1,"operator":"TOOL_USE","subgoal":"LOOKUP",
     "note":"call weather_lookup(date=2024-06-21, city=Paris)",
     "evidence":"tool","tool_call":"weather_lookup"},
    {"step":2,"operator":"INSPECT","subgoal":"EVIDENCE",
     "note":"tool returned sunrise=05:48 and sunset=21:58",
     "evidence":"tool_result"},
    {"step":3,"operator":"REASON","subgoal":"MEASURE",
     "note":"compute 21:58 - 05:48 = 16h10m = 58200 seconds",
     "evidence":"derived"},
    {"step":4,"operator":"COMMIT","subgoal":"ANSWER",
     "note":"answer: 58200 seconds","evidence":"final"}
  ]}
"""


def _build_prompt(sample: Dict[str, Any], source: str) -> str:
    """Construct the per-sample multihop decomposition prompt."""
    q = (sample.get("question") or sample.get("query") or "")[:QUESTION_CHAR_BUDGET]
    reasoning = (sample.get("answer_reasoning") or sample.get("reasoning") or "")
    reasoning = reasoning[:REASONING_CHAR_BUDGET]
    options = sample.get("options_block") or ""
    gold = sample.get("gold_answer") or sample.get("answer") or ""
    answer = sample.get("answer") or sample.get("answer_raw") or ""
    correct = sample.get("correct")
    modality = sample.get("modality") or ""

    # Pick a few-shot block per source so the model sees an apt example.
    if source in ("tir_bench", "visual_toolbench"):
        fewshot = _FEWSHOT_TIR
    else:
        fewshot = _FEWSHOT_VIDEO

    lines = [
        "TASK: Decompose the chain-of-thought below into ATOMIC REASONING HOPS.",
        "Each hop is exactly one cognitive move — looking at evidence, calling",
        "a tool, comparing options, deducing, or committing the final answer.",
        "",
        f"OPERATORS (choose exactly one per hop): {_OPS_LIST}",
        f"SUBGOALS  (choose exactly one per hop): {_SGS_LIST}",
        "",
        "RULES:",
        f"  - Output between {MIN_HOPS} and {MAX_HOPS} hops (most CoTs land 3-6).",
        "  - The LAST hop MUST be operator=COMMIT, subgoal=ANSWER (committing",
        "    the final answer).  If the chain has no clear answer, still",
        "    emit a final COMMIT/ANSWER hop tagged with whatever the model",
        "    actually concluded.",
        "  - Each hop's `note` is ≤ 25 words, concrete, references the",
        "    specific frame / element / tool / fact this hop actually uses.",
        "  - `evidence` ∈ {question, frame_<id>, option_<id>, passage,",
        "    tool, tool_result, derived, final, external}.",
        "  - When the hop is a tool invocation set `tool_call` to the tool",
        "    name (e.g. 'frame_pick', 'ocr', 'search', 'calculator').",
        "    Otherwise omit the field.",
        "  - Use REASON for inference / arithmetic / rule-out / deduction.",
        "  - Use TOOL_USE only when the chain visibly invokes an external",
        "    tool/function (tir / vtb mostly).",
        "  - Use INSPECT/EVIDENCE for grounding claims in frames/text.",
        "  - Use COMPARE for explicit weighing between candidate answers.",
        "  - Use VERIFY ONLY when the chain rechecks a candidate answer,",
        "    not for routine grounding.",
        "",
        fewshot,
        "===== SAMPLE TO DECOMPOSE =====",
        f"Source: {source}    Modality: {modality}",
        f"Question: {q}",
    ]
    if options:
        lines.append(f"Options:\n{options[:600]}")
    lines.append(f"Reasoning chain:\n{reasoning}")
    if answer:
        lines.append(f"Final answer (model said): {str(answer)[:200]}")
    if gold:
        lines.append(f"Gold answer (judge): {str(gold)[:200]}")
    if correct is not None:
        lines.append(f"Judge correct: {bool(correct)}")
    lines.extend([
        "",
        "OUTPUT FORMAT (strict JSON, no prose, no markdown fences):",
        '{"hops":[',
        '  {"step":0,"operator":"<OP>","subgoal":"<SG>","note":"<≤25 words>",',
        '   "evidence":"<source>","tool_call":"<optional>"},',
        '  ...',
        ']}',
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output post-processing
# ---------------------------------------------------------------------------

def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        m = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    return text


def _extract_top_level_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first balanced JSON object in *text*."""
    text = _strip_fence(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # Brute-force: find a balanced { ... } substring (handles trailing chatter).
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    start = -1
    return None


def _trim_note(raw: str) -> str:
    note = (raw or "").strip().strip("\"'`")
    if not note:
        return ""
    words = note.split()
    if len(words) > NOTE_WORD_BUDGET:
        words = words[:NOTE_WORD_BUDGET]
    return " ".join(words).rstrip(",.;:")


_VALID_EVIDENCE = frozenset({
    "question", "passage", "tool", "tool_result", "derived",
    "final", "external", "options",
})


def _clean_evidence(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return "derived"
    if s in _VALID_EVIDENCE:
        return s
    if s.startswith("frame") or s.startswith("clip"):
        return s.replace(" ", "_")
    if s.startswith("option"):
        return "options"
    if s.startswith("text") or s.startswith("doc") or s.startswith("passage"):
        return "passage"
    return s[:32]


def _normalize_hops(raw_hops: Any) -> List[Dict[str, Any]]:
    """Sanitise a list of hop objects coming from the LLM."""
    if not isinstance(raw_hops, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, h in enumerate(raw_hops):
        if not isinstance(h, dict):
            continue
        op, sg = normalize_dual_tag_qa(h)
        note = _trim_note(str(h.get("note") or h.get("intention_note") or ""))
        if not note:
            # Skip empty hops — they add no signal.
            continue
        ev = _clean_evidence(str(h.get("evidence") or ""))
        record: Dict[str, Any] = {
            "step": int(h.get("step", len(out))),
            "operator": op,
            "subgoal": sg,
            "note": note,
            "evidence": ev,
        }
        tc = (h.get("tool_call") or "").strip()
        if tc:
            record["tool_call"] = str(tc)[:64]
        out.append(record)
        if len(out) >= MAX_HOPS:
            break
    # Re-index step indices contiguously.
    for i, h in enumerate(out):
        h["step"] = i
    # Guarantee final hop is COMMIT/ANSWER for downstream consistency.
    if out and (out[-1]["operator"] != "COMMIT" or out[-1]["subgoal"] != "ANSWER"):
        out.append({
            "step": len(out),
            "operator": "COMMIT",
            "subgoal": "ANSWER",
            "note": "commit final answer",
            "evidence": "final",
        })
    return out


def _intentions_summary(hops: List[Dict[str, Any]]) -> str:
    """Render hops as a single ``[OP/SG] note ; ...`` string."""
    chunks: List[str] = []
    for h in hops:
        chunks.append(f"[{h['operator']}/{h['subgoal']}] {h['note']}")
    return " ; ".join(chunks)


# ---------------------------------------------------------------------------
# Per-sample LLM call
# ---------------------------------------------------------------------------

def _label_sample(sample: Dict[str, Any], *, source: str, model: str) -> Dict[str, Any]:
    """Run one GPT-5.4 multihop call; return enriched sample.

    Always returns a dict with ``hops`` populated (falling back to a
    single COMMIT/ANSWER hop when the LLM fails).
    """
    out = dict(sample)  # shallow copy; preserves original keys
    out["hops"] = []
    out["n_hops"] = 0
    out["intentions_summary"] = ""
    out["multihop_label_source"] = "fallback_default"
    out["multihop_label_error"] = None

    prompt = _build_prompt(sample, source=source)

    if ask_model is not None:
        try:
            raw = ask_model(
                f"{_SYSTEM_PROMPT}\n\n{prompt}",
                model=model,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        except Exception as exc:
            out["multihop_label_error"] = f"{type(exc).__name__}: {exc}"
            raw = None

        if raw and not str(raw).startswith("Error"):
            obj = _extract_top_level_object(str(raw))
            if obj is not None:
                hops = _normalize_hops(obj.get("hops") or obj.get("steps") or [])
                if hops:
                    out["hops"] = hops
                    out["n_hops"] = len(hops)
                    out["intentions_summary"] = _intentions_summary(hops)
                    out["multihop_label_source"] = "llm"
                    return out
                out["multihop_label_error"] = "empty_hops"
            else:
                out["multihop_label_error"] = "json_parse_failed"
        elif raw is not None and not out["multihop_label_error"]:
            out["multihop_label_error"] = f"llm_error: {str(raw)[:120]}"
    else:
        out["multihop_label_error"] = "ask_model_unavailable"

    # Fallback: single COMMIT/ANSWER hop derived from the answer field.
    answer = (sample.get("answer") or sample.get("answer_raw") or "").strip()
    fallback_note = _trim_note(answer or "answer committed without traced reasoning")
    out["hops"] = [{
        "step": 0,
        "operator": "COMMIT",
        "subgoal": "ANSWER",
        "note": fallback_note,
        "evidence": "final",
    }]
    out["n_hops"] = 1
    out["intentions_summary"] = _intentions_summary(out["hops"])
    return out


# ---------------------------------------------------------------------------
# Discovery / driver
# ---------------------------------------------------------------------------

KNOWN_QA_SOURCES = ("video_holmes", "siv_bench", "tir_bench", "visual_toolbench")
DEFAULT_MODELS = ("gpt-5.4", "claude", "gemini", "qwen")


def _discover_pairs(
    input_run: Path,
    *,
    sources: Tuple[str, ...],
    models: Tuple[str, ...],
) -> List[Tuple[str, str, Path]]:
    pairs: List[Tuple[str, str, Path]] = []
    for src in sources:
        sdir = input_run / src
        if not sdir.is_dir():
            logger.warning("source dir missing: %s", sdir)
            continue
        for mdl in models:
            mdir = sdir / mdl
            f = mdir / "samples.jsonl"
            if not f.exists():
                continue
            pairs.append((src, mdl, f))
    return pairs


def _process_pair(
    *,
    source: str,
    model_label: str,
    input_path: Path,
    output_dir: Path,
    label_model: str,
    workers: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    """Label all samples in one (source, model_label) pair."""
    samples: List[Dict[str, Any]] = []
    with input_path.open("r") as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except Exception:
                continue
    if limit is not None:
        samples = samples[:limit]
    n_total = len(samples)
    if n_total == 0:
        return {
            "source": source, "model_label": model_label,
            "n_total": 0, "n_ok": 0, "n_fallback": 0, "n_errors": 0,
            "skipped": True,
        }

    out_subdir = output_dir / source / model_label
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_file = out_subdir / "samples_with_hops.jsonl"

    t0 = time.time()
    ok = fallback = errors = 0
    op_hist: Dict[str, int] = {}
    sg_hist: Dict[str, int] = {}
    n_hops_total = 0

    with out_file.open("w") as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_label_sample, s, source=source, model=label_model): i
                for i, s in enumerate(samples)
            }
            for fu in as_completed(futures):
                try:
                    enriched = fu.result()
                except Exception as exc:
                    errors += 1
                    logger.warning("sample failed: %s", exc)
                    continue
                src_kind = enriched.get("multihop_label_source", "fallback_default")
                if src_kind == "llm":
                    ok += 1
                else:
                    fallback += 1
                for h in enriched.get("hops") or []:
                    op_hist[h["operator"]] = op_hist.get(h["operator"], 0) + 1
                    sg_hist[h["subgoal"]] = sg_hist.get(h["subgoal"], 0) + 1
                    n_hops_total += 1
                fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    elapsed = time.time() - t0

    summary = {
        "source": source, "model_label": model_label,
        "input_path": str(input_path),
        "output_path": str(out_file),
        "n_total": n_total,
        "n_ok": ok,
        "n_fallback": fallback,
        "n_errors": errors,
        "n_hops_total": n_hops_total,
        "mean_hops_per_sample": (n_hops_total / n_total) if n_total else 0.0,
        "operator_histogram": dict(sorted(op_hist.items(), key=lambda kv: -kv[1])),
        "subgoal_histogram": dict(sorted(sg_hist.items(), key=lambda kv: -kv[1])),
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_path = out_subdir / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "[%s/%s] done — %d samples, %d hops, llm=%d fallback=%d err=%d, %.1fs",
        source, model_label, n_total, n_hops_total, ok, fallback, errors, elapsed,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multihop CoT decomposition + intention labeling for QA.",
    )
    p.add_argument(
        "--input-run", type=Path, required=True,
        help="Path to qa_miniwob_labeled/run_<ts> (must contain "
             "<source>/<model>/samples.jsonl per pair).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output dir; default: labeling/qa_multihop_out/run_<utc-ts>.",
    )
    p.add_argument(
        "--sources", type=str, nargs="+", default=list(KNOWN_QA_SOURCES),
        help=f"Subset of QA sources (default: {', '.join(KNOWN_QA_SOURCES)}).",
    )
    p.add_argument(
        "--models", type=str, nargs="+", default=list(DEFAULT_MODELS),
        help=f"Models to process (default: {', '.join(DEFAULT_MODELS)}).",
    )
    p.add_argument(
        "--label-model", type=str, default=DEFAULT_MODEL,
        help=f"GPT model to call (default: {DEFAULT_MODEL}).",
    )
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap samples per pair (smoke test).")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.setLevel(logging.INFO)

    input_run: Path = args.input_run.resolve()
    if not input_run.is_dir():
        print(f"[label_qa_multihop] input run missing: {input_run}", file=sys.stderr)
        return 2

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = (CODEBASE_ROOT / "labeling" / "qa_multihop_out" / f"run_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_pairs(
        input_run,
        sources=tuple(args.sources),
        models=tuple(args.models),
    )
    if not pairs:
        print(f"[label_qa_multihop] no (source, model) pairs found under {input_run}",
              file=sys.stderr)
        return 2

    print(f"[label_qa_multihop] discovered {len(pairs)} (source, model) pairs:")
    for s, m, f in pairs:
        n = sum(1 for _ in f.open())
        print(f"  {s}/{m}: {n} samples → {f}")
    print(f"[label_qa_multihop] output dir: {output_dir}")
    print(f"[label_qa_multihop] label-model: {args.label_model}, workers: {args.workers}")
    if args.limit:
        print(f"[label_qa_multihop] LIMIT: {args.limit} samples per pair")
    print()

    run_meta = {
        "input_run": str(input_run),
        "output_dir": str(output_dir),
        "sources": list(args.sources),
        "models": list(args.models),
        "label_model": args.label_model,
        "workers": args.workers,
        "limit": args.limit,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "argv": sys.argv,
    }
    (output_dir / "_run_meta.json").write_text(json.dumps(run_meta, indent=2))

    summaries: List[Dict[str, Any]] = []
    for source, model_label, input_path in pairs:
        try:
            summary = _process_pair(
                source=source,
                model_label=model_label,
                input_path=input_path,
                output_dir=output_dir,
                label_model=args.label_model,
                workers=args.workers,
                limit=args.limit,
            )
        except Exception as exc:
            logger.error("pair %s/%s FAILED: %s", source, model_label, exc)
            traceback.print_exc()
            summary = {
                "source": source, "model_label": model_label,
                "error": f"{type(exc).__name__}: {exc}",
            }
        summaries.append(summary)

    # Aggregate run summary
    n_pairs_ok = sum(1 for s in summaries if "error" not in s)
    n_total = sum(s.get("n_total", 0) for s in summaries)
    n_ok = sum(s.get("n_ok", 0) for s in summaries)
    n_fb = sum(s.get("n_fallback", 0) for s in summaries)
    n_err = sum(s.get("n_errors", 0) for s in summaries)
    n_hops = sum(s.get("n_hops_total", 0) for s in summaries)
    op_total: Dict[str, int] = {}
    sg_total: Dict[str, int] = {}
    for s in summaries:
        for k, v in (s.get("operator_histogram") or {}).items():
            op_total[k] = op_total.get(k, 0) + v
        for k, v in (s.get("subgoal_histogram") or {}).items():
            sg_total[k] = sg_total.get(k, 0) + v

    aggregate = {
        "run_meta": run_meta,
        "n_pairs_total": len(summaries),
        "n_pairs_ok": n_pairs_ok,
        "n_samples_total": n_total,
        "n_samples_llm_ok": n_ok,
        "n_samples_fallback": n_fb,
        "n_samples_errors": n_err,
        "n_hops_total": n_hops,
        "mean_hops_per_sample": (n_hops / n_total) if n_total else 0.0,
        "operator_histogram": dict(sorted(op_total.items(), key=lambda kv: -kv[1])),
        "subgoal_histogram": dict(sorted(sg_total.items(), key=lambda kv: -kv[1])),
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "per_pair": summaries,
    }
    (output_dir / "_run_summary.json").write_text(json.dumps(aggregate, indent=2))

    print()
    print("=" * 70)
    print(f"[label_qa_multihop] DONE — {n_pairs_ok}/{len(summaries)} pairs ok, "
          f"{n_total} samples, {n_hops} hops")
    print(f"  llm_ok={n_ok}  fallback={n_fb}  errors={n_err}")
    print(f"  operator hist: {dict(list(op_total.items())[:6])}")
    print(f"  subgoal hist:  {dict(list(sg_total.items())[:6])}")
    print(f"[label_qa_multihop] summary: {output_dir / '_run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
