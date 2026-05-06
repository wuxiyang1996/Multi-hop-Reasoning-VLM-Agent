#!/usr/bin/env python
"""scripts.skillbridge_eval.run_actor_uplift
─────────────────────────────────────────────
Measure end-to-end actor accuracy uplift from injecting Crafter-minted
skills into the cold-start visual-reasoning actor prompt.

This is the experiment the smoke-runner / promotion-gate stack was
built to enable: it answers the user's original question — *does the
crafter+promotion path actually make the actor pick better answers?*
— rather than merely *does it generate diverse-looking proposals?*.

Pipeline per holdout sample (one VTB cold-start JSON, schema already
computed in a prior `generate_cold_start_actor_visual_reasoning.py` run
so we don't pay for the gpt-5.5 vision call again):

    schema_text + question + top-K relevant skills (from this mode's bank)
                              │
                              ▼
                  gpt-5.4 actor → predicted answer
                              │
                              ▼
                  gpt-5.4 judge → verdict (correct / incorrect / unscoreable)

Three modes share the SAME holdout sample list, the SAME judge model,
the SAME actor model — only the **bank injected into the prompt** differs:

* ``no_bank``      — empty skill block (this is the control; matches
                     the original cold-start actor by construction).
* ``rule_only``    — bank built from rule_only mode's smoke proposals
                     (deterministic Hypothesizer boilerplate).
* ``lane_b_llm``   — bank built from lane_b_llm mode's smoke proposals
                     (LLM Repairer's PatchProposals; only the Stage-0
                     survivors are eligible).

The smoke-attribution work proved the LLM mode mints proposals that
look more diverse and pass the gate at 60% (vs the rule path's
boilerplate at 100%). This experiment closes the loop: do those
LLM-minted skills, when handed back to the actor, actually shift its
answer distribution toward correctness?

Outputs:

  <output_dir>/<mode>/per_sample.jsonl
  <output_dir>/_uplift_summary.json
  <output_dir>/_uplift_summary.md

Hold-out window — by default we use VTB samples with sample_index
50 .. 50+N-1 from the most recent cold-start run, so we never train
and test on the same prompts (the smoke ran on samples 0..49). Pass
``--sample-offset 0`` to override.

Example::

    python -m scripts.skillbridge_eval.run_actor_uplift \\
        --smoke-dir labeling_supplement/episode_reflections_out/_smoke_attr_v2 \\
        --max-samples 50 \\
        --sample-offset 50 \\
        --actor-model gpt-5.4 \\
        --judge-model gpt-5.4 \\
        --output-dir labeling_supplement/episode_reflections_out/_actor_uplift_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# We piggy-back on the cold-start actor module:
#   * ``_bootstrap_api_keys_from_file`` populates OPENAI_API_KEY from
#     /workspace/api_keys.py (or the fallback search path the script
#     itself uses) — we want byte-identical key resolution here.
#   * ``_build_client_and_route`` produces the OpenAI client + routed
#     model identifier (handles OpenRouter slug rewriting etc.).
#   * ``_chat_completion`` is the rate-limited / retry-aware caller
#     the smoke pipeline uses; do NOT duplicate this logic.
#   * ``llm_judge_correct`` mirrors what ``--judge`` does in the
#     cold-start actor itself, so the verdict distribution lines up
#     with what the existing per-sample JSONs were graded on.
from cold_start.generate_cold_start_actor_visual_reasoning import (  # noqa: E402
    _build_client_and_route,
    _chat_completion,
    llm_judge_correct,
)

logger = logging.getLogger("skillbridge_eval.run_actor_uplift")

# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


def _find_latest_benchmark_run(
    samples_root: Path, benchmark: str,
) -> Optional[Path]:
    """Return the most-populated ``<benchmark>/`` directory under
    ``samples_root``. Two on-disk layouts coexist in this project:

    * **per-run tree** — ``Cold-start-out-visual-reasoning/<run>/<benchmark>/sample_*.json``
      (what the cold-start actor writes when invoked directly).
    * **flat tree**    — ``Cold-start-out-visual-reasoning/<benchmark>/sample_*.json``
      (the canonical pre-built corpus the smoke runs against —
      sample IDs aligned with the diversity manifest).

    We pick whichever has the most ``sample_*.json`` files (tie-break
    by most-recent mtime), so the holdout slice always lands in the
    largest available pool of samples."""
    if not samples_root.is_dir():
        return None
    candidates: List[Tuple[int, float, Path]] = []
    direct = samples_root / benchmark
    if direct.is_dir():
        n = len(list(direct.glob("sample_*.json")))
        if n > 0:
            candidates.append((n, direct.stat().st_mtime, direct))
    for run_dir in samples_root.iterdir():
        if not run_dir.is_dir() or run_dir.name == benchmark:
            continue
        bench_dir = run_dir / benchmark
        if bench_dir.is_dir():
            n = len(list(bench_dir.glob("sample_*.json")))
            if n > 0:
                candidates.append((n, bench_dir.stat().st_mtime, bench_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: (kv[0], kv[1]), reverse=True)
    return candidates[0][2]


def _load_holdout_samples(
    bench_dir: Path,
    *,
    offset: int,
    max_samples: int,
    benchmark: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return ``max_samples`` consecutive samples starting at ``offset``.

    The cold-start per-sample JSONs sometimes omit the ``benchmark``
    field (it's implicit from the directory name).  We backfill it
    from ``benchmark`` (defaults to the parent directory's name) so
    downstream consumers can rely on the field being populated.
    """
    files = sorted(bench_dir.glob("sample_*.json"))
    files = files[offset : offset + max_samples]
    inferred_benchmark = benchmark or bench_dir.name
    out: List[Dict[str, Any]] = []
    for f in files:
        try:
            blob = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("skipping malformed sample %s: %s", f, exc)
            continue
        # Only keep samples we have a usable schema + gold for, since
        # the bank can't help if there's nothing for the actor to
        # reason over.
        schema = (blob.get("schema") or "").strip()
        gold = (blob.get("gold_answer") or "").strip()
        question = (blob.get("question") or "").strip()
        if not (schema and gold and question):
            continue
        blob["__path"] = str(f)
        if not blob.get("benchmark"):
            blob["benchmark"] = inferred_benchmark
        out.append(blob)
    return out


# ---------------------------------------------------------------------------
# Bank → SkillCard projection (per mode)
# ---------------------------------------------------------------------------


@dataclass
class SkillCard:
    """A flat view of one Crafter-minted proposal, ready for prompt injection.

    We deliberately keep this leaner than the live ``SkillRecord`` —
    the actor only needs name + when-to-use + a couple of protocol
    bullets to decide whether the skill is relevant to the current
    question.  Anything heavier dilutes the prompt and starves the
    schema block of tokens.
    """
    skill_id: str
    name: str
    rationale: str
    protocol_steps: List[str] = field(default_factory=list)
    notes: str = ""

    def render(self) -> str:
        bullets = [f"- {s}" for s in self.protocol_steps[:5]]
        body = "\n".join(bullets) if bullets else "- (no protocol body)"
        head = f"### {self.name or self.skill_id}"
        return f"{head}\nWhen: {self.rationale.strip()}\nProtocol:\n{body}"


def _proposal_to_skill_card(row: Dict[str, Any]) -> Optional[SkillCard]:
    """Translate one ``proposal_to_json`` row into a SkillCard.

    We ignore ``ComposeProposal`` / ``GeneralizeProposal`` / ``RetireProposal``
    since the smoke modes never emit those — only the patch and
    hypothesis paths.  Anything else falls through to ``None``.
    """
    type_name = row.get("type", "")
    if type_name == "HypothesisProposal":
        proto = row.get("novel_protocol") or []
        return SkillCard(
            skill_id=row.get("proposal_id", ""),
            name=row.get("name", "") or row.get("proposal_id", "")[:24],
            rationale=row.get("rationale", ""),
            protocol_steps=_protocol_steps_to_strs(proto),
        )
    if type_name == "PatchProposal":
        proto = row.get("patched_protocol") or []
        # PatchProposal carries no ``name`` field (it patches an
        # existing skill); use the base id as the surface name.
        base = row.get("base_skill_id") or row.get("proposal_id", "")
        return SkillCard(
            skill_id=row.get("proposal_id", ""),
            name=f"patch_for_{base}",
            rationale=row.get("rationale", ""),
            protocol_steps=_protocol_steps_to_strs(proto),
        )
    return None


def _protocol_steps_to_strs(proto: List[Any]) -> List[str]:
    """Render a list of protocol hops (typed dicts or bare strings)
    into one-line bullets the actor can read at a glance."""
    out: List[str] = []
    for hop in proto:
        if isinstance(hop, dict):
            op = hop.get("op") or hop.get("action") or "STEP"
            note = hop.get("notes") or ""
            payload = hop.get("payload") or {}
            target = payload.get("target") if isinstance(payload, dict) else ""
            line = f"[{op}]"
            if target:
                line += f" target={target}"
            if note:
                line += f" — {note}"
            out.append(line)
        else:
            out.append(str(hop))
    return out


def _load_bank_for_mode(smoke_mode_dir: Path) -> List[SkillCard]:
    """Walk every ``proposals.jsonl`` under one mode's smoke output
    and project each row into a :class:`SkillCard`."""
    cards: List[SkillCard] = []
    for f in sorted(smoke_mode_dir.rglob("proposals.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            card = _proposal_to_skill_card(row)
            if card is not None:
                cards.append(card)
    return cards


# ---------------------------------------------------------------------------
# Token-Jaccard retrieval
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(*chunks: str) -> set:
    seen: set = set()
    for c in chunks:
        if not c:
            continue
        for t in _TOKEN_RE.findall(c.lower()):
            if len(t) >= 3:
                seen.add(t)
    return seen


def _retrieve_topk(
    question: str, schema: str, cards: List[SkillCard], *, k: int,
) -> List[SkillCard]:
    """Pick the ``k`` cards whose token set has the highest Jaccard with
    the question (and a touch of the schema, lightly weighted).  The
    *question* is the dominant signal — the schema is mostly entity
    descriptors that match every card spuriously."""
    if not cards or k <= 0:
        return []
    q_tokens = _tokenize(question, schema[:500])
    if not q_tokens:
        return cards[:k]
    scored: List[Tuple[float, SkillCard]] = []
    for c in cards:
        c_tokens = _tokenize(c.name, c.rationale, c.notes, " ".join(c.protocol_steps))
        if not c_tokens:
            continue
        inter = len(q_tokens & c_tokens)
        if inter == 0:
            continue
        union = len(q_tokens | c_tokens)
        scored.append((inter / union if union else 0.0, c))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return [c for _, c in scored[:k]]


# ---------------------------------------------------------------------------
# Actor call (schema + question [+ skill bank block])
# ---------------------------------------------------------------------------


_ACTOR_SYSTEM_PROMPT = (
    "You are an Actor Agent for the COS-PLAY visual-reasoning pipeline.\n"
    "On every step you receive (a) a structured ``<state>...</state>`` "
    "schema produced by a vision call on the input image, (b) the "
    "benchmark question, and OPTIONALLY (c) a small set of relevant "
    "skill cards drafted from past failures on similar questions.  The "
    "skill cards are advisory only — use them when their ``When:`` "
    "rationale and ``Protocol:`` bullets line up with the current "
    "question, otherwise ignore them and answer from the schema "
    "directly.\n\n"
    "Your job:\n"
    "1. Reason briefly (≤4 sentences) over the schema entities, citing "
    "entity ids (e1, e2, ...) when relevant.  If a skill card applies, "
    "name it once.\n"
    "2. Emit a concise free-form answer (a single phrase, number, or "
    "word).  No explanation, no markdown, no quoting.\n\n"
    "Always respond by calling the ``choose_answer`` function."
)


def _build_actor_messages(
    *,
    benchmark: str,
    schema: str,
    question: str,
    skill_bank_block: str,
    is_mcq: bool,
    valid_actions: List[str],
    options_block: Optional[str],
) -> List[Dict[str, str]]:
    """Mirror cold_start.generate_cold_start_actor_visual_reasoning's
    user prompt construction so MCQ vs free-form benchmarks are
    handled identically — the only difference is the appended skill
    bank block."""
    user_parts: List[str] = [
        f"Benchmark: {benchmark}",
        "Question:",
        question.strip(),
    ]
    if options_block:
        user_parts.extend(["", options_block.strip()])
    user_parts.extend([
        "",
        "Structured visual-state schema (from gpt-5.5 vision call):",
        schema.strip(),
    ])
    if skill_bank_block.strip():
        user_parts.extend([
            "",
            "Relevant skill cards (advisory — apply only if they fit "
            "this question; otherwise ignore):",
            skill_bank_block.strip(),
        ])
    user_parts.append("")
    if is_mcq and valid_actions:
        # MCQ — let the actor pick a letter from a closed set.  The
        # tool schema accepts any string but the system prompt + this
        # constraint keeps the output scoreable.
        user_parts.append(
            f"Valid answers: {', '.join(valid_actions)}.  "
            "Pick EXACTLY one letter and emit it as the choose_answer "
            "argument (no explanation, no quoting)."
        )
    else:
        user_parts.append(
            "Free-form QA: answer concisely (a single phrase, number, "
            "or word).  Match the wording the question expects."
        )
    user_parts.extend([
        "",
        "Think step-by-step over the schema, then call the choose_answer function.",
    ])
    return [
        {"role": "system", "content": _ACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


_ACTOR_TOOLS = [{
    "type": "function",
    "function": {
        "name": "choose_answer",
        "description": "Emit the actor's final free-form answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["answer"],
        },
    },
}]


def _ask_actor(
    client: Any, *, model: str, benchmark: str, schema: str, question: str,
    skill_bank_block: str, is_mcq: bool, valid_actions: List[str],
    options_block: Optional[str],
    temperature: float, max_tokens: int,
    reasoning_effort: Optional[str],
) -> Tuple[Optional[str], str, Optional[str]]:
    """Returns ``(answer, raw_message_or_args, error)``."""
    messages = _build_actor_messages(
        benchmark=benchmark, schema=schema, question=question,
        skill_bank_block=skill_bank_block, is_mcq=is_mcq,
        valid_actions=valid_actions, options_block=options_block,
    )
    try:
        resp = _chat_completion(
            client, model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            tools=_ACTOR_TOOLS,
            tool_choice={"type": "function", "function": {"name": "choose_answer"}},
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:                                       # noqa: BLE001
        return None, "", repr(exc)
    msg = resp.choices[0].message
    raw = msg.content or ""
    if getattr(msg, "tool_calls", None):
        tc = msg.tool_calls[0]
        raw_args = (
            getattr(tc, "arguments", None)
            or getattr(getattr(tc, "function", None), "arguments", None)
            or "{}"
        )
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except Exception:
            args = {}
        ans = str(args.get("answer", "")).strip()
        if ans:
            return ans, raw or json.dumps(args), None
    if raw.strip():
        return raw.strip(), raw, None
    return None, raw, "empty_response"


# ---------------------------------------------------------------------------
# Per-mode runner
# ---------------------------------------------------------------------------


@dataclass
class ModeUplift:
    label: str
    n_samples: int = 0
    n_correct: int = 0
    n_incorrect: int = 0
    n_unscoreable: int = 0
    n_actor_errors: int = 0
    n_judge_errors: int = 0
    cards_total: int = 0
    cards_per_sample_avg: float = 0.0
    elapsed_s: float = 0.0
    per_sample: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def correct_rate(self) -> float:
        n = self.n_samples - self.n_actor_errors - self.n_judge_errors
        return float(self.n_correct) / n if n > 0 else 0.0


def _process_one(
    *,
    sample: Dict[str, Any],
    cards: List[SkillCard],
    top_k: int,
    client: Any,
    actor_model: str,
    actor_routed: str,
    judge_routed: str,
    actor_temperature: float,
    actor_max_tokens: int,
    actor_reasoning_effort: Optional[str],
    judge_cache_dir: Path,
) -> Dict[str, Any]:
    sid = sample.get("sample_id", "")
    schema = sample.get("schema", "")
    question = sample.get("question", "")
    gold = sample.get("gold_answer", "")
    benchmark = sample.get("benchmark", "visual_toolbench")
    is_mcq = bool(sample.get("is_mcq"))
    valid_actions = list(sample.get("valid_actions") or [])
    options_block = sample.get("options_block")

    selected_cards = _retrieve_topk(question, schema, cards, k=top_k)
    bank_block = "\n\n".join(c.render() for c in selected_cards)

    t0 = time.time()
    answer, raw, actor_err = _ask_actor(
        client, model=actor_routed,
        benchmark=benchmark, schema=schema, question=question,
        skill_bank_block=bank_block, is_mcq=is_mcq,
        valid_actions=valid_actions, options_block=options_block,
        temperature=actor_temperature, max_tokens=actor_max_tokens,
        reasoning_effort=actor_reasoning_effort,
    )
    actor_elapsed = time.time() - t0

    if actor_err is not None or not answer:
        return {
            "sample_id": sid, "error": actor_err or "empty_answer",
            "selected_card_ids": [c.skill_id for c in selected_cards],
            "actor_elapsed_s": actor_elapsed,
        }

    if is_mcq and valid_actions:
        # MCQ — exact / canonicalised letter match.  Avoids paying
        # for an LLM judge call on every sample (TIR-Bench's 300
        # samples × 4 modes would be 1200 superfluous calls when a
        # one-character comparison suffices).
        verdict, correct, judge_err = _mcq_verdict(answer, gold, valid_actions)
        cached = False
    else:
        judge = llm_judge_correct(
            question=question, gold=gold, predicted=answer,
            benchmark=benchmark, client=client, routed_model=judge_routed,
            cache_dir=judge_cache_dir, sample_id=sid,
        )
        verdict = judge.get("verdict")
        correct = judge.get("correct")
        cached = bool(judge.get("cached"))
        judge_err = judge.get("error")
    return {
        "sample_id": sid, "question": question[:300],
        "gold": gold[:300], "predicted": answer[:300],
        "verdict": verdict, "judge_correct": correct,
        "judge_cached": cached, "judge_error": judge_err,
        "scoring_mode": "mcq_letter" if (is_mcq and valid_actions) else "llm_judge",
        "selected_card_ids": [c.skill_id for c in selected_cards],
        "selected_card_names": [c.name for c in selected_cards],
        "n_cards_in_bank": len(cards),
        "actor_elapsed_s": actor_elapsed,
    }


_MCQ_LETTER_RE = re.compile(r"\b([A-Z])\b")


def _mcq_verdict(
    predicted: str, gold: str, valid_actions: List[str],
) -> Tuple[str, Optional[bool], Optional[str]]:
    """Return ``(verdict, correct_bool, error)`` for an MCQ-style answer.

    The predicted string may include reasoning prefix or quotes; we
    canonicalise to the first uppercase letter that appears in the
    valid_actions set, then compare to gold.  Mirrors the
    ``_canonicalize_action`` helper in the cold-start actor (kept
    local to avoid the heavy import surface).
    """
    g = (gold or "").strip().upper()
    p = (predicted or "").strip().upper()
    valids = {a.strip().upper() for a in valid_actions if a.strip()}
    if not valids or g not in valids:
        return "unscoreable", None, "gold_not_in_valid_actions" if g else "empty_gold"
    # Try exact first, then leading char, then any single uppercase
    # letter token in the body that's in valids.
    if p in valids:
        return ("correct" if p == g else "incorrect", p == g, None)
    if p[:1] in valids:
        return ("correct" if p[:1] == g else "incorrect", p[:1] == g, None)
    for tok in _MCQ_LETTER_RE.findall(p):
        if tok in valids:
            return ("correct" if tok == g else "incorrect", tok == g, None)
    return "incorrect", False, None


def _run_one_mode(
    *,
    label: str,
    cards: List[SkillCard],
    samples: List[Dict[str, Any]],
    top_k: int,
    output_dir: Path,
    client: Any,
    actor_model: str,
    actor_routed: str,
    judge_routed: str,
    actor_temperature: float,
    actor_max_tokens: int,
    actor_reasoning_effort: Optional[str],
    num_workers: int,
) -> ModeUplift:
    rep = ModeUplift(label=label, cards_total=len(cards))
    out_dir = output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_cache_dir = out_dir / "judge_cache"
    judge_cache_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / "per_sample.jsonl"

    rep.n_samples = len(samples)
    cards_used: List[int] = []
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    if num_workers <= 1:
        for s in samples:
            rows.append(_process_one(
                sample=s, cards=cards, top_k=top_k, client=client,
                actor_model=actor_model, actor_routed=actor_routed,
                judge_routed=judge_routed,
                actor_temperature=actor_temperature,
                actor_max_tokens=actor_max_tokens,
                actor_reasoning_effort=actor_reasoning_effort,
                judge_cache_dir=judge_cache_dir,
            ))
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(
                _process_one,
                sample=s, cards=cards, top_k=top_k, client=client,
                actor_model=actor_model, actor_routed=actor_routed,
                judge_routed=judge_routed,
                actor_temperature=actor_temperature,
                actor_max_tokens=actor_max_tokens,
                actor_reasoning_effort=actor_reasoning_effort,
                judge_cache_dir=judge_cache_dir,
            ) for s in samples]
            for fut in as_completed(futures):
                rows.append(fut.result())
    rep.elapsed_s = time.time() - t0

    with per_sample_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            cards_used.append(len(row.get("selected_card_ids") or []))
            if row.get("error"):
                rep.n_actor_errors += 1
                continue
            if row.get("judge_error"):
                rep.n_judge_errors += 1
                continue
            v = row.get("verdict")
            if v == "correct":
                rep.n_correct += 1
            elif v == "incorrect":
                rep.n_incorrect += 1
            elif v == "unscoreable":
                rep.n_unscoreable += 1

    rep.cards_per_sample_avg = (
        sum(cards_used) / len(cards_used) if cards_used else 0.0
    )
    rep.per_sample = rows
    return rep


# ---------------------------------------------------------------------------
# Cross-mode summary
# ---------------------------------------------------------------------------


def _emit_summary(
    *,
    output_dir: Path,
    reports: List[ModeUplift],
    samples_count: int,
    actor_model: str,
    actor_routed: str,
    judge_routed: str,
    smoke_dir: Path,
    sample_offset: int,
    benchmark: str,
) -> Tuple[Path, Path]:
    blob = {
        "benchmark":      benchmark,
        "smoke_dir":      str(smoke_dir),
        "actor_model":    actor_model,
        "actor_routed":   actor_routed,
        "judge_routed":   judge_routed,
        "samples_count":  samples_count,
        "sample_offset":  sample_offset,
        "completed_at":   time.time(),
        "modes": [
            {
                "label":               r.label,
                "n_samples":           r.n_samples,
                "n_correct":           r.n_correct,
                "n_incorrect":         r.n_incorrect,
                "n_unscoreable":       r.n_unscoreable,
                "n_actor_errors":      r.n_actor_errors,
                "n_judge_errors":      r.n_judge_errors,
                "cards_total":         r.cards_total,
                "cards_per_sample_avg": r.cards_per_sample_avg,
                "correct_rate":        r.correct_rate,
                "elapsed_s":           r.elapsed_s,
            } for r in reports
        ],
    }
    js = output_dir / "_uplift_summary.json"
    js.write_text(json.dumps(blob, indent=2))

    lines: List[str] = []
    lines.append("# Actor uplift attribution (cold-start VTB holdout)\n")
    lines.append(f"- smoke_dir: `{smoke_dir}`")
    lines.append(f"- actor_model: `{actor_model}` (routed `{actor_routed}`)")
    lines.append(f"- judge_routed: `{judge_routed}`")
    lines.append(f"- holdout samples: `{samples_count}` (offset {sample_offset})\n")
    lines.append(
        "| mode | n samples | bank size | top-K avg | "
        "correct | incorrect | unscoreable | actor err | judge err | "
        "correct-rate |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in reports:
        lines.append(
            f"| `{r.label}` | {r.n_samples} | {r.cards_total} | "
            f"{r.cards_per_sample_avg:.2f} | {r.n_correct} | "
            f"{r.n_incorrect} | {r.n_unscoreable} | "
            f"{r.n_actor_errors} | {r.n_judge_errors} | "
            f"{r.correct_rate * 100:.1f}% |"
        )
    md = output_dir / "_uplift_summary.md"
    md.write_text("\n".join(lines) + "\n")
    return js, md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--smoke-dir", type=Path, required=True,
        help="The output_dir from run_smoke_attribution; subdirs are "
             "the mode labels we evaluate.",
    )
    p.add_argument(
        "--samples-root", type=Path,
        default=REPO_ROOT / "Cold-start-out-visual-reasoning",
        help="Cold-start VR run root (auto-picks the most-populated "
             "<run>/<benchmark>/ that has sample_*.json).",
    )
    p.add_argument(
        "--benchmark", type=str, default="visual_toolbench",
        choices=("visual_toolbench", "tir_bench", "video_holmes", "siv_bench"),
        help="Which VR benchmark to run uplift on. Default "
             "visual_toolbench (the smoke's training distribution).",
    )
    p.add_argument(
        "--samples-dir", type=Path, default=None,
        help="Override the auto-picked benchmark directory.",
    )
    p.add_argument(
        "--max-samples", type=int, default=50,
        help="Holdout sample count per mode (default 50).",
    )
    p.add_argument(
        "--sample-offset", type=int, default=50,
        help="Skip the first N samples (smoke used 0..49 by default; "
             "this offset puts the actor uplift on a disjoint slice).",
    )
    p.add_argument(
        "--top-k", type=int, default=3,
        help="Top-K skills to inject into the actor prompt per sample.",
    )
    p.add_argument(
        "--modes", nargs="+",
        default=["no_bank", "rule_only", "lane_b_llm"],
        help="Modes to evaluate. ``no_bank`` is the control (skill "
             "block stays empty); other modes are read from "
             "<smoke-dir>/<label>/.",
    )
    p.add_argument(
        "--actor-model", type=str, default="gpt-5.4",
        help="Actor backbone (default gpt-5.4 — same as the smoke).",
    )
    p.add_argument(
        "--judge-model", type=str, default="",
        help="Judge model (default = same as actor).",
    )
    p.add_argument(
        "--actor-temperature", type=float, default=0.4,
        help="Actor sampling temperature.",
    )
    p.add_argument(
        "--actor-max-tokens", type=int, default=2048,
        help="Actor max_tokens.",
    )
    p.add_argument(
        "--reasoning-effort", type=str, default=None,
        help="OpenAI reasoning_effort knob (forwarded to actor calls).",
    )
    p.add_argument(
        "--num-workers", type=int, default=8,
        help="Concurrent samples (default 8 — VTB is API-bound).",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write per-mode dirs + cross-mode summary.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if not args.smoke_dir.is_dir():
        logger.error("smoke-dir does not exist: %s", args.smoke_dir)
        return 1

    bench_dir = args.samples_dir or _find_latest_benchmark_run(
        args.samples_root, args.benchmark,
    )
    if bench_dir is None or not bench_dir.is_dir():
        logger.error(
            "no %s/ dir under %s; pass --samples-dir explicitly",
            args.benchmark, args.samples_root,
        )
        return 1
    logger.info("holdout %s dir: %s", args.benchmark, bench_dir)

    samples = _load_holdout_samples(
        bench_dir, offset=args.sample_offset, max_samples=args.max_samples,
        benchmark=args.benchmark,
    )
    if not samples:
        logger.error(
            "no holdout samples picked at offset=%d max=%d under %s",
            args.sample_offset, args.max_samples, vtb_dir,
        )
        return 1
    logger.info("loaded %d holdout samples", len(samples))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    actor_model = args.actor_model
    judge_model = args.judge_model or actor_model
    client, actor_routed = _build_client_and_route(model=actor_model)
    if client is None:
        logger.error("could not build actor client (no API key?)")
        return 1
    _, judge_routed = _build_client_and_route(model=judge_model)
    logger.info("actor model: %s → routed %s", actor_model, actor_routed)
    logger.info("judge model: %s → routed %s", judge_model, judge_routed)

    reports: List[ModeUplift] = []
    for label in args.modes:
        if label == "no_bank":
            cards: List[SkillCard] = []
        else:
            mode_dir = args.smoke_dir / label
            if not mode_dir.is_dir():
                logger.warning("smoke mode %s not found at %s; skipping",
                               label, mode_dir)
                continue
            cards = _load_bank_for_mode(mode_dir)
        logger.info("mode=%s bank_size=%d", label, len(cards))
        rep = _run_one_mode(
            label=label, cards=cards, samples=samples, top_k=args.top_k,
            output_dir=args.output_dir, client=client,
            actor_model=actor_model, actor_routed=actor_routed,
            judge_routed=judge_routed,
            actor_temperature=args.actor_temperature,
            actor_max_tokens=args.actor_max_tokens,
            actor_reasoning_effort=args.reasoning_effort,
            num_workers=args.num_workers,
        )
        reports.append(rep)
        logger.info(
            "[%s] correct=%d incorrect=%d unscoreable=%d "
            "actor_err=%d judge_err=%d  rate=%.1f%%  elapsed=%.1fs",
            rep.label, rep.n_correct, rep.n_incorrect,
            rep.n_unscoreable, rep.n_actor_errors, rep.n_judge_errors,
            rep.correct_rate * 100, rep.elapsed_s,
        )

    js, md = _emit_summary(
        output_dir=args.output_dir, reports=reports,
        samples_count=len(samples),
        actor_model=actor_model, actor_routed=actor_routed,
        judge_routed=judge_routed,
        smoke_dir=args.smoke_dir, sample_offset=args.sample_offset,
        benchmark=args.benchmark,
    )
    print()
    print("=== actor uplift summary ===")
    print(f"output_dir: {args.output_dir}")
    print(f"json:       {js}")
    print(f"markdown:   {md}\n")
    for r in reports:
        print(
            f"  {r.label:<14}  n={r.n_samples}  "
            f"correct={r.n_correct}/{r.n_samples}  "
            f"unscoreable={r.n_unscoreable}  "
            f"actor_err={r.n_actor_errors}  judge_err={r.n_judge_errors}  "
            f"rate={r.correct_rate * 100:.1f}%  elapsed={r.elapsed_s:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
