"""Failure synthesiser for the visual-reasoning cold-start corpus.

Source layout (one of):

    Cold-start-out-visual-reasoning/{visual_toolbench,tir_bench}/sample_<id>.json
    Cold-start-out-visual-reasoning/{video_holmes,siv_bench}/sample_<id>.json

Each per-sample JSON has the shape documented in
``visual_reasoning_wrapper/README.md`` §"Per-sample JSON shape":

    correct                : bool   — judge-derived when --judge, else strmatch
    correct_strmatch       : bool   — diagnostic-only
    judge.verdict          : "correct" | "incorrect" | "unscoreable"
    judge.reason           : free-text, MCQ benchmarks omit this block
    schema_recovery        : "strict" | "fenced" | "truncated" | "untagged" | "no_image"
    schema_finish_reason   : "stop" | "length" | ...
    answer                 : str
    answer_error           : str | null
    answer_reasoning       : str
    is_mcq                 : bool
    benchmark              : "visual_toolbench" | "tir_bench" | "video_holmes" | "siv_bench"

Because VR cold-start is a one-call pipeline (vision schema → actor
answer) there is no per-step ``experiences[]`` array; every failure
trace has ``failed_step_index = 0`` and the (synthetic) ``skill_id``
is left empty so the dispatcher routes to ``Hypothesizer`` /
``BANK_GAP`` rather than to ``Repairer`` (which needs an existing
base skill). The Crafter's ``hypothesize_min_recurrences`` /
related-skill Jaccard gates still apply, so a single isolated wrong
answer produces no proposal — that is the desired noise floor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from legacy.crafter.failure_memory import SEMANTIC_BUCKET_EXTRA_KEY
from data_structure.extensions.failure_trace import FailureTrace

# These thresholds match the conservative defaults in the gymv
# synthesiser (``reflect_per_episode_gpt54._synthesize_failures``).
# Tunable via the call site.
DEFAULT_MAX_FAILURES_PER_SAMPLE = 4

# Schema-recovery values that mean "the VLM call returned malformed
# output and the parser had to fall back". ``strict`` is the only
# clean tier; everything else carries diagnostic value for the
# Repairer / Hypothesizer.
_DIRTY_SCHEMA_RECOVERIES = frozenset({
    "fenced", "truncated", "untagged", "no_image",
})


def from_sample(
    sample: Dict[str, Any],
    *,
    domain: str = "visual_reasoning",
    sample_id: str = "",
    max_failures: int = DEFAULT_MAX_FAILURES_PER_SAMPLE,
) -> List[FailureTrace]:
    """Walk a per-sample VR / video JSON and emit ``FailureTrace[]``.

    ``domain`` should be one of:

      - ``"visual_reasoning"`` — VTB / TIR-Bench (image, free-form)
      - ``"video"``            — Video-Holmes / SIV-Bench (video MCQ)

    The benchmark id from ``sample["benchmark"]`` is preserved on
    every trace's ``extra.benchmark`` field so per-benchmark slicing
    is possible downstream.
    """
    benchmark = str(sample.get("benchmark") or "")
    sid = sample_id or str(sample.get("sample_id") or sample.get("task_id") or "")
    is_mcq = bool(sample.get("is_mcq"))
    common_extra: Dict[str, Any] = {
        "benchmark": benchmark,
        "sample_id": sid,
        "is_mcq": is_mcq,
    }

    # Fix-A semantic_bucket — ``(synthesis_signal, benchmark, mcq?)``
    # gives the FailureMemory dedup key enough resolution to keep
    # WRONG_ANSWER on tir_bench separate from WRONG_ANSWER on
    # visual_toolbench, and from UNSCOREABLE / EMPTY_ANSWER /
    # MALFORMED_SCHEMA on the same benchmark.  Without this, every
    # VR cold-start sample collapsed onto a single
    # ``(skill_id="", failure_class=INVARIANT_VIOLATION, idx=0)``
    # pattern and the LLM Hypothesizer mode-collapsed into 10
    # paraphrases of "evidence_gate_*" — see the v3 attribution
    # summary §"Diagnosis: LLM Hypothesizer mode collapse".
    def _bucket(signal: str) -> str:
        bench = benchmark or "unknown_bench"
        mode = "mcq" if is_mcq else "freeform"
        return f"{signal.lower()}/{bench}/{mode}"

    out: List[FailureTrace] = []

    # ── 1. WRONG_ANSWER — the headline signal ─────────────────────────
    # Prefer judge-derived correctness when present; fall back to
    # ``correct`` otherwise. We treat ``correct_strmatch`` separately
    # because it has known false-negatives on free-form benches.
    judge = sample.get("judge") or {}
    judge_verdict = (judge.get("verdict") or "").lower() if judge else ""
    is_correct = bool(sample.get("correct"))
    if not is_correct and judge_verdict in ("incorrect", ""):
        # ``""`` covers MCQ rows where --judge wasn't run; in that
        # case `correct` is letter-equality and is reliable.
        out.append(FailureTrace(
            skill_id="",
            skill_episode_id=f"{sid}#wrong_answer",
            domain=domain,
            failed_step_index=0,
            failure_class="INVARIANT_VIOLATION",
            abort_reason=_summarise_wrong_answer(sample, judge),
            extra={
                **common_extra,
                SEMANTIC_BUCKET_EXTRA_KEY: _bucket("WRONG_ANSWER"),
                "synthesis_signal": "WRONG_ANSWER",
                "answer": _truncate(sample.get("answer")),
                "gold_answer": _truncate(sample.get("gold_answer")),
                "judge_verdict": judge_verdict or None,
                "judge_reason": _truncate((judge or {}).get("reason"), n=400),
            },
        ))

    # ── 2. UNSCOREABLE — judge can't decide ───────────────────────────
    # Treat as a precondition / contract failure: the actor's output
    # was not in a form the judge could grade. Routed through
    # PRECONDITION_STRENGTHENING by FailureDiagnoser.
    if judge_verdict == "unscoreable":
        out.append(FailureTrace(
            skill_id="",
            skill_episode_id=f"{sid}#unscoreable",
            domain=domain,
            failed_step_index=0,
            failure_class="PRECONDITION_VIOLATION",
            abort_reason="judge_verdict=unscoreable",
            extra={
                **common_extra,
                SEMANTIC_BUCKET_EXTRA_KEY: _bucket("UNSCOREABLE"),
                "synthesis_signal": "UNSCOREABLE",
                "judge_reason": _truncate((judge or {}).get("reason"), n=400),
            },
        ))

    # ── 3. EMPTY_ANSWER — actor failed to commit ──────────────────────
    answer = sample.get("answer")
    answer_err = sample.get("answer_error")
    if (not answer or (isinstance(answer, str) and not answer.strip())) or answer_err:
        out.append(FailureTrace(
            skill_id="",
            skill_episode_id=f"{sid}#empty_answer",
            domain=domain,
            failed_step_index=0,
            failure_class="INVARIANT_VIOLATION",
            abort_reason=(
                f"answer_error={answer_err}" if answer_err
                else "answer_empty"
            ),
            extra={
                **common_extra,
                SEMANTIC_BUCKET_EXTRA_KEY: _bucket("EMPTY_ANSWER"),
                "synthesis_signal": "EMPTY_ANSWER",
                "answer_error": answer_err,
            },
        ))

    # ── 4. MALFORMED_SCHEMA — VLM vision call produced dirty schema ──
    schema_recovery = (sample.get("schema_recovery") or "").lower()
    if schema_recovery in _DIRTY_SCHEMA_RECOVERIES:
        out.append(FailureTrace(
            skill_id="",
            skill_episode_id=f"{sid}#malformed_schema",
            domain=domain,
            failed_step_index=0,
            failure_class="INVARIANT_VIOLATION",
            abort_reason=f"schema_recovery={schema_recovery}",
            extra={
                **common_extra,
                SEMANTIC_BUCKET_EXTRA_KEY: _bucket("MALFORMED_SCHEMA"),
                "synthesis_signal": "MALFORMED_SCHEMA",
                "schema_recovery": schema_recovery,
                "schema_source": sample.get("schema_source"),
                "schema_finish_reason": sample.get("schema_finish_reason"),
            },
        ))

    # Severity ordering is the list-construction order above.
    if len(out) > max_failures:
        out = out[:max_failures]
    return out


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _truncate(s: Optional[Any], *, n: int = 200) -> Optional[str]:
    """Defensive string trim; returns ``None`` for falsy / non-str."""
    if not s:
        return None
    txt = str(s)
    if len(txt) <= n:
        return txt
    return txt[: n - 1] + "…"


def _summarise_wrong_answer(
    sample: Dict[str, Any],
    judge: Dict[str, Any],
) -> str:
    """Build a one-line abort_reason that captures the wrong-answer
    locus without dumping the entire judge reasoning."""
    answer = _truncate(sample.get("answer"), n=80) or "<empty>"
    if judge and judge.get("reason"):
        return f"answer={answer!r}; judge={_truncate(judge.get('reason'), n=140)}"
    gold = _truncate(sample.get("gold_answer"), n=80) or "<missing>"
    return f"answer={answer!r}; gold={gold!r}"


__all__ = ["from_sample", "DEFAULT_MAX_FAILURES_PER_SAMPLE"]
