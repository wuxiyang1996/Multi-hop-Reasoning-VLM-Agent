"""Per-run JSONL instrumentation streams for reviewer-facing analysis.

This module owns 5 append-only logs that the trainer wires into via a
process-global facade. Each log answers one question that cannot be
recovered post-hoc from existing artifacts:

  * ``harness/rejections.jsonl`` — every per-event eligibility veto
    (status_not_runnable / domain_mismatch / no_adapter / …).  Drives
    the §4.3 failure-mode pie chart.
  * ``harness/validate.jsonl`` — every per-event ``validate_invocation``
    diagnostic (binding/precondition/evidence/adapter booleans, missing
    binding lists, failed predicate IDs).  Drives skill-failure case
    studies and §4.3 mutation→repair correspondence.
  * ``lifecycle/transitions.jsonl`` — every ``SkillStatus`` transition
    (DRAFT → PROVISIONAL → ACTIVE → DEPRECATED → RETIRED) with a
    timestamp and a reason tag.  Drives the §5.3 skill-lifetime
    distribution + promotion/deprecation curves.
  * ``intention/switches.jsonl`` — every per-step intention update
    (z_t prev → z_t new; whether it counts as a "sharp shift").  Drives
    §4.1 intention-trigger ablation.
  * ``runtime/component_timings.jsonl`` — per-component vLLM call
    counts, prompt/completion token counts, total latency.  Drives §6
    runtime-overhead analysis (NeurIPS checklist Q8).

Design constraints:

  * **Lazy open**: files are not created until the first ``log_*``
    call, so disabling a flag really means "no file written".
  * **Single fd per stream**: held in a module-level dict; threadsafe
    via a per-stream lock.
  * **Non-fatal**: any I/O error downgrades to a single ``logger.debug``
    line; the trainer's hot path never raises.
  * **No third-party deps**: stdlib ``json`` only.

The facade is ``set_run_dir(Path)``: the orchestrator calls it once at
startup; subsequent ``log_*`` calls fan out to the right file under that
directory.  A second call to ``set_run_dir`` re-targets all streams (used
when ``--resume`` is detected and the writer must point at the resumed
run dir).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TextIO

logger = logging.getLogger(__name__)

# ── Stream registry ─────────────────────────────────────────────────────

#: Subdir + filename for each stream id.
_STREAM_PATHS: Dict[str, tuple[str, str]] = {
    "harness_rejection": ("harness_log", "rejections.jsonl"),
    "harness_validate": ("harness_log", "validate.jsonl"),
    "lifecycle_transition": ("lifecycle_log", "transitions.jsonl"),
    "intention_switch": ("intention_log", "switches.jsonl"),
    "component_timing": ("runtime_log", "component_timings.jsonl"),
}

_run_dir_lock = threading.Lock()
_run_dir: Optional[Path] = None
_handles: Dict[str, TextIO] = {}
_stream_locks: Dict[str, threading.Lock] = {
    key: threading.Lock() for key in _STREAM_PATHS
}


def set_run_dir(path: Optional[Path]) -> None:
    """Point all loggers at ``path``.  Closes prior handles.

    Called once by the orchestrator at startup.  ``None`` disables all
    logging (subsequent ``log_*`` calls become no-ops).
    """
    global _run_dir
    with _run_dir_lock:
        for key, fh in list(_handles.items()):
            try:
                fh.close()
            except Exception:
                pass
        _handles.clear()
        _run_dir = Path(path) if path is not None else None


def _get_handle(stream: str) -> Optional[TextIO]:
    """Return an open append-mode fd for ``stream`` (lazy open).

    Returns ``None`` when ``set_run_dir`` has not been called or when
    file creation fails.
    """
    rd = _run_dir
    if rd is None:
        return None
    fh = _handles.get(stream)
    if fh is not None:
        return fh
    subdir, fname = _STREAM_PATHS[stream]
    target = rd / subdir / fname
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = open(target, "a", encoding="utf-8", buffering=1)  # line-buffered
        _handles[stream] = fh
        return fh
    except Exception as exc:
        logger.debug(
            "RunLoggers: failed to open %s (%s) — disabling stream",
            target, exc,
        )
        return None


def _emit(stream: str, row: Mapping[str, Any]) -> None:
    """Atomically write ``row`` (JSON-encoded) to ``stream``."""
    if _run_dir is None:
        return
    lock = _stream_locks[stream]
    with lock:
        fh = _get_handle(stream)
        if fh is None:
            return
        try:
            fh.write(json.dumps(row, default=str, ensure_ascii=False))
            fh.write("\n")
        except Exception as exc:
            logger.debug("RunLoggers: write to %s failed: %s", stream, exc)


# ── Public log_* facade ─────────────────────────────────────────────────


def log_harness_rejection(
    *,
    step: int,
    episode_id: str,
    game: str,
    domain: str,
    task: str,
    skill_id: str,
    veto: str,
    veto_reason: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Append one per-event eligibility-filter rejection.

    ``veto`` is one of the 8 codes defined in ``harness/eligibility.py``
    (``status_not_runnable / shadow_disallowed / domain_mismatch /
    task_mismatch / skill_type_mismatch / no_adapter / adapter_raised /
    adapter_cannot_handle``).
    """
    row: Dict[str, Any] = {
        "kind": "harness_rejection",
        "step": int(step),
        "episode_id": str(episode_id),
        "game": str(game),
        "domain": str(domain),
        "task": str(task or ""),
        "skill_id": str(skill_id),
        "veto": str(veto),
        "veto_reason": str(veto_reason or ""),
        "ts": time.time(),
    }
    if extra:
        row["extra"] = dict(extra)
    _emit("harness_rejection", row)


def log_harness_validate(
    *,
    step: int,
    episode_id: str,
    game: str,
    inner_step: int,
    skill_id: str,
    ok: bool,
    binding_ok: bool,
    precondition_ok: bool,
    evidence_ok: bool,
    adapter_ok: bool,
    shadow_only: bool = False,
    veto_reasons: Optional[list] = None,
    missing_bindings: Optional[list] = None,
    missing_evidence_in: Optional[list] = None,
    failed_preconditions: Optional[list] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Append one per-event ``validate_invocation`` diagnostic.

    Logged from inside the episode loop — one row per call to
    :meth:`SkillHarness.validate_invocation`.  Captures both successful
    invocations (for retrieval-frequency cross-validation) and vetoes
    (for case-study repair traces).
    """
    row: Dict[str, Any] = {
        "kind": "harness_validate",
        "step": int(step),
        "episode_id": str(episode_id),
        "game": str(game),
        "inner_step": int(inner_step),
        "skill_id": str(skill_id),
        "ok": bool(ok),
        "binding_ok": bool(binding_ok),
        "precondition_ok": bool(precondition_ok),
        "evidence_ok": bool(evidence_ok),
        "adapter_ok": bool(adapter_ok),
        "shadow_only": bool(shadow_only),
        "veto_reasons": list(veto_reasons or []),
        "missing_bindings": list(missing_bindings or []),
        "missing_evidence_in": list(missing_evidence_in or []),
        "failed_preconditions": list(failed_preconditions or []),
        "ts": time.time(),
    }
    if extra:
        row["extra"] = dict(extra)
    _emit("harness_validate", row)


def log_lifecycle_transition(
    *,
    skill_id: str,
    from_status: str,
    to_status: str,
    reason: str = "",
    step: Optional[int] = None,
    game: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Append one ``SkillStatus`` transition.

    Hooked into ``SkillLifecycleManager._set_status``.  ``step`` is the
    trainer outer step when known (best-effort: lifecycle may transition
    outside the trainer step boundary, e.g. during dump-driver bring-up).
    """
    row: Dict[str, Any] = {
        "kind": "lifecycle_transition",
        "skill_id": str(skill_id),
        "from_status": str(from_status),
        "to_status": str(to_status),
        "reason": str(reason or ""),
        "step": int(step) if step is not None else None,
        "game": str(game or ""),
        "ts": time.time(),
    }
    if extra:
        row["extra"] = dict(extra)
    _emit("lifecycle_transition", row)


def log_intention_switch(
    *,
    step: int,
    episode_id: str,
    game: str,
    inner_step: int,
    prev_intention: str,
    new_intention: str,
    switched: bool,
    sharp_shift: bool = False,
    summary_state_delta: str = "",
    urgency: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Append one per-step intention update.

    ``sharp_shift`` distinguishes meaningful subgoal jumps (tag-prefix
    inequality OR high urgency) from cosmetic re-phrasings, supporting
    the §4.1 sharp-shift threshold definition.
    """
    row: Dict[str, Any] = {
        "kind": "intention_switch",
        "step": int(step),
        "episode_id": str(episode_id),
        "game": str(game),
        "inner_step": int(inner_step),
        "prev_intention": str(prev_intention or ""),
        "new_intention": str(new_intention or ""),
        "switched": bool(switched),
        "sharp_shift": bool(sharp_shift),
        "summary_state_delta": str(summary_state_delta or "")[:512],
        "urgency": str(urgency or ""),
        "ts": time.time(),
    }
    if extra:
        row["extra"] = dict(extra)
    _emit("intention_switch", row)


def log_component_timing(
    *,
    step: int,
    component: str,
    n_calls: int,
    total_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Append a per-component aggregated runtime row for one trainer step.

    ``component`` should be one of: ``actor.action_taking`` /
    ``actor.skill_selection`` / ``actor.intention`` / ``actor.contract`` /
    ``segment`` / ``curator`` / ``crafter.deterministic`` / ``crafter.llm`` /
    ``promotion.judge`` / ``harness.validator`` / ``schema.profile``.

    Logged once per component per trainer step (orchestrator-driven).
    """
    row: Dict[str, Any] = {
        "kind": "component_timing",
        "step": int(step),
        "component": str(component),
        "n_calls": int(n_calls),
        "total_ms": float(total_ms),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "ts": time.time(),
    }
    if extra:
        row["extra"] = dict(extra)
    _emit("component_timing", row)


# ── Per-step component-timing aggregator (block A5 — non-actor side) ───────
#
# The actor's vLLM client maintains its own per-adapter aggregate (see
# ``AsyncVLLMClient.snapshot_per_component``).  35B / external API
# calls (crafter LLM, promotion judge, LLM harness validator, schema
# profile generation) go through ``API_func.ask_vllm`` outside that
# client, so they need a separate aggregator that the trainer's hot
# wrappers can pump into.  The orchestrator calls
# ``flush_component_timings(step)`` once per trainer step.

_component_agg_lock = threading.Lock()
_component_agg: Dict[str, Dict[str, float]] = {}


def record_component_call(
    component: str,
    *,
    latency_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    n_calls: int = 1,
) -> None:
    """Aggregate one external (35B / API) call under ``component``.

    Cheap (microsecond) book-keeping.  Drained per trainer step via
    :func:`flush_component_timings`.
    """
    if not component:
        return
    with _component_agg_lock:
        bucket = _component_agg.get(component)
        if bucket is None:
            bucket = {
                "n_calls": 0,
                "total_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
            _component_agg[component] = bucket
        bucket["n_calls"] += int(n_calls)
        bucket["total_ms"] += float(latency_ms)
        bucket["prompt_tokens"] += int(prompt_tokens)
        bucket["completion_tokens"] += int(completion_tokens)


def flush_component_timings(step: int) -> Dict[str, Dict[str, float]]:
    """Drain the in-process aggregator and emit one row per component.

    Called by the orchestrator at the end of each trainer step.  Returns
    the snapshot dict (also written to JSONL) so callers can fold it
    into a step-summary if desired.
    """
    with _component_agg_lock:
        snap = {k: dict(v) for k, v in _component_agg.items()}
        _component_agg.clear()
    for component, bucket in snap.items():
        log_component_timing(
            step=step,
            component=component,
            n_calls=int(bucket.get("n_calls", 0)),
            total_ms=float(bucket.get("total_ms", 0.0)),
            prompt_tokens=int(bucket.get("prompt_tokens", 0)),
            completion_tokens=int(bucket.get("completion_tokens", 0)),
        )
    return snap


import contextlib  # noqa: E402  (kept at module bottom — only used by helper)


@contextlib.contextmanager
def measure_component(component: str):
    """Context manager that wall-clocks a block and records the elapsed
    ms under ``component`` (token counts default to 0 — for 35B / API
    callers where token usage is not directly observable).

    Example::

        with measure_component("crafter.llm"):
            result = ask_vllm(prompt, model="...")
    """
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        record_component_call(component, latency_ms=elapsed_ms)
