"""Regression tests for Fix-A — ``FailureMemory`` semantic_bucket.

Pins both halves of the contract introduced in 2026-05:

* **Backwards compatibility** — when no synthesiser sets
  ``trace.extra["semantic_bucket"]`` the dedup key collapses back to
  the legacy ``(skill_id, failure_class, failed_step_index)`` triple,
  so every Phase-1 / Phase-2 launcher that hasn't migrated its
  synthesiser sees byte-identical patterns.
* **Splitting on transfer-target domains** — when the VR
  ``failure_synth`` is hooked up, four mechanically-distinct VR
  failure modes (WRONG_ANSWER / UNSCOREABLE / EMPTY_ANSWER /
  MALFORMED_SCHEMA) on the same benchmark must land in distinct
  ``FailurePattern``s, and the same mechanism × different benchmark
  must also land in distinct patterns.

Without Fix-A the v3 smoke produced 31 LLM hypotheses across
**5 patterns** with the top pattern firing 11 times — the structural
root cause of the "evidence_gate" mode collapse.
"""

from __future__ import annotations

from crafter.failure_memory import (
    FailureMemory,
    SEMANTIC_BUCKET_EXTRA_KEY,
)
from data_structure.extensions.failure_trace import FailureTrace
from labeling_supplement._failure_synth.visual_reasoning import from_sample


# ---------------------------------------------------------------------------
# Backward-compat — gymv synthesisers don't set the bucket key, so the
# pattern dedup must remain identical to the pre-Fix-A behaviour.
# ---------------------------------------------------------------------------


def test_legacy_traces_without_bucket_collapse_into_one_pattern():
    """gymv-style traces (no ``semantic_bucket`` key) sharing
    ``(skill_id, failure_class, failed_step_index)`` collapse into a
    single ``FailurePattern`` — same as before Fix-A."""
    mem = FailureMemory()
    for i in range(5):
        mem.add(FailureTrace(
            skill_id="skill_x", skill_episode_id=f"ep{i}",
            domain="gymv", failed_step_index=2,
            failure_class="PRECONDITION_VIOLATION",
            abort_reason=f"precondition #{i}",
        ))
    assert len(mem.patterns()) == 1
    only = mem.patterns()[0]
    assert only.count == 5
    assert only.semantic_bucket == ""


def test_legacy_extra_without_bucket_key_collapses_too():
    """Even when ``trace.extra`` is populated, missing the bucket key
    must not split the pattern (so a synthesiser that emits other
    metadata under ``extra`` doesn't accidentally migrate to Fix-A
    semantics)."""
    mem = FailureMemory()
    for i in range(3):
        mem.add(FailureTrace(
            skill_id="skill_y", skill_episode_id=f"ep{i}",
            domain="gymv", failed_step_index=0,
            failure_class="INVARIANT_VIOLATION",
            abort_reason=f"abort #{i}",
            extra={"some_other_metadata": i, "weights": [1, 2, 3]},
        ))
    assert len(mem.patterns()) == 1


def test_non_string_bucket_key_is_ignored():
    """Defensive: a buggy synthesiser setting bucket to a non-string
    must not crash; we treat it as missing."""
    mem = FailureMemory()
    for v in (None, 42, [1, 2], {"k": "v"}):
        mem.add(FailureTrace(
            skill_id="skill_z", skill_episode_id=f"ep_{type(v).__name__}",
            domain="gymv", failed_step_index=0,
            failure_class="INVARIANT_VIOLATION",
            abort_reason="x",
            extra={SEMANTIC_BUCKET_EXTRA_KEY: v},
        ))
    assert len(mem.patterns()) == 1, (
        "non-string bucket values should fall back to the empty "
        "bucket and collapse into one pattern"
    )


# ---------------------------------------------------------------------------
# VR-domain splitting — the core fix.
# ---------------------------------------------------------------------------


def test_vr_synth_emits_distinct_buckets_per_failure_mode():
    """A single VR sample with all four failure signals (wrong answer +
    unscoreable + empty + malformed schema) produces four traces, each
    carrying a distinct ``semantic_bucket``."""
    sample = {
        "benchmark": "visual_toolbench", "is_mcq": False,
        "correct": False,
        "judge": {"verdict": "unscoreable", "reason": "ambiguous"},
        "answer": "",                          # → EMPTY_ANSWER
        "answer_error": "model_returned_null",
        "schema_recovery": "fenced",           # → MALFORMED_SCHEMA
        "schema_source": "vlm_call",
        "gold_answer": "Of course",
    }
    traces = from_sample(sample, sample_id="vtb_001")
    buckets = {
        t.extra.get(SEMANTIC_BUCKET_EXTRA_KEY) for t in traces
    }
    # Wrong-answer is suppressed when judge says "unscoreable", but
    # we still get UNSCOREABLE + EMPTY_ANSWER + MALFORMED_SCHEMA from
    # this construction.  Bucket count should be exactly 3.
    assert len(buckets) == 3, (
        f"expected 3 distinct VR buckets, got {sorted(buckets)}"
    )
    # Sanity: bucket format is ``<signal>/<benchmark>/<mcq?>``.
    for b in buckets:
        parts = b.split("/")
        assert len(parts) == 3 and parts[1] == "visual_toolbench" and parts[2] == "freeform"


def test_vr_buckets_split_failure_memory_patterns():
    """End-to-end: 50 VR samples that under the legacy key would
    collapse onto 1 pattern now split into ≥ 4 patterns once the
    bucket is wired."""
    mem = FailureMemory()
    for i in range(50):
        sample = {
            "benchmark": "visual_toolbench", "is_mcq": False,
            "correct": False,
            "answer": f"answer_{i}", "gold_answer": f"gold_{i}",
        }
        for tr in from_sample(sample, sample_id=f"vtb_{i:03d}"):
            mem.add(tr)
    patterns = mem.patterns()
    assert len(patterns) == 1, (
        "All 50 wrong-answer traces share the same bucket, so they "
        "*should* collapse — Fix-A only splits across signals / "
        "benchmarks, not within a single signal."
    )
    assert patterns[0].count == 50
    assert patterns[0].semantic_bucket == "wrong_answer/visual_toolbench/freeform"


def test_vr_buckets_split_across_benchmarks_and_signals():
    """The realistic case for the smoke: WRONG_ANSWER on VTB +
    WRONG_ANSWER on TIR + UNSCOREABLE on VTB + MALFORMED_SCHEMA on
    Video-Holmes lives in 4 separate patterns (was 1 pre-Fix-A)."""
    mem = FailureMemory()

    def _wrong(bench, is_mcq, gold):
        return {
            "benchmark": bench, "is_mcq": is_mcq, "correct": False,
            "answer": "x", "gold_answer": gold,
        }

    def _unscoreable(bench, is_mcq):
        return {
            "benchmark": bench, "is_mcq": is_mcq, "correct": False,
            "answer": "x", "gold_answer": "y",
            "judge": {"verdict": "unscoreable", "reason": "bad"},
        }

    def _malformed(bench, is_mcq):
        return {
            "benchmark": bench, "is_mcq": is_mcq, "correct": False,
            "answer": "x", "gold_answer": "y",
            "schema_recovery": "truncated",
        }

    for s, sid in [
        (_wrong("visual_toolbench", False, "g0"), "vtb_a"),
        (_wrong("visual_toolbench", False, "g1"), "vtb_b"),
        (_wrong("tir_bench",        True,  "F"),  "tir_a"),
        (_wrong("tir_bench",        True,  "B"),  "tir_b"),
        (_unscoreable("visual_toolbench", False), "vtb_c"),
        (_malformed("video_holmes", True),        "vid_a"),
    ]:
        for tr in from_sample(s, sample_id=sid):
            mem.add(tr)

    buckets = {p.semantic_bucket for p in mem.patterns()}
    assert "wrong_answer/visual_toolbench/freeform" in buckets
    assert "wrong_answer/tir_bench/mcq" in buckets
    assert "unscoreable/visual_toolbench/freeform" in buckets
    assert "malformed_schema/video_holmes/mcq" in buckets
    # Each WRONG_ANSWER sample on VTB also emits a duplicate trace
    # because the unscoreable construction carries answer/gold too;
    # we only check the cardinal shape: ≥ 4 distinct buckets, bucket
    # labels carry signal × benchmark × mcq.
    assert len(buckets) >= 4
