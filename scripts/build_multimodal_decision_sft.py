"""Build high-quality decision-agent SFT JSONL from multi-model multimodal corpora.

Each benchmark draws ``correct=True`` samples from EVERY frontier model that
labeled it, so the SFT pool is the union over models (deduped by
``(sample_id, model)``).  We deliberately exclude rollouts produced by our own
9B/35B trainees (``Qwen/Qwen3.5-9B``, ``Qwen/Qwen3.5-35B-A3B``) — distilling
their outputs back into themselves is degenerate.

Beyond the QA + miniwob sources, this script also pulls the **rollout-corpus**
sources used by the original ``labeling/build_decision_sft_jsonl.py`` pipeline:

    gymv          (8 retro Genesis games × 4 frontier models)
        gpt-5.4      labeled  (intentions + skill_query + active_skill)
        claude-4.6   raw      (state + action + reward only)
        gemini-3.1   raw
        qwen3-vl-235b raw

    env_wrappers  (4 games: tetris, twenty_forty_eight, candy_crush, super_mario)
        gpt-5.4      labeled  (no other model rollouts available)

Multi-model source coverage:

    video_holmes         Cold-start gpt-5.4   396  +  Claude 4.6   458
                         Gemini 3.1 pro        64  +  Qwen3-VL-235B 357
                         => ~1275 correct rows across 4 frontier models.
    siv_bench            Cold-start gpt-5.4   220  +  Claude 4.6   245
                         Gemini 3.1 pro       131  +  Qwen3-VL-235B 211
                         => ~807 correct rows.
    tir_bench            gpt-5.4  102 + Claude 81 + Gemini 56 + Qwen3-VL 65
                         => ~304 correct rows (141 MCQ).
    visual_toolbench     gpt-5.4   29 + Claude 14 + Gemini  7 + Qwen3-VL 25
                         => ~75 correct rows (all open-ended).
    miniwob/browsergym   Cold-start gpt-5.4 (75 ep / 215 step)
                         Claude 4.6        (82 ep / 227 step)
                         Gemini 3.1 pro    (82 ep / 237 step)
                         Qwen3-VL-235B     (73 ep / 226 step)
                         => ~312 successful episodes / ~905 step rows.

Output layout (mirrors ``labeling/decision_sft_jsonl/run_<ts>``):

    labeling/decision_sft_jsonl/run_multimodal_<ts>/
    ├── _run_summary.json
    ├── video_holmes/action_taking.jsonl
    ├── siv_bench/action_taking.jsonl
    ├── tir_bench/action_taking.jsonl
    ├── visual_toolbench/action_taking.jsonl
    ├── miniwob/action_taking.jsonl

Row schema matches ``labeling/build_decision_sft_jsonl.py:build_action_taking_row``
verbatim so the existing trainer (``trainer/SFT/data_loader.py``) ingests it
without changes.  Each row carries an extra ``source_model`` field so we can
later weight, ablate, or filter by model provenance.

Usage:
    python scripts/build_multimodal_decision_sft.py [--out-root DIR] [--dry-run] \
        [--sources video_holmes,siv_bench,tir_bench,visual_toolbench,miniwob] \
        [--limit N] [--include-open]

By default ``--include-open`` is OFF: we skip open-ended QA (visual_toolbench
+ tir_bench-open) because the decision agent expects discrete actions.  Pass
the flag to synthesize 2-option binary rows (gold-answer vs. DEFER) for them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reuse the canonical row builders so our format stays bit-identical with the
# legacy ``labeling/build_decision_sft_jsonl.py`` outputs that the trainer
# already knows how to ingest.
sys.path.insert(0, str(REPO_ROOT))
from labeling.build_decision_sft_jsonl import (  # noqa: E402
    build_action_taking_row as _legacy_build_action_taking_row,
    build_skill_selection_row as _legacy_build_skill_selection_row,
    _resolve_image as _legacy_resolve_image,
)

# Model labels we *exclude* — these are our own trainee/perceptor weights.
# Distilling their own outputs back is degenerate and wastes capacity.
EXCLUDED_MODEL_PREFIXES = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-35B-A3B")


def _short_model_label(raw: str) -> str:
    """Compact, filesystem-safe model label (e.g. ``claude-4.6``)."""
    if not raw:
        return "unknown"
    last = raw.split("/")[-1].lower()
    if "claude" in raw.lower():
        return "claude-4.6"
    if "gemini" in raw.lower():
        return "gemini-3.1-pro"
    if "qwen3-vl" in raw.lower() or "qwen3-vl-235b" in raw.lower():
        return "qwen3-vl-235b"
    if "gpt-5.4" in raw.lower():
        return "gpt-5.4"
    if "gpt-5.5" in raw.lower():
        return "gpt-5.5"
    return last.replace(":", "_")[:32]

# ---------------------------------------------------------------------------
# Static prompt templates  (verbatim from labeling/build_decision_sft_jsonl.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert game-playing agent. "
    "You receive a game state and must choose exactly one action by its NUMBER.\n\n"
    "Rules:\n"
    "- Study the state carefully before choosing.\n"
    "- Consider which action makes the most progress toward winning.\n"
    "- NEVER repeat the same action more than 2 times in a row.\n"
    "- If recent actions got zero reward, change strategy.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences>\n"
    "ACTION: <number>\n"
)

# Trainer subgoal vocabulary -- composite tags fall back to EXECUTE.
SUBGOAL_TAGS = frozenset({
    "SETUP", "CLEAR", "MERGE", "ATTACK", "DEFEND",
    "NAVIGATE", "POSITION", "COLLECT", "BUILD", "SURVIVE",
    "OPTIMIZE", "EXPLORE", "EXECUTE",
})

# Map QA dimensions / question_types onto an "active_skill" handle.  These
# strings only need to be stable identifiers -- the SFT loader treats them
# as opaque skill labels, but the curator can pick them up later for merge.
VIDEO_HOLMES_SKILL_MAP = {
    "MHR": "video_qa/MULTIHOP_REASONING",
    "SR":  "video_qa/SOCIAL_REASONING",
    "TCI": "video_qa/CAUSE_INFERENCE",
    "IMC": "video_qa/INTENT_MOTIVE",
    "PAR": "video_qa/PHYSICAL_ACTION",
    "TA":  "video_qa/TEMPORAL_AWARENESS",
    "CTI": "video_qa/CROSS_TIME",
}

SIV_BENCH_SKILL_MAP = {
    "Emotion Inference":         "social_qa/EMOTION_INFERENCE",
    "Mental State":              "social_qa/MENTAL_STATE",
    "Social Norm":               "social_qa/SOCIAL_NORM",
    "Intent Inference":          "social_qa/INTENT_INFERENCE",
    "Relation Inference":        "social_qa/RELATION_INFERENCE",
    "Behavior Prediction":       "social_qa/BEHAVIOR_PREDICTION",
    "Counterfactual Prediction": "social_qa/COUNTERFACTUAL",
    "Environment Perception":    "social_qa/ENVIRONMENT_PERCEPTION",
    "Attitude Inference":        "social_qa/ATTITUDE_INFERENCE",
    "Activity Recognition":      "social_qa/ACTIVITY_RECOGNITION",
    "Causal Reasoning":          "social_qa/CAUSAL_REASONING",
    "Identity Recognition":      "social_qa/IDENTITY_RECOGNITION",
    "Temporal Inference":        "social_qa/TEMPORAL_INFERENCE",
}

TIR_BENCH_SKILL_MAP = {
    "rotation":           "image_qa/SPATIAL_ROTATION",
    "counting":           "image_qa/COUNTING",
    "spatial":            "image_qa/SPATIAL_REASONING",
    "ocr":                "image_qa/OCR_READING",
    "geometry":           "image_qa/GEOMETRY",
    "perception":         "image_qa/PERCEPTION",
    "math":               "image_qa/VISUAL_MATH",
    "visual_search":      "image_qa/VISUAL_SEARCH",
    "table":              "image_qa/TABLE_READING",
    "chart":              "image_qa/CHART_READING",
}


# ---------------------------------------------------------------------------
# Helpers shared with build_decision_sft_jsonl.py
# ---------------------------------------------------------------------------
def _format_numbered_actions(actions: List[str]) -> str:
    return "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))


def _action_index_1based(action: str, valid_actions: List[str]) -> int:
    target = (action or "").strip()
    for i, a in enumerate(valid_actions):
        if a == action:
            return i + 1
    target_low = target.lower()
    for i, a in enumerate(valid_actions):
        if (a or "").strip().lower() == target_low:
            return i + 1
    return 1


def _trim(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _emit_row(
    schema_text: str,
    valid_actions: List[str],
    correct_action: str,
    reasoning: str,
    *,
    game: str,
    corpus: str,
    episode_id: str,
    step_idx: int,
    intention_subgoal: str = "EXECUTE",
    intention_note: str = "",
    active_skill: str = "",
    skill_pass_rate: Optional[float] = None,
    skill_n_instances: Optional[int] = None,
    reward: Optional[float] = 1.0,
    image: Optional[Dict[str, Any]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a single action_taking.jsonl row following the canonical shape."""
    user = (
        f"Game state:\n\n{schema_text}\n\n"
        f"Available actions (pick ONE by number):\n"
        f"{_format_numbered_actions(valid_actions)}\n\n"
        f"Choose the best action. Output REASONING then ACTION number."
    )
    prompt = SYSTEM_PROMPT + "\n" + user

    action_num = _action_index_1based(correct_action, valid_actions)
    note = _trim(reasoning or "Expert demonstration.", 200)
    completion = f"REASONING: {note}\nACTION: {action_num}"

    sg = intention_subgoal.upper() if intention_subgoal else "EXECUTE"
    if sg not in SUBGOAL_TAGS:
        sg = "EXECUTE"
    intention_short = f"[{sg}] {_trim(intention_note or note, 160)}"

    row: Dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        "intention": intention_short,
        "active_skill": active_skill,
        "game": game,
        "corpus": corpus,
        "episode_id": episode_id,
        "step_idx": step_idx,
        "valid_actions": list(valid_actions),
        "intention_full": f"[{sg}] {_trim(intention_note or note, 400)}",
        "intention_operator": sg,
        "intention_subgoal": sg,
        "reward": reward,
    }
    if active_skill:
        row["skill_execution_hint"] = _trim(intention_note or note, 240)
        if skill_pass_rate is not None:
            row["skill_pass_rate"] = skill_pass_rate
        if skill_n_instances is not None:
            row["skill_n_instances"] = skill_n_instances
    if image is not None:
        row["image"] = image
    if extra_fields:
        row.update(extra_fields)
    return row


# ---------------------------------------------------------------------------
# Source: video_holmes / siv_bench / tir_bench (MCQ)
# ---------------------------------------------------------------------------
def _qa_sample_to_row(
    sample: Dict[str, Any],
    *,
    bench: str,
    corpus: str,
    skill_map: Dict[str, str],
    intent_key: str,  # which raw_sample field carries the question_type
) -> Optional[Dict[str, Any]]:
    if not sample.get("correct"):
        return None
    if not sample.get("is_mcq"):
        return None
    schema = (sample.get("schema") or "").strip()
    if not schema:
        return None
    valid_actions = list(sample.get("valid_actions") or [])
    if len(valid_actions) < 2:
        return None
    gold = (sample.get("gold_answer") or "").strip()
    if not gold or gold not in valid_actions:
        return None

    raw_sample = sample.get("raw_sample") or {}
    qtype = (raw_sample.get(intent_key)
             or raw_sample.get("question_type")
             or raw_sample.get("dimension")
             or "UNKNOWN")
    skill_id = skill_map.get(qtype, f"{bench}/UNKNOWN")
    note = sample.get("answer_reasoning") or ""

    sample_id = str(sample.get("sample_id") or sample.get("task_id") or "qa")
    episode_id = f"{bench}__{sample_id}"

    image = None
    video_meta = sample.get("video_meta") or {}
    if video_meta.get("video_path"):
        image = {"path": video_meta["video_path"], "mime_type": "video/mp4"}

    question_text = (sample.get("question") or "").strip()
    options_block = (sample.get("options_block") or "").strip()
    # Some benches (tir_bench) inline the options into ``question``; avoid
    # duplicating them when both representations exist.
    has_inline_opts = bool(re.search(r"(?:^|\n)\s*[A-F]\.\s+", question_text))
    if has_inline_opts or not options_block:
        schema_with_q = f"{schema}\n\n<question>\n{question_text}"
    else:
        schema_with_q = f"{schema}\n\n<question>\n{question_text}\n\n{options_block}"

    return _emit_row(
        schema_with_q,
        valid_actions,
        gold,
        note,
        game=bench,
        corpus=corpus,
        episode_id=episode_id,
        step_idx=0,
        intention_subgoal="EXECUTE",
        intention_note=f"[{qtype}] {_trim(note, 200)}",
        active_skill=skill_id,
        skill_pass_rate=1.0,
        skill_n_instances=1,
        reward=1.0,
        image=image,
        extra_fields={"question_type": qtype, "sample_id": sample_id},
    )


# ---------------------------------------------------------------------------
# Source: visual_toolbench + tir_bench (open-ended)  [behind --include-open]
# ---------------------------------------------------------------------------
DEFER_ACTION = "DEFER (insufficient evidence in current state)"


def _qa_open_to_row(
    sample: Dict[str, Any],
    *,
    bench: str,
    corpus: str,
) -> Optional[Dict[str, Any]]:
    """Synthesize a 2-option row: (truncated gold answer) vs. DEFER."""
    if not sample.get("correct"):
        return None
    if sample.get("is_mcq"):
        return None
    schema = (sample.get("schema") or "").strip()
    if not schema:
        return None
    gold = _trim((sample.get("gold_answer") or "").strip(), 80)
    if not gold:
        return None
    valid_actions = [gold, DEFER_ACTION]
    note = sample.get("answer_reasoning") or ""
    sample_id = str(sample.get("sample_id") or "open")
    episode_id = f"{bench}__{sample_id}"
    schema_with_q = (
        f"{schema}\n\n<question>\n{sample.get('question','').strip()}"
    )
    return _emit_row(
        schema_with_q,
        valid_actions,
        gold,
        note,
        game=bench,
        corpus=corpus,
        episode_id=episode_id,
        step_idx=0,
        intention_subgoal="EXECUTE",
        intention_note=_trim(note, 200),
        active_skill=f"{bench}/OPEN_QA",
        skill_pass_rate=1.0,
        skill_n_instances=1,
        reward=1.0,
        extra_fields={"sample_id": sample_id, "is_mcq": False},
    )


# ---------------------------------------------------------------------------
# Source: miniwob (browsergym episode-step decomposition)
# ---------------------------------------------------------------------------
_BID_RE = re.compile(r"e\d+\[[^\]]*?bid=(\d+)[^\]]*?\]")
_CLICKABLE_RE = re.compile(r"\bclickable\b")
_TYPEABLE_RE = re.compile(r"\binput\b|\bedit\b|\beditable\b")


def _extract_clickable_bids_from_schema(schema_text: str) -> List[Tuple[str, str]]:
    """Return [(bid, label_for_action)] for entities whose attribute line
    contains 'clickable'.

    The miniwob schema layout is::

        <entities>
        e1[type=element, label=button 'Submit', bid=12, ...]
        ...
        <attributes>
        e1.state=visible,clickable
    """
    if not schema_text:
        return []
    # Map entity_id -> bid
    eid_to_bid: Dict[str, str] = {}
    for ln in schema_text.splitlines():
        m = re.match(r"\s*(e\d+)\[[^\]]*?bid=(\d+)", ln)
        if m:
            eid_to_bid[m.group(1)] = m.group(2)
    bids: List[Tuple[str, str]] = []
    for ln in schema_text.splitlines():
        m = re.match(r"\s*(e\d+)\.state=([^\n]+)", ln)
        if not m:
            continue
        eid, attrs = m.group(1), m.group(2)
        bid = eid_to_bid.get(eid)
        if bid and _CLICKABLE_RE.search(attrs):
            bids.append((bid, eid))
    return bids


def _build_miniwob_valid_actions(step: Dict[str, Any]) -> List[str]:
    """Combine browsergym navigation primitives with bid-specific clicks."""
    nav = list(step.get("available_actions") or [])
    schema = ((step.get("metadata") or {}).get("schema_canonical")
              or (step.get("metadata") or {}).get("schema")
              or "")
    bid_clicks = [f'click("{bid}")' for bid, _ in _extract_clickable_bids_from_schema(schema)]
    seen = set()
    out: List[str] = []
    for a in bid_clicks + nav:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _episode_total_reward(ep: Dict[str, Any]) -> float:
    """Best-effort total-reward extraction for a BrowserGym episode.

    Order of preference:
      1. ``rollout_metadata.total_reward`` (set by the cold-start runner;
         present on both miniwob and webshop episodes).
      2. ``total_reward`` at the top level (older snapshots).
      3. The ``reward`` of the last experience (miniwob's binary terminal
         signal, webshop's terminal partial-credit signal).
    """
    rm = ep.get("rollout_metadata") or {}
    if isinstance(rm, dict) and rm.get("total_reward") is not None:
        try:
            return float(rm["total_reward"])
        except (TypeError, ValueError):
            pass
    if ep.get("total_reward") is not None:
        try:
            return float(ep["total_reward"])
        except (TypeError, ValueError):
            pass
    exps = ep.get("experiences") or []
    if exps:
        last = exps[-1] or {}
        try:
            return float(last.get("reward") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _miniwob_episode_to_rows(
    ep: Dict[str, Any],
    *,
    game: str,
    corpus: str,
    source_model: str = "unknown",
    success_threshold: float = 0.5,
    skill_prefix: str = "web",
) -> List[Dict[str, Any]]:
    """Turn one BrowserGym (miniwob or webshop) episode into action_taking rows.

    *success_threshold* is the minimum total episode reward required for any
    rows to be emitted.  miniwob is binary so 0.5 maps to "successful";
    webshop's reward is granular (0, 0.33, 0.5, 0.67, 0.75, 1.0) so 0.5 means
    "≥ half the credit".  Pass 0.0 to keep every episode.

    *skill_prefix* tags the synthesised ``active_skill`` (``web`` for miniwob,
    ``webshop`` for webshop) so the downstream skill bank can keep the two
    domains separable when they share a corpus name.
    """
    final_r = _episode_total_reward(ep)
    if final_r < success_threshold:
        return []
    rows: List[Dict[str, Any]] = []
    episode_id = f"{ep.get('episode_id') or ''}__{source_model}"
    for i, step in enumerate(ep.get("experiences") or []):
        action = step.get("action") or ""
        if not action:
            continue
        schema = ((step.get("metadata") or {}).get("schema_canonical")
                  or (step.get("metadata") or {}).get("schema")
                  or step.get("state") or "")
        if not schema:
            continue
        valid_actions = _build_miniwob_valid_actions(step)
        if action not in valid_actions:
            valid_actions = [action] + valid_actions
        note = (step.get("intentions") or "").strip() or "Web task step."
        if action.startswith("click"):
            sg = "EXECUTE"
        elif action.startswith(("type", "fill", "press")):
            sg = "EXECUTE"
        elif action.startswith(("scroll", "go_back", "go_forward")):
            sg = "NAVIGATE"
        else:
            sg = "EXECUTE"

        verb = action.split("(")[0].upper() or "ACTION"
        row = _emit_row(
            schema,
            valid_actions,
            action,
            note,
            game=game,
            corpus=corpus,
            episode_id=episode_id,
            step_idx=i,
            intention_subgoal=sg,
            intention_note=note,
            active_skill=f"{skill_prefix}/{verb}",
            skill_pass_rate=float(final_r),
            skill_n_instances=len(ep.get("experiences") or []),
            reward=float(step.get("reward") or 0.0),
            extra_fields={"task_goal": (step.get("goal") or step.get("tasks") or "")[:200]},
        )
        row["source_model"] = source_model
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def _process_qa_source(
    samples_path: Path,
    *,
    bench: str,
    corpus: str,
    skill_map: Dict[str, str],
    intent_key: str,
    include_open: bool,
    rows_out: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Read one ``samples.jsonl`` file (one model's labels) and append rows.

    Each row gets ``source_model`` so we can break things down later.  Rows
    from EXCLUDED_MODEL_PREFIXES are skipped so we never SFT-distill our own
    9B/35B onto itself.
    """
    if not samples_path.exists():
        return {"path": str(samples_path), "error": "missing"}

    n_total = n_correct = n_mcq_emit = n_open_emit = n_skipped_noschema = 0
    n_skipped_excluded = 0
    qtype_counter: Counter = Counter()
    model_label = "unknown"

    with samples_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except Exception:
                continue
            raw_model = (sample.get("model_routed") or sample.get("model") or "").strip()
            if any(raw_model.startswith(p) for p in EXCLUDED_MODEL_PREFIXES):
                n_skipped_excluded += 1
                continue
            model_label = _short_model_label(raw_model)
            n_total += 1
            if not sample.get("correct"):
                continue
            n_correct += 1
            if not (sample.get("schema") or "").strip():
                n_skipped_noschema += 1
                continue

            row = None
            if sample.get("is_mcq"):
                row = _qa_sample_to_row(
                    sample, bench=bench, corpus=corpus,
                    skill_map=skill_map, intent_key=intent_key,
                )
                if row is not None:
                    n_mcq_emit += 1
                    qtype_counter[row.get("question_type", "UNKNOWN")] += 1
            elif include_open:
                row = _qa_open_to_row(sample, bench=bench, corpus=corpus)
                if row is not None:
                    n_open_emit += 1
                    qtype_counter["__open__"] += 1

            if row is not None:
                row["source_model"] = model_label
                row["episode_id"] = f"{row['episode_id']}__{model_label}"
                rows_out.append(row)

    return {
        "path": str(samples_path),
        "model": model_label,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_mcq_emitted": n_mcq_emit,
        "n_open_emitted": n_open_emit,
        "n_skipped_no_schema": n_skipped_noschema,
        "n_skipped_excluded_model": n_skipped_excluded,
        "by_question_type": dict(qtype_counter),
    }


def _process_qa_bench(
    sources: List[Path],
    *,
    bench: str,
    corpus: str,
    out_path: Path,
    skill_map: Dict[str, str],
    intent_key: str,
    include_open: bool,
    dry_run: bool,
    limit: Optional[int],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    per_source: List[Dict[str, Any]] = []
    for src in sources:
        per_source.append(_process_qa_source(
            src, bench=bench, corpus=corpus,
            skill_map=skill_map, intent_key=intent_key,
            include_open=include_open, rows_out=rows,
        ))
        if limit is not None and len(rows) >= limit:
            rows = rows[:limit]
            break

    if not dry_run and rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_model = Counter(r.get("source_model", "unknown") for r in rows)
    by_qtype = Counter(r.get("question_type", "UNKNOWN") for r in rows)
    return {
        "bench": bench,
        "out": str(out_path),
        "n_sources": len(sources),
        "n_rows_written": len(rows) if not dry_run else 0,
        "n_rows_dry_run": len(rows) if dry_run else 0,
        "by_model": dict(by_model),
        "by_question_type": dict(by_qtype),
        "per_source": per_source,
        "preview": rows[:3] if rows else [],
    }


def _process_miniwob_root(
    root: Path,
    *,
    source_model: str,
    rows_out: List[Dict[str, Any]],
    only_miniwob: bool = True,
    task_glob: Optional[str] = None,
    success_threshold: float = 0.5,
    skill_prefix: str = "web",
) -> Dict[str, Any]:
    """Process one model's browsergym rollout root directory.

    ``task_glob`` overrides the per-task pattern (default ``miniwob.*`` when
    ``only_miniwob`` is True, ``*`` otherwise).  This is what lets webshop
    reuse the same code path with ``task_glob='webshop.*'``.
    """
    if not root.exists():
        return {"path": str(root), "error": "missing"}

    n_tasks = n_succ_eps = n_steps = 0
    if task_glob is not None:
        pattern = task_glob
    else:
        pattern = "miniwob.*" if only_miniwob else "*"
    for task_dir in sorted(root.glob(pattern)):
        if task_dir.name.startswith("_"):
            continue
        rollouts = task_dir / "rollouts.jsonl"
        if not rollouts.exists():
            continue
        n_tasks += 1
        task_name = task_dir.name
        with rollouts.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except Exception:
                    continue
                ep_rows = _miniwob_episode_to_rows(
                    ep, game=task_name, corpus="browsergym",
                    source_model=source_model,
                    success_threshold=success_threshold,
                    skill_prefix=skill_prefix,
                )
                if ep_rows:
                    n_succ_eps += 1
                    n_steps += len(ep_rows)
                    rows_out.extend(ep_rows)
    return {
        "path": str(root),
        "model": source_model,
        "n_task_dirs": n_tasks,
        "n_successful_episodes": n_succ_eps,
        "n_step_rows": n_steps,
        "task_glob": pattern,
        "success_threshold": success_threshold,
    }


# ---------------------------------------------------------------------------
# Source: rollout-corpus (gymv + env_wrappers, episode_*.json layout)
# ---------------------------------------------------------------------------
def _process_rollout_episode(
    ep_path: Path,
    *,
    game: str,
    corpus: str,
    source_model: str,
    rows_action: List[Dict[str, Any]],
    rows_skill: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """Reuse the legacy row builders for one ``episode_*.json`` file.

    Tags every row with ``source_model`` and suffixes the episode_id so rows
    from different models for the same episode_id stay distinct.
    """
    try:
        ep = json.loads(ep_path.read_text())
    except Exception:
        return 0, 0
    base_id = str(ep.get("episode_id") or ep_path.stem)
    episode_id = f"{base_id}__{source_model}"
    exps = ep.get("experiences") or ep.get("steps") or []
    image_root = ep_path.parent
    n_act = n_skill = 0
    for i, step in enumerate(exps):
        image = _legacy_resolve_image(step, image_root)
        ar = _legacy_build_action_taking_row(
            step, game=game, episode_id=episode_id,
            step_idx=i, image=image, corpus=corpus,
        )
        if ar:
            ar["source_model"] = source_model
            rows_action.append(ar)
            n_act += 1
        sr = _legacy_build_skill_selection_row(
            step, game=game, episode_id=episode_id,
            step_idx=i, image=image, corpus=corpus,
        )
        if sr:
            sr["source_model"] = source_model
            rows_skill.append(sr)
            n_skill += 1
    return n_act, n_skill


def _process_rollout_corpus(
    sources: List[Tuple[Path, str, Optional[List[str]]]],
    *,
    corpus: str,
    out_root: Path,
    dry_run: bool,
    limit: Optional[int],
) -> Dict[str, Any]:
    """Process a list of ``(corpus_root, model_label, game_filter)`` sources.

    For each source root we walk ``<root>/<game>/episode_*.json`` and emit one
    pair of ``action_taking.jsonl`` + ``skill_selection.jsonl`` *per game*
    (across ALL models combined).
    """
    rows_per_game_action: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows_per_game_skill:  Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_source: List[Dict[str, Any]] = []

    for root, model, game_filter in sources:
        if not root.exists():
            per_source.append({"path": str(root), "model": model, "error": "missing"})
            continue
        games = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and not d.name.startswith("_")
            and (game_filter is None or d.name in game_filter)
        )
        n_eps_total = n_steps_total = n_act_total = n_skill_total = 0
        for game in games:
            game_dir = root / game
            ep_files = sorted(game_dir.glob("episode_*.json"))
            for ep_path in ep_files:
                if "buffer" in ep_path.stem:
                    continue
                n_eps_total += 1
                n_act, n_skill = _process_rollout_episode(
                    ep_path, game=game, corpus=corpus,
                    source_model=model,
                    rows_action=rows_per_game_action[game],
                    rows_skill=rows_per_game_skill[game],
                )
                n_act_total += n_act
                n_skill_total += n_skill
            n_steps_total += len(rows_per_game_action[game])  # cumulative; re-derived below
        per_source.append({
            "path": str(root),
            "model": model,
            "n_games": len(games),
            "n_episodes": n_eps_total,
            "n_action_rows": n_act_total,
            "n_skill_rows": n_skill_total,
        })
        total_rows = sum(len(v) for v in rows_per_game_action.values())
        if limit is not None and total_rows >= limit:
            break

    # Apply global limit (per game) by truncating
    if limit is not None:
        cap = max(1, limit // max(1, len(rows_per_game_action)))
        for g in rows_per_game_action:
            rows_per_game_action[g] = rows_per_game_action[g][:cap]
            rows_per_game_skill[g]  = rows_per_game_skill[g][:cap]

    n_act_written = n_skill_written = 0
    by_game_act: Counter = Counter()
    by_game_skill: Counter = Counter()
    by_model_act: Counter = Counter()
    if not dry_run:
        for game, rows in rows_per_game_action.items():
            if not rows:
                continue
            game_dir = out_root / game
            game_dir.mkdir(parents=True, exist_ok=True)
            with (game_dir / "action_taking.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_act_written += len(rows)
            by_game_act[game] = len(rows)
            for r in rows:
                by_model_act[r.get("source_model", "unknown")] += 1
        for game, rows in rows_per_game_skill.items():
            if not rows:
                continue
            game_dir = out_root / game
            game_dir.mkdir(parents=True, exist_ok=True)
            with (game_dir / "skill_selection.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_skill_written += len(rows)
            by_game_skill[game] = len(rows)
    else:
        for game, rows in rows_per_game_action.items():
            by_game_act[game] = len(rows)
            for r in rows:
                by_model_act[r.get("source_model", "unknown")] += 1
        for game, rows in rows_per_game_skill.items():
            by_game_skill[game] = len(rows)
        n_act_written = sum(by_game_act.values())
        n_skill_written = sum(by_game_skill.values())

    preview = []
    for game, rows in rows_per_game_action.items():
        if rows:
            preview.append(rows[0])
            break
    return {
        "corpus": corpus,
        "n_sources": len(sources),
        "n_games": len(by_game_act),
        "n_action_rows_written": n_act_written,
        "n_skill_rows_written": n_skill_written,
        "by_game_action": dict(by_game_act),
        "by_game_skill": dict(by_game_skill),
        "by_model_action": dict(by_model_act),
        "per_source": per_source,
        "preview": preview,
    }


def _process_miniwob(
    sources: List[Tuple[Path, str]],
    *,
    out_path: Path,
    dry_run: bool,
    limit: Optional[int],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    per_source: List[Dict[str, Any]] = []
    for root, model in sources:
        per_source.append(_process_miniwob_root(
            root, source_model=model, rows_out=rows,
        ))
        if limit is not None and len(rows) >= limit:
            rows = rows[:limit]
            break

    if not dry_run and rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_model = Counter(r.get("source_model", "unknown") for r in rows)
    by_task = Counter(r.get("game", "unknown") for r in rows)
    return {
        "bench": "miniwob",
        "out": str(out_path),
        "n_sources": len(sources),
        "n_rows_written": len(rows) if not dry_run else 0,
        "n_rows_dry_run": len(rows) if dry_run else 0,
        "by_model": dict(by_model),
        "rows_per_task_top10": dict(by_task.most_common(10)),
        "per_source": per_source,
        "preview": rows[:3] if rows else [],
    }


def _process_webshop(
    sources: List[Tuple[Path, str]],
    *,
    out_path: Path,
    dry_run: bool,
    limit: Optional[int],
    success_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Drive the per-task webshop scan, mirroring ``_process_miniwob``.

    Each entry in *sources* is ``(model_root, source_model_label)`` where
    ``model_root`` is a ``webshop_50task_<tag>/`` directory holding
    ``webshop.<idx>/rollouts.jsonl`` files (one per task).  The threshold
    is configurable because webshop reward is granular (cf. miniwob's
    binary signal); the default 0.5 treats partial successes (≥ half
    credit) as positive SFT signal.
    """
    rows: List[Dict[str, Any]] = []
    per_source: List[Dict[str, Any]] = []
    for root, model in sources:
        per_source.append(_process_miniwob_root(
            root, source_model=model, rows_out=rows,
            task_glob="webshop.*",
            success_threshold=success_threshold,
            skill_prefix="webshop",
        ))
        if limit is not None and len(rows) >= limit:
            rows = rows[:limit]
            break

    if not dry_run and rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_model = Counter(r.get("source_model", "unknown") for r in rows)
    by_task = Counter(r.get("game", "unknown") for r in rows)
    return {
        "bench": "webshop",
        "out": str(out_path),
        "n_sources": len(sources),
        "n_rows_written": len(rows) if not dry_run else 0,
        "n_rows_dry_run": len(rows) if dry_run else 0,
        "success_threshold": success_threshold,
        "by_model": dict(by_model),
        "rows_per_task_top10": dict(by_task.most_common(10)),
        "per_source": per_source,
        "preview": rows[:3] if rows else [],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
_OR_TX = REPO_ROOT / "openrouter-transfer-baselines-out" / "2026-05-01_08-06-44"
_OR_GYMV = REPO_ROOT / "openrouter-baselines-out" / "openrouter_skip8_e16_s80_20260503_093707"
_LABELED_GPT54 = REPO_ROOT / "labeling" / "skill_actions_out" / "run_20260430_064325"

# 8 gymv games for which we have parallel claude/gemini/qwen rollouts.
GYMV_8_GAMES = [
    "Temporal_Airstriker-v0",
    "Temporal_AlteredBeast-v0",
    "Temporal_Columns-v0",
    "Temporal_DynamiteHeaddy-v0",
    "Temporal_SpaceHarrierII-v0",
    "Temporal_StreetsOfRage2-v0",
    "Temporal_Strider-v0",
    "Temporal_ThunderForceIII-v0",
]
ENV_WRAPPERS_4_GAMES = ["tetris", "twenty_forty_eight", "candy_crush", "super_mario"]

SOURCE_CONFIGS = {
    "video_holmes": {
        "kind": "qa",
        "samples": [
            REPO_ROOT / "Cold-start-out-visual-reasoning-video/video_holmes/samples.jsonl",
            _OR_TX / "claude/vr_video/video_holmes/samples.jsonl",
            _OR_TX / "gemini/vr_video/video_holmes/samples.jsonl",
            _OR_TX / "qwen/vr_video/video_holmes/samples.jsonl",
        ],
        "corpus": "visual-reasoning-video",
        "skill_map": VIDEO_HOLMES_SKILL_MAP,
        "intent_key": "question_type",
    },
    "siv_bench": {
        "kind": "qa",
        "samples": [
            REPO_ROOT / "Cold-start-out-visual-reasoning-video/siv_bench/samples.jsonl",
            _OR_TX / "claude/vr_video/siv_bench/samples.jsonl",
            _OR_TX / "gemini/vr_video/siv_bench/samples.jsonl",
            _OR_TX / "qwen/vr_video/siv_bench/samples.jsonl",
        ],
        "corpus": "visual-reasoning-video",
        "skill_map": SIV_BENCH_SKILL_MAP,
        "intent_key": "dimension",
    },
    "tir_bench": {
        "kind": "qa",
        "samples": [
            REPO_ROOT / "Cold-start-out-visual-reasoning/tir_bench/samples.jsonl",
            _OR_TX / "claude/vr_image/tir_bench/samples.jsonl",
            _OR_TX / "gemini/vr_image/tir_bench/samples.jsonl",
            _OR_TX / "qwen/vr_image/tir_bench/samples.jsonl",
        ],
        "corpus": "visual-reasoning",
        "skill_map": TIR_BENCH_SKILL_MAP,
        "intent_key": "task_type",
    },
    "visual_toolbench": {
        "kind": "qa",
        "samples": [
            REPO_ROOT / "Cold-start-out-visual-reasoning/visual_toolbench/samples.jsonl",
            _OR_TX / "claude/vr_image/visual_toolbench/samples.jsonl",
            _OR_TX / "gemini/vr_image/visual_toolbench/samples.jsonl",
            _OR_TX / "qwen/vr_image/visual_toolbench/samples.jsonl",
        ],
        "corpus": "visual-reasoning",
        "skill_map": {},
        "intent_key": "category",
    },
    "miniwob": {
        "kind": "miniwob",
        # (browsergym root, source_model_label).  Cold-start gpt-5.4 lives
        # at the canonical location; multi-model copies live under the
        # transfer-baseline run that has full miniwob coverage.
        "roots": [
            (REPO_ROOT / "Cold-start-out-browsergym", "gpt-5.4"),
            (_OR_TX / "claude/browsergym",            "claude-4.6"),
            (_OR_TX / "gemini/browsergym",            "gemini-3.1-pro"),
            (_OR_TX / "qwen/browsergym",              "qwen3-vl-235b"),
        ],
    },
    "webshop": {
        "kind": "webshop",
        # (webshop_50task_<tag> root, source_model_label).  All four
        # frontier rollouts live under the same Cold-start-out-browsergym
        # tree but in tag-specific directories — one per model.  Output is
        # written to ``<out_root>/webshop/action_taking.jsonl`` parallel to
        # miniwob; the trainer ingests it via the same data_loader path.
        "roots": [
            (REPO_ROOT / "Cold-start-out-browsergym/webshop_50task_low",    "gpt-5.4"),
            (REPO_ROOT / "Cold-start-out-browsergym/webshop_50task_claude", "claude-4.6"),
            (REPO_ROOT / "Cold-start-out-browsergym/webshop_50task_gemini", "gemini-3.1-pro"),
            (REPO_ROOT / "Cold-start-out-browsergym/webshop_50task_qwen",   "qwen3-vl-235b"),
        ],
        # Webshop reward is granular (0, 0.33, 0.5, 0.67, 0.75, 1.0).  0.5
        # captures the partial-success regime where the agent grounded the
        # right product type but missed a constraint.  Override on the CLI
        # with ``--webshop-min-reward`` to widen / tighten.
        "success_threshold": 0.5,
    },
    # ----- rollout-corpus sources (episode_*.json layout) -----
    "gymv": {
        "kind": "rollout_corpus",
        "corpus": "gym_v",
        # (root, model_label, game_filter).  gpt-5.4 pulls from the LABELED
        # skill_actions output (intentions + skill_query candidates baked in);
        # the other three are RAW rollouts (state+action+reward only) so the
        # legacy row builders will fall back to ``[EXECUTE] act in the game``
        # for intention.  All four contribute action_taking rows; only the
        # gpt-5.4 source produces non-empty skill_selection rows.
        "sources": [
            (_LABELED_GPT54 / "gym_v",       "gpt-5.4",         GYMV_8_GAMES),
            (_OR_GYMV / "claude/gymv",       "claude-4.6",      GYMV_8_GAMES),
            (_OR_GYMV / "gemini/gymv",       "gemini-3.1-pro",  GYMV_8_GAMES),
            (_OR_GYMV / "qwen/gymv",         "qwen3-vl-235b",   GYMV_8_GAMES),
        ],
    },
    "env_wrappers": {
        "kind": "rollout_corpus",
        "corpus": "env_wrappers",
        # Only gpt-5.4 has labeled cold-start data for these 4 games.  No
        # multi-model rollouts exist for env_wrappers in our trees.
        "sources": [
            (_LABELED_GPT54 / "env_wrappers", "gpt-5.4", ENV_WRAPPERS_4_GAMES),
        ],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=None,
                    help="Output directory (default: labeling/decision_sft_jsonl/run_multimodal_<ts>).")
    ap.add_argument("--sources", default=",".join(SOURCE_CONFIGS.keys()),
                    help="Comma-separated subset of sources to build.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do not write JSONL; just print stats + 3 row preview.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total rows per source (for fast iteration).")
    ap.add_argument("--include-open", action="store_true",
                    help="Also synthesize 2-option rows for open-ended visual_toolbench / tir_bench-open.")
    ap.add_argument("--webshop-min-reward", type=float, default=None,
                    help="Override the per-episode total-reward threshold "
                         "for webshop SFT inclusion.  Default is the value "
                         "in SOURCE_CONFIGS['webshop']['success_threshold'] "
                         "(0.5).  Pass 0.0 to keep every episode (noisy), "
                         "or e.g. 1.0 to keep only fully-correct ones.")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else (
        REPO_ROOT / "labeling" / "decision_sft_jsonl" / f"run_multimodal_{ts}"
    )
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    requested = [s.strip() for s in args.sources.split(",") if s.strip()]
    summary: Dict[str, Any] = {
        "run_id": out_root.name,
        "out_root": str(out_root),
        "include_open": args.include_open,
        "dry_run": args.dry_run,
        "sources": {},
    }

    for src in requested:
        cfg = SOURCE_CONFIGS.get(src)
        if cfg is None:
            print(f"[skip] unknown source: {src}")
            continue
        kind = cfg["kind"]
        out_path = out_root / src / "action_taking.jsonl"
        print(f"\n=== {src}  ({kind}) ===")
        if kind == "qa":
            stats = _process_qa_bench(
                cfg["samples"],
                bench=src,
                corpus=cfg["corpus"],
                out_path=out_path,
                skill_map=cfg["skill_map"],
                intent_key=cfg["intent_key"],
                include_open=args.include_open,
                dry_run=args.dry_run,
                limit=args.limit,
            )
        elif kind == "miniwob":
            stats = _process_miniwob(
                cfg["roots"],
                out_path=out_path,
                dry_run=args.dry_run,
                limit=args.limit,
            )
        elif kind == "webshop":
            ws_thresh = cfg.get("success_threshold", 0.5)
            if args.webshop_min_reward is not None:
                ws_thresh = float(args.webshop_min_reward)
            stats = _process_webshop(
                cfg["roots"],
                out_path=out_path,
                dry_run=args.dry_run,
                limit=args.limit,
                success_threshold=ws_thresh,
            )
        elif kind == "rollout_corpus":
            # Rollout corpora write per-game under out_root directly (no src
            # subdir prefix), matching the legacy build_decision_sft_jsonl.py
            # layout that the trainer's load_decision_adapter_data expects.
            stats = _process_rollout_corpus(
                cfg["sources"],
                corpus=cfg["corpus"],
                out_root=out_root,
                dry_run=args.dry_run,
                limit=args.limit,
            )
        else:
            stats = {"bench": src, "error": f"unknown kind {kind}"}
        summary["sources"][src] = {k: v for k, v in stats.items() if k != "preview"}
        for k, v in stats.items():
            if k == "preview":
                continue
            print(f"  {k}: {v}")
        if stats.get("preview"):
            print("  --- preview row[0] ---")
            row0 = stats["preview"][0]
            print("  prompt[:240] :", row0["prompt"][:240].replace("\n", " ⏎ "))
            print("  ...prompt tail:", row0["prompt"][-220:].replace("\n", " ⏎ "))
            print("  completion   :", row0["completion"].replace("\n", " ⏎ "))
            print("  intention    :", row0.get("intention"))
            print("  active_skill :", row0.get("active_skill"))
            print("  valid_actions:", row0.get("valid_actions"))

    if not args.dry_run:
        with (out_root / "_run_summary.json").open("w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nWrote summary -> {out_root / '_run_summary.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
