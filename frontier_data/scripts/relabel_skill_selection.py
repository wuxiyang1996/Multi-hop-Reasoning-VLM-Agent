#!/usr/bin/env python3
"""Relabel skill_selection SFT data: degenerate REASONING+SKILL → EFFECTS+DECISION+SKILL.

Reads existing skill_selection.jsonl + action_taking.jsonl (for rewards),
pre-computes deterministic EFFECTS via StateEffectObserver, then calls
gpt-5-mini to label DECISION (CONTINUE/SWITCH) and refine EFFECTS.

Usage:
    python -m frontier_data.scripts.relabel_skill_selection \
        --data-dir SFT_Data/high_reward/decision_sft \
        --out-dir  SFT_Data/high_reward/decision_sft_v2 \
        --games twenty_forty_eight tetris \
        --workers 8

    # Dry run (no LLM calls, deterministic-only):
    python -m frontier_data.scripts.relabel_skill_selection \
        --data-dir SFT_Data/high_reward/decision_sft \
        --out-dir  SFT_Data/high_reward/decision_sft_v2 \
        --dry-run
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger("relabel_skill_selection")

# ── Keys bootstrap ────────────────────────────────────────────────────
try:
    _keys_path = Path(__file__).resolve().parents[3] / "keys.py"
    if _keys_path.exists():
        import importlib.util
        _spec = importlib.util.spec_from_file_location("_keys", str(_keys_path))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        for attr in ("openrouter_api_key", "openai_api_key"):
            val = getattr(_mod, attr, "")
            env_key = attr.upper()
            if val and not os.environ.get(env_key):
                os.environ[env_key] = val
except Exception:
    pass

from decision_agents.agent_helper import extract_game_facts
from decision_agents.protocol_utils import (
    EFFECT_REGISTRY,
    TASK_EFFECT_SUBSET,
    StateEffectObserver,
    get_valid_effects,
)
from decision_agents.skill_decision_core import (
    SKILL_SELECTION_SYSTEM_PROMPT,
    format_candidates_for_selection,
    parse_skill_selection,
)

try:
    from API_func import ask_model
except ImportError:
    ask_model = None

# ── Constants ─────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-5-mini"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 120
MAX_RETRIES = 3

ALL_GAME_TASKS = [
    "twenty_forty_eight", "tetris", "candy_crush", "super_mario",
    "Temporal_Airstriker-v0", "Temporal_AlteredBeast-v0",
    "Temporal_Columns-v0", "Temporal_DynamiteHeaddy-v0",
    "Temporal_SpaceHarrierII-v0", "Temporal_StreetsOfRage2-v0",
    "Temporal_Strider-v0", "Temporal_ThunderForceIII-v0",
]

# ── Data loading ──────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _load_paired_data(
    data_dir: Path, task: str,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Any]]]:
    """Load skill_selection rows + build a reward lookup from action_taking."""
    task_dir = data_dir / task
    if not task_dir.exists():
        for corpus in ("gym_v", "env_wrappers"):
            candidate = data_dir / corpus / task
            if candidate.exists():
                task_dir = candidate
                break

    ss_path = task_dir / "skill_selection.jsonl"
    at_path = task_dir / "action_taking.jsonl"

    if not ss_path.exists():
        logger.warning("No skill_selection.jsonl for %s", task)
        return [], {}

    ss_rows = _read_jsonl(ss_path)

    reward_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if at_path.exists():
        for row in _read_jsonl(at_path):
            key = (row.get("episode_id", ""), row.get("step_idx", -1))
            reward_map[key] = row

    return ss_rows, reward_map


# ── Phase A: deterministic effects ────────────────────────────────────

def _extract_state_text_from_prompt(prompt: str) -> str:
    """Pull the <state>...</state> block from the SFT prompt."""
    m = re.search(r"<state>(.*?)</state>", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_facts_from_schema(state_text: str, game_name: str) -> Dict[str, str]:
    """Extract game facts from <state> XML schema format.

    The SFT prompts contain schemas like:
        e4[type=region, label=empty_cells] / e4.value=14
        e5[type=text, label=highest_tile] / e5.value=256
    We parse these into the same key=value format that
    extract_game_facts returns from raw game text.
    """
    facts: Dict[str, str] = {}

    entity_labels: Dict[str, str] = {}
    for m in re.finditer(r"(e\d+)\[.*?label=(\w+)", state_text):
        entity_labels[m.group(1)] = m.group(2)

    for m in re.finditer(r"(e\d+)\.value=(\S+)", state_text):
        eid, val = m.group(1), m.group(2)
        label = entity_labels.get(eid, "")
        if label == "empty_cells":
            facts["empty"] = val
        elif label == "highest_tile":
            facts["highest"] = val
        elif label == "score":
            facts["score"] = val
        elif label == "stack_height" or label == "stack_h":
            facts["stack_h"] = val
        elif label == "holes":
            facts["holes"] = val
        elif label in ("tile_2", "tile_4", "tile_8", "tile_16", "tile_32",
                        "tile_64", "tile_128", "tile_256", "tile_512", "tile_1024"):
            pass
        elif label == "moves" or label == "moves_left":
            facts["moves"] = val
        elif label == "pairs":
            facts["pairs"] = val
        elif label == "piece" or label == "current_piece":
            facts["piece"] = val

    for m in re.finditer(r"(e\d+)\.state=(\w+)", state_text):
        eid, val = m.group(1), m.group(2)
        label = entity_labels.get(eid, "")
        if label == "mario":
            facts["mario"] = val

    for m in re.finditer(r"(e\d+)\[.*?label=(\w+).*?pos=([^,\]]+(?:,[^,\]]+)*)", state_text):
        eid, label, pos = m.group(1), m.group(2), m.group(3)
        if label == "mario":
            facts["mario"] = pos

    gn = game_name.lower().replace("-", "_").replace(" ", "_")
    if not facts:
        facts = extract_game_facts(state_text, gn)

    return facts


def compute_deterministic_effects(
    episode_rows: List[Dict[str, Any]],
    reward_map: Dict[Tuple[str, int], Dict[str, Any]],
    game_name: str,
) -> List[Dict[str, str]]:
    """Run StateEffectObserver over an episode's rows in step order.

    Resets the observer when the active skill changes to match the
    runtime StepTracker behavior (effects are per-skill-stint, not
    per-episode).
    """
    observer = StateEffectObserver()
    results = []
    prev_skill = None

    for row in episode_rows:
        ep_id = row.get("episode_id", "")
        step_idx = row.get("step_idx", 0)
        active_skill = row.get("active_skill", "")
        state_text = _extract_state_text_from_prompt(row.get("prompt", ""))

        if active_skill != prev_skill:
            observer.reset()
            prev_skill = active_skill

        at_row = reward_map.get((ep_id, step_idx), {})
        reward = at_row.get("reward", 0.0)
        action = at_row.get("completion", "")
        if "ACTION:" in action:
            action = action.split("ACTION:")[-1].strip().split("\n")[0]

        facts = _extract_facts_from_schema(state_text, game_name)
        effects = observer.observe(facts, reward=reward, action=action, game_name=game_name)
        results.append(dict(effects))

    return results


# ── Phase B: gpt-5-mini labeling ──────────────────────────────────────

def _build_labeling_prompt(
    *,
    game_name: str,
    step_idx: int,
    active_skill: str,
    candidates: List[str],
    reward: float,
    cum_reward: float,
    steps_on_skill: int,
    det_effects: List[str],
    valid_tags: List[str],
    prev_summary: str,
    curr_summary: str,
    sel_idx: int,
) -> str:
    """Build a compact prompt for gpt-5-mini DECISION labeling."""
    cands_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))
    det_str = ", ".join(det_effects) if det_effects else "none"
    tags_str = ", ".join(valid_tags)

    return (
        f"Game: {game_name} | Step {step_idx} | Active skill: {active_skill}\n"
        f"Reward this step: {reward} | Cumulative on skill: {cum_reward:.1f} | "
        f"Steps on skill: {steps_on_skill}\n"
        f"Observer-detected effects: [{det_str}]\n"
        f"Valid effect tags: [{tags_str}]\n\n"
        f"Previous state: {prev_summary[:500]}\n"
        f"Current state: {curr_summary[:500]}\n\n"
        f"Candidates:\n{cands_text}\n\n"
        f"Based on state changes and reward, output exactly 3 lines:\n"
        f"EFFECTS: <achieved effects from valid tags>\n"
        f"DECISION: CONTINUE or SWITCH\n"
        f"SKILL: {sel_idx}"
    )


def _call_llm_with_retry(
    prompt: str, model: str = DEFAULT_MODEL,
) -> Optional[str]:
    if ask_model is None:
        return None
    for attempt in range(MAX_RETRIES):
        try:
            raw = ask_model(
                prompt,
                model=model,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            if raw and not raw.startswith("Error"):
                return raw.strip()
        except Exception as exc:
            logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
    return None


def _parse_llm_completion(
    raw: str, valid_set: set, n_candidates: int, fallback_idx: int,
) -> Tuple[List[str], str, int]:
    """Parse 3-line LLM output, validate tags, return (effects, decision, skill_idx)."""
    effects_m = re.search(r"EFFECTS?\s*:\s*(.+?)(?=\nDECISION|\nSKILL|\Z)", raw, re.I | re.DOTALL)
    decision_m = re.search(r"DECISION\s*:\s*(CONTINUE|SWITCH)", raw, re.I)
    skill_m = re.search(r"SKILL\s*:\s*(\d+)", raw, re.I)

    effects = []
    if effects_m:
        for tag in effects_m.group(1).split(","):
            tag = tag.strip().lower()
            if tag in valid_set:
                effects.append(tag)

    decision = decision_m.group(1).upper() if decision_m else "CONTINUE"

    idx = fallback_idx
    if skill_m:
        parsed = int(skill_m.group(1)) - 1
        if 0 <= parsed < n_candidates:
            idx = parsed

    return effects, decision, idx


# ── Phase C: prompt regeneration ──────────────────────────────────────

def _build_new_prompt(
    *,
    state_text: str,
    intention: str,
    candidates: List[str],
    active_skill: str,
    achieved_effects: List[str],
    valid_tags: List[str],
    game_name: str,
) -> str:
    """Build the new-format prompt matching runtime build_skill_selection_prompt."""
    cands_for_prompt = []
    for c in candidates:
        cands_for_prompt.append({"skill_name": c, "skill_id": c})

    candidates_text = format_candidates_for_selection(cands_for_prompt)
    valid_tags_str = ", ".join(valid_tags)

    skill_ctx = ""
    if active_skill:
        skill_ctx = f"Active skill: {active_skill}\n"

    progress_ctx = ""
    if achieved_effects:
        progress_ctx = f"Achieved effects: {', '.join(sorted(achieved_effects))}\n"

    user_content = (
        f"Current state:\n{state_text[:3500]}\n\n"
        f"Intention: {intention[:500]}\n"
        f"Valid effect tags: [{valid_tags_str}]\n"
        f"{skill_ctx}"
        f"{progress_ctx}\n"
        f"Strategies:\n{candidates_text}\n\n"
        f"EFFECTS: <list achieved effects from the valid set>\n"
        f"DECISION: CONTINUE or SWITCH\n"
        f"SKILL: <number>"
    )
    return SKILL_SELECTION_SYSTEM_PROMPT + "\n" + user_content


# ── Main relabeling pipeline ─────────────────────────────────────────

def relabel_task(
    data_dir: Path,
    out_dir: Path,
    task: str,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    workers: int = 4,
) -> int:
    """Relabel all skill_selection data for one task. Returns row count."""
    game_name = task.lower().replace("-", "_").replace(" ", "_")
    valid_tags = get_valid_effects(task)
    valid_set = set(valid_tags)

    ss_rows, reward_map = _load_paired_data(data_dir, task)
    if not ss_rows:
        logger.info("Skipping %s: no data", task)
        return 0

    # Group by episode, sort by step_idx
    episodes: Dict[str, List[Dict[str, Any]]] = {}
    for row in ss_rows:
        ep = row.get("episode_id", "unknown")
        episodes.setdefault(ep, []).append(row)
    for ep_rows in episodes.values():
        ep_rows.sort(key=lambda r: r.get("step_idx", 0))

    output_rows: List[Dict[str, Any]] = []

    for ep_id, ep_rows in episodes.items():
        # Phase A: deterministic effects
        det_effects_list = compute_deterministic_effects(ep_rows, reward_map, game_name)

        # Track cumulative reward and steps per skill stint
        cum_reward = 0.0
        steps_on_skill = 0
        prev_skill = None
        prev_summary = ""

        for i, row in enumerate(ep_rows):
            step_idx = row.get("step_idx", i)
            active_skill = row.get("active_skill", "")
            candidates = row.get("candidates", [])
            intention = row.get("intention", "")
            sel_skill_id = row.get("selected_skill_id", "")

            at_row = reward_map.get((ep_id, step_idx), {})
            reward = at_row.get("reward", 0.0)

            # Track skill stint
            if active_skill != prev_skill:
                cum_reward = reward
                steps_on_skill = 1
                prev_skill = active_skill
            else:
                cum_reward += reward
                steps_on_skill += 1

            state_text = _extract_state_text_from_prompt(row.get("prompt", ""))
            curr_summary = state_text[:300]

            # Deterministic effects (filter to valid set)
            det_effects = det_effects_list[i] if i < len(det_effects_list) else {}
            det_tags = [k for k, v in det_effects.items() if v == "true" and k in valid_set]

            # Selected skill index (1-based in completion)
            if sel_skill_id and sel_skill_id in candidates:
                sel_idx = candidates.index(sel_skill_id) + 1
            else:
                sel_idx = 1

            # Phase B: LLM labeling
            if not dry_run and ask_model is not None:
                labeling_prompt = _build_labeling_prompt(
                    game_name=task,
                    step_idx=step_idx,
                    active_skill=active_skill,
                    candidates=candidates,
                    reward=reward,
                    cum_reward=cum_reward,
                    steps_on_skill=steps_on_skill,
                    det_effects=det_tags,
                    valid_tags=valid_tags,
                    prev_summary=prev_summary,
                    curr_summary=curr_summary,
                    sel_idx=sel_idx,
                )
                raw_reply = _call_llm_with_retry(labeling_prompt, model=model)
                if raw_reply:
                    effects, decision, _ = _parse_llm_completion(
                        raw_reply, valid_set, len(candidates), sel_idx - 1,
                    )
                else:
                    effects = det_tags
                    decision = "CONTINUE" if (reward > 0 or steps_on_skill < 3) else "SWITCH"
            else:
                effects = det_tags
                decision = "CONTINUE" if (reward > 0 or steps_on_skill < 3) else "SWITCH"

            # Phase C: build new completion + prompt
            effects_str = ", ".join(effects) if effects else "state_observed"
            completion = f"EFFECTS: {effects_str}\nDECISION: {decision}\nSKILL: {sel_idx}"

            new_prompt = _build_new_prompt(
                state_text=state_text,
                intention=intention,
                candidates=candidates,
                active_skill=active_skill,
                achieved_effects=effects,
                valid_tags=valid_tags,
                game_name=task,
            )

            new_row = {
                "prompt": new_prompt,
                "completion": completion,
                "intention": intention,
                "active_skill": active_skill,
                "game": row.get("game", task),
                "corpus": row.get("corpus", ""),
                "episode_id": ep_id,
                "step_idx": step_idx,
                "candidates": candidates,
                "selected_skill_id": sel_skill_id,
                "effects": effects,
                "decision": decision,
            }
            if "image" in row:
                new_row["image"] = row["image"]

            output_rows.append(new_row)
            prev_summary = curr_summary

    # Write output
    if output_rows:
        task_out = out_dir / task
        task_out.mkdir(parents=True, exist_ok=True)
        out_path = task_out / "skill_selection.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in output_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("[%s] Wrote %d rows → %s", task, len(output_rows), out_path)

    return len(output_rows)


def relabel_task_parallel(
    data_dir: Path,
    out_dir: Path,
    task: str,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    workers: int = 8,
) -> int:
    """Relabel with parallel LLM calls across episodes."""
    game_name = task.lower().replace("-", "_").replace(" ", "_")
    valid_tags = get_valid_effects(task)
    valid_set = set(valid_tags)

    ss_rows, reward_map = _load_paired_data(data_dir, task)
    if not ss_rows:
        logger.info("Skipping %s: no data", task)
        return 0

    episodes: Dict[str, List[Dict[str, Any]]] = {}
    for row in ss_rows:
        ep = row.get("episode_id", "unknown")
        episodes.setdefault(ep, []).append(row)
    for ep_rows in episodes.values():
        ep_rows.sort(key=lambda r: r.get("step_idx", 0))

    all_jobs: List[Tuple[Dict[str, Any], Dict[str, str], str]] = []

    for ep_id, ep_rows in episodes.items():
        det_effects_list = compute_deterministic_effects(ep_rows, reward_map, game_name)

        cum_reward = 0.0
        steps_on_skill = 0
        prev_skill = None
        prev_summary = ""

        for i, row in enumerate(ep_rows):
            step_idx = row.get("step_idx", i)
            active_skill = row.get("active_skill", "")
            candidates = row.get("candidates", [])
            sel_skill_id = row.get("selected_skill_id", "")

            at_row = reward_map.get((ep_id, step_idx), {})
            reward = at_row.get("reward", 0.0)

            if active_skill != prev_skill:
                cum_reward = reward
                steps_on_skill = 1
                prev_skill = active_skill
            else:
                cum_reward += reward
                steps_on_skill += 1

            state_text = _extract_state_text_from_prompt(row.get("prompt", ""))
            curr_summary = state_text[:300]
            det_effects = det_effects_list[i] if i < len(det_effects_list) else {}
            det_tags = [k for k, v in det_effects.items() if v == "true" and k in valid_set]

            if sel_skill_id and sel_skill_id in candidates:
                sel_idx = candidates.index(sel_skill_id) + 1
            else:
                sel_idx = 1

            labeling_prompt = _build_labeling_prompt(
                game_name=task,
                step_idx=step_idx,
                active_skill=active_skill,
                candidates=candidates,
                reward=reward,
                cum_reward=cum_reward,
                steps_on_skill=steps_on_skill,
                det_effects=det_tags,
                valid_tags=valid_tags,
                prev_summary=prev_summary,
                curr_summary=curr_summary,
                sel_idx=sel_idx,
            )

            job_ctx = {
                "row": row, "det_tags": det_tags, "sel_idx": sel_idx,
                "reward": reward, "steps_on_skill": steps_on_skill,
                "state_text": state_text, "candidates": candidates,
                "valid_tags": valid_tags, "valid_set": valid_set,
            }
            all_jobs.append((job_ctx, labeling_prompt))
            prev_summary = curr_summary

    output_rows: List[Dict[str, Any]] = []

    if dry_run or ask_model is None:
        for job_ctx, _ in all_jobs:
            row = job_ctx["row"]
            det_tags = job_ctx["det_tags"]
            sel_idx = job_ctx["sel_idx"]
            reward = job_ctx["reward"]
            steps_on_skill = job_ctx["steps_on_skill"]
            effects = det_tags
            decision = "CONTINUE" if (reward > 0 or steps_on_skill < 3) else "SWITCH"
            output_rows.append(_finalize_row(job_ctx, effects, decision))
    else:
        def _do_one(idx_prompt):
            idx, (ctx, prompt) = idx_prompt
            raw = _call_llm_with_retry(prompt, model=model)
            return idx, ctx, raw

        indexed = list(enumerate(all_jobs))
        results = [None] * len(indexed)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_do_one, ip): ip[0] for ip in indexed}
            done_count = 0
            for fut in as_completed(futures):
                idx, ctx, raw = fut.result()
                results[idx] = (ctx, raw)
                done_count += 1
                if done_count % 200 == 0:
                    logger.info("[%s] %d / %d LLM calls done", task, done_count, len(indexed))

        for ctx, raw in results:
            if raw:
                effects, decision, _ = _parse_llm_completion(
                    raw, ctx["valid_set"], len(ctx["candidates"]), ctx["sel_idx"] - 1,
                )
            else:
                effects = ctx["det_tags"]
                reward = ctx["reward"]
                steps = ctx["steps_on_skill"]
                decision = "CONTINUE" if (reward > 0 or steps < 3) else "SWITCH"
            output_rows.append(_finalize_row(ctx, effects, decision))

    if output_rows:
        task_out = out_dir / task
        task_out.mkdir(parents=True, exist_ok=True)
        out_path = task_out / "skill_selection.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in output_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("[%s] Wrote %d rows → %s", task, len(output_rows), out_path)

    return len(output_rows)


def _finalize_row(
    ctx: Dict[str, Any], effects: List[str], decision: str,
) -> Dict[str, Any]:
    row = ctx["row"]
    sel_idx = ctx["sel_idx"]
    candidates = ctx["candidates"]
    state_text = ctx["state_text"]
    valid_tags = ctx["valid_tags"]
    intention = row.get("intention", "")
    active_skill = row.get("active_skill", "")

    effects_str = ", ".join(effects) if effects else "state_observed"
    completion = f"EFFECTS: {effects_str}\nDECISION: {decision}\nSKILL: {sel_idx}"

    new_prompt = _build_new_prompt(
        state_text=state_text,
        intention=intention,
        candidates=candidates,
        active_skill=active_skill,
        achieved_effects=effects,
        valid_tags=valid_tags,
        game_name=row.get("game", ""),
    )

    return {
        "prompt": new_prompt,
        "completion": completion,
        "intention": intention,
        "active_skill": active_skill,
        "game": row.get("game", ""),
        "corpus": row.get("corpus", ""),
        "episode_id": row.get("episode_id", ""),
        "step_idx": row.get("step_idx", 0),
        "candidates": candidates,
        "selected_skill_id": row.get("selected_skill_id", ""),
        "effects": effects,
        "decision": decision,
        **({"image": row["image"]} if "image" in row else {}),
    }


# ── CLI ───────────────────────────────────────────────────────────────

def _relabel_one_task(args_tuple):
    """Wrapper for ProcessPoolExecutor — relabels one task."""
    data_dir, out_dir, task, model, dry_run, workers = args_tuple
    t0 = time.time()
    try:
        if dry_run or workers <= 1:
            n = relabel_task(data_dir, out_dir, task, model=model, dry_run=dry_run)
        else:
            n = relabel_task_parallel(
                data_dir, out_dir, task, model=model,
                dry_run=dry_run, workers=workers,
            )
    except Exception:
        logger.exception("[%s] failed", task)
        n = 0
    elapsed = time.time() - t0
    logger.info("[%s] %d rows in %.1fs", task, n, elapsed)
    return task, n


def main():
    parser = argparse.ArgumentParser(description="Relabel skill_selection SFT data")
    parser.add_argument("--data-dir", type=str,
                        default="SFT_Data/high_reward/decision_sft")
    parser.add_argument("--out-dir", type=str,
                        default="SFT_Data/high_reward/decision_sft_v2")
    parser.add_argument("--games", nargs="*", default=None,
                        help="Tasks to relabel (default: all 12 games)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=32,
                        help="LLM concurrency per task (default: 32)")
    parser.add_argument("--task-parallelism", type=int, default=4,
                        help="Number of tasks (games) to relabel concurrently")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls, use deterministic effects + heuristic DECISION")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    tasks = args.games or ALL_GAME_TASKS

    tp = args.task_parallelism
    if tp <= 1 or len(tasks) == 1:
        total = 0
        for task in tasks:
            logger.info("=== Relabeling %s ===", task)
            _, n = _relabel_one_task(
                (data_dir, out_dir, task, args.model, args.dry_run, args.workers),
            )
            total += n
    else:
        from concurrent.futures import ProcessPoolExecutor
        job_args = [
            (data_dir, out_dir, t, args.model, args.dry_run, args.workers)
            for t in tasks
        ]
        total = 0
        logger.info(
            "Launching %d tasks with task_parallelism=%d, workers=%d each",
            len(tasks), tp, args.workers,
        )
        with ProcessPoolExecutor(max_workers=tp) as pool:
            for task_name, n in pool.map(_relabel_one_task, job_args):
                logger.info("=== Finished %s: %d rows ===", task_name, n)
                total += n

    logger.info("=== Done: %d total rows across %d tasks ===", total, len(tasks))


if __name__ == "__main__":
    main()
