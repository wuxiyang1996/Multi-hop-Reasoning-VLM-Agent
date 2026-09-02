#!/usr/bin/env python3
"""Collect frozen Qwen3.5-9B answers over shared CLEVRER graph facts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import runpy
import sys
import time
from typing import Any

import requests


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from motif_transfer.clevrer_compact_event_graph import compact_event_graph  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(path)
    return value


def _extract(text: str, answer_specs: dict[str, int]) -> dict[str, str]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    value = json.loads(cleaned)
    if isinstance(value, dict) and isinstance(value.get("answers"), dict):
        value = value["answers"]
    if not isinstance(value, dict) or set(value) != set(answer_specs):
        raise ValueError("response keys do not exactly match task IDs")
    answers = {key: str(item).strip().casefold() for key, item in value.items()}
    for task_id, choice_count in answer_specs.items():
        if choice_count and (
            len(answers[task_id]) != choice_count
            or any(char not in "01" for char in answers[task_id])
        ):
            raise ValueError(f"{task_id} requires a {choice_count}-character 0/1 bitstring")
    return answers


def _call(*, url: str, key: str, model: str, prompt: str, answer_specs: dict[str, int], retries: int) -> dict[str, Any]:
    payload = {
        "model": model, "temperature": 0, "max_tokens": 256,
        "messages": [
            {"role": "system", "content": (
                "You are a frozen neural decision actor. Answer using only the supplied "
                "video event-graph facts. Descriptive answers must be the shortest exact label. "
                "For each multiple-choice question, judge EVERY choice independently and return a bit "
                "string in displayed order (1=yes/correct, 0=no/incorrect). The bitstring length MUST "
                "equal the number of choices. It is NOT a choice index. Return one JSON object mapping "
                "every task_id to answer."
            )},
            {"role": "user", "content": prompt},
        ],
        "reasoning": {"enabled": False},
    }
    error = None
    for attempt in range(retries):
        try:
            response = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=180)
            response.raise_for_status(); data = response.json()
            raw = data["choices"][0]["message"]["content"]
            answers = _extract(raw, answer_specs)
            return {"answers": answers, "raw_response": raw, "usage": data.get("usage", {}),
                    "provider_request_id": response.headers.get("x-request-id")}
        except Exception as exc:  # transport/schema retry, never outcome-driven
            error = repr(exc); time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(error)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--grounder-config", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--max-videos", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--one-task-per-call", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("actor artifact is immutable")
    runtime = _read(args.runtime); config = _read(args.grounder_config)
    if runtime.get("status") != "CLEVRER_SHARED_RUNTIME_FROZEN" or runtime.get("answers_read"):
        raise ValueError("runtime authority invalid")
    executor_root = args.official_root / "executor"; sys.path.insert(0, str(executor_root))
    from executor import Executor  # type: ignore  # noqa: E402
    from simulation import Simulation  # type: ignore  # noqa: E402
    secrets = runpy.run_path(str(args.keys)); key = secrets.get("OPENROUTER_API_KEY") or secrets.get("openrouter_api_key")
    if not key: raise RuntimeError("OPENROUTER_API_KEY missing")

    selected_ids = sorted({int(row["video_id"]) for row in runtime["videos"]})[:args.max_videos]
    tasks_by_video = {video_id: [] for video_id in selected_ids}
    for task in runtime["tasks"]:
        if int(task["video_id"]) in tasks_by_video: tasks_by_video[int(task["video_id"])].append(task)
    jobs = []
    for video_id in selected_ids:
        path = Path(config["prediction_root"]) / f"sim_{video_id:05d}.json"
        executor = Executor(Simulation(str(path), use_event_ann=True))
        facts = compact_event_graph(executor)
        question_blocks = []
        for row in tasks_by_video[video_id]:
            choices = ""
            if row["choices"]:
                choices = "\nCHOICES:\n" + "\n".join(
                    f"{index}: {choice['choice']}" for index, choice in enumerate(row["choices"])
                )
            question_blocks.append((row, f"TASK_ID: {row['task_id']}\nQUESTION: {row['question']}{choices}"))
        groups = [[item] for item in question_blocks] if args.one_task_per_call else [question_blocks]
        for group_index, group in enumerate(groups):
            task_ids = [str(row["task_id"]) for row, _ in group]
            answer_specs = {str(row["task_id"]): len(row["choices"]) for row, _ in group}
            prompt = facts + "\n\n" + "\n\n".join(block for _, block in group)
            jobs.append((f"{video_id}:{group_index}", video_id, task_ids, answer_specs,
                         prompt, stable_hash(facts)))

    results = {}
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_call, url=endpoint, key=key, model=args.model, prompt=prompt,
                               answer_specs=answer_specs, retries=args.retries): (job_id, video_id, task_ids, facts_sha)
                   for job_id, video_id, task_ids, answer_specs, prompt, facts_sha in jobs}
        for future in as_completed(futures):
            job_id, video_id, task_ids, facts_sha = futures[future]
            value = future.result(); results[job_id] = {**value, "video_id": video_id,
                                                        "task_ids": task_ids, "facts_sha256": facts_sha}
            print(f"completed job={job_id} ({len(results)}/{len(jobs)})", flush=True)

    rows = []
    for job_id in sorted(results):
        value = results[job_id]; video_id = int(value["video_id"])
        for task_id in value["task_ids"]:
            rows.append({"task_id": task_id, "video_id": video_id,
                         "prediction": value["answers"][task_id],
                         "facts_sha256": value["facts_sha256"],
                         "answer_read": False, "official_program_read": False})
    usage = {key: sum(int(value.get("usage", {}).get(key, 0) or 0) for value in results.values())
             for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    body = {"schema_version": "clevrer-qwen35-9b-shared-graph-actor-v1",
            "status": "NEURAL_ACTOR_PREDICTIONS_FROZEN_BEFORE_OUTCOMES",
            "model": args.model, "thinking_enabled": False,
            "shared_runtime_sha256": runtime["runtime_sha256"],
            "grounding": "SHARED_CACHED_NSDR_COMPACT_TYPED_FACTS",
            "video_count": len(selected_ids), "rows": rows, "usage": usage,
            "provider_calls": len(results), "answers_read": False,
            "official_programs_read": False}
    body["runtime_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "videos": len(selected_ids), "rows": len(rows),
                      "usage": usage, "runtime_sha256": body["runtime_sha256"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
