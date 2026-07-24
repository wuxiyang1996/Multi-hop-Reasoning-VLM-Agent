#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.vtb_evaluator import (  # noqa: E402
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    official_judge_prompt,
    parse_official_judge_response,
    parse_rubric_blob,
    score_vtb_task,
)


def _sha(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _load_row(parquet: Path, sample_id: str) -> tuple[int, dict[str, Any]]:
    import duckdb

    connection = duckdb.connect()
    columns = [row[0] for row in connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(parquet)]
    ).fetchall()]
    if sample_id.startswith("row:"):
        row_index = int(sample_id.split(":", 1)[1])
        row = connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 1 OFFSET ?", [str(parquet), row_index]
        ).fetchone()
    else:
        matches = connection.execute(
            "SELECT row_number() OVER () - 1 AS row_index, * FROM read_parquet(?) WHERE id = ?",
            [str(parquet), sample_id],
        ).fetchall()
        if len(matches) != 1:
            raise ValueError(f"sample id must resolve exactly once: {sample_id}")
        row_index, *row = matches[0]
    if row is None:
        raise ValueError(f"sample does not exist: {sample_id}")
    return int(row_index), dict(zip(columns, row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-faithful VisualToolBench rubric judge.")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--responses", type=Path, required=True,
                        help='JSON object with exact shape {"turns":[{"model_answer":"..."}, ...]}')
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", default="o4-mini")
    parser.add_argument("--openai-base-url", default="https://us.api.openai.com/v1")
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    row_index, row = _load_row(args.parquet, args.sample_id)
    response_payload = json.loads(args.responses.read_text(encoding="utf-8"))
    response_turns = response_payload.get("turns") or []
    prompts = list(row.get("turn_prompts") or [])
    golden_answers = list(row.get("turn_golden_answers") or [])
    rubric_blobs = list(row.get("rubrics_by_turn") or [])
    if not (len(response_turns) == len(prompts) == len(golden_answers) == len(rubric_blobs)):
        raise SystemExit(
            "NOT_EVALUABLE: response, prompt, gold, and rubric turn counts must match exactly "
            f"({len(response_turns)}, {len(prompts)}, {len(golden_answers)}, {len(rubric_blobs)})"
        )
    if not response_turns:
        raise SystemExit("NOT_EVALUABLE: task has no turns")

    key = runpy.run_path(str(args.keys)).get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is missing")
    from openai import OpenAI

    client = OpenAI(base_url=args.openai_base_url, api_key=str(key), timeout=180.0)
    rubric_turns = tuple(parse_rubric_blob(blob) for blob in rubric_blobs)
    verdict_turns = []
    call_receipts = []
    for turn_index, (question, gold, response_row, rubrics) in enumerate(zip(
        prompts, golden_answers, response_turns, rubric_turns
    )):
        model_answer = str(response_row.get("model_answer") or "").strip()
        if not model_answer:
            raise SystemExit(f"NOT_EVALUABLE: turn {turn_index} has no model_answer")
        turn_verdicts = []
        for rubric in rubrics:
            judge_prompt = official_judge_prompt(str(question), str(gold), rubric, model_answer)
            last_error: Exception | None = None
            for attempt in range(args.attempts):
                try:
                    completion = client.chat.completions.create(
                        model=args.judge_model,
                        messages=[{"role": "user", "content": judge_prompt}],
                        response_format={"type": "json_object"},
                    )
                    raw = str(completion.choices[0].message.content or "")
                    verdict = parse_official_judge_response(
                        rubric.rubric_id, raw,
                        prompt_sha256=_sha(judge_prompt), response_sha256=_sha(raw),
                    )
                    usage = completion.usage.model_dump() if completion.usage is not None else {}
                    call_receipts.append({
                        "turn_index": turn_index,
                        "rubric_id": rubric.rubric_id,
                        "attempt": attempt,
                        "prompt_sha256": _sha(judge_prompt),
                        "response_sha256": _sha(raw),
                        "usage": usage,
                    })
                    turn_verdicts.append(verdict)
                    break
                except Exception as exc:  # fail closed after bounded retry
                    last_error = exc
                    if attempt + 1 < args.attempts:
                        time.sleep(1)
            else:
                raise RuntimeError(
                    f"judge failed for turn={turn_index} rubric={rubric.rubric_id}: {last_error}"
                )
        verdict_turns.append(tuple(turn_verdicts))

    task_id = str(row.get("id") or args.sample_id)
    score = score_vtb_task(task_id, rubric_turns, tuple(verdict_turns))
    output = {
        "schema_version": 1,
        "evaluator": "visualtoolbench_official_reproduction",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "judge_model": args.judge_model,
        "critical_rule": "weight>=4",
        "row_index": row_index,
        "task_id": task_id,
        "turncase": row.get("turncase"),
        "num_turns": len(prompts),
        "response_receipt_sha256": _sha(args.responses.read_bytes()),
        "score": score.to_json(),
        "judge_call_receipts": call_receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "task_id": task_id,
        "num_turns": len(prompts),
        "rubrics": sum(len(turn) for turn in rubric_turns),
        "ars": score.ars,
        "apr_pass": score.apr_pass,
        "judge_calls": len(call_receipts),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
