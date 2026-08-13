#!/usr/bin/env python3
"""Attach outcome-blind neural applicability receipts to video candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.active_video_transfer import stable_hash  # noqa: E402
from run_active_video_wrapper_transfer import _questions  # noqa: E402


def _judge_input(
    receipt: Mapping[str, Any], row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "question": str(row["Question"]),
        "options": dict(row["Options"]),
        "question_independent_audio_scout": [
            {
                "start_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
                "description": segment["description"],
            }
            for segment in receipt["audio_overview"]
        ],
        "candidate_tests": [
            {
                "candidate_id": candidate["candidate_id"],
                "start_sec": candidate["arguments"]["start_sec"],
                "end_sec": candidate["arguments"]["end_sec"],
                "hypothesis": candidate["hypothesis"],
            }
            for candidate in receipt["candidates"]
        ],
    }


def _judge(
    receipt: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    api_key: str,
) -> dict[str, Any]:
    judge_config = config["outcome_blind_applicability"]
    payload = _judge_input(receipt, row)
    client = OpenAI(
        api_key=api_key,
        base_url=str(judge_config["base_url"]),
        timeout=float(judge_config["timeout_seconds"]),
        max_retries=int(judge_config["max_retries"]),
    )
    response = client.chat.completions.create(
        model=str(judge_config["model"]),
        temperature=0,
        response_format={"type": "json_object"},
        max_completion_tokens=int(judge_config["max_tokens"]),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a target-native intervention applicability judge. "
                    "Do not answer the multiple-choice question. Score every "
                    "candidate test for expected ability to discriminate the "
                    "listed options using causal relevance, temporal coverage, "
                    "and ability to falsify alternatives. Return JSON "
                    "{\"scores\":[{\"candidate_id\":string,"
                    "\"expected_information_gain\":number 0..1,"
                    "\"expected_answer_change_probability\":number 0..1,"
                    "\"outcome_balance\":number 0..1,\"reason\":string}]}. "
                    "Do not output an answer letter or claim evidence outside "
                    "the supplied low-bandwidth scout."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("applicability judge returned no JSON")
    parsed = json.loads(raw)
    scores = list(parsed.get("scores") or ())
    expected_ids = [str(row["candidate_id"]) for row in payload["candidate_tests"]]
    by_id = {str(score.get("candidate_id")): score for score in scores}
    if set(by_id) != set(expected_ids) or len(by_id) != len(scores):
        raise ValueError("applicability scores do not align with candidate IDs")
    canonical = []
    for candidate_id in expected_ids:
        score = by_id[candidate_id]
        values = {
            name: float(score[name])
            for name in (
                "expected_information_gain",
                "expected_answer_change_probability",
                "outcome_balance",
            )
        }
        if any(not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("applicability score outside [0, 1]")
        canonical.append({
            "candidate_id": candidate_id,
            **values,
            "reason": str(score.get("reason") or ""),
        })
    usage = response.usage
    return {
        "input_sha256": stable_hash(payload),
        "scores": canonical,
        "model": str(response.model),
        "finish_reason": str(response.choices[0].finish_reason),
        "prompt_tokens": int(usage.prompt_tokens if usage else 0),
        "completion_tokens": int(usage.completion_tokens if usage else 0),
        "response_sha256": stable_hash(canonical),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = [
        row
        for path in args.receipts
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]
    by_id = {str(row["sample_id"]): row for row in rows}
    expected = list(map(str, config["splits"]["adaptation"]))
    if len(by_id) != len(rows) or set(by_id) != set(expected):
        raise SystemExit("receipt identities do not match frozen adaptation IDs")
    output = {sample_id: copy.deepcopy(by_id[sample_id]) for sample_id in expected}
    questions = _questions(args.dataset_root, "train")
    judge_config = config["outcome_blind_applicability"]
    keys = runpy.run_path(str(args.keys))
    key_name = str(judge_config["api_key_name"])
    api_key = keys.get(key_name)
    if not api_key:
        raise SystemExit(f"{key_name} is missing")
    pending = [
        sample_id for sample_id in expected
        if "outcome_blind_applicability_receipt" not in output[sample_id]
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _judge,
                output[sample_id],
                row=questions[sample_id],
                config=config,
                api_key=str(api_key),
            ): sample_id
            for sample_id in pending
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            receipt = future.result()
            output[sample_id]["outcome_blind_applicability_receipt"] = receipt
            score_index = {
                row["candidate_id"]: row for row in receipt["scores"]
            }
            for candidate in output[sample_id]["candidates"]:
                candidate["outcome_blind_applicability"] = score_index[
                    candidate["candidate_id"]
                ]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    [output[value] for value in expected],
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "completed": sample_id,
                "progress": f"{sum('outcome_blind_applicability_receipt' in row for row in output.values())}/{len(expected)}",
            }), flush=True)
    contract = {
        "kind": "OUTCOME_BLIND_TARGET_NATIVE_NEURAL_APPLICABILITY",
        "config_sha256": stable_hash(config),
        "input_files_sha256": [
            stable_hash(json.loads(path.read_text(encoding="utf-8")))
            for path in args.receipts
        ],
        "forbidden_judge_inputs": [
            "gold_answer", "baseline.answer", "candidate.answer",
            "candidate.wrapper_receipt.result", "candidate evidence panel",
        ],
    }
    contract["contract_sha256"] = stable_hash(contract)
    for row in output.values():
        row["outcome_blind_applicability_contract"] = contract
    args.output.write_text(
        json.dumps([output[value] for value in expected], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "COMPLETE",
        "samples": len(output),
        "contract_sha256": contract["contract_sha256"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
