#!/usr/bin/env python3
"""Run a frozen zero-trajectory target-schema LLM synthesis baseline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
from typing import Any, Mapping

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.target_schema_synthesis import (  # noqa: E402
    parse_program_response,
    score_program,
    synthesis_prompt,
)


DEFAULT_CONFIG = REPO / "configs/target_schema_synthesis_v29.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "reported_cost_usd": None,
        }
    extra = getattr(usage, "model_extra", None) or {}
    cost = getattr(usage, "cost", None)
    if cost is None:
        cost = extra.get("cost")
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(
            getattr(usage, "completion_tokens", 0) or 0
        ),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "reported_cost_usd": None if cost is None else float(cost),
    }


def _call(
    client: OpenAI, config: Mapping[str, Any], *, target: str,
    interface: str, replicate: int,
) -> dict[str, Any]:
    prompt = synthesis_prompt(target, interface, variant=replicate)
    request_body = {
        "model": str(config["model"]["id"]),
        "temperature": float(config["model"]["temperature"]),
        "max_tokens": int(config["model"]["maximum_output_tokens"]),
        "target": target,
        "replicate": replicate,
        "prompt_sha256": stable_hash(prompt),
    }
    started = time.monotonic()
    error = None
    response = None
    for attempt in range(int(config["model"]["max_retries"]) + 1):
        try:
            response = client.chat.completions.create(
                model=request_body["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=request_body["temperature"],
                max_tokens=request_body["max_tokens"],
                response_format={"type": "json_object"},
            )
            break
        except Exception as exception:  # provider errors are evidence
            error = f"{type(exception).__name__}: {exception}"
            if attempt >= int(config["model"]["max_retries"]):
                break
            time.sleep(2**attempt)
    elapsed = time.monotonic() - started
    if response is None:
        return {
            **request_body,
            "status": "PROVIDER_ERROR",
            "error": error,
            "wall_clock_seconds": elapsed,
            "complete_target_trajectories_read": 0,
            "target_outcomes_read": 0,
        }
    message = response.choices[0].message
    text = str(message.content or "")
    try:
        program = parse_program_response(text)
        score = score_program(target, program)
        status = "PARSED"
        parse_error = None
    except (ValueError, json.JSONDecodeError) as exception:
        program = None
        score = {
            "target": target,
            "exact_program_match": False,
            "field_matches": {},
        }
        status = "PARSE_ERROR"
        parse_error = f"{type(exception).__name__}: {exception}"
    return {
        **request_body,
        "status": status,
        "provider_model": str(getattr(response, "model", "")),
        "provider_response_id_sha256": stable_hash(
            str(getattr(response, "id", ""))
        ),
        "response_sha256": stable_hash(text),
        "response_text": text,
        "program": program,
        "score": score,
        "parse_error": parse_error,
        "usage": _usage(response),
        "wall_clock_seconds": elapsed,
        "complete_target_trajectories_read": 0,
        "target_outcomes_read": 0,
    }


def run(config_path: Path, keys_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    _self_hash(config, "config_sha256")
    if config.get("status") != "FROZEN_BEFORE_TARGET_SCHEMA_CALLS":
        raise ValueError("target synthesis protocol is not frozen")
    for hash_field, path_field in config["dependency_fields"].items():
        if _sha(REPO / str(config[path_field])) != config[hash_field]:
            raise ValueError(f"dependency changed: {config[path_field]}")
    values = runpy.run_path(str(keys_path))
    api_key = str(
        values.get("OPENROUTER_API_KEY")
        or values.get("openrouter_api_key") or ""
    )
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")
    client = OpenAI(
        api_key=api_key,
        base_url=str(config["model"]["base_url"]),
        timeout=float(config["model"]["timeout_seconds"]),
    )
    jobs = [
        (target, str(spec["interface_description"]), replicate)
        for target, spec in sorted(config["targets"].items())
        for replicate in range(int(config["replicates_per_target"]))
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=int(config["workers"])) as executor:
        futures = {
            executor.submit(
                _call, client, config, target=target,
                interface=interface, replicate=replicate,
            ): (target, replicate)
            for target, interface, replicate in jobs
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (str(row["target"]), int(row["replicate"])))
    per_target = {}
    for target in sorted(config["targets"]):
        target_rows = [row for row in rows if row["target"] == target]
        per_target[target] = {
            "calls": len(target_rows),
            "provider_successes": sum(
                row["status"] != "PROVIDER_ERROR" for row in target_rows
            ),
            "parsed": sum(row["status"] == "PARSED" for row in target_rows),
            "exact_program_matches": sum(
                bool((row.get("score") or {}).get("exact_program_match"))
                for row in target_rows
            ),
            "complete_target_trajectories_read": 0,
            "target_outcomes_read": 0,
        }
    usage_rows = [row.get("usage") or {} for row in rows]
    costs = [
        float(row["reported_cost_usd"])
        for row in usage_rows if row.get("reported_cost_usd") is not None
    ]
    gates = {
        "all_provider_calls_returned": all(
            row["status"] != "PROVIDER_ERROR" for row in rows
        ),
        "all_responses_parsed": all(row["status"] == "PARSED" for row in rows),
        "zero_complete_target_trajectories_read": all(
            row["complete_target_trajectories_read"] == 0 for row in rows
        ),
        "zero_target_outcomes_read": all(
            row["target_outcomes_read"] == 0 for row in rows
        ),
        "all_targets_received_frozen_call_budget": all(
            row["calls"] == int(config["replicates_per_target"])
            for row in per_target.values()
        ),
    }
    body = {
        "schema_version": "target-schema-synthesis-v29-report",
        "status": (
            "TARGET_SCHEMA_SYNTHESIS_BASELINE_COMPLETE"
            if all(gates.values())
            else "TARGET_SCHEMA_SYNTHESIS_BASELINE_INCOMPLETE"
        ),
        "config_sha256": str(config["config_sha256"]),
        "claim_boundary": str(config["claim_boundary"]),
        "model": dict(config["model"]),
        "rows": rows,
        "per_target": per_target,
        "resource_accounting": {
            "provider_calls": len(rows),
            "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in usage_rows),
            "completion_tokens": sum(
                int(row.get("completion_tokens", 0)) for row in usage_rows
            ),
            "total_tokens": sum(int(row.get("total_tokens", 0)) for row in usage_rows),
            "reported_cost_usd": sum(costs) if costs else None,
            "wall_clock_seconds_sum": sum(
                float(row["wall_clock_seconds"]) for row in rows
            ),
            "target_environment_interactions": 0,
            "complete_target_trajectories": 0,
        },
        "gates": gates,
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    config = _read(config_path)
    output = REPO / str(config["output"])
    if output.exists():
        raise SystemExit(f"refusing to overwrite synthesis baseline: {output}")
    report = run(config_path, args.keys)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "per_target": report["per_target"],
        "resource_accounting": report["resource_accounting"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
        "output": str(output),
    }, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
