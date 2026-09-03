#!/usr/bin/env python3
"""Complete only missing V17 rows after provider-side JSON truncation.

The semantic request, model, images, and parser are unchanged.  This fallback only
raises the provider completion budget so a receipt that was cut off in transit can
finish.  Existing receipts are never regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_natural_video_cross_model_proof_v17 as v17  # noqa: E402
import collect_natural_video_recovery_v15 as paired  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-completion-tokens", type=int, default=16_000)
    parser.add_argument("--transport-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.max_completion_tokens <= 5_200:
        raise ValueError("transport fallback budget must exceed the frozen initial budget")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    inputs = json.loads(Path(config["input_receipts"]).read_text(encoding="utf-8"))
    ordered_keys = [
        (str(row["benchmark"]), str(row["sample_id"])) for row in inputs
    ]
    contract_sha256 = v17._content_hash({
        "config_sha256": _sha256(args.config),
        "input_receipts_sha256": _sha256(Path(config["input_receipts"])),
        "collector_sha256": _sha256(REPO / "scripts/collect_natural_video_cross_model_proof_v17.py"),
        "ordered_keys": ordered_keys,
    })
    existing = {
        (str(row["benchmark"]), str(row["sample_id"])): row
        for row in json.loads(args.output.read_text(encoding="utf-8"))
    }
    for row in existing.values():
        if row.get("collection_contract_sha256") != contract_sha256:
            raise ValueError("cached cross-model receipt contract mismatch")
    row_by_key = {key: row for key, row in zip(ordered_keys, inputs)}
    pending = [key for key in ordered_keys if key not in existing]
    if not pending:
        print(json.dumps({"status": "already_complete", "samples": len(existing)}))
        return

    keys = runpy.run_path(str(args.keys))
    api_key = keys.get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("configured cross-model API key is missing")

    original_json_call = paired.media_helpers._json_call

    def expanded_budget_json_call(
        client: OpenAI,
        *,
        model: str,
        system: str,
        content: list[dict[str, Any]],
        max_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = None
        payload = None
        raw = ""
        parse_errors = []
        for transport_attempt in range(1, args.transport_attempts + 1):
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_completion_tokens=args.max_completion_tokens,
                response_format={"type": "json_object"},
                # OpenRouter counts Gemini's hidden reasoning against the completion
                # budget.  Low effort changes transport allocation, not the public
                # evidence, typed proof contract, or recovery decision.
                extra_body={"reasoning": {"effort": "low"}},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            )
            raw = response.choices[0].message.content or ""
            try:
                if not raw:
                    raise ValueError("model returned no JSON content")
                payload = json.loads(raw)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                parse_errors.append(f"attempt {transport_attempt}: {exc}")
        if response is None or payload is None:
            raise ValueError("transport JSON retries exhausted: " + " | ".join(parse_errors))
        usage = response.usage
        return payload, {
            "model": str(response.model),
            "finish_reason": str(response.choices[0].finish_reason),
            "prompt_tokens": int(usage.prompt_tokens if usage else 0),
            "completion_tokens": int(usage.completion_tokens if usage else 0),
            "cost": float(getattr(usage, "cost", 0.0) or 0.0),
            "response_sha256": v17._content_hash(payload),
            "transport_fallback": "completion_budget_and_hidden_reasoning_allocation",
            "hidden_reasoning_effort": "low",
            "initial_max_completion_tokens": int(max_tokens),
            "fallback_max_completion_tokens": args.max_completion_tokens,
            "transport_attempts_used": len(parse_errors) + 1,
            "prior_transport_parse_errors": parse_errors,
        }

    paired.media_helpers._json_call = expanded_budget_json_call
    try:
        for key in pending:
            row = v17._collect_one(
                row_by_key[key],
                config=config,
                api_key=str(api_key),
                contract_sha256=contract_sha256,
            )
            row["transport_fallback"] = {
                "reason": "provider_response_truncated_before_valid_json",
                "semantic_prompt_changed": False,
                "model_changed": False,
                "media_changed": False,
                "max_completion_tokens": args.max_completion_tokens,
                "hidden_reasoning_effort": "low",
            }
            existing[key] = row
            args.output.write_text(
                json.dumps(
                    [existing[item] for item in ordered_keys if item in existing],
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "completed": list(key),
                "progress": f"{len(existing)}/{len(ordered_keys)}",
            }), flush=True)
    finally:
        paired.media_helpers._json_call = original_json_call

    missing = [key for key in ordered_keys if key not in existing]
    if missing:
        raise SystemExit(f"transport fallback incomplete: {missing}")
    print(json.dumps({
        "status": "NATURAL_VIDEO_V17_TRANSPORT_COMPLETE",
        "samples": len(existing),
        "fallback_rows": len(pending),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
