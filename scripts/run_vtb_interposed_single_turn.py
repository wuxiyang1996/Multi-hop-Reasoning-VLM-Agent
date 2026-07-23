#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from functools import partial
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.exact_request_cache import ExactRequestCache  # noqa: E402
from motif_transfer.transfer_matrix import REQUIRED_TARGET_CONDITIONS  # noqa: E402
from motif_transfer.vtb_capabilities import OFFICIAL_REQUIRED_KEYS, audit_vtb_runtime  # noqa: E402
from motif_transfer.vtb_interposition import (  # noqa: E402
    VTBInterpositionHarness,
    VTBReviewVerdict,
    VTBToolProposal,
    VTBToolReceipt,
    VTBVerificationVerdict,
    pad_json_to_exact_tokens,
    parse_harness_review,
    parse_harness_verification,
)


SOURCE_CONDITIONS = {
    "authentic_game_source", "renamed_game_source", "shuffled_game_source", "other_game_source",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_row(parquet: Path, sample_id: str) -> dict[str, Any]:
    import duckdb

    if not sample_id.startswith("row:"):
        raise ValueError("v2 uses frozen row:<zero-based index> IDs")
    index = int(sample_id.split(":", 1)[1])
    connection = duckdb.connect()
    columns = [row[0] for row in connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(parquet)]
    ).fetchall()]
    row = connection.execute(
        "SELECT * FROM read_parquet(?) LIMIT 1 OFFSET ?", [str(parquet), index]
    ).fetchone()
    if row is None:
        raise ValueError(f"sample does not exist: {sample_id}")
    return dict(zip(columns, row))


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _serialize_message(message: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": row.id,
                "type": "function",
                "function": {"name": row.function.name, "arguments": row.function.arguments},
            }
            for row in message.tool_calls
        ]
    return result


def _harness_system() -> str:
    return """You are an advisory-only Motif/Harness Agent. You never choose a target tool,
write tool arguments, or answer the target question. Review an already selected Decision Agent
proposal using only the supplied treatment evidence and live receipt IDs. Return JSON only with
exactly: verdict (ADMIT, REPLAN, or ABSTAIN), reason, expected_transition, termination_test,
source_receipt_ids, live_receipt_ids. Do not invent receipt IDs. REPLAN means the Decision Agent
must independently choose again; ABSTAIN disables source assistance and admits the current target
proposal under target-only fallback. expected_transition must describe evidence expected in the
tool observation; merely saying EXECUTE_TOOL or TOOL_RETURNS is not evidence."""


def _verification_system() -> str:
    return """You are an advisory-only verifier. You cannot choose tools or answer the task.
Compare the prior prediction with the cited real tool receipt. Return JSON only with exactly:
verdict (SUPPORTED, REFUTED, or INCONCLUSIVE), reason, receipt_id. Use the supplied receipt_id
exactly and do not infer unobserved state."""


def _judge_success(observation: Mapping[str, Any]) -> bool:
    if observation.get("ok") in {True, "true"}:
        return True
    if str(observation.get("status") or "").lower() == "success":
        return True
    return not bool(observation.get("error") or observation.get("stderr"))


def _cached_harness_completion(
    client: Any,
    cache: ExactRequestCache | None,
    *,
    model: str,
    system: str,
    payload_text: str,
) -> tuple[str, dict[str, Any], bool]:
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": payload_text},
        ],
        "response_format": {"type": "json_object"},
    }
    cached = cache.get(request) if cache is not None else None
    if cached is not None:
        return str(cached["content"]), dict(cached.get("usage") or {}), True
    completion = client.chat.completions.create(**request)
    content = str(completion.choices[0].message.content or "")
    usage = completion.usage.model_dump() if completion.usage is not None else {}
    if cache is not None:
        cache.put(request, {"content": content, "usage": usage})
    return content, usage, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Online two-agent interposition over pinned VTB tools.")
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "configs/vtb_single_turn_manifest_v2.json")
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--condition", choices=REQUIRED_TARGET_CONDITIONS, required=True)
    parser.add_argument("--treatment", type=Path,
                        help="Frozen SOURCE_SUPPORTED treatment artifact for source conditions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--decision-model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--harness-model", default="gpt-5-mini")
    parser.add_argument("--openai-base-url", default="https://us.api.openai.com/v1")
    parser.add_argument("--max-tool-rounds", type=int, default=20)
    parser.add_argument("--max-replans", type=int, default=1)
    parser.add_argument("--decision-cache", type=Path,
                        help="Shared exact-request cache required by matched conditions")
    parser.add_argument("--harness-cache", type=Path,
                        help="Shared exact-request cache for reproducible Harness calls")
    parser.add_argument("--harness-input-tokens", type=int, default=6000,
                        help="Exact o200k_base tokens per Harness user payload")
    parser.add_argument("--allow-degraded-adaptation", action="store_true")
    parser.add_argument("--protocol-smoke", action="store_true",
                        help="Allow a sub-20 cap only on the frozen adaptation item")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.sample_id not in {manifest["adaptation_id"], *manifest["test_ids"]}:
        raise SystemExit("sample is outside the frozen v2 manifest")
    adaptation = args.sample_id == manifest["adaptation_id"]
    if args.max_tool_rounds != int(manifest["official_tool_call_cap"]):
        if not (args.protocol_smoke and adaptation and 0 < args.max_tool_rounds < 20):
            raise SystemExit("only an adaptation protocol smoke may use a sub-20 cap")
    if not args.protocol_smoke and (args.decision_cache is None or args.harness_cache is None):
        raise SystemExit("matched run requires shared --decision-cache and --harness-cache")

    keys = runpy.run_path(str(args.keys))
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", *OFFICIAL_REQUIRED_KEYS):
        if keys.get(name) and not os.environ.get(name):
            os.environ[name] = str(keys[name])
    audit = audit_vtb_runtime(
        args.official_repo.resolve(),
        key_presence={name: bool(os.environ.get(name)) for name in OFFICIAL_REQUIRED_KEYS},
    )
    degraded = not audit.paper_faithful_full_tool_ready
    if degraded and not (args.allow_degraded_adaptation and adaptation):
        raise SystemExit("NOT_RUNNABLE: full-tool preflight failed outside adaptation diagnostic")
    if not audit.official_inference_ready:
        raise SystemExit("NOT_RUNNABLE: pinned official checkout or Python runtime is invalid")

    treatment: dict[str, Any] = {}
    known_source_receipts: set[str] = set()
    if args.condition in SOURCE_CONDITIONS:
        if args.treatment is None:
            raise SystemExit("source condition requires a frozen treatment artifact")
        treatment = json.loads(args.treatment.read_text(encoding="utf-8"))
        if treatment.get("condition") != args.condition:
            raise SystemExit("treatment condition mismatch")
        if treatment.get("source_lifecycle") != "SOURCE_SUPPORTED":
            raise SystemExit("confirmatory source treatment is not SOURCE_SUPPORTED")
        known_source_receipts = {str(row) for row in treatment.get("source_receipt_ids") or ()}
        if not known_source_receipts:
            raise SystemExit("source treatment contains no frozen source receipts")
    elif args.condition == "generic_reasoning":
        if args.treatment is not None:
            treatment = json.loads(args.treatment.read_text(encoding="utf-8"))
            if treatment.get("condition") != "generic_reasoning":
                raise SystemExit("generic treatment condition mismatch")
            if treatment.get("source_lifecycle") != "CONTROL":
                raise SystemExit("generic treatment must be labeled CONTROL")
            if treatment.get("source_receipt_ids"):
                raise SystemExit("generic treatment may not carry real source receipts")
        elif not args.protocol_smoke:
            raise SystemExit("matched generic condition requires a compiled treatment artifact")
    elif args.treatment is not None:
        raise SystemExit("target-only condition may not receive treatment")
    if treatment and treatment.get("payload_sha256") != stable_hash(treatment.get("payload") or {}):
        raise SystemExit("treatment payload hash mismatch")

    record = _load_row(args.parquet, args.sample_id)
    prompts = list(record.get("turn_prompts") or [])
    if str(record.get("turncase")) != "single-turn" or len(prompts) != 1:
        raise SystemExit("interposed v2 runner accepts exactly one single-turn item")
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    image_hashes = []
    for index, image in enumerate(record.get("images") or []):
        raw = image.get("bytes") if isinstance(image, dict) else None
        if not raw:
            raise SystemExit(f"image {index} has no inline bytes")
        suffix = Path(str(image.get("path") or "image.png")).suffix or ".png"
        path = args.asset_dir / f"input_{index}{suffix}"
        path.write_bytes(raw)
        image_paths.append(path.resolve())
        image_hashes.append(_sha(raw))

    official_scripts = args.official_repo.resolve() / "scripts"
    sys.path.insert(0, str(official_scripts))
    from prompt import system_prompt_high
    from tools import VisionTools

    functions = {
        "python_image_processing": partial(
            VisionTools.python_image_processing,
            processed_image_save_path=str(args.asset_dir.resolve()),
        ),
        "python_interpreter": VisionTools.python_interpreter,
        "google_search": VisionTools.google_search,
        "browser_get_page_text": VisionTools.browser_get_page_text,
        "historical_weather": VisionTools.historical_weather,
        "calculator": VisionTools.safe_calculator,
    }
    from openai import OpenAI

    decision = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=str(os.environ.get("OPENROUTER_API_KEY") or ""), timeout=180.0,
    )
    harness_client = OpenAI(
        base_url=args.openai_base_url,
        api_key=str(os.environ.get("OPENAI_API_KEY") or ""),
        timeout=180.0,
    )
    decision_cache = (
        ExactRequestCache(args.decision_cache, {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": args.decision_model,
            "temperature": 0,
        })
        if args.decision_cache is not None else None
    )
    harness_cache = (
        ExactRequestCache(args.harness_cache, {
            "provider": "openai",
            "base_url": args.openai_base_url,
            "model": args.harness_model,
            "response_format": "json_object",
        })
        if args.harness_cache is not None else None
    )
    initial_content = [{"type": "text", "text": str(prompts[0])}]
    initial_content.extend({"type": "image_url", "image_url": {"url": _data_url(path)}} for path in image_paths)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt_high},
        {"role": "user", "content": initial_content},
    ]
    live_receipts: list[VTBToolReceipt] = []
    raw_tool_trace = []
    reviews = []
    verifications = []
    model_content = []
    decision_usage = []
    harness_usage = []
    harness_input_token_counts = []
    schema_hashes = []
    source_enabled = args.condition in SOURCE_CONDITIONS
    replan_count = 0
    final_answer: str | None = None
    termination = "OFFICIAL_CAP_EXHAUSTED"

    for round_index in range(args.max_tool_rounds):
        tools = VisionTools.get_tools([str(path) for path in image_paths], str(args.asset_dir.resolve()))
        schema_hashes.append(stable_hash(tools))
        boundary = VTBInterpositionHarness(tools, audit.tool_contract_sha256)
        decision_request = {
            "model": args.decision_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
        }
        cached = decision_cache.get(decision_request) if decision_cache is not None else None
        if cached is None:
            response = decision.chat.completions.create(**decision_request)
            message = _serialize_message(response.choices[0].message)
            usage = response.usage.model_dump() if response.usage is not None else {}
            if decision_cache is not None:
                decision_cache.put(decision_request, {"message": message, "usage": usage})
            cache_hit = False
        else:
            message = dict(cached["message"])
            usage = dict(cached.get("usage") or {})
            cache_hit = True
        decision_usage.append({"cache_hit": cache_hit, "usage": usage})
        model_content.append({"round_index": round_index, "content": message.get("content")})
        tool_calls = list(message.get("tool_calls") or [])
        if not tool_calls:
            final_answer = str(message.get("content") or "").strip()
            termination = "MODEL_FINAL_ANSWER" if final_answer else "EMPTY_MODEL_FINAL"
            break
        assistant = message
        proposals = []
        for call in tool_calls:
            function = call["function"]
            arguments = json.loads(function["arguments"])
            proposal = VTBToolProposal.create(
                round_index, str(call["id"]), str(function["name"]), arguments,
            )
            boundary.validate_proposal(proposal)
            proposals.append(proposal)

        review = None
        if args.condition != "target_only" and (args.condition == "generic_reasoning" or source_enabled):
            review_payload = {
                "condition": args.condition,
                "proposals": [
                    {
                        "proposal_id": row.proposal_id,
                        "tool_name": row.tool_name,
                        "arguments": dict(row.arguments),
                    }
                    for row in proposals
                ],
                "treatment": treatment.get("payload", {}) if treatment else {
                    "kind": "diagnostic_generic_review_control",
                    "instruction": "test expected information gain; detect exact repetition; define a stop test",
                },
                "known_source_receipt_ids": sorted(known_source_receipts),
                "known_live_receipt_ids": [row.receipt_id for row in live_receipts],
                "recent_live_receipts": [
                    {"receipt_id": row.receipt_id, "success": row.success,
                     "observation_sha256": row.observation_sha256}
                    for row in live_receipts[-4:]
                ],
            }
            review_text = pad_json_to_exact_tokens(review_payload, args.harness_input_tokens)
            review_raw, review_usage, review_cache_hit = _cached_harness_completion(
                harness_client, harness_cache,
                model=args.harness_model, system=_harness_system(), payload_text=review_text,
            )
            harness_input_token_counts.append({
                "phase": "review", "round_index": round_index,
                "o200k_base_tokens": args.harness_input_tokens,
            })
            harness_usage.append({
                "phase": "review", "round_index": round_index,
                "cache_hit": review_cache_hit, "usage": review_usage,
            })
            review = parse_harness_review(review_raw)
            boundary.validate_review(
                review, condition=args.condition,
                known_source_receipts=known_source_receipts,
                known_live_receipts={row.receipt_id for row in live_receipts},
            )
            reviews.append({
                "round_index": round_index,
                "reviewed_proposal_ids": [row.proposal_id for row in proposals],
                **asdict(review),
            })
            if review.verdict == VTBReviewVerdict.REPLAN and replan_count < args.max_replans:
                replan_count += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "An advisory-only reviewer requested reconsideration. Its non-binding evidence test was: "
                        f"{review.reason}; expected={review.expected_transition}; stop={review.termination_test}. "
                        "Independently choose a target-native tool call or final answer."
                    ),
                })
                continue
            if review.verdict in {VTBReviewVerdict.ABSTAIN, VTBReviewVerdict.REPLAN}:
                source_enabled = False

        messages.append(assistant)
        for proposal in proposals:
            try:
                observation = functions[proposal.tool_name](**dict(proposal.arguments))
                if not isinstance(observation, Mapping):
                    observation = {"result": observation}
            except Exception as exc:
                observation = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            output_paths = [Path(row) for row in observation.get("output_paths", []) if Path(row).is_file()]
            output_hashes = tuple(_sha(path.read_bytes()) for path in output_paths)
            receipt = VTBToolReceipt.create(
                proposal,
                tool_contract_sha256=audit.tool_contract_sha256,
                observation=observation,
                success=_judge_success(observation),
                output_paths_sha256=output_hashes,
            )
            boundary.validate_receipt(proposal, receipt)
            live_receipts.append(receipt)
            raw_tool_trace.append({
                "proposal": {
                    "proposal_id": proposal.proposal_id,
                    "round_index": proposal.round_index,
                    "call_id": proposal.call_id,
                    "tool_name": proposal.tool_name,
                    "arguments": dict(proposal.arguments),
                    "agent_id": proposal.agent_id,
                },
                "observation": dict(observation),
                "receipt": asdict(receipt),
            })
            messages.append({
                "tool_call_id": proposal.call_id,
                "role": "tool",
                "name": proposal.tool_name,
                "content": str(dict(observation)),
            })
            if output_paths:
                image_paths.extend(output_paths)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Transformed images from {proposal.call_id}."},
                        *({"type": "image_url", "image_url": {"url": _data_url(path)}} for path in output_paths),
                    ],
                })

            if review is not None:
                verify_payload = {
                    "prediction": review.expected_transition,
                    "termination_test": review.termination_test,
                    "receipt": asdict(receipt),
                    "observation": dict(observation),
                    "recent_live_receipts": [asdict(row) for row in live_receipts[-4:]],
                    "observation_hash_seen_before": any(
                        row.observation_sha256 == receipt.observation_sha256
                        for row in live_receipts[:-1]
                    ),
                }
                verify_text = pad_json_to_exact_tokens(verify_payload, args.harness_input_tokens)
                verify_raw, verify_usage, verify_cache_hit = _cached_harness_completion(
                    harness_client, harness_cache,
                    model=args.harness_model, system=_verification_system(), payload_text=verify_text,
                )
                harness_input_token_counts.append({
                    "phase": "verify", "round_index": round_index,
                    "o200k_base_tokens": args.harness_input_tokens,
                })
                harness_usage.append({
                    "phase": "verify", "round_index": round_index,
                    "cache_hit": verify_cache_hit, "usage": verify_usage,
                })
                verification = parse_harness_verification(verify_raw)
                boundary.validate_verification(
                    verification, {row.receipt_id for row in live_receipts}
                )
                verifications.append({"round_index": round_index, **asdict(verification)})
                if source_enabled and verification.verdict == VTBVerificationVerdict.REFUTED:
                    source_enabled = False

    payload = {
        "schema_version": 1,
        "executor": "vtb_two_agent_online_interposition",
        "sample_id": args.sample_id,
        "split": "adaptation" if adaptation else "test",
        "condition": args.condition,
        "claim_label": (
            "PROTOCOL_SMOKE" if args.protocol_smoke else
            "CAPABILITY_DEGRADED_ADAPTATION_DIAGNOSTIC" if degraded else "MATCHED_RUN"
        ),
        "official_commit": audit.expected_commit,
        "tool_contract_sha256": audit.tool_contract_sha256,
        "dynamic_tool_schema_sha256_by_round": schema_hashes,
        "decision_model": args.decision_model,
        "harness_model": args.harness_model if args.condition != "target_only" else None,
        "max_tool_rounds": args.max_tool_rounds,
        "replan_count": replan_count,
        "termination_reason": termination,
        "final_answer": final_answer,
        "final_answer_present": bool(final_answer),
        "image_sha256": image_hashes,
        "source_enabled_at_end": source_enabled,
        "tool_call_count": len(live_receipts),
        "reviews": reviews,
        "verifications": verifications,
        "tool_trace": raw_tool_trace,
        "model_content": model_content,
        "decision_usage": decision_usage,
        "harness_usage": harness_usage,
        "harness_input_token_counts": harness_input_token_counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_id": args.sample_id,
        "condition": args.condition,
        "termination_reason": termination,
        "tool_calls": len(live_receipts),
        "reviews": len(reviews),
        "verifications": len(verifications),
        "replans": replan_count,
        "final_answer_present": bool(final_answer),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
