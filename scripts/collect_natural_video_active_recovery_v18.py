#!/usr/bin/env python3
"""Collect active, wrapper-executed verification on consumed natural-video dev."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_natural_video_focused_verify_v16 as focused  # noqa: E402
import run_active_video_wrapper_transfer as media_helpers  # noqa: E402
import run_structured_video_transfer as structured  # noqa: E402
from motif_transfer.natural_video_active_recovery import (  # noqa: E402
    ACTIVE_PREDICATE_KINDS,
    authentic_recovery_decision,
    parse_active_arbitration,
    parse_active_probe,
    source_compatible,
)
from motif_transfer.natural_video_recovery import PROOF_KINDS  # noqa: E402
from motif_transfer.sokoban_video_recovery import validate_source_receipt  # noqa: E402
from motif_transfer.visual_wrapper_bridge import (  # noqa: E402
    build_video_registry,
    execute_video_intervention,
    route_question,
    video_tool_schemas,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _image_content(value: bytes) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(value).decode("ascii")
        },
    }


def _decode_json_object(raw: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Decode provider JSON, allowing only a non-semantic outer text wrapper."""

    try:
        payload = json.loads(raw)
        if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
            payload = payload[0]
            wrapper = {
                "kind": "singleton_list",
                "prefix_chars": "0",
                "suffix_chars": "0",
                "prefix_sha256": hashlib.sha256(b"").hexdigest(),
                "suffix_sha256": hashlib.sha256(b"").hexdigest(),
            }
        else:
            wrapper = {
                "kind": "none",
                "prefix_chars": "0",
                "suffix_chars": "0",
                "prefix_sha256": hashlib.sha256(b"").hexdigest(),
                "suffix_sha256": hashlib.sha256(b"").hexdigest(),
            }
    except json.JSONDecodeError as direct_error:
        start = raw.find("{")
        if start < 0:
            raise direct_error
        try:
            payload, end = json.JSONDecoder().raw_decode(raw, start)
        except json.JSONDecodeError:
            raise direct_error
        prefix = raw[:start].strip()
        suffix = raw[end:].strip()
        # The JSON object remains the only semantic payload.  Preserve the exact
        # discarded wrapper in the usage receipt so normalization is auditable.
        if (
            len(prefix) + len(suffix) > 4096
            or any(token in prefix + suffix for token in ("{", "}"))
        ):
            raise ValueError("provider response does not contain one bounded JSON object")
        wrapper = {
            "kind": (
                "markdown_code_fence"
                if "```" in prefix + suffix
                else "bounded_outer_text"
            ),
            "prefix_chars": str(len(prefix)),
            "suffix_chars": str(len(suffix)),
            "prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
            "suffix_sha256": hashlib.sha256(suffix.encode("utf-8")).hexdigest(),
        }
    if not isinstance(payload, dict):
        raise ValueError("provider JSON root must be an object")
    return payload, wrapper


def _json_call(
    client: OpenAI,
    *,
    model: str,
    system: str,
    content: list[dict[str, Any]],
    max_tokens: int,
    transport_retries: int,
    hidden_reasoning_effort: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    failures = []
    for attempt in range(1, transport_retries + 1):
        kwargs: dict[str, Any] = {}
        if hidden_reasoning_effort:
            kwargs["extra_body"] = {"reasoning": {"effort": hidden_reasoning_effort}}
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            **kwargs,
        )
        choice = response.choices[0]
        raw = choice.message.content or ""
        try:
            payload, transport_wrapper = _decode_json_object(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            usage = response.usage
            failures.append(
                f"attempt {attempt}: {exc}; model={response.model}; "
                f"finish_reason={choice.finish_reason}; raw_chars={len(raw)}; "
                f"completion_tokens={int(usage.completion_tokens if usage else 0)}; "
                f"refusal={getattr(choice.message, 'refusal', None)!r}"
            )
            continue
        usage = response.usage
        return payload, {
            "model": str(response.model),
            "finish_reason": str(response.choices[0].finish_reason),
            "prompt_tokens": int(usage.prompt_tokens if usage else 0),
            "completion_tokens": int(usage.completion_tokens if usage else 0),
            "cost": float(getattr(usage, "cost", 0.0) or 0.0),
            "response_sha256": _content_hash(payload),
            "transport_attempts": attempt,
            "prior_transport_errors": failures,
            "hidden_reasoning_effort": hidden_reasoning_effort,
            "transport_wrapper_removed": transport_wrapper,
        }
    raise ValueError("JSON transport retries exhausted: " + " | ".join(failures))


def _claim_binding(row: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    primary = str(row["primary"]["answer"])
    alternative = str(row["cross_model_proof"]["answer"])
    slots = [primary, alternative]
    if int(hashlib.sha256(
        f"{row['benchmark']}|{row['sample_id']}".encode("utf-8")
    ).hexdigest()[:2], 16) % 2:
        slots.reverse()
    claim_ids = ["C0", "C1"]
    return claim_ids, dict(zip(claim_ids, slots))


def _plan_probe(
    client: OpenAI,
    *,
    row: Mapping[str, Any],
    panels: Sequence[bytes],
    duration_seconds: float,
    tool_schemas: Sequence[Mapping[str, Any]],
    claim_ids: Sequence[str],
    claim_to_slot: Mapping[str, str],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    public = row["sample_public"]
    planner = config["planner_model"]
    sensing = config["active_sensing"]
    max_span = min(
        float(sensing["maximum_window_seconds"]),
        float(sensing["maximum_window_fraction"]) * duration_seconds,
    )
    claims = {
        claim_id: {
            "slot": claim_to_slot[claim_id],
            "text": public["options"][claim_to_slot[claim_id]],
        }
        for claim_id in claim_ids
    }
    prompt = (
        "Choose one real wrapper sensing action that best distinguishes the two "
        "blinded answer claims using a temporally focused observation not present "
        "at this density in the overview. Do not answer the question and do not "
        "guess which claim came from an earlier agent. Bind the action to one typed "
        "observable predicate. Use clip-relative seconds. "
        f"The clip duration is {duration_seconds:.3f}s and the window span must be "
        f"at most {max_span:.3f}s. The controller will override n to "
        f"{int(sensing['frames_per_probe'])}. Actual wrapper schema: "
        f"{json.dumps(list(tool_schemas), ensure_ascii=False)}\n"
        f"Question: {public['question']}\n"
        f"Options: {json.dumps(public['options'], ensure_ascii=False)}\n"
        f"Blinded compared claims: {json.dumps(claims, ensure_ascii=False)}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(panels):
        content.extend([
            {"type": "text", "text": f"Low-density chronological overview {index + 1}/{len(panels)}:"},
            _image_content(panel),
        ])
    last_error = ""
    for _ in range(int(planner["schema_retries"])):
        attempt = list(content)
        if last_error:
            attempt.append({"type": "text", "text": "Schema error: " + last_error})
        payload, usage = _json_call(
            client,
            model=str(planner["id"]),
            system=(
                "Return JSON only: {claim_ids:[\"C0\",\"C1\"],tool:"
                "\"sample_frames\",predicate_kind:ENTITY_GROUNDING|EVENT_OCCURRENCE|"
                "TEMPORAL_ORDER|CAUSAL_LINK,start_sec:number,end_sec:number,"
                "expected_facts:{C0:string,C1:string},why_discriminative:string}."
            ),
            content=attempt,
            max_tokens=int(planner["max_tokens"]),
            transport_retries=int(planner["transport_retries"]),
            hidden_reasoning_effort=str(planner.get("hidden_reasoning_effort") or "") or None,
        )
        try:
            parsed = parse_active_probe(
                payload,
                claim_ids=claim_ids,
                duration_seconds=duration_seconds,
                frames_per_probe=int(sensing["frames_per_probe"]),
                maximum_window_fraction=float(sensing["maximum_window_fraction"]),
                maximum_window_seconds=float(sensing["maximum_window_seconds"]),
            )
            return parsed, payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("active probe schema retries exhausted: " + last_error)


def _arbitrate(
    client: OpenAI,
    *,
    row: Mapping[str, Any],
    overview_panels: Sequence[bytes],
    evidence_panel: bytes,
    probe: Mapping[str, Any],
    wrapper_receipt: Mapping[str, Any],
    claim_ids: Sequence[str],
    claim_to_slot: Mapping[str, str],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    public = row["sample_public"]
    arbiter = config["arbiter_model"]
    claims = {
        claim_id: {
            "slot": claim_to_slot[claim_id],
            "text": public["options"][claim_to_slot[claim_id]],
            "expected_fact": probe["expected_facts"][claim_id],
        }
        for claim_id in claim_ids
    }
    prompt = (
        "Independently arbitrate a video question after a real focused sensing "
        "action. You are blind to which compared claim was the earlier commitment "
        "and blind to all previous model conclusions. Use the overview only for "
        "context and the focused panel to test the typed predicate. Evaluate both "
        "blinded claims with exactly five proof steps in the required order. Mark "
        "ANSWER_ENTAILMENT REFUTED only for visible contradiction, SUPPORTED only "
        "when the necessary facts are visibly established, otherwise UNKNOWN. "
        "Return calibrated probabilities for every native option. No gold, official "
        "program, graph, or relation annotation is available.\n"
        f"Question: {public['question']}\n"
        f"Options: {json.dumps(public['options'], ensure_ascii=False)}\n"
        f"Blinded claims: {json.dumps(claims, ensure_ascii=False)}\n"
        f"Executed typed probe: {json.dumps(probe, ensure_ascii=False)}\n"
        "Wrapper receipt (image-free): "
        + json.dumps({
            "tool": wrapper_receipt["tool"],
            "arguments": wrapper_receipt["arguments"],
            "proxy_frame_indices": wrapper_receipt["proxy_frame_indices"],
        }, ensure_ascii=False)
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, panel in enumerate(overview_panels):
        content.extend([
            {"type": "text", "text": f"Context overview {index + 1}/{len(overview_panels)}:"},
            _image_content(panel),
        ])
    content.extend([
        {"type": "text", "text": "New focused wrapper observation:"},
        _image_content(evidence_panel),
    ])
    last_error = ""
    for _ in range(int(arbiter["schema_retries"])):
        attempt = list(content)
        if last_error:
            attempt.append({"type": "text", "text": "Schema error: " + last_error})
        payload, usage = _json_call(
            client,
            model=str(arbiter["id"]),
            system=(
                "Return JSON only: {answer:string,probabilities:{slot:number},"
                "candidate_proofs:[{claim_id:string,proof_steps:[{kind:string,"
                "status:SUPPORTED|REFUTED|UNKNOWN,confidence:number,visible_fact:"
                "string}]}],observed_evidence:[string],unresolved_uncertainties:"
                "[string],reason:string}. candidate_proofs must preserve C0,C1; "
                "each proof must preserve " + ",".join(PROOF_KINDS) + "."
            ),
            content=attempt,
            max_tokens=int(arbiter["max_tokens"]),
            transport_retries=int(arbiter["transport_retries"]),
            hidden_reasoning_effort=str(arbiter.get("hidden_reasoning_effort") or "") or None,
        )
        try:
            parsed = parse_active_arbitration(
                payload,
                slots=tuple(public["answer_slots"]),
                claim_ids=claim_ids,
            )
            return parsed, payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("active arbitration schema retries exhausted: " + last_error)


def _collect_one(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    public = row["sample_public"]
    sensing = config["active_sensing"]
    planner_client = OpenAI(
        api_key=api_key,
        base_url=str(config["planner_model"]["base_url"]),
        timeout=float(config["planner_model"]["timeout_seconds"]),
        max_retries=int(config["planner_model"]["max_retries"]),
    )
    arbiter_client = OpenAI(
        api_key=api_key,
        base_url=str(config["arbiter_model"]["base_url"]),
        timeout=float(config["arbiter_model"]["timeout_seconds"]),
        max_retries=int(config["arbiter_model"]["max_retries"]),
    )
    overview_panels = focused._proof_panels(row, config)
    proxy_frames, metadata = structured._sample_clip(
        Path(public["video_path"]),
        start_sec=float(public["clip_start_seconds"]),
        end_sec=(
            float(public["clip_end_seconds"])
            if public.get("clip_end_seconds") is not None else None
        ),
        frame_count=int(sensing["proxy_frame_count"]),
        max_side=int(sensing["proxy_frame_max_side"]),
    )
    duration = float(metadata["duration_seconds"])
    registry, proxy_fps = build_video_registry(
        proxy_frames,
        duration_seconds=duration,
        wrapper_root=config["wrapper_root"],
        required_tools=("sample_frames",),
    )
    schemas = video_tool_schemas(registry, allowed_tools=("sample_frames",))
    routing = route_question(
        str(public["question"]), modality="video", wrapper_root=config["wrapper_root"],
    ).as_dict()
    claim_ids, claim_to_slot = _claim_binding(row)
    try:
        probe, probe_raw, planner_usage = _plan_probe(
            planner_client,
            row=row,
            panels=overview_panels,
            duration_seconds=duration,
            tool_schemas=schemas,
            claim_ids=claim_ids,
            claim_to_slot=claim_to_slot,
            config=config,
        )
    except Exception as exc:
        raise RuntimeError(f"planner stage failed: {exc}") from exc
    selected, wrapper_receipt = execute_video_intervention(
        registry,
        proxy_frames,
        tool=probe["tool"],
        arguments=probe["arguments"],
    )
    indices = wrapper_receipt["proxy_frame_indices"]
    evidence_panel = media_helpers._panel_bytes(
        selected,
        labels=[f"A{index} {index / proxy_fps:.2f}s" for index in indices],
        frame_width=int(sensing["focused_frame_width"]),
        quality=int(sensing["jpeg_quality"]),
    )
    try:
        arbitration, arbitration_raw, arbiter_usage = _arbitrate(
            arbiter_client,
            row=row,
            overview_panels=overview_panels,
            evidence_panel=evidence_panel,
            probe=probe,
            wrapper_receipt=wrapper_receipt,
            claim_ids=claim_ids,
            claim_to_slot=claim_to_slot,
            config=config,
        )
    except Exception as exc:
        raise RuntimeError(f"arbiter stage failed: {exc}") from exc
    primary = str(row["primary"]["answer"])
    alternative = str(row["cross_model_proof"]["answer"])
    recover = authentic_recovery_decision(
        arbitration,
        claim_to_slot=claim_to_slot,
        primary_slot=primary,
        alternative_slot=alternative,
    )
    authentic_answer = alternative if recover else primary
    gold = str(row["gold_answer"])
    return {
        "schema_version": 18,
        "execution_status": "ACTIVE_VERIFICATION_COMPLETED",
        "benchmark": row["benchmark"],
        "split": "development",
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "family": row["family"],
        "gold_answer": gold,
        "primary": row["primary"],
        "primary_correct": bool(row["primary_correct"]),
        "cross_model_proof_answer": alternative,
        "cross_model_proof_correct": bool(row["proof_correct"]),
        "claim_to_slot": claim_to_slot,
        "active_probe": probe,
        "active_probe_raw": probe_raw,
        "wrapper_routing": routing,
        "wrapper_receipt": wrapper_receipt,
        "arbitration": arbitration,
        "arbitration_raw": arbitration_raw,
        "active_direct_correct": arbitration["answer"] == gold,
        "authentic_recover": recover,
        "authentic_answer": authentic_answer,
        "authentic_correct": authentic_answer == gold,
        "authentic_uplift": int(authentic_answer == gold) - int(bool(row["primary_correct"])),
        "usage": {"planner": planner_usage, "arbiter": arbiter_usage},
        "video_metadata": {**metadata, "wrapper_proxy_fps": proxy_fps},
        "overview_panel_sha256": [hashlib.sha256(value).hexdigest() for value in overview_panels],
        "active_evidence_sha256": hashlib.sha256(evidence_panel).hexdigest(),
        "input_v17_row_sha256": _content_hash(row),
        "collection_contract_sha256": contract_sha256,
        "source_compatibility_rule": "OBSERVED_ACTION_EFFECT_ONLY",
        "runtime_saw_gold_or_official_structure": False,
    }


def _no_op_row(
    row: Mapping[str, Any], *, error: Exception, contract_sha256: str,
) -> dict[str, Any]:
    """Fail closed to the immutable primary when a TEST cannot be verified."""

    primary = str(row["primary"]["answer"])
    gold = str(row["gold_answer"])
    return {
        "schema_version": 18,
        "execution_status": "NOOP_TO_PRIMARY",
        "benchmark": row["benchmark"],
        "split": "development",
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "family": row["family"],
        "gold_answer": gold,
        "primary": row["primary"],
        "primary_correct": bool(row["primary_correct"]),
        "cross_model_proof_answer": str(row["cross_model_proof"]["answer"]),
        "cross_model_proof_correct": bool(row["proof_correct"]),
        "active_probe": None,
        "wrapper_receipt": None,
        "arbitration": None,
        "active_direct_correct": None,
        "authentic_recover": False,
        "authentic_answer": primary,
        "authentic_correct": primary == gold,
        "authentic_uplift": 0,
        "fail_closed_error": f"{type(error).__name__}: {error}",
        "input_v17_row_sha256": _content_hash(row),
        "collection_contract_sha256": contract_sha256,
        "source_compatibility_rule": "OBSERVED_ACTION_EFFECT_ONLY",
        "runtime_saw_gold_or_official_structure": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lineage_paths = {
        "source_receipt_sha256": Path(config["source_receipt"]),
        "input_v15_receipts_sha256": Path(config["input_v15_receipts"]),
        "input_v17_receipts_sha256": Path(config["input_v17_receipts"]),
        "collector_sha256": Path(__file__).resolve(),
        "contract_module_sha256": REPO / "src/motif_transfer/natural_video_active_recovery.py",
        "wrapper_bridge_sha256": REPO / "src/motif_transfer/visual_wrapper_bridge.py",
    }
    for key, path in lineage_paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V18 frozen lineage mismatch: {key}")
    validate_source_receipt(json.loads(
        Path(config["source_receipt"]).read_text(encoding="utf-8")
    ))
    v15 = json.loads(Path(config["input_v15_receipts"]).read_text(encoding="utf-8"))
    v17 = json.loads(Path(config["input_v17_receipts"]).read_text(encoding="utf-8"))
    public_by_key = {
        (str(row["benchmark"]), str(row["sample_id"])): row for row in v15
    }
    candidates = []
    for proof_row in v17:
        key = (str(proof_row["benchmark"]), str(proof_row["sample_id"]))
        if not source_compatible(str(proof_row["benchmark"]), str(proof_row["family"])):
            continue
        if proof_row["primary"]["answer"] == proof_row["cross_model_proof"]["answer"]:
            continue
        base = public_by_key[key]
        candidates.append({**proof_row, "sample_public": base["sample_public"]})
    ordered_keys = [(str(row["benchmark"]), str(row["sample_id"])) for row in candidates]
    if len(ordered_keys) != int(config["expected_active_rows"]):
        raise ValueError("V18 active row count drift")
    contract_sha256 = _content_hash({
        "config_sha256": _sha256(args.config),
        "input_v15_receipts_sha256": _sha256(Path(config["input_v15_receipts"])),
        "input_v17_receipts_sha256": _sha256(Path(config["input_v17_receipts"])),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "contract_module_sha256": _sha256(REPO / "src/motif_transfer/natural_video_active_recovery.py"),
        "ordered_keys": ordered_keys,
    })
    key_values = runpy.run_path(str(args.keys))
    api_key = key_values.get(config["api_key_name"])
    if not api_key:
        raise SystemExit(f"{config['api_key_name']} is missing")
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V18 receipt contract mismatch")
            existing[(str(row["benchmark"]), str(row["sample_id"]))] = row
    row_by_key = {key: row for key, row in zip(ordered_keys, candidates)}
    pending = [key for key in ordered_keys if key not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(
            [existing[key] for key in ordered_keys if key in existing],
            ensure_ascii=False,
            indent=2,
        ) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one,
                row_by_key[key],
                config=config,
                api_key=str(api_key),
                contract_sha256=contract_sha256,
            ): key for key in pending
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                existing[key] = future.result()
            except Exception as exc:
                existing[key] = _no_op_row(
                    row_by_key[key], error=exc, contract_sha256=contract_sha256,
                )
                save()
                print(json.dumps({
                    "no_op": list(key),
                    "error": f"{type(exc).__name__}: {exc}",
                    "progress": f"{len(existing)}/{len(ordered_keys)}",
                }), flush=True)
                continue
            save()
            print(json.dumps({
                "completed": list(key),
                "progress": f"{len(existing)}/{len(ordered_keys)}",
            }), flush=True)
    missing = [key for key in ordered_keys if key not in existing]
    if missing:
        raise SystemExit(f"incomplete V18 active recovery; rerun: {missing}")
    rows = [existing[key] for key in ordered_keys]
    print(json.dumps({
        "status": "NATURAL_VIDEO_V18_ACTIVE_DEVELOPMENT_COLLECTED",
        "samples": len(rows),
        "authentic_recoveries": sum(row["authentic_recover"] for row in rows),
        "primary_correct": sum(row["primary_correct"] for row in rows),
        "active_direct_evaluated": sum(
            row["active_direct_correct"] is not None for row in rows
        ),
        "active_direct_correct": sum(
            bool(row["active_direct_correct"]) for row in rows
        ),
        "authentic_correct": sum(row["authentic_correct"] for row in rows),
        "authentic_uplift_counts": {
            str(value): sum(row["authentic_uplift"] == value for row in rows)
            for value in (-1, 0, 1)
        },
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
