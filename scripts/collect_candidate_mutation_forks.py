#!/usr/bin/env python3
"""Collect matched STAR BIND->MUTATE candidate program receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import collect_candidate_claim_video_forks as claim  # noqa: E402
import run_active_video_wrapper_transfer as media_helpers  # noqa: E402
import run_structured_video_transfer as structured  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_source_edge(source: Mapping[str, Any], required: Mapping[str, str]) -> None:
    edges = source.get("effect_ir", {}).get("edges", ())
    expected = (required["from"], required["to"], required["guard"])
    observed = {(row.get("from"), row.get("to"), row.get("guard")) for row in edges}
    if expected not in observed:
        raise ValueError(f"source IR lacks required edge {expected}")


def _compile_mutations(
    client: OpenAI, *, config: Mapping[str, Any], sample: Mapping[str, Any],
    base_candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = config["model"]
    inputs = [{
        "slot": str(row["slot"]),
        "candidate_claim": str(row["claim"]),
        "frozen_bind_entity": str(row["bind_entity_visual_description"]),
    } for row in base_candidates]
    prompt = (
        "Compile each candidate into one typed video MUTATE predicate. You do not "
        "know the answer and must preserve the slots, candidate meaning, and frozen "
        "bind entity exactly. A mutation is an observable state or action transition "
        "of the bound carrier, possibly after a visible boundary event. Express the "
        "state before and after; if the candidate describes an action, use not-yet-"
        "performed -> performed as the state transition. Do not score, rank, or "
        "eliminate candidates. Return all slots exactly once.\nQuestion: "
        + str(sample["question"])
        + "\nCandidate programs: " + json.dumps(inputs, ensure_ascii=False)
    )
    last_error = ""
    expected = [row["slot"] for row in inputs]
    for _ in range(int(model["schema_retries"])):
        payload, usage = media_helpers._json_call(
            client, model=str(model["id"]),
            system=(
                "Return JSON only: {candidate_mutations:[{slot:string,"
                "mutation_claim:string,boundary_event:string,before_state:string,"
                "after_state:string}]}. No correctness estimates."
            ),
            content=[{"type": "text", "text": prompt + (f"\nSchema error: {last_error}" if last_error else "")}],
            max_tokens=int(model["max_compile_tokens"]),
        )
        try:
            rows = list(payload.get("candidate_mutations") or ())
            if [str(row.get("slot")) for row in rows] != expected:
                raise ValueError("mutation compiler must preserve all candidate slots and order")
            parsed = []
            for base, row in zip(base_candidates, rows):
                values = {
                    key: str(row.get(key) or "").strip()
                    for key in ("mutation_claim", "boundary_event", "before_state", "after_state")
                }
                if not all(values.values()):
                    raise ValueError("typed mutation fields must be nonempty")
                parsed.append({
                    "slot": str(row["slot"]),
                    "bind_entity_visual_description": str(base["bind_entity_visual_description"]),
                    **values,
                })
            return parsed, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(f"mutation compiler schema failed: {last_error}")


def _ground_mutation(
    client: OpenAI, *, config: Mapping[str, Any], program: Mapping[str, Any],
    panel: bytes, observation_kind: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = config["model"]
    handle = (
        "Magenta boxes are a proposed persistent identity handle produced by an "
        "earlier BIND action. Use them only for identity; a box is not evidence of "
        "the transition. "
        if observation_kind != "UNBOUND_DUAL_VIEW" else
        "No persistent identity handle is supplied; infer the candidate carrier "
        "from the full frames and generic center views. "
    )
    prompt = (
        handle
        + "Measure this one typed MUTATE predicate from the ordered video views. "
        "Do not answer or compare any multiple-choice question. Bound carrier: "
        + str(program["bind_entity_visual_description"])
        + ". Candidate mutation claim: " + str(program["mutation_claim"])
        + ". Boundary event: " + str(program["boundary_event"])
        + ". Before state: " + str(program["before_state"])
        + ". After state: " + str(program["after_state"])
        + ". Return P(the specified before->after transition occurs in the correct "
        "temporal order | only visible evidence). Report uncertainty if the carrier, "
        "boundary, or either state is not visible."
    )
    last_error = ""
    for _ in range(int(model["schema_retries"])):
        payload, usage = media_helpers._json_call(
            client, model=str(model["id"]),
            system=(
                "Return JSON only: {support_probability:number,sensor_reliability:"
                "number,before_observed:bool,after_observed:bool,observed_facts:"
                "[string],measurement:string}."
            ),
            content=[
                {"type": "text", "text": prompt + (f" Schema error: {last_error}" if last_error else "")},
                media_helpers._image_content(panel),
            ],
            max_tokens=int(model["max_mutation_tokens"]),
        )
        try:
            parsed = claim._parse_relation(payload)
            parsed["before_observed"] = bool(payload.get("before_observed"))
            parsed["after_observed"] = bool(payload.get("after_observed"))
            return parsed, payload, usage
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError(f"mutation grounder schema failed: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--base-forks-file", required=True)
    parser.add_argument("--output-file", default="candidate_mutation_forks.json")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = Path(config["source"]["typed_summary"])
    if _sha256(source_path) != config["source"]["typed_summary_sha256"]:
        raise ValueError("source gate hash mismatch")
    source_gate = json.loads(source_path.read_text(encoding="utf-8"))
    if source_gate.get("status") != "SOURCE_TYPED_GATE_PASSED":
        raise ValueError("source typed gate did not pass")
    _validate_source_edge(source_gate, config["source"]["required_edge"])
    receipts = json.loads((args.run_dir / "receipts.json").read_text(encoding="utf-8"))
    selected = receipts[args.sample_offset:args.sample_offset + args.sample_count]
    if not selected:
        raise ValueError("empty mutation shard")
    base_path = args.run_dir / args.base_forks_file
    base = {str(row["sample_id"]): row for row in json.loads(base_path.read_text(encoding="utf-8"))}
    missing = [str(row["sample_id"]) for row in selected if str(row["sample_id"]) not in base]
    if missing or not all(bool(base[str(row["sample_id"])].get("complete")) for row in selected):
        raise ValueError(f"complete BIND base forks are required: {missing}")
    output_path = args.run_dir / args.output_file
    existing = {}
    if output_path.is_file():
        existing = {str(row["sample_id"]): row for row in json.loads(output_path.read_text(encoding="utf-8"))}
    keys = runpy.run_path(str(args.keys))
    model = config["model"]
    client = OpenAI(
        api_key=str(keys[model["api_key_name"]]), base_url=str(model["base_url"]),
        timeout=float(model["timeout_seconds"]), max_retries=int(model["max_retries"]),
    )

    def save() -> None:
        ordered = [existing[str(source["sample_id"])] for source in selected if str(source["sample_id"]) in existing]
        output_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for source in selected:
        sample_id = str(source["sample_id"])
        row = existing.get(sample_id)
        if row and row.get("complete"):
            continue
        base_row = base[sample_id]
        frames, metadata = structured._sample_clip(
            Path(source["sample"]["video_path"]),
            start_sec=float(source["video_metadata"]["clip_start_seconds"]),
            end_sec=float(source["video_metadata"]["clip_end_seconds"]),
            frame_count=int(config["media"]["proxy_frame_count"]),
            max_side=int(config["media"]["proxy_frame_max_side"]),
        )
        seconds = metadata["proxy_sample_seconds"]
        indices = list(base_row["candidates"][0]["track_indices"])
        global_panel, _ = claim._dual_view_panel(
            frames, indices, seconds, config=config, prefix="U",
        )
        if row is None:
            programs, usage = _compile_mutations(
                client, config=config, sample=source["sample"],
                base_candidates=base_row["candidates"],
            )
            candidates = []
            for program, base_candidate in zip(programs, base_row["candidates"]):
                candidates.append({
                    **program,
                    "track": base_candidate["track"],
                    "track_indices": base_candidate["track_indices"],
                    "track_evidence_sha256": base_candidate["track_evidence_sha256"],
                    "identity_verification": base_candidate["identity_verification"],
                })
            row = {
                "schema_version": 1, "benchmark": source["benchmark"],
                "sample_id": sample_id, "complete": False,
                "source_gate_sha256": _sha256(source_path),
                "base_bind_forks_sha256": _sha256(base_path),
                "config_sha256": _sha256(args.config),
                "compiler_usage": usage, "candidates": candidates,
                "compiler_saw_question_and_candidates": True,
                "compiler_saw_gold_or_official_program": False,
                "mutation_grounders_saw_full_question_option_set_or_gold": False,
            }
            existing[sample_id] = row
            save()
        candidates = row["candidates"]
        for index, candidate in enumerate(candidates):
            panels = {"unbound_mutation": (global_panel, "UNBOUND_DUAL_VIEW", 0)}
            bound_panel, bound_fallbacks = claim._dual_view_panel(
                frames, indices, seconds, config=config,
                track_indices=candidate["track_indices"],
                tracks=candidate["track"]["tracks"], prefix="B",
            )
            wrong = candidates[(index + 1) % len(candidates)]
            wrong_panel, wrong_fallbacks = claim._dual_view_panel(
                frames, indices, seconds, config=config,
                track_indices=wrong["track_indices"], tracks=wrong["track"]["tracks"],
                prefix="W",
            )
            panels["bound_mutation"] = (bound_panel, "BOUND_DUAL_VIEW", bound_fallbacks)
            panels["wrong_guard_mutation"] = (wrong_panel, "WRONG_GUARD_DUAL_VIEW", wrong_fallbacks)
            for key, (panel, observation_kind, fallbacks) in panels.items():
                if key in candidate:
                    continue
                parsed, raw, usage = _ground_mutation(
                    client, config=config, program=candidate, panel=panel,
                    observation_kind=observation_kind,
                )
                candidate[key] = {
                    **parsed, "raw": raw, "usage": usage,
                    "panel_sha256": hashlib.sha256(panel).hexdigest(),
                    "overlay_fallback_count": fallbacks,
                }
                if key == "wrong_guard_mutation":
                    candidate[key]["wrong_track_slot"] = str(wrong["slot"])
                save()
        row["complete"] = True
        save()
        print(json.dumps({
            "sample_id": sample_id,
            "deltas": [round(abs(float(c["bound_mutation"]["support_probability"]) - float(c["unbound_mutation"]["support_probability"])), 3) for c in candidates],
        }), flush=True)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
