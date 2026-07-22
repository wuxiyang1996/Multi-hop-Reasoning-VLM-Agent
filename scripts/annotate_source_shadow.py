#!/usr/bin/env python3
"""Annotate immutable source-policy receipts with a non-acting shadow Agent."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import EvidenceVerdict, stable_hash  # noqa: E402
from motif_transfer.frozen_motif_agent import OpenAICompatibleBackend  # noqa: E402
from motif_transfer.instrumented_import import import_native_source_batch  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_object(raw: str, keys: set[str]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"expected exact object keys {sorted(keys)}")
    return value


def _text_state(state: Mapping[str, Any]) -> str:
    return str(state.get("observable_state", ""))[:1500]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    backend = OpenAICompatibleBackend(
        args.endpoint,
        {"shadow_predict": args.model, "shadow_verify": args.model},
        json_mode=True,
        temperature=0.0,
        request_overrides={
            "max_tokens": 256,
            "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        transport_attempts=1,
    )
    episodes = import_native_source_batch(args.evidence)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        records = episode.records
        if args.max_steps is not None:
            records = records[:args.max_steps]
        for record in records:
            base = {
                "schema_version": 1,
                "episode_id": record.episode_id,
                "step": record.step,
                "source_transition_receipt_id": record.transition.receipt_id,
                "before_hash": record.transition.before_hash,
                "action": record.action,
                "after_hash": record.transition.after_hash,
            }
            predict_raw = None
            try:
                action_ordinal = record.before.native_actions.index(record.action)
                predict_payload = {
                    "before_observation": _text_state(record.before.state),
                    "already_selected_native_action": record.action,
                    "already_selected_action_ordinal": action_ordinal,
                    "recorded_decision_reasoning": record.action_reasoning[:500],
                }
                predict_system = (
                    "The source Decision Agent already selected the displayed native action. "
                    "You have no action authority. Predict only its next observable delta or "
                    "abstain. Do not propose, rank, replace, or map actions and do not invent "
                    "hidden state. Return exactly one JSON object with keys decision, "
                    "action_ordinal, predicted_observable_delta, rationale. decision is "
                    "PREDICT or ABSTAIN. predicted_observable_delta may be any JSON value."
                )
                predict_raw = backend.complete(
                    "shadow_predict", predict_system, predict_payload,
                )
                prediction = _exact_object(
                    predict_raw,
                    {"decision", "action_ordinal", "predicted_observable_delta", "rationale"},
                )
                if prediction["decision"] not in {"PREDICT", "ABSTAIN"}:
                    raise ValueError("invalid prediction decision")
                claimed_ordinal = prediction["action_ordinal"]
                if prediction["decision"] == "PREDICT":
                    if int(claimed_ordinal) != action_ordinal:
                        raise ValueError("shadow Agent changed the frozen action ordinal")
                elif claimed_ordinal is not None and int(claimed_ordinal) != action_ordinal:
                    raise ValueError("abstaining Agent referenced a different action ordinal")
                if not isinstance(prediction["rationale"], str):
                    raise ValueError("rationale must be a string")
                predict_receipt = {
                    "payload_sha256": stable_hash(predict_payload),
                    "raw_response": predict_raw,
                    "raw_response_sha256": stable_hash(predict_raw),
                    "usage": dict(backend.last_usage),
                }
                if prediction["decision"] == "ABSTAIN":
                    rows.append({
                        **base,
                        "status": "AGENT_ABSTAINED",
                        "action_ordinal": action_ordinal,
                        "prediction": prediction,
                        "prediction_receipt": predict_receipt,
                    })
                    continue

                verify_payload = {
                    "before_observation": _text_state(record.before.state),
                    "already_executed_native_action": record.action,
                    "prediction": prediction,
                    "after_observation": _text_state(record.after.state),
                    "official_reward": record.reward,
                    "official_done": record.after.terminal,
                    "source_transition_receipt_id": record.transition.receipt_id,
                }
                verify_system = (
                    "Compare one prediction with the supplied immutable before/action/after "
                    "receipt. Do not infer hidden state or select an action. Return exactly one "
                    "JSON object with keys verdict and evidence_claim. verdict is SUPPORTED, "
                    "REFUTED, or INCONCLUSIVE."
                )
                verify_raw = backend.complete(
                    "shadow_verify", verify_system, verify_payload,
                )
                verification = _exact_object(
                    verify_raw, {"verdict", "evidence_claim"},
                )
                EvidenceVerdict(str(verification["verdict"]))
                if not isinstance(verification["evidence_claim"], str):
                    raise ValueError("evidence_claim must be a string")
                rows.append({
                    **base,
                    "status": "VALID_CLOSED_LOOP_ANNOTATION",
                    "action_ordinal": action_ordinal,
                    "prediction": prediction,
                    "prediction_receipt": predict_receipt,
                    "verification": verification,
                    "verification_receipt": {
                        "payload_sha256": stable_hash(verify_payload),
                        "raw_response": verify_raw,
                        "raw_response_sha256": stable_hash(verify_raw),
                        "usage": dict(backend.last_usage),
                    },
                })
            except Exception as exc:
                rows.append({
                    **base,
                    "status": "EXCLUDED_PROTOCOL_GAP",
                    "error": f"{type(exc).__name__}:{exc}",
                    "unparsed_prediction_response": predict_raw,
                })

    args.output.mkdir(parents=True, exist_ok=False)
    annotations = args.output / "shadow_annotations.jsonl"
    annotations.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    source_files = {
        name: _sha256(args.evidence / name)
        for name in ("manifest.json", "events.jsonl", "episodes.jsonl")
    }
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in sorted({row["status"] for row in rows})
    }
    manifest = {
        "schema_version": 1,
        "authority": "OFFLINE_SHADOW_ANNOTATION_OF_IMMUTABLE_SOURCE_RECEIPTS",
        "causal_boundary": "PREDICTION_CALL_EXCLUDES_AFTER_OBSERVATION",
        "source_evidence": str(args.evidence.resolve()),
        "source_files_sha256": source_files,
        "backend_identity": backend.identity,
        "annotator_sha256": _sha256(Path(__file__)),
        "annotation_file": annotations.name,
        "annotation_file_sha256": _sha256(annotations),
        "row_count": len(rows),
        "status_counts": counts,
        "claim_limit": (
            "Agent fields are untrusted claims. The Harness validates only schema, exact "
            "action ordinal, immutable source receipt lineage, and recorded hashes."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "status_counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
