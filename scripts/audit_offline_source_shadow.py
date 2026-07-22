#!/usr/bin/env python3
"""Mechanically audit offline shadow annotations against immutable source receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import EvidenceVerdict, stable_hash  # noqa: E402
from motif_transfer.instrumented_import import import_native_source_batch  # noqa: E402
from motif_transfer.phase1_assets import read_jsonl  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_state(state) -> str:
    return str(state.get("observable_state", ""))[:1500]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.annotation / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("authority") != "OFFLINE_SHADOW_ANNOTATION_OF_IMMUTABLE_SOURCE_RECEIPTS":
        raise ValueError("unsupported annotation authority")
    if manifest.get("causal_boundary") != "PREDICTION_CALL_EXCLUDES_AFTER_OBSERVATION":
        raise ValueError("annotation did not record the causal prompt boundary")
    expected_source_hashes = manifest.get("source_files_sha256") or {}
    for name in ("manifest.json", "events.jsonl", "episodes.jsonl"):
        if expected_source_hashes.get(name) != _sha256(args.evidence / name):
            raise ValueError(f"immutable source file hash mismatch: {name}")
    annotation_path = args.annotation / str(manifest["annotation_file"])
    if manifest.get("annotation_file_sha256") != _sha256(annotation_path):
        raise ValueError("annotation file hash mismatch")

    source_records = {
        (episode.episode_id, record.step): record
        for episode in import_native_source_batch(args.evidence)
        for record in episode.records
    }
    rows = read_jsonl(annotation_path)
    row_keys = [(str(row.get("episode_id")), int(row.get("step", -1))) for row in rows]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("duplicate annotation for one source step")
    gaps: list[str] = []
    valid = 0
    abstained = 0
    for row, key in zip(rows, row_keys):
        record = source_records.get(key)
        if record is None:
            gaps.append(f"{key}:UNKNOWN_SOURCE_STEP")
            continue
        if (
            row.get("source_transition_receipt_id") != record.transition.receipt_id
            or row.get("before_hash") != record.transition.before_hash
            or row.get("action") != record.action
            or row.get("after_hash") != record.transition.after_hash
        ):
            gaps.append(f"{key}:SOURCE_LINEAGE_MISMATCH")
            continue
        status = str(row.get("status"))
        if status == "EXCLUDED_PROTOCOL_GAP":
            continue
        action_ordinal = record.before.native_actions.index(record.action)
        if row.get("action_ordinal") != action_ordinal:
            gaps.append(f"{key}:ACTION_ORDINAL_MISMATCH")
            continue
        prediction = row.get("prediction") or {}
        predict_receipt = row.get("prediction_receipt") or {}
        predict_payload = {
            "before_observation": _text_state(record.before.state),
            "already_selected_native_action": record.action,
            "already_selected_action_ordinal": action_ordinal,
            "recorded_decision_reasoning": record.action_reasoning[:500],
        }
        raw_prediction = predict_receipt.get("raw_response")
        try:
            if predict_receipt.get("payload_sha256") != stable_hash(predict_payload):
                raise ValueError("prediction payload hash")
            if predict_receipt.get("raw_response_sha256") != stable_hash(raw_prediction):
                raise ValueError("prediction response hash")
            if json.loads(raw_prediction) != prediction:
                raise ValueError("prediction parse mismatch")
            if set(prediction) != {
                "decision", "action_ordinal", "predicted_observable_delta", "rationale",
            }:
                raise ValueError("prediction exact-key mismatch")
            if prediction.get("decision") not in {"PREDICT", "ABSTAIN"}:
                raise ValueError("prediction decision")
            if not isinstance(prediction.get("rationale"), str):
                raise ValueError("prediction rationale")
            claimed_ordinal = prediction["action_ordinal"]
            if prediction.get("decision") == "PREDICT":
                if int(claimed_ordinal) != action_ordinal:
                    raise ValueError("prediction action ordinal")
            elif claimed_ordinal is not None and int(claimed_ordinal) != action_ordinal:
                raise ValueError("abstention action ordinal")
        except Exception as exc:
            gaps.append(f"{key}:INVALID_PREDICTION_RECEIPT:{exc}")
            continue
        if status == "AGENT_ABSTAINED" and prediction.get("decision") == "ABSTAIN":
            abstained += 1
            continue
        if status != "VALID_CLOSED_LOOP_ANNOTATION" or prediction.get("decision") != "PREDICT":
            gaps.append(f"{key}:STATUS_PREDICTION_MISMATCH")
            continue
        verification = row.get("verification") or {}
        verify_receipt = row.get("verification_receipt") or {}
        verify_payload = {
            "before_observation": _text_state(record.before.state),
            "already_executed_native_action": record.action,
            "prediction": prediction,
            "after_observation": _text_state(record.after.state),
            "official_reward": record.reward,
            "official_done": record.after.terminal,
            "source_transition_receipt_id": record.transition.receipt_id,
        }
        raw_verification = verify_receipt.get("raw_response")
        try:
            if verify_receipt.get("payload_sha256") != stable_hash(verify_payload):
                raise ValueError("verification payload hash")
            if verify_receipt.get("raw_response_sha256") != stable_hash(raw_verification):
                raise ValueError("verification response hash")
            if json.loads(raw_verification) != verification:
                raise ValueError("verification parse mismatch")
            if set(verification) != {"verdict", "evidence_claim"}:
                raise ValueError("verification exact-key mismatch")
            EvidenceVerdict(str(verification["verdict"]))
            if not isinstance(verification.get("evidence_claim"), str):
                raise ValueError("verification evidence claim")
        except Exception as exc:
            gaps.append(f"{key}:INVALID_VERIFICATION_RECEIPT:{exc}")
            continue
        valid += 1

    report = {
        "schema_version": 1,
        "authority": "MECHANICAL_OFFLINE_SHADOW_AUDIT_NO_SEMANTIC_INTERPRETATION",
        "source_step_count": len(source_records),
        "annotation_row_count": len(rows),
        "valid_closed_loop_annotations": valid,
        "agent_abstentions": abstained,
        "excluded_protocol_gaps": sum(
            row.get("status") == "EXCLUDED_PROTOCOL_GAP" for row in rows
        ),
        "audit_gaps": gaps,
        "descriptive_checks": {
            "all_source_steps_covered": set(row_keys) == set(source_records),
            "no_duplicate_annotations": len(row_keys) == len(set(row_keys)),
            "all_nonexcluded_receipts_valid": not gaps,
            "source_trajectory_cannot_be_modified_by_annotation": True,
        },
        "claim_limit": (
            "This validates lineage and protocol coverage, not the semantic quality or "
            "cross-domain value of an Agent prediction."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["descriptive_checks"], sort_keys=True))


if __name__ == "__main__":
    main()
