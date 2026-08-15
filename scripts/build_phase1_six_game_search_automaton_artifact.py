#!/usr/bin/env python3
"""Authorize one common target artifact only after all six formal source gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase1_common_search_ir import (  # noqa: E402
    analyze_common_search_ir,
    canonical_policy_sha256,
    read_jsonl,
    validate_option_template_artifact,
)
from motif_transfer.real_source_interventions import validate_plan  # noqa: E402
from motif_transfer.sokoban_search_automaton_v16 import (  # noqa: E402
    ACTIONS,
    EVENTS,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_collected_rows(
    *,
    game: str,
    rows: list[dict],
    audit: dict,
    maximum_attempts_per_snapshot: int,
) -> None:
    observed_hashes = []
    for row in rows:
        body = dict(row)
        claimed = body.pop("row_sha256", None)
        if claimed != stable_hash(body):
            raise ValueError(f"{game}: collected row self-hash mismatch")
        if row.get("game") != game:
            raise ValueError(f"{game}: collected row game mismatch")
        observed_hashes.append(str(claimed))

    accepted_hashes = []
    for snapshot_id, snapshot in audit["snapshots"].items():
        attempts = list(snapshot.get("attempts") or [])
        accepted_index = snapshot.get("accepted_attempt_index")
        if not attempts or not isinstance(accepted_index, int):
            raise ValueError(f"{game}: incomplete audit for {snapshot_id}")
        if not 0 <= accepted_index < len(attempts):
            raise ValueError(f"{game}: invalid accepted attempt for {snapshot_id}")
        if len(attempts) > maximum_attempts_per_snapshot:
            raise ValueError(f"{game}: retry budget exceeded for {snapshot_id}")
        accepted_hashes.extend(
            map(str, attempts[accepted_index].get("row_sha256s") or [])
        )
    if sorted(accepted_hashes) != sorted(observed_hashes):
        raise ValueError(f"{game}: accepted collection receipts mismatch")


def _validate_manifest(manifest: dict) -> None:
    body = dict(manifest)
    claimed = body.pop("manifest_sha256", None)
    if claimed != stable_hash(body):
        raise ValueError("formal manifest self-hash mismatch")
    if manifest.get("status") not in {
        "FROZEN_BEFORE_FORMAL_SOURCE_COLLECTION",
        "FROZEN_BEFORE_STREETS_V2_FORMAL_COLLECTION",
    }:
        raise ValueError("formal manifest is not frozen")
    if manifest.get("target_data_read_for_freeze") is not False:
        raise ValueError("formal source protocol read target data")


def _validate_frozen_protocol_receipts(manifest: dict) -> dict:
    """Resolve inherited manifests and verify the frozen source code bytes."""

    source_manifest = manifest
    inherited = manifest.get("inherited_v1_manifest")
    if inherited:
        inherited_path = Path(inherited["path"])
        if _sha256(inherited_path) != inherited["file_sha256"]:
            raise ValueError("inherited V1 manifest file hash mismatch")
        source_manifest = _read(inherited_path)
        _validate_manifest(source_manifest)
        if source_manifest["manifest_sha256"] != inherited["manifest_sha256"]:
            raise ValueError("inherited V1 manifest identity mismatch")
        if manifest.get("protocol") != source_manifest.get("protocol"):
            raise ValueError("combined manifest changed the frozen protocol")
    code_receipts = list(source_manifest.get("code_receipts") or [])
    if not code_receipts:
        raise ValueError("formal manifest has no frozen code receipts")
    for receipt in code_receipts:
        path = Path(receipt["path"])
        if _sha256(path) != receipt["file_sha256"]:
            raise ValueError(f"frozen source code changed: {path}")
    return source_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase1_common_search_ir_formal_v1/manifest.json",
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=REPO / "runs/phase1_common_search_ir_formal_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            REPO / "runs/phase1_common_search_ir_formal_v1/"
            "common_search_automaton_artifact.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    manifest = _read(args.manifest)
    _validate_manifest(manifest)
    _validate_frozen_protocol_receipts(manifest)
    required_policy_hash = str(
        manifest["protocol"]["required_canonical_policy_sha256"]
    )
    config_receipts = {
        str(row["game"]): row for row in manifest["config_receipts"]
    }
    lineages = []
    policies = []
    for game in map(str, manifest["games"]):
        config_path = Path(config_receipts[game]["path"])
        if _sha256(config_path) != config_receipts[game]["file_sha256"]:
            raise ValueError(f"{game}: config changed after formal freeze")
        config = _read(config_path)
        run_dir = Path(
            config_receipts[game].get("run_dir") or (args.run_root / game)
        )
        plan_path = run_dir / "plan.json"
        rows_path = run_dir / "rows.jsonl"
        report_path = run_dir / "report.json"
        audit_path = rows_path.with_suffix(
            rows_path.suffix + ".collection-audit.json"
        )
        plan = _read(plan_path)
        validate_plan(plan)
        rows = read_jsonl(rows_path)
        report = _read(report_path)
        audit = _read(audit_path)
        if audit.get("plan_sha256") != plan.get("plan_sha256"):
            raise ValueError(f"{game}: collection audit/plan mismatch")
        if set(audit.get("snapshots") or {}) != {
            str(row["snapshot_id"]) for row in plan["snapshots"]
        }:
            raise ValueError(f"{game}: incomplete collection checkpoint set")
        _validate_collected_rows(
            game=game,
            rows=rows,
            audit=audit,
            maximum_attempts_per_snapshot=int(
                config["maximum_infrastructure_attempts_per_snapshot"]
            ),
        )
        if any(
            snapshot.get("retry_decision_read_outcome") is not False
            for snapshot in audit["snapshots"].values()
        ):
            raise ValueError(f"{game}: outcome-dependent retry detected")
        recomputed = analyze_common_search_ir(
            rows,
            primary_horizon=int(config["horizon"]),
            source_gate_requirements=config["source_gate"],
            minimum_eligible_fraction_each_split=float(
                config["minimum_eligible_fraction_each_split"]
            ),
            expected_policy_sha256=str(config["expected_policy_sha256"]),
            maximum_intervention_failed_rows=int(
                config["maximum_intervention_failed_rows"]
            ),
        )
        expected_report = recomputed | {
            "game": game,
            "rows_path": str(rows_path.resolve()),
            "rows_sha256": _sha256(rows_path),
        }
        if report != expected_report:
            raise ValueError(f"{game}: formal report does not recompute exactly")
        if not report.get("source_gate_passed"):
            raise ValueError(f"{game}: formal source gate failed")
        if report.get("canonical_policy_sha256") != required_policy_hash:
            raise ValueError(f"{game}: common-policy equivalence failed")
        if report["ledger_audit"].get("native_action_tokens_exported_to_ir"):
            raise ValueError(f"{game}: native action leaked into common IR")
        policy = dict(report["automaton_gate"]["learned_policy"])
        if set(policy) != set(EVENTS) or set(policy.values()) != set(ACTIONS):
            raise ValueError(f"{game}: invalid learned policy")
        if canonical_policy_sha256(policy) != required_policy_hash:
            raise ValueError(f"{game}: policy hash mismatch")
        policies.append(policy)
        template_receipt = None
        if config.get("option_template_artifact"):
            template_path = Path(config["option_template_artifact"])
            template = _read(template_path)
            validate_option_template_artifact(
                template, game=game, horizon=int(config["horizon"])
            )
            template_receipt = {
                "path": str(template_path.resolve()),
                "file_sha256": _sha256(template_path),
                "artifact_sha256": template["artifact_sha256"],
            }
            streets_receipts = manifest.get("streets_v2_design_receipts")
            if game == "gymv_streets_of_rage_2" and streets_receipts:
                if _sha256(template_path) != streets_receipts[
                    "v2_template_file_sha256"
                ]:
                    raise ValueError("Streets V2 template changed after freeze")
                if template["artifact_sha256"] != streets_receipts[
                    "v2_template_artifact_sha256"
                ]:
                    raise ValueError("Streets V2 template identity mismatch")
        lineages.append({
            "game": game,
            "config_file_sha256": _sha256(config_path),
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": _sha256(plan_path),
            "rows_file_sha256": _sha256(rows_path),
            "report_file_sha256": _sha256(report_path),
            "collection_audit_file_sha256": _sha256(audit_path),
            "eligible_ledgers": report["ledger_audit"]["eligible_ledgers"],
            "fresh_eligible_states": report["ledger_audit"][
                "split_eligible"
            ]["heldout"],
            "template_receipt": template_receipt,
        })
    if any(policy != policies[0] for policy in policies[1:]):
        raise ValueError("six formal source policies disagree")

    body = {
        "schema_version": "sokoban-search-automaton-artifact-v16",
        "status": "SOURCE_SEARCH_AUTOMATON_FROZEN",
        "target_authorized": True,
        "learned_policy": policies[0],
        "canonical_policy_sha256": required_policy_hash,
        "source_lineage": {
            "kind": "SIX_INDEPENDENT_PHASE1_GAME_FORMAL_CONSENSUS",
            "formal_manifest_sha256": manifest["manifest_sha256"],
            "formal_manifest_file_sha256": _sha256(args.manifest),
            "games": list(map(str, manifest["games"])),
        },
        "source_lineages": lineages,
        "transfer_contract": {
            "state": "ABSTRACT_EVENT_PLUS_ATTEMPT_LEDGER",
            "advance_only_after_observed_effect": True,
            "unknown_event": "ABSTAIN",
            "target_permission": (
                "TARGET_BINDS_EVENTS_CANDIDATES_AND_NATIVE_ACTIONS_FROM_ITS_"
                "OWN_NEURAL_GROUNDING;SOURCE_NATIVE_ACTIONS_AND_CANDIDATE_"
                "IDENTITIES_FORBIDDEN"
            ),
        },
        "claim_boundary": (
            "SIX_FORMALLY_QUALIFIED_PHASE1_SOURCE_GAMES;COMMON_SYMBOLIC_"
            "EVENT_ROUTING_ONLY;TARGET_NATIVE_NEURAL_GROUNDING_REQUIRED;NO_"
            "SOURCE_ACTION_TOKEN_CANDIDATE_ID_OR_ORDER_EXPORTED"
        ),
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "artifact": str(args.output.resolve()),
        "artifact_sha256": artifact["artifact_sha256"],
        "games": len(lineages),
        "canonical_policy_sha256": required_policy_hash,
        "target_authorized": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
