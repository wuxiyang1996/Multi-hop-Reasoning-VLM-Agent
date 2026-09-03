#!/usr/bin/env python3
"""Validate and smoke-dispatch the frozen four-domain skill library."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.neurosymbolic_skill_library import (  # noqa: E402
    DispatchVerdict,
    EvidenceTier,
    FrozenNeurosymbolicSkillLibrary,
    TargetRequest,
    validate_dispatch_receipt,
)


def _jsonable_receipt(receipt) -> dict:
    value = asdict(receipt)
    value["verdict"] = receipt.verdict.value
    value["evidence_tier"] = (
        receipt.evidence_tier.value if receipt.evidence_tier else None
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry_payload = json.loads(args.registry.read_text(encoding="utf-8"))
    library = FrozenNeurosymbolicSkillLibrary.load(args.registry, repo=REPO)

    positive = []
    for row in registry_payload.get("verification_requests") or ():
        request = TargetRequest.create(
            row["domain"], row["interface"], row["capabilities"],
        )
        receipt = library.dispatch(
            request, minimum_evidence=EvidenceTier(str(row["minimum_evidence"])),
        )
        validate_dispatch_receipt(receipt)
        expected = str(row["expected_skill_id"])
        if receipt.verdict != DispatchVerdict.SELECT_SKILL or receipt.skill_id != expected:
            raise SystemExit(
                f"positive dispatch failed for {request.domain}: {receipt.reason}"
            )
        positive.append(_jsonable_receipt(receipt))

    negative = []
    for row in registry_payload.get("negative_verification_requests") or ():
        request = TargetRequest.create(
            row["domain"], row["interface"], row.get("capabilities") or (),
        )
        receipt = library.dispatch(
            request,
            minimum_evidence=EvidenceTier(str(
                row.get("minimum_evidence") or "MECHANISM"
            )),
        )
        validate_dispatch_receipt(receipt)
        if receipt.verdict != DispatchVerdict.ABSTAIN:
            raise SystemExit(f"negative dispatch did not abstain: {request}")
        negative.append(_jsonable_receipt(receipt))

    domains = [row["request"]["domain"] for row in positive]
    required_domains = ["webshop", "alfworld", "discoveryworld", "tir"]
    all_four = sorted(domains) == sorted(required_domains)
    payload = {
        "schema_version": "four-domain-neurosymbolic-library-audit-v1",
        "status": (
            "FOUR_DOMAIN_FRESH_FORMAL_SKILL_DISPATCH_VALIDATED"
            if all_four and all(
                row["evidence_tier"] == EvidenceTier.FRESH_FORMAL.value
                for row in positive
            )
            else "FOUR_DOMAIN_SKILL_DISPATCH_INCOMPLETE"
        ),
        "claim_boundary": (
            "The harness validates evidence-bound selection and target-native "
            "authority for four exact interfaces. It does not claim that one "
            "universal skill transfers to every domain or unsupported interface."
        ),
        "registry_file_sha256": library.registry_sha256,
        "domains": domains,
        "all_four_exact_routes_selected": all_four,
        "all_selected_routes_target_native_action_authority": all(
            row["action_authority"] == "TARGET_NATIVE_GROUNDER_AND_EXECUTOR"
            for row in positive
        ),
        "unsupported_routes_abstain": all(
            row["verdict"] == DispatchVerdict.ABSTAIN.value for row in negative
        ),
        "dispatch_receipts": positive,
        "negative_dispatch_receipts": negative,
    }
    payload["report_sha256"] = stable_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "domains": domains,
        "report_sha256": payload["report_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
