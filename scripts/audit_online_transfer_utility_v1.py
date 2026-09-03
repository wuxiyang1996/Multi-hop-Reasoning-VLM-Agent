#!/usr/bin/env python3
"""Audit calibrated post-replication selection/abstention for transfer routes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.online_transfer_utility import (  # noqa: E402
    ApplicabilityReceipt,
    OnlineTransferUtilityGate,
    PairedOutcome,
)


VALID = ApplicabilityReceipt(True, True, True, True, True)


def read(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_receipt(
    *, wins: int, losses: int, ties: int,
    applicability: ApplicabilityReceipt = VALID,
    evidence_status: str,
) -> dict[str, Any]:
    gate = OnlineTransferUtilityGate()
    before = gate.decision(applicability)
    gate.update_many(
        [PairedOutcome(True, False)] * wins
        + [PairedOutcome(False, True)] * losses
        + [PairedOutcome(True, True)] * ties
    )
    after = gate.decision(applicability)
    return {
        "evidence_status": evidence_status,
        "paired_outcomes": {"wins": wins, "losses": losses, "ties": ties},
        "applicability": asdict(applicability),
        "cold_start_decision": before.decision,
        "post_replication": {
            "decision": after.decision,
            "reason": after.reason,
            "posterior_mean_win_probability_on_disagreement": (
                after.posterior_mean_win_probability
            ),
            "posterior_95pct_lower_win_probability_on_disagreement": (
                after.posterior_lower_win_probability
            ),
            "observed_disagreement_rate": after.observed_disagreement_rate,
        },
    }


def paired(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(row["wins"]), int(row["losses"]), int(row["ties"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-domain", type=Path, required=True)
    parser.add_argument("--tir-rotation", type=Path, required=True)
    parser.add_argument("--alfworld-multiplicity", type=Path, required=True)
    parser.add_argument("--discoveryworld-normal", type=Path, required=True)
    parser.add_argument("--clevrer", type=Path, required=True)
    parser.add_argument("--latent-ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    four = read(args.four_domain)
    rotation = read(args.tir_rotation)
    multiplicity = read(args.alfworld_multiplicity)
    normal = read(args.discoveryworld_normal)
    clevrer = read(args.clevrer)
    ontology = read(args.latent_ontology)

    routes: dict[str, Any] = {}
    for domain, row in four["domains"].items():
        wins, losses, ties = paired(row["paired"])
        routes[f"replication/{domain}"] = route_receipt(
            wins=wins, losses=losses, ties=ties,
            evidence_status=str(row["evidence_status"]),
        )
    formal_rotation = rotation["fresh_formal"]
    routes["extension/tir_rotation"] = route_receipt(
        wins=int(formal_rotation["wins"]),
        losses=int(formal_rotation["losses"]),
        ties=int(formal_rotation["ties"]),
        evidence_status=str(rotation["status"]),
    )
    wins, losses, ties = paired(multiplicity["paired_authentic_vs_target"])
    routes["extension/alfworld_multiplicity"] = route_receipt(
        wins=wins, losses=losses, ties=ties,
        evidence_status=str(multiplicity["status"]),
    )
    normal_pair = normal["paired_authentic_vs_target"]
    routes["extension/discoveryworld_normal"] = route_receipt(
        wins=int(normal_pair["wins"]), losses=int(normal_pair["losses"]),
        ties=int(normal_pair["ties"]), evidence_status=str(normal["status"]),
    )
    clevrer_pair = clevrer["paired_authentic"]["target_explicit_no_recovery"]
    routes["extension/clevrer_event_graph"] = route_receipt(
        wins=int(clevrer_pair["wins"]), losses=int(clevrer_pair["losses"]),
        ties=int(clevrer_pair["ties"]), evidence_status=str(clevrer["status"]),
    )
    invalid_ontology = ApplicabilityReceipt(True, True, True, False, True)
    routes["candidate/learned_latent_ontology"] = route_receipt(
        wins=0, losses=0, ties=0, applicability=invalid_ontology,
        evidence_status=str(ontology["status"]),
    )

    selected = sorted(
        name for name, row in routes.items()
        if row["post_replication"]["decision"] == "SELECT_SKILL"
    )
    abstained = sorted(set(routes) - set(selected))
    report = {
        "schema_version": "online-transfer-utility-audit-v1",
        "status": "CALIBRATED_ROUTE_SELECTION_AND_ABSTENTION_AUDITED",
        "estimand": (
            "Posterior probability that transfer wins rather than loses, conditional "
            "on a paired disagreement, for a future task on the exact registered route."
        ),
        "protocol": {
            "prior": "Beta(1,1)",
            "credible_lower_bound": 0.95,
            "selection_rule": (
                "SELECT_SKILL only when the one-sided 95% lower bound exceeds 0.5 "
                "and all exact structural applicability predicates pass."
            ),
            "ties_count_as_exposure_not_directional_evidence": True,
            "current_task_outcome_can_affect_its_own_decision": False,
        },
        "routes": routes,
        "selected_routes": selected,
        "abstained_routes": abstained,
        "integrity": {
            "input_file_sha256": {
                "four_domain": file_sha256(args.four_domain),
                "tir_rotation": file_sha256(args.tir_rotation),
                "alfworld_multiplicity": file_sha256(args.alfworld_multiplicity),
                "discoveryworld_normal": file_sha256(args.discoveryworld_normal),
                "clevrer": file_sha256(args.clevrer),
                "latent_ontology": file_sha256(args.latent_ontology),
            },
            "gate_implementation_sha256": file_sha256(
                REPO / "src/motif_transfer/online_transfer_utility.py"
            ),
        },
        "limitations": [
            "Calibration is route-level and post-replication; it is not a learned state-level applicability predictor.",
            "Exact structural applicability predicates are designed and fail closed; arbitrary-domain analogy detection is not claimed.",
            "The selector governs future use and never retroactively selects on the evaluated task's outcome.",
        ],
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "selected_routes": selected,
        "abstained_routes": abstained, "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
