#!/usr/bin/env python3
"""Freeze a procedural-game value artifact with an ALFWorld-native grounder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.hierarchical_skill_transfer import (  # noqa: E402
    HierarchicalValueExample,
    fit_value_ensemble,
    marginal_value_control,
    phase_permuted_control,
    serialize_ensemble,
    shuffled_value_control,
)
from motif_transfer.procedural_workflow_game import (  # noqa: E402
    SourceGameCollection,
    collect_intervention_examples,
)


CONDITIONS = (
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
    "phase_permuted_source_plus_target",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect(config: dict[str, Any], *, evaluation: bool) -> SourceGameCollection:
    prefix = "evaluation" if evaluation else "train"
    workflow = config["workflow"]
    return collect_intervention_examples(
        surfaces=tuple(map(str, config[f"{prefix}_surfaces"])),
        domains_per_surface=int(config[f"{prefix}_domains_per_surface"]),
        states_per_domain=int(config[f"{prefix}_states_per_domain"]),
        replicates_per_action=int(config["replicates_per_action"]),
        seed=int(config[f"{prefix}_seed"]),
        minimum_budget=int(workflow["minimum_budget"]),
        maximum_budget=int(workflow["maximum_budget"]),
        completion_probability_range=workflow["completion_probability_range"],
        failure_cost_range=workflow["failure_cost_range"],
        progress_reward=float(workflow["progress_reward"]),
        invalid_option_cost=float(workflow["invalid_option_cost"]),
        retain_receipts=evaluation,
    )


def _mse(model, rows: Sequence[HierarchicalValueExample]) -> float:
    predicted, _ = model.predict([row.features for row in rows])
    expected = np.asarray([row.value for row in rows], dtype=np.float64)
    return float(np.mean((predicted - expected) ** 2))


def _receipt_hash(collection: SourceGameCollection) -> str:
    return stable_hash([asdict(row) for row in collection.receipts])


def build_candidate(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    source = config["source"]
    train = _collect(source, evaluation=False)
    evaluation = _collect(source, evaluation=True)
    rows = {
        "authentic_source_plus_target": train.examples,
        "shuffled_source_plus_target": shuffled_value_control(
            train.examples, seed=int(source["control_seed"]),
        ),
        "source_marginal_plus_target": marginal_value_control(train.examples),
        "phase_permuted_source_plus_target": phase_permuted_control(train.examples),
    }
    models = {
        name: fit_value_ensemble(
            values,
            seed=int(source["model_seed"]),
            ensemble_size=int(source["ensemble_size"]),
            alpha=float(source["ridge_alpha"]),
        )
        for name, values in rows.items()
    }
    mse = {name: _mse(model, evaluation.examples) for name, model in models.items()}
    authentic = mse["authentic_source_plus_target"]
    improvements = {
        name: (value - authentic) / max(value, 1e-12)
        for name, value in mse.items()
        if name != "authentic_source_plus_target"
    }
    minimum = float(source["minimum_relative_mse_improvement_over_each_control"])
    source_gate = all(value >= minimum for value in improvements.values())

    target_path = (REPO / config["target"]["base_artifact"]).resolve()
    target = json.loads(target_path.read_text(encoding="utf-8"))
    if target.get("status") != "QUALIFICATION_AUTHORIZED":
        raise RuntimeError("base target-native grounder is not frozen/authorized")
    if not target.get("target_grounder_gate", {}).get("passed"):
        raise RuntimeError("base target-native grounder gate did not pass")

    artifact: dict[str, Any] = {
        "schema_version": "procedural-game-to-alfworld-candidate-v1",
        "status": "QUALIFICATION_AUTHORIZED" if source_gate else "BLOCKED_AT_SOURCE_GATE",
        "claim_boundary": config["claim_boundary"],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "target_grounder": target["target_grounder"],
        "target_grounder_gate": target["target_grounder_gate"],
        "target_grounder_lineage": {
            "base_artifact": str(target_path),
            "base_artifact_sha256": _sha256(target_path),
            "qualification_or_heldout_used_for_retraining": False,
        },
        "source": {
            "kind": "matched-intervention-procedural-workflow-game-mdp-v1",
            "train_surfaces": source["train_surfaces"],
            "evaluation_surfaces": source["evaluation_surfaces"],
            "surface_overlap": sorted(
                set(source["train_surfaces"]) & set(source["evaluation_surfaces"])
            ),
            "train_domains": train.domains,
            "evaluation_domains": evaluation.domains,
            "train_states": train.states,
            "evaluation_states": evaluation.states,
            "train_examples": len(train.examples),
            "evaluation_examples": len(evaluation.examples),
            "evaluation_intervention_receipts": len(evaluation.receipts),
            "evaluation_receipts_sha256": _receipt_hash(evaluation),
            "matched_actions_per_state": 5,
            "replicates_per_action": int(source["replicates_per_action"]),
            "alpha_renamed_native_actions_per_domain": True,
            "raw_action_tokens_transferred": False,
            "target_action_tokens_used_for_source_training": False,
            "heldout_value_mse": mse,
            "relative_mse_improvement_over_control": improvements,
            "minimum_relative_mse_improvement_over_each_control": minimum,
            "gate_passed": source_gate,
            "models": {
                name: serialize_ensemble(model) for name, model in models.items()
            },
        },
        "qualification_or_heldout_used_for_training": False,
        "cross_domain_transfer_supported": False,
    }
    artifact["artifact_content_sha256"] = stable_hash(artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact = build_candidate(config, config_path=config_path)
    output = (REPO / config["target"]["artifact"]).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite candidate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": artifact["status"],
        "source_gate": artifact["source"]["gate_passed"],
        "source_mse": artifact["source"]["heldout_value_mse"],
        "relative_improvements": artifact["source"][
            "relative_mse_improvement_over_control"
        ],
        "evaluation_receipts": artifact["source"]["evaluation_intervention_receipts"],
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "QUALIFICATION_AUTHORIZED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
