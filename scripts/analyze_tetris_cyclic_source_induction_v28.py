#!/usr/bin/env python3
"""Analyze fresh source-only cyclic induction and its frozen TIR bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.cyclic_identity_induction import (  # noqa: E402
    evaluate_cyclic_program,
    induce_cyclic_identity_program,
    permute_recovery_effect_bindings,
    permute_terminal_labels,
    subset_cyclic_dataset,
    validate_cyclic_dataset,
)


DEFAULT_CONFIG = REPO / "configs/tetris_cyclic_source_induction_v28.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _ordered_ids(
    dataset: Mapping[str, Any], namespace: str,
) -> list[str]:
    return sorted(
        (str(row["episode_id"]) for row in dataset["episodes"]),
        key=lambda episode_id: stable_hash({
            "namespace": namespace, "episode_id": episode_id,
        }),
    )


def _curve(
    development: Mapping[str, Any], qualification: Mapping[str, Any],
    *, namespace: str, minimum_episodes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    order = _ordered_ids(development, namespace)
    rows = [{
        "complete_source_intervention_episodes": 0,
        "status": "ABSTAIN_NO_SOURCE_INTERVENTION_FORK",
        "qualification_all_forks_classified": False,
    }]
    first = None
    for budget in range(1, len(order) + 1):
        selected = subset_cyclic_dataset(development, order[:budget])
        program = induce_cyclic_identity_program(
            selected, minimum_episodes=minimum_episodes,
        )
        evaluation = evaluate_cyclic_program(program, qualification)
        row = {
            "complete_source_intervention_episodes": budget,
            "status": str(program["status"]),
            "selected_relation": program["selected_relation"],
            "program_sha256": str(program["program_sha256"]),
            "retained_candidate_forks": int(
                program["diagnostics"]["candidate_forks"]
            ),
            "retained_all_fork_primitive_transitions": int(
                program["diagnostics"]["primitive_transitions"]
            ),
            "non_self_inverse_successes": int(
                program["diagnostics"]["non_self_inverse_successes"]
            ),
            "qualification": evaluation,
            "qualification_all_forks_classified": bool(
                evaluation["all_forks_classified"]
            ),
        }
        rows.append(row)
        if (
            program["status"] == "SOURCE_CYCLIC_IDENTITY_PROGRAM_INDUCED"
            and evaluation["all_forks_classified"]
            and first is None
        ):
            first = row | {"program": program}
    return rows, first


def _controls(
    development: Mapping[str, Any], *, minimum_episodes: int,
) -> dict[str, Any]:
    label_control = induce_cyclic_identity_program(
        permute_terminal_labels(development),
        minimum_episodes=minimum_episodes,
    )
    binding_control = induce_cyclic_identity_program(
        permute_recovery_effect_bindings(development),
        minimum_episodes=minimum_episodes,
    )
    return {
        "terminal_label_permuted_status": str(label_control["status"]),
        "terminal_label_permuted_relation": label_control[
            "selected_relation"
        ],
        "recovery_binding_permuted_status": str(binding_control["status"]),
        "recovery_binding_permuted_relation": binding_control[
            "selected_relation"
        ],
        "both_controls_abstain": (
            str(label_control["status"]).startswith("ABSTAIN")
            and str(binding_control["status"]).startswith("ABSTAIN")
        ),
    }


def _target_bridge(
    summary: Mapping[str, Any], old_source: Mapping[str, Any],
    program: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _self_hash(summary, "summary_sha256")
    _self_hash(old_source, "artifact_sha256")
    formal = summary["fresh_formal"]
    selected = None if program is None else program.get("selected_relation")
    semantic_match = bool(
        selected == "COMPOSE_PROBE_RECOVERY_TO_IDENTITY"
        and "compose inverse element" in str(
            old_source["transferred_program"]["recovery"]
        )
    )
    return {
        "new_induced_relation": selected,
        "old_compiler_relation": str(
            old_source["transferred_program"]["recovery"]
        ),
        "execution_semantic_match": semantic_match,
        "existing_fresh_target_utility": {
            "tasks": int(formal["tasks"]),
            "source_semantics_correct": int(formal["authentic_correct"]),
            "target_written_isomorphic_correct": int(
                formal["target_written_isomorphic_correct"]
            ),
            "raw_correct": int(formal["raw_correct"]),
            "destructive_control_correct": dict(
                formal["destructive_control_correct"]
            ),
            "wins": int(formal["wins"]),
            "losses": int(formal["losses"]),
            "exact_two_sided_p": float(formal["exact_two_sided_p"]),
        },
        "target_written_isomorphic_equivalence": (
            int(formal["authentic_correct"])
            == int(formal["target_written_isomorphic_correct"])
        ),
        "prospective_boundary": (
            "The TIR execution reserve predates this inducer and used an "
            "extensionally identical compiler artifact. It is utility context, "
            "not a newly prospective target result for the V28 artifact."
        ),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    _self_hash(config, "config_sha256")
    if config.get("status") != "FROZEN_BEFORE_FRESH_SOURCE_COLLECTION":
        raise ValueError("Tetris cyclic protocol is not frozen")
    for hash_field, path_field in config["dependency_fields"].items():
        path = Path(str(config[path_field]))
        if not path.is_absolute():
            path = REPO / path
        if _sha(path) != config[hash_field]:
            raise ValueError(f"dependency changed: {path}")

    development = _read(REPO / str(config["outputs"]["development"]))
    qualification = _read(REPO / str(config["outputs"]["qualification"]))
    validate_cyclic_dataset(development)
    validate_cyclic_dataset(qualification)
    if development["config_sha256"] != config["config_sha256"]:
        raise ValueError("development source dataset is from another protocol")
    if qualification["config_sha256"] != config["config_sha256"]:
        raise ValueError("qualification source dataset is from another protocol")
    curve, first = _curve(
        development, qualification,
        namespace=str(config["acquisition_order_namespace"]),
        minimum_episodes=int(config["minimum_induction_episodes"]),
    )
    controls = _controls(
        development,
        minimum_episodes=int(config["minimum_induction_episodes"]),
    )
    reserve_path = REPO / str(config["outputs"]["reserve"])
    reserve = None
    if reserve_path.exists():
        reserve_dataset = _read(reserve_path)
        validate_cyclic_dataset(reserve_dataset)
        if reserve_dataset["config_sha256"] != config["config_sha256"]:
            raise ValueError("reserve source dataset is from another protocol")
        reserve = evaluate_cyclic_program(
            (first or {}).get("program", {}), reserve_dataset,
        )
    summary = _read(REPO / str(config["target_utility_summary"]))
    old_source = _read(REPO / str(config["old_source_artifact"]))
    bridge = _target_bridge(
        summary, old_source, (first or {}).get("program"),
    )
    qualification_gates = {
        "source_k0_abstains": curve[0]["status"].startswith("ABSTAIN"),
        "fresh_development_has_minimum_episodes": (
            len(development["episodes"])
            >= int(config["minimum_development_episodes"])
        ),
        "fresh_qualification_has_minimum_episodes": (
            len(qualification["episodes"])
            >= int(config["minimum_qualification_episodes"])
        ),
        "program_induced_within_budget": first is not None,
        "qualification_all_forks_classified": bool(
            first and first["qualification"]["all_forks_classified"]
        ),
        "qualification_zero_false_positive_support": bool(
            first and first["qualification"]["false_positive_support"] == 0
        ),
        "effect_and_label_controls_abstain": bool(
            controls["both_controls_abstain"]
        ),
        "no_raw_source_action_export": (
            development["raw_source_action_tokens_exported"] is False
            and qualification["raw_source_action_tokens_exported"] is False
        ),
        "no_target_data_read_for_source_induction": (
            development["target_data_read"] is False
            and qualification["target_data_read"] is False
        ),
        "matches_frozen_target_execution_semantics": bool(
            bridge["execution_semantic_match"]
        ),
        "target_written_isomorphic_equivalence_retained": bool(
            bridge["target_written_isomorphic_equivalence"]
        ),
    }
    qualification_passed = all(qualification_gates.values())
    reserve_gates = {
        "reserve_collected_only_after_qualification": bool(
            reserve is not None and qualification_passed
        ),
        "reserve_all_forks_classified": bool(
            reserve and reserve["all_forks_classified"]
        ),
        "reserve_zero_false_positive_support": bool(
            reserve and reserve["false_positive_support"] == 0
        ),
    }
    full_passed = qualification_passed and all(reserve_gates.values())
    body = {
        "schema_version": "tetris-cyclic-source-induction-v28-report",
        "status": (
            "THIRD_PROGRAM_FAMILY_SOURCE_RESERVE_VALIDATED"
            if full_passed
            else (
                "SOURCE_CYCLIC_QUALIFICATION_PASSED_RESERVE_UNOPENED"
                if qualification_passed and reserve is None
                else "SOURCE_CYCLIC_INDUCTION_GATE_FAILED"
            )
        ),
        "config_sha256": str(config["config_sha256"]),
        "program_family": "ALGEBRAIC_CYCLIC_IDENTITY_RECOVERY",
        "claim_boundary": str(config["claim_boundary"]),
        "development": {
            "episodes": len(development["episodes"]),
            "candidate_forks": sum(
                len(row["candidates"]) for row in development["episodes"]
            ),
            "all_fork_primitive_transitions": sum(
                len(candidate["primitive_transitions"])
                for row in development["episodes"]
                for candidate in row["candidates"]
            ),
            "acquisition_curve": curve,
            "first_qualified": first,
        },
        "qualification": {
            "episodes": len(qualification["episodes"]),
            "evaluation": None if first is None else first["qualification"],
        },
        "controls": controls,
        "reserve": reserve,
        "target_bridge": bridge,
        "qualification_gates": qualification_gates,
        "reserve_gates": reserve_gates,
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    report = run(config_path)
    config = _read(config_path)
    output = REPO / str(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "first_qualified_budget": (
            report["development"]["first_qualified"] or {}
        ).get("complete_source_intervention_episodes"),
        "qualification_gates": report["qualification_gates"],
        "reserve_gates": report["reserve_gates"],
        "report_sha256": report["report_sha256"],
        "output": str(output),
    }, indent=2))
    return 0 if report["status"] != "SOURCE_CYCLIC_INDUCTION_GATE_FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
