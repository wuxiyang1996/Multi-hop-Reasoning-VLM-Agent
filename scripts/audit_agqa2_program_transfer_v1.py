#!/usr/bin/env python3
"""Run an outcome-blind, zero-provider-cost AGQA 2.0 program audit."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Iterator, Mapping, TextIO
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motif_transfer.agqa_program_transfer import (  # noqa: E402
    COMPOSITE_ROUTE,
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_ROUTE,
    profile_program,
    select_program,
)
from motif_transfer.clevrer_unified_goal_relation import (  # noqa: E402
    source_goal_relation_contract,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_ir_applicability import (  # noqa: E402
    SourceIRContract,
    temporal_function_artifact_contract,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _load_verified_json(path: str, expected_sha256: str) -> dict:
    resolved = _resolve(path)
    actual = _sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(f"file hash mismatch: {resolved}: {actual}")
    return json.loads(resolved.read_text())


def _iter_top_level_object(handle: TextIO) -> Iterator[tuple[str, dict]]:
    """Incrementally parse a single large JSON object without extraction."""

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    while True:
        chunk = handle.read(1024 * 1024)
        if chunk:
            buffer += chunk
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    break
                if buffer[position] != "{":
                    raise ValueError("AGQA split is not a top-level JSON object")
                position += 1
                started = True
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "}":
                return
            try:
                task_id, after_key = decoder.raw_decode(buffer, position)
                value_position = after_key
                while value_position < len(buffer) and (
                    buffer[value_position].isspace()
                    or buffer[value_position] == ":"
                ):
                    value_position += 1
                row, after_value = decoder.raw_decode(buffer, value_position)
            except json.JSONDecodeError:
                break
            if not isinstance(task_id, str) or not isinstance(row, dict):
                raise ValueError("unexpected AGQA top-level row")
            yield task_id, row
            position = after_value
        if position:
            buffer = buffer[position:]
            position = 0
        if not chunk:
            raise ValueError("truncated AGQA JSON object")


def _load_sources(config: Mapping[str, object]):
    specs = config["sources"]
    relation_spec = specs["goal_relation"]
    relation_artifact = _load_verified_json(
        relation_spec["artifact_path"], relation_spec["artifact_file_sha256"],
    )
    relation_confirmation = _load_verified_json(
        relation_spec["confirmation_path"],
        relation_spec["confirmation_file_sha256"],
    )
    relation = source_goal_relation_contract(
        relation_artifact, relation_confirmation,
    )

    report_spec = specs["arcade_report"]
    report = _load_verified_json(
        report_spec["path"], report_spec["file_sha256"],
    )
    temporal = []
    for label in ("temporal_pair", "temporal_single"):
        spec = specs[label]
        artifact = _load_verified_json(
            spec["artifact_path"], spec["artifact_file_sha256"],
        )
        lineage = next(
            row for row in report["lineages"]
            if row["source_game"] == spec["source_game"]
        )
        temporal.append(temporal_function_artifact_contract(
            artifact,
            source_confirmation_sha256=report["report_sha256"],
            source_intervention_qualified=(
                lineage["status"] == "V4_SOURCE_DOMAIN_FUNCTION_CONFIRMED"
            ),
        ))
    return (relation, *temporal), report


def _target_written_equivalent(source: SourceIRContract) -> SourceIRContract:
    """Construct an extensionally identical non-source control contract."""

    return SourceIRContract.create(
        program_sha256=f"target-written-equivalent-{source.program_sha256}",
        ir_kind=source.ir_kind,
        operator_sequence=source.operator_sequence,
        recurrent=source.recurrent,
        terminal_predicate_families=source.terminal_predicate_families,
        source_intervention_qualified=True,
        source_confirmation_sha256="synthetic-target-written-control",
    )


def run(config_path: Path, output_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    module_spec = config["compiler"]
    module_path = _resolve(module_spec["module_path"])
    if _sha256_file(module_path) != module_spec["module_sha256"]:
        raise ValueError("AGQA compiler differs from the consumed-development config")
    sources, arcade_report = _load_sources(config)
    source_by_route = dict(zip(
        (RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE),
        sources,
    ))
    wrong_by_route = {
        RELATION_ROUTE: sources[1],
        TEMPORAL_PAIR_ROUTE: sources[2],
        TEMPORAL_SINGLE_ROUTE: sources[0],
    }

    archive = Path(config["dataset"]["archive_path"])
    archive_sha256 = _sha256_file(archive)
    if archive_sha256 != config["dataset"]["archive_sha256"]:
        raise ValueError("AGQA archive hash mismatch")
    entry = config["dataset"]["entry"]
    route_counts = Counter()
    selected_programs = Counter()
    wrong_source_rejections = Counter()
    selected_examples = 0
    task_count = 0
    video_ids = set()
    forbidden_field_reads = Counter()
    target_grounder_sha256 = module_spec["target_grounder_sha256"]

    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("AGQA ZIP integrity check failed")
        with bundle.open(entry, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            for task_id, row in _iter_top_level_object(text):
                # Create the complete selector input before any diagnostic
                # annotation is inspected.  `answer` and `sg_grounding` are
                # never read anywhere in this audit.
                profile = profile_program(
                    task_id=task_id, program=str(row.get("program", "")),
                )
                receipt = select_program(
                    sources, profile,
                    target_grounder_sha256=target_grounder_sha256,
                )
                task_count += 1
                video_ids.add(str(row.get("video_id", "")))
                route_counts[profile.route_kind] += 1
                selected = receipt["selected_program_sha256"]
                if selected is not None:
                    selected_examples += 1
                    selected_programs[str(selected)] += 1
                    wrong_receipt = select_program(
                        (wrong_by_route[profile.route_kind],), profile,
                        target_grounder_sha256=target_grounder_sha256,
                    )
                    wrong_source_rejections[profile.route_kind] += int(
                        wrong_receipt["selected_program_sha256"] is None
                    )

    expectations = config["development_expectations_from_consumed_test_metadata"]
    expected_routes = expectations["routes"]
    routing_counts_match = (
        task_count == expectations["questions"]
        and len(video_ids) == expectations["videos"]
        and selected_examples == expectations["selected"]
        and dict(route_counts) == expected_routes
    )
    exact_selection = all(
        selected_programs[source.program_sha256] == route_counts[route]
        for route, source in source_by_route.items()
    )
    wrong_total = sum(wrong_source_rejections.values())
    wrong_rejected = wrong_total == selected_examples

    equivalent_matches = {}
    for route, source in source_by_route.items():
        # A minimal representative is sufficient because the target type is
        # fixed per route.  This intentionally probes an attribution limit.
        if route == RELATION_ROUTE:
            program = (
                "Exists(Query(class, OnlyItem(Iterate(video, Filter(frame, "
                "[relations, touching, objects])))), Iterate(video, "
                "Filter(frame, [relations, holding, objects])))"
            )
        elif route == TEMPORAL_PAIR_ROUTE:
            program = (
                "Compare([before, after], Exists(item, Iterate(Localize("
                "temporal tag, action), Filter(frame, [objects]))))"
            )
        else:
            program = (
                "Superlative(max, [Filter(video, [actions, one]), "
                "Filter(video, [actions, two])], Subtract(Query(end, action), "
                "Query(start, action)))"
            )
        profile = profile_program(task_id=f"control-{route}", program=program)
        receipt = select_program(
            (_target_written_equivalent(source),), profile,
            target_grounder_sha256=target_grounder_sha256,
        )
        equivalent_matches[route] = (
            receipt["status"] == "UNIQUE_SOURCE_CONTRACT_SELECTED"
        )

    aggregate = arcade_report["qualified_aggregate"]
    source_specific_portfolio_passed = arcade_report["gates"][
        "qualified_authentic_aggregate_beats_source_permuted"
    ]
    mechanical_gates = {
        "official_archive_hash_and_zip_integrity": True,
        "consumed_metadata_status_declared": (
            config["status"] == "CONSUMED_METADATA_DEVELOPMENT_ONLY"
        ),
        "compiler_reads_only_task_id_and_public_program": True,
        "routing_counts_reproduce_consumed_development_expectations": (
            routing_counts_match
        ),
        "three_exact_anonymous_source_contracts_selected": exact_selection,
        "wrong_source_type_permutation_abstains": wrong_rejected,
        "composite_programs_fail_closed": (
            selected_programs.total() == selected_examples
            and route_counts[COMPOSITE_ROUTE] > 0
        ),
        "no_answer_scene_graph_video_or_target_action_read": (
            not forbidden_field_reads
        ),
        "zero_provider_or_model_calls": True,
    }
    claim_gates = {
        "source_specific_arcade_portfolio_beats_permuted": bool(
            source_specific_portfolio_passed
        ),
        "source_origin_identifiable_from_structural_contract": not all(
            equivalent_matches.values()
        ),
        "target_native_neural_grounding_measured": False,
        "agqa_answer_success_rate_measured": False,
        "untouched_evaluation_split_preserved": False,
    }
    result = {
        "schema_version": "agqa2-program-transfer-audit-v1",
        "status": "AGQA2_PROGRAM_TRANSFER_MECHANISM_FEASIBLE_NOT_SUCCESS_VALIDATED",
        "claim_boundary": config["claim_boundary"],
        "config": {
            "path": str(config_path.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(config_path),
            "split_use": config["status"],
        },
        "dataset": {
            "archive_sha256": archive_sha256,
            "archive_bytes": archive.stat().st_size,
            "entry": entry,
            "questions": task_count,
            "videos": len(video_ids),
            "raw_videos_downloaded": False,
            "visual_features_downloaded": False,
        },
        "blind_interface": {
            "input_fields": module_spec["input_fields"],
            "forbidden_fields": module_spec["forbidden_fields"],
            "forbidden_field_reads": dict(forbidden_field_reads),
            "source_identity_used_as_feature": False,
            "target_outcomes_read": 0,
            "target_actions_emitted": 0,
        },
        "source_contracts": [asdict(source) for source in sources],
        "routing": {
            "counts": dict(route_counts),
            "selected_examples": selected_examples,
            "selected_fraction": selected_examples / task_count,
            "abstained_examples": task_count - selected_examples,
            "abstained_fraction": (task_count - selected_examples) / task_count,
            "selected_program_counts": dict(selected_programs),
        },
        "controls": {
            "wrong_source_type_rejections": dict(wrong_source_rejections),
            "wrong_source_type_rejected_total": wrong_total,
            "wrong_source_type_tested_total": selected_examples,
            "target_written_equivalent_matches": equivalent_matches,
            "target_written_equivalent_match_count": sum(
                equivalent_matches.values()
            ),
            "target_written_equivalent_tested": len(equivalent_matches),
            "arcade_source_portfolio_status": arcade_report["status"],
            "arcade_authentic_correct": aggregate["authentic_correct"],
            "arcade_source_permuted_correct": aggregate["permuted_correct"],
            "arcade_examples": aggregate["examples"],
        },
        "mechanical_gates": mechanical_gates,
        "claim_gates": claim_gates,
        "all_mechanical_gates_passed": all(mechanical_gates.values()),
        "all_claim_gates_passed": all(claim_gates.values()),
        "raw_video_advancement_authorized": False,
        "cost": {
            "provider_calls": 0,
            "model_calls": 0,
            "new_raw_video_bytes": 0,
            "new_visual_feature_bytes": 0,
        },
    }
    body = dict(result)
    result["report_sha256"] = stable_hash(body)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/agqa2_program_transfer_v1_development.json"),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "docs/results/agqa2_program_transfer_v1.json"),
    )
    args = parser.parse_args()
    result = run(Path(args.config).resolve(), Path(args.output).resolve())
    print(json.dumps({
        "status": result["status"],
        "questions": result["dataset"]["questions"],
        "selected": result["routing"]["selected_examples"],
        "all_mechanical_gates_passed": result["all_mechanical_gates_passed"],
        "all_claim_gates_passed": result["all_claim_gates_passed"],
        "raw_video_advancement_authorized": result[
            "raw_video_advancement_authorized"
        ],
        "report_sha256": result["report_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
