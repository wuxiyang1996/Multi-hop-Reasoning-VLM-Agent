#!/usr/bin/env python3
"""Compile AGQA V2 receipts with strict native-to-sampled temporal boundaries.

This is a compatibility wrapper around the frozen V2 compiler.  It changes
only three target-native acquisition operations before V2 compilation:

* lower temporal bounds use the first sampled frame at or after the bound;
* upper temporal bounds use the last sampled frame at or before the bound;
* entity tracks without evidence inside the requested window fail closed.

The wrapper writes a new content-addressed backend identity.  It never reads
answers, official STSGs, functional programs, source controllers, or outcomes.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from motif_transfer.agqa_query_grounder_v2 import (
    QueryGroundingV2Receipt,
    query_grounding_v2_from_dict,
)
from motif_transfer.agqa_query_object_grounder import canonical_object_label
from motif_transfer.agqa_strict_temporal_projection import (
    action_localization_native_view,
    project_native_window_strict,
    rebind_nested_action_patients,
    recenter_action_anchor_events,
    recenter_degenerate_boundary_action_events,
    strict_track_for_label,
)
from motif_transfer.contracts import stable_hash
from scripts import compile_agqa_action_genome_query_grounder_v2 as frozen_v2


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_track(tracks, label: str, lower: int, upper: int):
    return strict_track_for_label(
        tracks, label, lower, upper, canonicalize=canonical_object_label,
    )


def _strict_query_window(plan: dict, sampled_native_indices: list[int]) -> tuple[int, int]:
    native = frozen_v2.native_temporal_window(plan)
    if native is None:
        return tuple(frozen_v2.dense_window(plan))
    return project_native_window_strict(sampled_native_indices, native[0], native[1])


def _strict_obligation_segment(
    obligation: dict, sampled_native_indices: list[int], frame_count: int,
) -> tuple[int, int, int]:
    view = list(action_localization_native_view(obligation))
    if not view:
        return frozen_v2._segment(int(obligation["argmax_window"]), frame_count)
    lower, upper = project_native_window_strict(
        sampled_native_indices, min(view), max(view),
    )
    return lower, upper, round((lower + upper) / 2) if lower <= upper else 0


def _path_argument(argv: list[str], name: str) -> tuple[int, Path]:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return index + 1, Path(argv[index + 1])
        if value.startswith(f"{name}="):
            return index, Path(value.split("=", 1)[1])
    raise ValueError(f"{name} is required")


def main() -> int:
    argv = list(sys.argv)
    center_all = "--action-event-center" in argv
    center_boundary = "--action-event-boundary-center" in argv
    if center_all and center_boundary:
        raise ValueError("choose one action-event temporal representation")
    argv = [value for value in argv if value not in {
        "--action-event-center", "--action-event-boundary-center",
    }]
    temporal_representation = (
        "RECEPTIVE_FIELD_CENTER_POINT" if center_all else
        "BOUNDARY_DEGENERACY_CENTER_POINT" if center_boundary else
        "RECEPTIVE_FIELD_INTERVAL"
    )
    output_index, output = _path_argument(argv, "--output")
    _, semantic_runtime_path = _path_argument(argv, "--semantic-runtime")
    _, query_plans_path = _path_argument(argv, "--query-plans")
    if output.exists():
        raise FileExistsError("strict-temporal V2 grounder artifact is immutable")
    implementation_sha = _file_hash(Path(__file__))
    with tempfile.TemporaryDirectory(prefix="agqa-qgv2-strict-") as directory:
        temporary = Path(directory) / "v2-parent.json"
        if argv[output_index].startswith("--output="):
            argv[output_index] = f"--output={temporary}"
        else:
            argv[output_index] = str(temporary)
        original_argv = sys.argv
        original_query = frozen_v2._query_sgdet_window
        original_segment = frozen_v2._obligation_sgdet_segment
        original_track = frozen_v2._track_for_label
        try:
            sys.argv = argv
            frozen_v2._query_sgdet_window = _strict_query_window
            frozen_v2._obligation_sgdet_segment = _strict_obligation_segment
            frozen_v2._track_for_label = _strict_track
            result = frozen_v2.main()
        finally:
            sys.argv = original_argv
            frozen_v2._query_sgdet_window = original_query
            frozen_v2._obligation_sgdet_segment = original_segment
            frozen_v2._track_for_label = original_track
        if result:
            return int(result)
        report = json.loads(temporary.read_text())

    parent_backend_sha = str(report["grounder_backend_sha256"])
    backend_sha = stable_hash({
        "protocol": "ACTION_GENOME_SGDET_SLOWFAST_QUERY_GROUNDER_V2_STRICT_TEMPORAL_V1",
        "parent_backend_sha256": parent_backend_sha,
        "implementation_sha256": implementation_sha,
        "native_window_projection": "LOWER_CEIL_UPPER_FLOOR_FAIL_CLOSED",
        "track_evidence_policy": "IN_WINDOW_REQUIRED",
        "action_localization_view_policy": "DENSE_TYPED_FIELD_THEN_NATIVE_FALLBACK",
        "action_event_temporal_representation": temporal_representation,
    })
    semantic_runtime = json.loads(semantic_runtime_path.read_text())
    semantics = {
        str(row["task_id"]): frozen_v2._semantic(row["receipt"])
        for row in semantic_runtime["rows"]
    }
    query_plans = json.loads(query_plans_path.read_text())
    plans = {str(row["task_id"]): row for row in query_plans["rows"]}
    rows = []
    rebound_action_events = 0
    recentered_action_events = 0
    for raw in report["rows"]:
        parent = query_grounding_v2_from_dict(raw["receipt"])
        events, rebound = rebind_nested_action_patients(
            parent.events, parent.tracks, semantics[parent.task_id],
        )
        rebound_action_events += rebound
        if center_all:
            events, recentered = recenter_action_anchor_events(
                events, plans[parent.task_id], parent.selected_frame_indices,
            )
        elif center_boundary:
            events, recentered = recenter_degenerate_boundary_action_events(
                events, plans[parent.task_id], parent.selected_frame_indices,
            )
        else:
            recentered = 0
        recentered_action_events += recentered
        receipt = QueryGroundingV2Receipt.create(
            task_id=parent.task_id,
            video_sha256=parent.video_sha256,
            semantic_slots_sha256=parent.semantic_slots_sha256,
            selected_frame_indices=parent.selected_frame_indices,
            selected_frame_sha256s=parent.selected_frame_sha256s,
            tracks=parent.tracks,
            events=events,
            candidates=parent.candidates,
            public_ontology_sha256=parent.public_ontology_sha256,
            grounder_backend_sha256=backend_sha,
            provider_calls=parent.provider_calls,
        )
        rows.append({
            **raw,
            "receipt": asdict(receipt),
            "temporal_coordinate_space": "SGDET_SAMPLED_POSITION",
            "temporal_projection": "LOWER_CEIL_UPPER_FLOOR_FAIL_CLOSED",
        })
    report.update({
        "schema_version": "agqa-action-genome-query-grounder-v2-strict-temporal-v1",
        "grounder_backend_sha256": backend_sha,
        "rows": rows,
        "strict_temporal_projection": True,
        "in_window_track_evidence_required": True,
        "nested_action_patient_coreference": True,
        "separate_action_localization_view": True,
        "action_event_temporal_representation": temporal_representation,
        "rebound_nested_action_events": rebound_action_events,
        "recentered_action_events": recentered_action_events,
        "parent_backend_sha256": parent_backend_sha,
        "strict_temporal_implementation_sha256": implementation_sha,
    })
    report.pop("report_sha256", None)
    report["report_sha256"] = stable_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "tasks": len(rows),
        "supported": sum(
            bool(row["receipt"]["candidates"])
            and row["receipt"]["candidates"][0]["status"] == "SUPPORTED"
            for row in rows
        ),
        "rebound_nested_action_events": rebound_action_events,
        "recentered_action_events": recentered_action_events,
        "strict_temporal_projection": True,
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
