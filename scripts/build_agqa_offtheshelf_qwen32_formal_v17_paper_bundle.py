#!/usr/bin/env python3
"""Build and audit the paper-facing AGQA Qwen32 compositional V17 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.agqa_layer_b_harness import ARMS
from motif_transfer.contracts import stable_hash


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_label(path: Path, roots: list[Path]) -> str:
    """Return a stable repo-relative label for artifacts stored outside code root."""
    resolved = path.resolve()
    for root in roots:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return str(path)


def _stable(value: dict, field: str) -> None:
    claimed = value.get(field)
    body = {key: item for key, item in value.items() if key != field}
    if not isinstance(claimed, str) or stable_hash(body) != claimed:
        raise ValueError(f"invalid embedded {field}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-evaluation", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--semantic-runtime", type=Path, required=True)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--grounding-view", type=Path, action="append", required=True)
    parser.add_argument("--routed-grounding", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--preoutcome", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--anonymous-controller", type=Path, required=True)
    parser.add_argument("--development-evaluation", type=Path, required=True)
    parser.add_argument("--slowfast-development-evaluation", type=Path, required=True)
    parser.add_argument("--artifact-label-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.verify_existing:
        raise FileExistsError("AGQA V17 paper bundle is immutable")
    if len(args.grounding_view) != 2:
        raise ValueError("paper bundle requires the ordered 48/96 grounding pair")

    formal = _load(args.formal_evaluation)
    cohort = _load(args.cohort)
    manifest = _load(args.manifest)
    protocol = _load(args.protocol)
    runtime = _load(args.semantic_runtime)
    download = _load(args.download_receipt)
    views = [_load(path) for path in args.grounding_view]
    routed = _load(args.routed_grounding)
    fallback = _load(args.fallback)
    pre = _load(args.preoutcome)
    source = _load(args.source_capabilities)
    controller = _load(args.anonymous_controller)
    development = _load(args.development_evaluation)
    slowfast = _load(args.slowfast_development_evaluation)
    for value, field in (
        (formal, "report_sha256"), (pre, "receipt_sha256"),
        (runtime, "runtime_sha256"), (routed, "report_sha256"),
        (fallback, "report_sha256"), (source, "artifact_sha256"),
        (controller, "artifact_sha256"), (development, "report_sha256"),
        (slowfast, "report_sha256"),
    ):
        _stable(value, field)
    for view in views:
        _stable(view, "report_sha256")

    if formal.get("status") != "AGQA_QWEN32_COMPOSITIONAL_FRESH_FORMAL_TRANSFER_VALIDATED":
        raise ValueError("formal transfer did not validate")
    if not all(formal.get("gates", {}).values()):
        raise ValueError("formal transfer gates are incomplete")
    if protocol.get("status") != "QWEN32_COMPOSITIONAL_FORMAL_PROTOCOL_FROZEN_AFTER_DEVELOPMENT":
        raise ValueError("formal protocol is invalid")
    if manifest.get("status") != "AGQA_QWEN32_COMPOSITIONAL_FRESH_FORMAL_V17_FROZEN":
        raise ValueError("formal cohort manifest is invalid")
    if manifest.get("protocol_file_sha256") != _sha256(args.protocol):
        raise ValueError("cohort does not bind protocol")
    if manifest.get("public_cohort_sha256") != cohort.get("cohort_sha256"):
        raise ValueError("manifest/cohort mismatch")
    if download.get("status") != "COMPLETE":
        raise ValueError("formal video download is incomplete")
    if download.get("selection_manifest_sha256") != manifest.get("download_selection_sha256"):
        raise ValueError("download receipt does not bind frozen selection")
    if formal.get("formal_protocol_file_sha256") != _sha256(args.protocol):
        raise ValueError("formal evaluation does not bind protocol")
    if formal.get("formal_manifest_file_sha256") != _sha256(args.manifest):
        raise ValueError("formal evaluation does not bind manifest")
    if formal.get("preoutcome_file_sha256") != _sha256(args.preoutcome):
        raise ValueError("formal evaluation does not bind preoutcome")
    if pre.get("formal_protocol_file_sha256") != _sha256(args.protocol):
        raise ValueError("preoutcome does not bind protocol")
    if pre.get("formal_manifest_file_sha256") != _sha256(args.manifest):
        raise ValueError("preoutcome does not bind manifest")
    if pre.get("source_capability_sha256") != source.get("artifact_sha256"):
        raise ValueError("formal source capability mismatch")
    if pre.get("anonymous_controller_sha256") != controller.get("artifact_sha256"):
        raise ValueError("formal anonymous controller mismatch")
    if [view.get("frame_budget") for view in views] != protocol["runtime"]["frame_views"]:
        raise ValueError("formal frame views differ from protocol")
    if [view.get("grounder_backend_sha256") for view in views] != protocol["runtime"]["qwen32_grounder_backend_sha256s"]:
        raise ValueError("formal grounding backend differs from development-qualified backend")
    if routed.get("candidate_report_sha256s") != [view["report_sha256"] for view in views]:
        raise ValueError("routed grounding does not bind ordered views")
    if fallback.get("grounding_report_sha256") != routed.get("report_sha256"):
        raise ValueError("fallback does not bind shared routed grounding")
    cohort_sha = cohort["cohort_sha256"]
    if len({cohort_sha, runtime.get("cohort_sha256"), routed.get("cohort_sha256"),
            fallback.get("cohort_sha256"), pre.get("cohort_sha256"),
            *(view.get("cohort_sha256") for view in views)}) != 1:
        raise ValueError("formal runtime artifacts do not share a cohort")
    if development.get("status") != "V18_COMPOSITIONAL_DEVELOPMENT_TRANSFER_SIGNAL_PASSED":
        raise ValueError("Qwen32 development qualification did not pass")
    if slowfast.get("status") != "V15_COMPOSITIONAL_DEVELOPMENT_TRANSFER_SIGNAL_FAILED":
        raise ValueError("SlowFast diagnostic status changed")

    n = len(formal["rows"])
    if n != int(manifest["rows"]) or n != 256:
        raise ValueError("formal cohort size differs from frozen protocol")
    main_table = [{
        "arm": arm,
        "correct": int(formal["summaries"][arm]["correct"]),
        "total": int(formal["summaries"][arm]["total"]),
        "accuracy": float(formal["summaries"][arm]["accuracy"]),
        "symbolic_commits": int(formal["summaries"][arm]["symbolic_commits"]),
    } for arm in ARMS]
    grounding = {
        "frame_views": [view["frame_budget"] for view in views],
        "total_frame_presentations_per_task": sum(int(view["frame_budget"]) for view in views),
        "provider_calls": sum(int(view.get("provider_calls", 0)) for view in views),
        "provider_receipt_cost_usd": sum(float(view.get("reported_receipt_provider_cost_usd", 0)) for view in views),
        "incremental_provider_cost_usd": sum(float(view.get("incremental_provider_cost_usd", 0)) for view in views),
        "provider_errors_fail_closed": sum(
            row.get("provider_error") is not None for view in views for row in view["rows"]
        ),
        "empty_event_graphs": sum(
            not row["grounding_receipt"]["events"] for view in views for row in view["rows"]
        ),
        "events": sum(
            len(row["grounding_receipt"]["events"]) for view in views for row in view["rows"]
        ),
        "routed_candidate_counts": routed["route_counts"],
        "downloaded_videos": len(download["videos"]),
        "downloaded_bytes": sum(int(row["file_size"]) for row in download["videos"]),
    }
    body = {
        "schema_version": "agqa-qwen32-compositional-formal-paper-bundle-v17",
        "status": "AGQA_QWEN32_COMPOSITIONAL_V17_PAPER_BUNDLE_VALIDATED",
        "claim": (
            "A source-only anonymous controller induced from game interventions significantly "
            "improves final AGQA duration-compositional QA on a one-shot fresh video/task-disjoint "
            "balanced-train reserve under shared answer-blind raw-video grounding."
        ),
        "claim_scope": formal["claim_scope"],
        "tasks": n,
        "videos": manifest["videos"],
        "semantic_roots": manifest["semantic_roots"],
        "main_table": main_table,
        "paired_ablations": formal["comparisons"],
        "root_breakdown": formal["root_breakdown"],
        "failure_taxonomy": formal["failure_taxonomy"],
        "formal_gates": formal["gates"],
        "secondary_target": formal["secondary_target"],
        "preoutcome_metrics": pre["metrics"],
        "grounding": grounding,
        "development_diagnostics": {
            "slowfast_v15_status": slowfast["status"],
            "slowfast_v15_source_accuracy": slowfast["summaries"]["source_induced"]["accuracy"],
            "slowfast_v15_source_vs_neural": slowfast["comparisons"]["neural_only"],
            "qwen32_v16_status": development["status"],
            "qwen32_v16_source_accuracy": development["summaries"]["source_induced"]["accuracy"],
            "qwen32_v16_source_vs_neural": development["comparisons"]["neural_only"],
            "development_results_are_not_formal_evidence": True,
        },
        "shared_invariants": {
            "all_arms_share_raw_frames_grounding_parser_executor_and_fallback": True,
            "only_symbolic_harness_varies": True,
            "source_permuted_is_matched_control": True,
            "target_written_isomorphic_prediction_equivalence": 1.0,
            "generic_is_reported_ceiling_not_pass_gate": True,
            "formal_outcomes_opened_after_all_five_arm_decisions": True,
        },
        "paper_boundaries": {
            "AGQA_official_test_claimed": False,
            "full_AGQA_distribution_claimed": False,
            "raw_video_QA_SOTA_claimed": False,
            "target_native_grounding_is_frozen_off_the_shelf_engineering": True,
            "source_provenance_necessary_against_isomorphic_controller_claimed": False,
            "secondary_overall_accuracy_above_55_percent": formal["secondary_target"]["source_overall_accuracy_strictly_above_55_percent"],
        },
        "artifact_file_sha256s": {
            _artifact_label(path, args.artifact_label_root): _sha256(path) for path in (
                args.formal_evaluation, args.cohort, args.manifest, args.protocol,
                args.semantic_runtime, args.download_receipt, *args.grounding_view, args.routed_grounding,
                args.fallback, args.preoutcome, args.source_capabilities,
                args.anonymous_controller, args.development_evaluation,
                args.slowfast_development_evaluation,
            )
        },
    }
    body["bundle_sha256"] = stable_hash(body)
    if args.verify_existing:
        if not args.output.exists() or _load(args.output) != body:
            raise ValueError("existing V17 paper bundle does not reproduce exactly")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "tasks": n, "main_table": main_table,
        "source_vs_neural": formal["comparisons"]["neural_only"],
        "grounding": grounding, "bundle_sha256": body["bundle_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
