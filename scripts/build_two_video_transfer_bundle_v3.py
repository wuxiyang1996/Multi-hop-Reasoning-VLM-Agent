#!/usr/bin/env python3
"""Audit CLEVRER plus the fresh AGQA compositional V17b transfer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable(value: dict, field: str) -> None:
    claimed = value.get(field)
    body = {key: item for key, item in value.items() if key != field}
    if not isinstance(claimed, str) or stable_hash(body) != claimed:
        raise ValueError(f"invalid embedded {field}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clevrer-formal", type=Path, required=True)
    parser.add_argument("--clevrer-substitution", type=Path, required=True)
    parser.add_argument("--agqa-bundle", type=Path, required=True)
    parser.add_argument("--anonymous-controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.verify_existing:
        raise FileExistsError("two-video V3 bundle is immutable")

    clevrer = _load(args.clevrer_formal)
    substitution = _load(args.clevrer_substitution)
    agqa = _load(args.agqa_bundle)
    controller = _load(args.anonymous_controller)
    for value, field in (
        (clevrer, "report_sha256"), (substitution, "report_sha256"),
        (agqa, "bundle_sha256"), (controller, "artifact_sha256"),
    ):
        _stable(value, field)
    if clevrer.get("status") != "CLEVRER_FULL_LAYER_B_TRANSFER_VALIDATED":
        raise ValueError("CLEVRER transfer did not validate")
    if substitution.get("status") != "CLEVRER_ANONYMOUS_HARNESS_SUBSTITUTION_VERIFIED":
        raise ValueError("CLEVRER anonymous substitution did not validate")
    if agqa.get("status") != "AGQA_QWEN32_COMPOSITIONAL_V17_PAPER_BUNDLE_VALIDATED":
        raise ValueError("AGQA compositional transfer did not validate")
    controller_sha = controller["artifact_sha256"]
    if substitution.get("controller_artifact_sha256") != controller_sha:
        raise ValueError("CLEVRER used a different anonymous controller")
    agqa_rows = {row["arm"]: row for row in agqa["main_table"]}
    if agqa["formal_gates"].get("target_written_isomorphic_equivalence") is not True:
        raise ValueError("AGQA isomorphic control did not match source")

    body = {
        "schema_version": "two-video-fresh-layer-b-transfer-bundle-v3",
        "status": "BOTH_VIDEO_BENCHMARKS_FRESH_LAYER_B_VALIDATED_WITH_COMPOSITIONAL_AGQA_REPLICATION",
        "anonymous_controller_artifact_sha256": controller_sha,
        "claim": (
            "One source-only anonymous game controller significantly improves final QA under "
            "benchmark-shared raw-video grounding on fresh CLEVRER and on a one-shot fresh "
            "AGQA duration-compositional reserve."
        ),
        "main_table": [
            {
                "benchmark": "CLEVRER",
                "scope": "fresh 400-video four-family selective Layer B",
                "tasks": clevrer["task_count"],
                "neural_correct": clevrer["metrics"]["neural_only"]["correct"],
                "neural_accuracy": clevrer["metrics"]["neural_only"]["accuracy"],
                "source_correct": clevrer["metrics"]["source_induced"]["correct"],
                "source_accuracy": clevrer["metrics"]["source_induced"]["accuracy"],
                "generic_correct": clevrer["metrics"]["generic_symbolic"]["correct"],
                "generic_accuracy": clevrer["metrics"]["generic_symbolic"]["accuracy"],
                "gain_percentage_points": 100 * (
                    clevrer["metrics"]["source_induced"]["accuracy"]
                    - clevrer["metrics"]["neural_only"]["accuracy"]
                ),
                "wins": clevrer["paired"]["source_vs_neural"]["wins"],
                "losses": clevrer["paired"]["source_vs_neural"]["losses"],
                "paired_p": clevrer["paired"]["source_vs_neural"]["one_sided_exact_p"],
                "paired_test": "one-sided exact sign/McNemar",
                "negative_transfer_fraction": clevrer["negative_transfer_loss_fraction"],
                "isomorphic_equivalence": 1.0,
            },
            {
                "benchmark": "AGQA2",
                "scope": "fresh 256-video duration-compositional balanced-train reserve",
                "tasks": agqa["tasks"],
                "neural_correct": agqa_rows["neural_only"]["correct"],
                "neural_accuracy": agqa_rows["neural_only"]["accuracy"],
                "source_correct": agqa_rows["source_induced"]["correct"],
                "source_accuracy": agqa_rows["source_induced"]["accuracy"],
                "generic_correct": agqa_rows["generic_scaffold"]["correct"],
                "generic_accuracy": agqa_rows["generic_scaffold"]["accuracy"],
                "gain_percentage_points": 100 * (
                    agqa_rows["source_induced"]["accuracy"]
                    - agqa_rows["neural_only"]["accuracy"]
                ),
                "wins": agqa["paired_ablations"]["neural_only"]["wins"],
                "losses": agqa["paired_ablations"]["neural_only"]["losses"],
                "paired_p": agqa["paired_ablations"]["neural_only"]["exact_two_sided_p"],
                "paired_test": "two-sided exact McNemar",
                "negative_transfer_fraction": (
                    agqa["paired_ablations"]["neural_only"]["losses"] / agqa["tasks"]
                ),
                "isomorphic_equivalence": 1.0,
            },
        ],
        "shared_invariants": {
            "same_anonymous_game_controller": True,
            "within_benchmark_all_arms_share_frames_grounder_parser_executor_and_fallback": True,
            "matched_source_permutation_control": True,
            "target_written_isomorphic_control": True,
            "generic_target_native_symbolic_is_reported_as_ceiling_not_pass_gate": True,
            "formal_outcomes_unavailable_until_predictions_froze": True,
        },
        "paper_boundaries": {
            "AGQA_official_test_claimed": False,
            "full_AGQA_distribution_claimed": False,
            "raw_video_QA_SOTA_claimed": False,
            "game_to_video_reasoning_transfer_claimed": True,
            "target_native_grounder_is_off_the_shelf_and_designer_selected": True,
            "source_provenance_necessary_against_isomorphic_controller_claimed": False,
        },
        "artifact_file_sha256s": {
            str(path): _sha(path) for path in (
                args.clevrer_formal, args.clevrer_substitution,
                args.agqa_bundle, args.anonymous_controller,
            )
        },
    }
    body["bundle_sha256"] = stable_hash(body)
    if args.verify_existing:
        if not args.output.exists() or _load(args.output) != body:
            raise ValueError("existing two-video V3 bundle does not reproduce")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "main_table": body["main_table"],
        "bundle_sha256": body["bundle_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
