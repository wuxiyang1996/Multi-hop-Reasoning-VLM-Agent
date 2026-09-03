#!/usr/bin/env python3
"""Add development-only CLEVRER/AGQA2 grounding to the four-domain SFT set.

The video rows train only the anonymous symbolic Harness.  Raw frames, public
questions, native relation/action names, answers, official programs, scene
graphs, and evaluation outcomes are never serialized into model-visible data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

try:  # Direct script execution places ``scripts/`` on sys.path.
    from build_target_harness_sft_pilot_v1 import (  # type: ignore  # noqa: E402
        _candidate_pairs,
        _format_example,
        _json_text,
        _load_programs,
        _read,
        _resolve,
        _serialized_probability,
        _sha256,
    )
except ModuleNotFoundError:  # Imported as ``scripts.*`` by unit tests.
    from scripts.build_target_harness_sft_pilot_v1 import (  # noqa: E402
        _candidate_pairs,
        _format_example,
        _json_text,
        _load_programs,
        _read,
        _resolve,
        _serialized_probability,
        _sha256,
    )
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_source_function_induction import QUALIFIED  # noqa: E402
from motif_transfer.phase3_typed_effect_induction import TYPED_EFFECTS  # noqa: E402
from motif_transfer.target_harness_sft import (  # noqa: E402
    GroundedTargetState,
    TargetHarnessSFTExample,
)
from motif_transfer.video_proof_grounder import (  # noqa: E402
    V14_FEATURE_NAMES,
    validate_v14_artifact,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _stable_object(value: Mapping[str, Any], hash_field: str) -> bool:
    body = dict(value)
    claimed = str(body.pop(hash_field, ""))
    return bool(claimed) and stable_hash(body) == claimed


def _split_for_video(video_id: str, rule: Mapping[str, Any]) -> str:
    if (
        rule.get("kind") != "STABLE_VIDEO_ID_HASH_MODULO"
        or rule.get("identity_only_no_outcome_selection") is not True
    ):
        raise ValueError("unsupported or outcome-dependent video split rule")
    modulus = int(rule["modulus"])
    validation_bucket = int(rule["validation_bucket"])
    if modulus < 2 or not 0 <= validation_bucket < modulus:
        raise ValueError("invalid video hash split")
    bucket = int(stable_hash({"video_identity": str(video_id)})[:16], 16) % modulus
    return "validation" if bucket == validation_bucket else "train"


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _video_provenance(
    *, state: GroundedTargetState, benchmark: str, video_group_sha256: str,
    input_receipt_sha256: str, adapter_kind: str,
) -> dict[str, Any]:
    return {
        "state_receipt_sha256": state.state_receipt_sha256,
        "target_task_sha256": state.target_task_sha256,
        "target_benchmark": benchmark,
        "split": state.split,
        "video_group_sha256": video_group_sha256,
        "input_receipt_sha256": str(input_receipt_sha256),
        "grounder_artifact_sha256": state.grounder_artifact_sha256,
        "effect_adapter_kind": adapter_kind,
        "raw_frames_exposed_to_harness": False,
        "question_or_answer_exposed_to_harness": False,
        "native_action_or_relation_exposed_to_harness": False,
        "formal_or_reserve_target_used": False,
        "target_outcome_used_for_controller_label": False,
    }


def _clevrer_states(
    config: Mapping[str, Any], *, split_rule: Mapping[str, Any], precision: int,
) -> tuple[list[GroundedTargetState], dict[str, dict[str, Any]], dict[str, Any]]:
    receipt_path = _resolve(str(config["development_receipts"]))
    audit_path = _resolve(str(config["development_role_audit"]))
    artifact_path = _resolve(str(config["grounder_artifact"]))
    expected = {
        receipt_path: str(config["development_receipts_sha256"]),
        audit_path: str(config["development_role_audit_sha256"]),
        artifact_path: str(config["grounder_artifact_file_sha256"]),
    }
    for path, digest in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise SystemExit(f"CLEVRER frozen input hash mismatch: {path}")

    development = _read(receipt_path)
    audit = _read(audit_path)
    artifact = _read(artifact_path)
    proof_model, _, _, threshold = validate_v14_artifact(artifact)
    if (
        development.get("status") != "CLEVRER_V14_PROOF_DEVELOPMENT_COLLECTED"
        or tuple(development.get("feature_names") or ()) != V14_FEATURE_NAMES
        or development.get("compiler_exact_on_all_rows") is not True
    ):
        raise SystemExit("CLEVRER development proof receipts are not qualified")
    if (
        audit.get("status") != "CLEVRER_UNIFIED_V15_DEVELOPMENT_GATE_PASSED"
        or audit.get("role") != "consumed_v14_formal_repurposed_as_v15_development"
        or not all((audit.get("gates") or {}).values())
        or audit["lineage"].get("grounder_artifact_file_sha256")
        != _sha256(artifact_path)
    ):
        raise SystemExit("CLEVRER evidence is not frozen as consumed development")
    if artifact.get("development_receipts_file_sha256") != _sha256(receipt_path):
        raise SystemExit("CLEVRER proof grounder/development lineage mismatch")

    rows = list(development.get("rows") or ())
    features = [row.get("features") for row in rows]
    if not rows or any(
        not isinstance(vector, list) or len(vector) != len(V14_FEATURE_NAMES)
        for vector in features
    ):
        raise SystemExit("CLEVRER development feature rows malformed")
    scores = proof_model.predict(features)
    adapter = config["score_adapter"]
    temperature = float(adapter["temperature"])
    if (
        adapter.get("kind") != "FROZEN_UPLIFT_LOGISTIC_TO_TYPED_EFFECTS"
        or adapter.get("broadcast_across_effect_heads") is not True
        or temperature <= 0.0
        or int(adapter["candidate_order_variants"]) != 2
    ):
        raise SystemExit("unsupported CLEVRER typed-effect adapter")

    states: list[GroundedTargetState] = []
    provenance: dict[str, dict[str, Any]] = {}
    for row, score in zip(rows, scores):
        sample_id = str(row.get("sample_id") or "")
        proof_receipt = str(row.get("proof_receipts_sha256") or "")
        if ".mp4." not in sample_id or len(proof_receipt) != 64:
            raise SystemExit("CLEVRER row lacks an outcome-blind proof receipt")
        video_id = sample_id.split(".mp4.", 1)[0]
        split = _split_for_video(video_id, split_rule)
        video_group = stable_hash({"video_benchmark": "clevrer", "video": video_id})
        z = (float(score) - float(threshold)) / temperature
        probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
        values = (
            _serialized_probability(1.0 - probability, precision),
            _serialized_probability(probability, precision),
        )
        native_candidates = [
            {
                "candidate_key": stable_hash({
                    "clevrer_representation": index,
                    "proof_receipt": proof_receipt,
                }),
                "effects": {name: value for name in TYPED_EFFECTS},
            }
            for index, value in enumerate(values)
        ]
        for reverse in (False, True):
            ordered = list(reversed(native_candidates)) if reverse else native_candidates
            candidates = tuple({
                "candidate_id": f"C{index}", "effects": dict(candidate["effects"]),
            } for index, candidate in enumerate(ordered))
            state = GroundedTargetState(
                target_domain="video",
                target_task_sha256=video_group,
                split=split,
                state_receipt_sha256=stable_hash({
                    "proof_receipt": proof_receipt,
                    "grounder": artifact["artifact_sha256"],
                    "score": _serialized_probability(float(score), precision),
                    "candidate_order": [row_["candidate_key"] for row_ in ordered],
                }),
                grounder_artifact_sha256=str(artifact["artifact_sha256"]),
                candidates=candidates,
            )
            if not state.validate() or state.state_receipt_sha256 in provenance:
                raise SystemExit("invalid or duplicate CLEVRER grounded state")
            states.append(state)
            provenance[state.state_receipt_sha256] = _video_provenance(
                state=state, benchmark="clevrer", video_group_sha256=video_group,
                input_receipt_sha256=proof_receipt,
                adapter_kind=str(adapter["kind"]),
            )
    return states, provenance, {
        "development_receipts": str(receipt_path),
        "development_receipts_sha256": _sha256(receipt_path),
        "development_role_audit": str(audit_path),
        "development_role_audit_sha256": _sha256(audit_path),
        "grounder_artifact": str(artifact_path),
        "grounder_file_sha256": _sha256(artifact_path),
        "grounder_artifact_sha256": str(artifact["artifact_sha256"]),
        "raw_development_rows": len(rows),
        "grounded_candidate_sets": len(states),
        "grounder_was_trained_with_consumed_development_uplift": True,
        "controller_labels_use_target_outcomes": False,
        "formal_or_reserve_rows_used": 0,
    }


def _agqa_view_effects(
    attempt: Mapping[str, Any], config: Mapping[str, Any], *, precision: int,
) -> tuple[dict[str, float], str]:
    payload = attempt.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("AGQA neural view omitted a structured payload")
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("AGQA neural view omitted observations")
    supported = [
        row for row in observations
        if row.get("observability") == "OBSERVED"
        and row.get("start_frame") is not None
        and row.get("end_frame") is not None
        and bool(row.get("evidence_frames"))
    ]
    confidences = [_bounded(float(row.get("confidence", 0.0))) for row in supported]
    coverage_scores = config["coverage_scores"]
    coverage = float(coverage_scores.get(str(payload.get("coverage")), 0.0))
    sampled_frames = int(config["expected_sampled_frames"])
    evidence_normalizer = float(config["evidence_frame_normalizer"])
    durations = [
        max(0, int(row["end_frame"]) - int(row["start_frame"]) + 1)
        for row in supported
    ]
    evidence_frames = sum(len(row.get("evidence_frames") or ()) for row in supported)
    values = (
        max(confidences, default=0.0),
        sum(confidences) / max(1, len(observations)),
        _bounded(sum(durations) / max(1, sampled_frames)),
        _bounded(0.5 * coverage + 0.5 * min(1.0, evidence_frames / evidence_normalizer)),
    )
    usage = attempt.get("usage") or {}
    key = stable_hash({
        "response_sha256": str(usage.get("response_sha256") or ""),
        "view_receipt_payload": payload,
    })
    return {
        name: _serialized_probability(value, precision)
        for name, value in zip(TYPED_EFFECTS, values)
    }, key


def _agqa_consensus_effects(
    receipts: Sequence[Mapping[str, Any]], view_effects: Sequence[Mapping[str, float]],
    config: Mapping[str, Any], *, precision: int,
) -> tuple[dict[str, float], str]:
    if not receipts:
        raise ValueError("AGQA consensus candidate requires receipts")
    authorized = sum(bool(row.get("authorized")) for row in receipts) / len(receipts)
    sampled_frames = int(config["expected_sampled_frames"])
    intervals = [
        row.get("consensus_interval") for row in receipts
        if isinstance(row.get("consensus_interval"), list)
        and len(row["consensus_interval"]) == 2
    ]
    duration = sum(
        max(0, int(value[1]) - int(value[0]) + 1) / max(1, sampled_frames)
        for value in intervals
    ) / max(1, len(receipts))
    spreads = [
        float(row["maximum_endpoint_spread"]) for row in receipts
        if row.get("maximum_endpoint_spread") is not None
    ]
    spread_limit = float(config["maximum_consensus_endpoint_spread"])
    persistence = sum(
        _bounded(1.0 - value / spread_limit) for value in spreads
    ) / max(1, len(receipts))
    values = (
        sum(float(row[TYPED_EFFECTS[0]]) for row in view_effects)
        / max(1, len(view_effects)),
        authorized,
        _bounded(duration),
        _bounded(persistence),
    )
    key = stable_hash({
        "consensus_receipts": [str(row.get("receipt_sha256") or "") for row in receipts]
    })
    return {
        name: _serialized_probability(value, precision)
        for name, value in zip(TYPED_EFFECTS, values)
    }, key


def _agqa_states(
    config: Mapping[str, Any], *, split_rule: Mapping[str, Any], precision: int,
) -> tuple[list[GroundedTargetState], dict[str, dict[str, Any]], dict[str, Any]]:
    report_path = _resolve(str(config["development_report"]))
    config_path = _resolve(str(config["development_config"]))
    for path, digest in (
        (report_path, str(config["development_report_sha256"])),
        (config_path, str(config["development_config_sha256"])),
    ):
        if not path.is_file() or _sha256(path) != digest:
            raise SystemExit(f"AGQA2 frozen development input hash mismatch: {path}")
    report = _read(report_path)
    if (
        not _stable_object(report, "report_sha256")
        or report.get("status") != "AGQA2_TEMPORAL_LOCALIZED_QUERY_QUALIFIED"
        or report.get("split") != "train_consumed_development"
        or report.get("full_agqa_distribution_claim") is not False
        or report.get("grounder_qualified") is not True
        or int(report.get("sample_count", -1)) != 40
    ):
        raise SystemExit("AGQA2 input is not the frozen consumed-development report")
    adapter = config["effect_adapter"]
    if adapter.get("kind") != "MULTIVIEW_NEURAL_EVIDENCE_TO_TYPED_EFFECTS":
        raise SystemExit("unsupported AGQA2 effect adapter")
    maximum_views = int(adapter["maximum_view_candidates"])
    subset_sizes = tuple(map(int, adapter["candidate_subset_sizes"]))
    order_variants = int(adapter["candidate_order_variants"])
    if maximum_views < 2 or order_variants != 2 or not set(subset_sizes) <= {2, 3}:
        raise SystemExit("invalid AGQA2 candidate augmentation contract")

    states: list[GroundedTargetState] = []
    provenance: dict[str, dict[str, Any]] = {}
    blind_rows = 0
    for row in report.get("rows") or ():
        if (
            row.get("runtime_answer_read") is not False
            or row.get("runtime_functional_program_read") is not False
            or row.get("runtime_scene_graph_read") is not False
            or row.get("runtime_source_identity_read") is not False
            or row.get("official_answer_first_read_after_all_runtime_rows_froze")
            is not True
            or row.get("official_scene_graph_read_by_evaluator") is not False
        ):
            raise SystemExit("AGQA2 development runtime crossed the blind boundary")
        video_id = str(row.get("video_id") or "")
        runtime_receipt = str(row.get("runtime_receipt_sha256") or "")
        grounder_sha = str(row.get("grounder_sha256") or "")
        if not video_id or len(runtime_receipt) != 64 or grounder_sha != report["grounder_sha256"]:
            raise SystemExit("AGQA2 runtime receipt lineage mismatch")
        attempts = list(row.get("anchor_attempts") or ())[:maximum_views]
        if len(attempts) < 2:
            raise SystemExit("AGQA2 row has fewer than two neural grounding views")
        native_candidates = []
        view_effects = []
        for attempt in attempts:
            effects, candidate_key = _agqa_view_effects(
                attempt, adapter, precision=precision,
            )
            view_effects.append(effects)
            native_candidates.append({
                "candidate_key": candidate_key, "effects": effects,
            })
        consensus = list(row.get("anchor_consensus_receipts") or ())
        if consensus:
            effects, candidate_key = _agqa_consensus_effects(
                consensus, view_effects, adapter, precision=precision,
            )
            native_candidates.append({
                "candidate_key": candidate_key, "effects": effects,
            })

        split = _split_for_video(video_id, split_rule)
        video_group = stable_hash({"video_benchmark": "agqa2", "video": video_id})
        ordered_native = sorted(
            native_candidates, key=lambda value: str(value["candidate_key"]),
        )
        for size in subset_sizes:
            for subset in combinations(ordered_native, size):
                for reverse in (False, True):
                    ordered = tuple(reversed(subset)) if reverse else tuple(subset)
                    candidates = tuple({
                        "candidate_id": f"C{index}",
                        "effects": dict(candidate["effects"]),
                    } for index, candidate in enumerate(ordered))
                    state = GroundedTargetState(
                        target_domain="video",
                        target_task_sha256=video_group,
                        split=split,
                        state_receipt_sha256=stable_hash({
                            "runtime_receipt": runtime_receipt,
                            "grounder": grounder_sha,
                            "candidate_order": [
                                value["candidate_key"] for value in ordered
                            ],
                        }),
                        grounder_artifact_sha256=grounder_sha,
                        candidates=candidates,
                    )
                    if not state.validate():
                        raise SystemExit("invalid AGQA2 grounded state")
                    if state.state_receipt_sha256 in provenance:
                        continue
                    states.append(state)
                    provenance[state.state_receipt_sha256] = _video_provenance(
                        state=state, benchmark="agqa2",
                        video_group_sha256=video_group,
                        input_receipt_sha256=runtime_receipt,
                        adapter_kind=str(adapter["kind"]),
                    )
        blind_rows += 1
    return states, provenance, {
        "development_report": str(report_path),
        "development_report_sha256": _sha256(report_path),
        "development_config": str(config_path),
        "development_config_sha256": _sha256(config_path),
        "grounder_artifact_sha256": str(report["grounder_sha256"]),
        "raw_development_rows": int(report["sample_count"]),
        "raw_development_videos": int(report["unique_video_count"]),
        "blind_runtime_rows": blind_rows,
        "grounded_candidate_sets": len(states),
        "reported_historical_provider_cost_usd": float(
            report["reported_provider_cost_usd"]
        ),
        "new_provider_calls_for_sft_build": 0,
        "controller_labels_use_target_outcomes": False,
        "formal_or_reserve_rows_used": 0,
    }


def _load_parent(config: Mapping[str, Any]):
    root = _resolve(str(config["directory"]))
    paths = {
        "manifest": root / "manifest.json",
        "structured": root / "structured.jsonl",
        "train": root / "train.jsonl",
        "validation": root / "validation.jsonl",
    }
    for name, path in paths.items():
        if not path.is_file() or _sha256(path) != str(config[f"{name}_sha256"]):
            raise SystemExit(f"parent four-domain dataset {name} hash mismatch")
    manifest = _read(paths["manifest"])
    if (
        manifest.get("status") != "FROZEN_FOUR_TARGET_DEVELOPMENT_HARNESS_SUPERVISION"
        or not all((manifest.get("gates") or {}).values())
        or manifest.get("video_target_data_used") is not False
        or manifest.get("target_outcome_used_for_controller_labels") is not False
        or manifest.get("formal_or_qualification_targets_used") is not False
    ):
        raise SystemExit("parent four-domain dataset is not qualified")
    return (
        manifest,
        _read_jsonl(paths["structured"]),
        {split: _read_jsonl(paths[split]) for split in ("train", "validation")},
        paths,
    )


def _select_video_pairs(
    pairs: Sequence[tuple[TargetHarnessSFTExample, TargetHarnessSFTExample]],
    provenance: Mapping[str, Mapping[str, Any]], *, benchmark: str, split: str,
    quota: int, forbidden_prompts: set[str],
) -> list[tuple[TargetHarnessSFTExample, TargetHarnessSFTExample]]:
    by_family = defaultdict(list)
    for pair in pairs:
        state_receipt = str(pair[0].evidence_receipt_ids[0])
        meta = provenance.get(state_receipt)
        if meta and meta["target_benchmark"] == benchmark and pair[0].split == split:
            by_family[pair[0].source_family].append(pair)
    for family in by_family:
        by_family[family].sort(key=lambda pair: (pair[0].pair_id, pair[0].example_id))
    families = sorted(by_family)
    if not families:
        raise SystemExit(f"no candidate pairs for video/{benchmark}/{split}")
    offsets = {family: 0 for family in families}
    selected = []
    while len(selected) < quota:
        progress = False
        for family in families:
            rows = by_family[family]
            while offsets[family] < len(rows):
                pair = rows[offsets[family]]
                offsets[family] += 1
                prompts = {_json_text(row.input_payload) for row in pair}
                if len(prompts) != 2 or prompts & forbidden_prompts:
                    continue
                selected.append(pair)
                forbidden_prompts.update(prompts)
                progress = True
                break
            if len(selected) >= quota:
                break
        if not progress:
            found = sum(offsets.values())
            raise SystemExit(
                f"insufficient unique video pairs for {benchmark}/{split}: "
                f"needed {quota}, exhausted {found} candidates"
            )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/target_harness_sft_five_domain_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/target_harness_sft_five_domain_v1",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    config = _read(args.config)
    if (
        config.get("schema_version") != "target-harness-sft-five-domain-config-v1"
        or config.get("status") != "FROZEN_BEFORE_FIVE_DOMAIN_DATASET_BUILD"
    ):
        raise SystemExit("five-domain target Harness config is not frozen")

    parent_manifest, parent_structured, parent_model_rows, parent_paths = _load_parent(
        config["parent_four_domain_dataset"]
    )
    programs, program_receipts = _load_programs(
        _resolve(str(config["source_program_root"])),
        tuple(map(str, config["source_families"])),
    )
    qualified_families = sorted(
        family for family, program in programs.items()
        if program.get("status") == QUALIFIED
    )
    precision = int(config["controller_training"]["probability_decimal_places"])
    video_config = config["video"]
    split_rule = video_config["split_rule"]
    clevrer_states, clevrer_provenance, clevrer_audit = _clevrer_states(
        video_config["clevrer"], split_rule=split_rule, precision=precision,
    )
    agqa_states, agqa_provenance, agqa_audit = _agqa_states(
        video_config["agqa2"], split_rule=split_rule, precision=precision,
    )
    provenance = {**clevrer_provenance, **agqa_provenance}
    if len(provenance) != len(clevrer_provenance) + len(agqa_provenance):
        raise SystemExit("video state receipt collision across benchmarks")
    candidate_pairs = _candidate_pairs([*clevrer_states, *agqa_states], programs)
    forbidden_prompts = {_json_text(row["input_payload"]) for row in parent_structured}
    quota = video_config["quota_per_benchmark"]
    selected_pairs = []
    for split, quota_key in (
        ("train", "train_authentic"),
        ("validation", "validation_authentic"),
    ):
        for benchmark in ("clevrer", "agqa2"):
            selected_pairs.extend(_select_video_pairs(
                candidate_pairs, provenance, benchmark=benchmark, split=split,
                quota=int(quota[quota_key]), forbidden_prompts=forbidden_prompts,
            ))
    video_examples = [row for pair in selected_pairs for row in pair]
    if not video_examples or not all(row.validate() for row in video_examples):
        raise SystemExit("video Harness examples failed validation")

    video_structured = [asdict(row) for row in video_examples]
    all_structured = [*parent_structured, *video_structured]
    selected_state_receipts = {
        str(row.evidence_receipt_ids[0]) for row in video_examples
    }
    selected_provenance = {
        key: value for key, value in provenance.items() if key in selected_state_receipts
    }
    if selected_state_receipts != set(selected_provenance):
        raise SystemExit("selected video example lacks provenance")

    def model_row(row: TargetHarnessSFTExample) -> dict[str, Any]:
        output = _format_example(row)
        output["target_benchmark_audit_only"] = selected_provenance[
            str(row.evidence_receipt_ids[0])
        ]["target_benchmark"]
        return output

    all_model_rows = {
        split: [
            *parent_model_rows[split],
            *[model_row(row) for row in video_examples if row.split == split],
        ]
        for split in ("train", "validation")
    }
    domains = {str(row["target_domain"]) for row in all_structured}
    pairs_by_id = Counter(str(row["pair_id"]) for row in all_structured)
    domain_split_kind = Counter(
        (
            str(row["target_domain"]), str(row["split"]),
            "authentic" if row["control_variant"]
            == "AUTHENTIC_TARGET_NEURAL_GROUNDING" else "control",
        )
        for row in all_structured
    )
    video_benchmark_split_kind = Counter()
    for row in video_structured:
        meta = selected_provenance[str(row["evidence_receipt_ids"][0])]
        video_benchmark_split_kind[(
            str(meta["target_benchmark"]), str(row["split"]),
            "authentic" if row["control_variant"]
            == "AUTHENTIC_TARGET_NEURAL_GROUNDING" else "control",
        )] += 1
    prompts = [row["prompt"] for rows in all_model_rows.values() for row in rows]
    lower_prompts = "\n".join(prompts).lower()
    lower_completions = "\n".join(
        row["completion"] for rows in all_model_rows.values() for row in rows
    ).lower()
    source_families = {str(row["source_family"]) for row in all_structured}
    control_variants = {str(row["control_variant"]) for row in all_structured}
    task_sets = {
        (domain, split): {
            str(row["target_task_sha256"]) for row in all_structured
            if row["target_domain"] == domain and row["split"] == split
        }
        for domain in domains for split in ("train", "validation")
    }
    video_groups = {
        split: {
            str(row["video_group_sha256"])
            for row in selected_provenance.values() if row["split"] == split
        }
        for split in ("train", "validation")
    }
    expected_domain = {"train": 450, "validation": 150}
    expected_benchmark = {"train": 225, "validation": 75}
    forbidden_tokens = (
        "webshop", "alfworld", "discoveryworld", "tirbench", "video",
        "clevrer", "agqa", "selected_action", "expert_action",
        "official_success", "official_reward", "native_actions",
        "target_domain", "target_benchmark", "raw_frame", "question",
        "answer", "functional_program", "scene_graph", "anchor",
        "relation", "trajectory", "explicit_relation", "zoom_region",
        "extract_colors", "teleport_to_object", "pickup", '"use"',
        "go to ", "click('", "fill('", "move ", "take ",
    )
    gates = {
        "exact_five_target_domains": domains
        == {"webshop", "alfworld", "discoveryworld", "tirbench", "video"},
        "exact_two_video_benchmarks": {
            str(row["target_benchmark"]) for row in selected_provenance.values()
        } == {"clevrer", "agqa2"},
        "all_video_examples_valid": all(row.validate() for row in video_examples),
        "exact_one_control_per_authentic": set(pairs_by_id.values()) == {2},
        "exact_domain_split_kind_quotas": all(
            domain_split_kind[(domain, split, kind)] == count
            for domain in domains for split, count in expected_domain.items()
            for kind in ("authentic", "control")
        ),
        "exact_video_benchmark_split_kind_quotas": all(
            video_benchmark_split_kind[(benchmark, split, kind)] == count
            for benchmark in ("clevrer", "agqa2")
            for split, count in expected_benchmark.items()
            for kind in ("authentic", "control")
        ),
        "example_ids_unique": len({row["example_id"] for row in all_structured})
        == len(all_structured),
        "prompts_unique": len(set(prompts)) == len(prompts),
        "train_validation_prompts_disjoint": {
            row["prompt"] for row in all_model_rows["train"]
        }.isdisjoint({row["prompt"] for row in all_model_rows["validation"]}),
        "target_tasks_disjoint_across_splits": all(
            task_sets[(domain, "train")].isdisjoint(
                task_sets[(domain, "validation")]
            ) for domain in domains
        ),
        "video_groups_disjoint_across_splits": video_groups["train"].isdisjoint(
            video_groups["validation"]
        ),
        "target_benchmark_native_and_outcome_absent_from_prompts": not any(
            token in lower_prompts for token in forbidden_tokens
        ),
        "source_identity_absent_from_prompts": not any(
            family.lower() in lower_prompts for family in source_families
        ),
        "control_identity_absent_from_prompts": not any(
            control.lower() in lower_prompts for control in control_variants
        ),
        "outcome_native_identity_absent_from_completions": not any(
            token in lower_completions for token in (
                "reward", "success", "native_action", "question", "answer",
            )
        ),
        "all_video_provenance_preserves_authority": all(
            row["raw_frames_exposed_to_harness"] is False
            and row["question_or_answer_exposed_to_harness"] is False
            and row["native_action_or_relation_exposed_to_harness"] is False
            and row["formal_or_reserve_target_used"] is False
            and row["target_outcome_used_for_controller_label"] is False
            for row in selected_provenance.values()
        ),
        "agqa_all_consumed_development_runtime_rows_blind": (
            agqa_audit["blind_runtime_rows"] == agqa_audit["raw_development_rows"]
        ),
        "no_new_video_provider_calls": agqa_audit["new_provider_calls_for_sft_build"] == 0,
        "target_outcomes_not_used_for_controller_labels": all(
            row.get("target_outcome_used") is False for row in all_structured
        ),
        "parent_four_domain_gates_preserved": all(parent_manifest["gates"].values()),
    }
    if not all(gates.values()):
        raise SystemExit(f"five-domain target Harness gates failed: {gates}")

    args.output_dir.mkdir(parents=True)
    structured_path = args.output_dir / "structured.jsonl"
    with structured_path.open("w", encoding="utf-8") as stream:
        for row in sorted(all_structured, key=lambda value: value["example_id"]):
            stream.write(_json_text(row) + "\n")
    split_paths = {}
    for split, rows in all_model_rows.items():
        path = args.output_dir / f"{split}.jsonl"
        split_paths[split] = path
        with path.open("w", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda value: value["example_id"]):
                stream.write(_json_text(row) + "\n")
    provenance_path = args.output_dir / "video_provenance.jsonl"
    with provenance_path.open("w", encoding="utf-8") as stream:
        for key in sorted(selected_provenance):
            stream.write(_json_text(selected_provenance[key]) + "\n")

    summary = {
        "examples": len(all_structured),
        "pairs": len(all_structured) // 2,
        "target_domains": sorted(domains),
        "video_benchmarks": ["agqa2", "clevrer"],
        "qualified_source_families": qualified_families,
        "domain_split_kind_counts": {
            "/".join(key): value for key, value in sorted(domain_split_kind.items())
        },
        "video_benchmark_split_kind_counts": {
            "/".join(key): value
            for key, value in sorted(video_benchmark_split_kind.items())
        },
        "decision_counts": dict(sorted(Counter(
            str(row["target_payload"]["decision"]) for row in all_structured
        ).items())),
        "control_variant_counts": dict(sorted(Counter(
            str(row["control_variant"]) for row in all_structured
        ).items())),
        "source_family_counts": dict(sorted(Counter(
            str(row["source_family"]) for row in all_structured
        ).items())),
        "video_unique_groups": {
            split: len(video_groups[split]) for split in ("train", "validation")
        },
    }
    manifest = {
        "schema_version": "target-harness-sft-five-domain-dataset-v1",
        "status": "FROZEN_FIVE_TARGET_DEVELOPMENT_HARNESS_SUPERVISION",
        "authority": (
            "FIVE_TARGET_NATIVE_NEURAL_GROUNDERS;"
            "VIDEO_FROM_CONSUMED_DEVELOPMENT_ONLY;"
            "LABELS_FROM_FROZEN_SOURCE_PROGRAM_EXECUTOR"
        ),
        "build_config": str(args.config.resolve()),
        "build_config_sha256": _sha256(args.config),
        "parent_four_domain_dataset": {
            "manifest": str(parent_paths["manifest"]),
            "manifest_sha256": _sha256(parent_paths["manifest"]),
            "structured_sha256": _sha256(parent_paths["structured"]),
            "train_sha256": _sha256(parent_paths["train"]),
            "validation_sha256": _sha256(parent_paths["validation"]),
        },
        "source_program_receipts": program_receipts,
        "video": {
            "target_domain": "video",
            "benchmarks": {"clevrer": clevrer_audit, "agqa2": agqa_audit},
            "selected_grounded_state_receipts": len(selected_provenance),
            "raw_frames_exposed_to_harness": False,
            "questions_answers_or_native_relations_exposed_to_harness": False,
            "formal_or_reserve_video_targets_used": False,
            "controller_labels_use_target_outcomes": False,
            "new_provider_calls": 0,
        },
        "files": {
            "structured": {
                "path": str(structured_path.resolve()), "sha256": _sha256(structured_path),
            },
            "train": {
                "path": str(split_paths["train"].resolve()),
                "sha256": _sha256(split_paths["train"]),
            },
            "validation": {
                "path": str(split_paths["validation"].resolve()),
                "sha256": _sha256(split_paths["validation"]),
            },
            "video_provenance": {
                "path": str(provenance_path.resolve()),
                "sha256": _sha256(provenance_path),
            },
        },
        "summary": summary,
        "gates": gates,
        "prompt_policy": config["prompt_policy"],
        "controller_training_contract": {
            **config["controller_training"], "exact_tokenizer_audit_required": True,
        },
        "target_data_used": True,
        "target_outcome_used_for_controller_labels": False,
        "formal_or_qualification_targets_used": False,
        "video_target_data_used": True,
        "clevrer_grounder_used_consumed_development_outcome_labels": True,
        "claim_boundary": config["claim_boundary"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"], "summary": summary,
        "video": manifest["video"], "gates": gates,
        "output_dir": str(args.output_dir),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
