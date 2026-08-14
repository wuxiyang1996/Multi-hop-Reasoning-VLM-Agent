#!/usr/bin/env python3
"""Collect consumed/development CLEVRER paired proof receipts for V14 grounding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motif_transfer.clevrer_proof_receipts import (  # noqa: E402
    PROOF_FEATURE_NAMES,
    paired_proof_features,
)
from motif_transfer.clevrer_query_compiler import (  # noqa: E402
    compile_choice,
    compile_question,
    normalize_official_program,
)
from motif_transfer.sokoban_video_recovery import validate_source_receipt  # noqa: E402
from motif_transfer.video_recovery_cate import FEATURE_NAMES, build_features  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_parts(sample_id: str) -> tuple[int, int]:
    video, question = sample_id.split(".mp4.Q", 1)
    return int(video.rsplit("_", 1)[1]), int(question)


def _verify_inputs(config: Mapping[str, Any]) -> None:
    for spec in config["consumed_reports"]:
        path = Path(spec["path"])
        if _sha256(path) != spec["sha256"]:
            raise ValueError(f"consumed report hash mismatch: {path}")
    manifest_path = Path(config["development"]["split_manifest"])
    if _sha256(manifest_path) != config["development"]["split_manifest_sha256"]:
        raise ValueError("V14 split manifest hash mismatch")
    paths = {
        "source_receipt_sha256": Path(config["source_receipt"]),
        "collector_sha256": Path(__file__).resolve(),
        "proof_module_sha256": REPO / "src/motif_transfer/clevrer_proof_receipts.py",
        "compiler_module_sha256": REPO / "src/motif_transfer/clevrer_query_compiler.py",
        "base_feature_module_sha256": REPO / "src/motif_transfer/video_recovery_cate.py",
    }
    for key, path in paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V14 frozen lineage mismatch for {key}: {path}")
    source = json.loads(Path(config["source_receipt"]).read_text(encoding="utf-8"))
    validate_source_receipt(source)


def _sample_batches(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    output = []
    for spec in config["consumed_reports"]:
        report = json.loads(Path(spec["path"]).read_text(encoding="utf-8"))
        output.extend((str(row["sample_id"]), str(spec["name"])) for row in report["rows"])
    development = config["development"]
    manifest = json.loads(Path(development["split_manifest"]).read_text(encoding="utf-8"))
    development_ids = manifest["benchmarks"]["clevrer"]["splits"]["development"]
    if len(development_ids) != int(development["expected_samples"]):
        raise ValueError("unexpected V14 development sample count")
    output.extend(
        (str(sample_id), str(development["batch_name"]))
        for sample_id in development_ids
    )
    ids = [sample_id for sample_id, _ in output]
    if len(ids) != len(set(ids)):
        raise ValueError("proof-receipt collection contains duplicate IDs")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _verify_inputs(config)
    sample_batches = _sample_batches(config)

    target = config["target"]
    executor_root = Path(target["official_root"]) / "executor"
    sys.path.insert(0, str(executor_root))
    from executor import Executor  # type: ignore
    from simulation import Simulation  # type: ignore

    annotations_path = executor_root / "data/validation.json"
    annotations = {
        int(row["scene_index"]): row
        for row in json.loads(annotations_path.read_text(encoding="utf-8"))
    }
    prediction_hash_cache: dict[str, str] = {}
    rows = []
    for sample_id, batch in sample_batches:
        scene_id, question_id = _sample_parts(sample_id)
        question = next(
            row for row in annotations[scene_id]["questions"]
            if int(row["question_id"]) == question_id
        )
        family = str(question["question_type"])
        question_program = compile_question(str(question["question"]), family)
        choice_programs = [
            compile_choice(str(choice["choice"]), family)
            for choice in question["choices"]
        ]
        paths = {
            "explicit": executor_root / "data/propnet_preds"
            / str(target["explicit_relation_prediction_directory"])
            / f"sim_{scene_id:05d}.json",
            "trajectory": executor_root / "data/propnet_preds"
            / str(target["trajectory_prediction_directory"])
            / f"sim_{scene_id:05d}.json",
        }
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(paths)
        for path in paths.values():
            prediction_hash_cache.setdefault(str(path.resolve()), _sha256(path))
        explicit_executor = Executor(Simulation(str(paths["explicit"]), use_event_ann=True))
        trajectory_executor = Executor(Simulation(str(paths["trajectory"]), use_event_ann=False))
        proof_features, proof_receipts = paired_proof_features(
            explicit_executor, trajectory_executor, question_program, choice_programs,
        )
        explicit_raw = [str(value["explicit_answer"]) for value in proof_receipts]
        trajectory_raw = [str(value["trajectory_answer"]) for value in proof_receipts]
        explicit_answer = "".join("1" if value == "yes" else "0" for value in explicit_raw)
        trajectory_answer = "".join("1" if value == "yes" else "0" for value in trajectory_raw)
        base_features = build_features(
            family=family,
            question_program=question_program,
            choice_programs=choice_programs,
            explicit_answer=explicit_answer,
            trajectory_answer=trajectory_answer,
            explicit_error_count=explicit_raw.count("error"),
        )
        gold_answer = "".join(
            "1" if choice["answer"] == "correct" else "0"
            for choice in question["choices"]
        )
        explicit_correct = explicit_answer == gold_answer
        trajectory_correct = trajectory_answer == gold_answer
        rows.append({
            "sample_id": sample_id,
            "source_batch": batch,
            "family": family,
            "feature_names": list(FEATURE_NAMES + PROOF_FEATURE_NAMES),
            "features": list(map(float, base_features + proof_features)),
            "explicit_answer": explicit_answer,
            "trajectory_answer": trajectory_answer,
            "explicit_raw_executor_results": explicit_raw,
            "trajectory_raw_executor_results": trajectory_raw,
            "explicit_correct": explicit_correct,
            "trajectory_correct": trajectory_correct,
            "uplift": int(trajectory_correct) - int(explicit_correct),
            "compiler_question_exact": question_program
            == normalize_official_program(question["program"]),
            "compiler_choices_exact": all(
                compiled == normalize_official_program(choice["program"])
                for compiled, choice in zip(choice_programs, question["choices"])
            ),
            "proof_receipts_sha256": _content_hash(proof_receipts),
        })

    report = {
        "schema_version": 14,
        "status": "CLEVRER_V14_PROOF_DEVELOPMENT_COLLECTED",
        "samples": len(rows),
        "batch_counts": {
            batch: sum(row["source_batch"] == batch for row in rows)
            for batch in sorted({row["source_batch"] for row in rows})
        },
        "family_counts": {
            family: sum(row["family"] == family for row in rows)
            for family in sorted({row["family"] for row in rows})
        },
        "feature_names": list(FEATURE_NAMES + PROOF_FEATURE_NAMES),
        "compiler_exact_on_all_rows": all(
            row["compiler_question_exact"] and row["compiler_choices_exact"]
            for row in rows
        ),
        "uplift_counts": {
            str(value): sum(row["uplift"] == value for row in rows)
            for value in (-1, 0, 1)
        },
        "rows": rows,
        "lineage": {
            "config_sha256": _sha256(args.config),
            "annotations_sha256": _sha256(annotations_path),
            "prediction_file_sha256": prediction_hash_cache,
        },
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "samples": report["samples"],
        "batch_counts": report["batch_counts"],
        "family_counts": report["family_counts"],
        "uplift_counts": report["uplift_counts"],
        "compiler_exact_on_all_rows": report["compiler_exact_on_all_rows"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
