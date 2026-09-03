#!/usr/bin/env python3
"""One-shot six-arm evaluation on a preregistered fresh AGQA reserve."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from motif_transfer.agqa_stsg_typed_executor import AGQATypedSTSGExecutor
from motif_transfer.agqa_oracle_query_mdp import load_agqa_id_to_text
from motif_transfer.contracts import stable_hash
from motif_transfer.official_video_event_graph import load_builtin_only_pickle
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object


ARMS = ("neural_only", "source_induced", "source_permuted",
        "generic_scaffold", "target_written_isomorphic", "oracle_program")


def clean(value) -> str:
    return " ".join(str(value).replace("_", " ").casefold().split())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def binomial_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if not n: return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--runtime-programs", type=Path, required=True)
    parser.add_argument("--neural-predictions", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/test_balanced.txt")
    parser.add_argument("--stsg", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("formal result is immutable")
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg["status"] != "FORMAL_AUTHORIZED": raise ValueError("formal gate not authorized")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime_programs.read_text(encoding="utf-8"))
    neural = json.loads(args.neural_predictions.read_text(encoding="utf-8"))
    controls = {name: json.loads((args.controls_dir / f"{name}.json").read_text(encoding="utf-8")) for name in ARMS}
    if runtime["cohort_sha256"] != cohort["cohort_sha256"] or neural["cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("runtime/cohort mismatch")
    runtime_by_id = {row["task_id"]: row for row in runtime["rows"]}
    neural_by_id = {row["task_id"]: clean(row["prediction"]) for row in neural["rows"]}
    wanted = {row["task_id"] for row in cohort["rows"]}
    if set(runtime_by_id) != wanted or set(neural_by_id) != wanted:
        raise ValueError("formal runtime coverage mismatch")
    official = {}
    with zipfile.ZipFile(args.archive) as z, z.open(args.entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            if task_id in wanted:
                official[task_id] = {
                    "answer": clean(row["answer"]), "program": str(row["program"]),
                    "structural": str(row.get("structural", "")),
                    "semantic": str(row.get("semantic", "")),
                }
                if len(official) == len(wanted): break
    if set(official) != wanted: raise ValueError("formal evaluator join incomplete")
    corpus = load_builtin_only_pickle(args.stsg); ontology = load_agqa_id_to_text(args.ontology)
    stsg_sha = sha(args.stsg); rows_out = []; correct = {arm: 0 for arm in ARMS}
    routes = Counter(); families: dict[str, Counter[str]] = {}
    for public in cohort["rows"]:
        task_id, video = public["task_id"], public["video_id"]
        graph_hash = stable_hash({"stsg_sha256": stsg_sha, "video_id": video})
        predictions = {"neural_only": neural_by_id[task_id]}
        receipts = {}
        for arm in ARMS[1:]:
            control = controls[arm]
            executor = AGQATypedSTSGExecutor(
                graph=corpus[video], id_to_text=ontology, graph_sha256=graph_hash,
                authorized_operators=control["authorized_operators"],
                authorized_compositions=control.get("authorized_compositions"),
            )
            program = official[task_id]["program"] if arm == "oracle_program" else runtime_by_id[task_id]["predicted_program"]
            receipt = executor.execute(program, functional_program_source=("EVALUATOR_ORACLE" if arm == "oracle_program" else "FROZEN_COMPILER"))
            receipts[arm] = receipt
            predictions[arm] = clean(receipt.prediction) if receipt.status == "COMMITTED" else predictions["neural_only"]
            routes[(arm, receipt.status)] += 1
        answer = official[task_id]["answer"]
        row_correct = {arm: predictions[arm] == answer for arm in ARMS}
        for arm, value in row_correct.items(): correct[arm] += int(value)
        family = official[task_id]["structural"]
        bucket = families.setdefault(family, Counter()); bucket["rows"] += 1
        for arm, value in row_correct.items(): bucket[f"{arm}_correct"] += int(value)
        rows_out.append({
            "task_id": task_id, "video_id": video,
            "structural": family, "semantic": official[task_id]["semantic"],
            "answer": answer, "predictions": predictions, "correct": row_correct,
            "receipt_sha256s": {arm: receipt.receipt_sha256 for arm, receipt in receipts.items()},
        })
    n = len(rows_out); comparisons = {}
    for baseline in ("neural_only", "source_permuted", "generic_scaffold", "target_written_isomorphic"):
        wins = sum(row["correct"]["source_induced"] and not row["correct"][baseline] for row in rows_out)
        losses = sum(not row["correct"]["source_induced"] and row["correct"][baseline] for row in rows_out)
        comparisons[f"source_induced_vs_{baseline}"] = {
            "wins": wins, "losses": losses, "net_wins": wins - losses,
            "mcnemar_exact_p": binomial_two_sided(wins, losses),
        }
    gates = {
        "source_beats_neural": comparisons["source_induced_vs_neural_only"]["net_wins"] > 0 and comparisons["source_induced_vs_neural_only"]["mcnemar_exact_p"] < .05,
        "source_beats_permuted": comparisons["source_induced_vs_source_permuted"]["net_wins"] > 0 and comparisons["source_induced_vs_source_permuted"]["mcnemar_exact_p"] < .05,
        "source_beats_generic": comparisons["source_induced_vs_generic_scaffold"]["net_wins"] > 0 and comparisons["source_induced_vs_generic_scaffold"]["mcnemar_exact_p"] < .05,
        "source_matches_target_written_ceiling": abs(correct["source_induced"] - correct["target_written_isomorphic"]) / n <= .01,
        "negative_transfer_within_gate": comparisons["source_induced_vs_neural_only"]["losses"] <= max(5, int(.01 * n)),
        "all_structural_families_present": len(families) >= 5,
    }
    body = {
        "schema_version": "agqa-full-neurosymbolic-transfer-formal-v1",
        "status": "FULL_AGQA_TRANSFER_VALIDATED" if all(gates.values()) else "FULL_AGQA_TRANSFER_GATE_FAILED",
        "claim_boundary": "FRESH_QUESTIONS_AND_OUTCOMES;VIDEOS_AND_OFFICIAL_STSG_REUSED",
        "rows": n, "correct": correct,
        "accuracy": {arm: correct[arm] / n for arm in ARMS},
        "routes": {f"{arm}:{status}": value for (arm, status), value in sorted(routes.items())},
        "comparisons": comparisons, "gates": gates,
        "by_structural_family": {k: dict(v) for k, v in sorted(families.items())},
        "cohort_sha256": cohort["cohort_sha256"],
        "runtime_sha256": runtime["runtime_sha256"],
        "neural_runtime_sha256": neural["runtime_sha256"],
        "preregistration_sha256": prereg["preregistration_sha256"],
        "rows_detail": rows_out,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: body[k] for k in ("status", "rows", "accuracy", "comparisons", "gates", "report_sha256")}, indent=2))
    return 0 if body["status"] == "FULL_AGQA_TRANSFER_VALIDATED" else 1


if __name__ == "__main__": raise SystemExit(main())
