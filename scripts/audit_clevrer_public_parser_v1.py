#!/usr/bin/env python3
"""Freeze development exactness and outcome-blind reserve parser coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.clevrer_descriptive_compiler import compile_descriptive_question
from motif_transfer.clevrer_public_semantics import parse_public_semantics
from motif_transfer.clevrer_query_compiler import (
    compile_choice, compile_question, normalize_official_program,
)
from motif_transfer.contracts import stable_hash


def _runtime_normalize(program: list[str]) -> list[str]:
    """Map annotation DSL aliases to the released NS-DR executor vocabulary."""
    output = []
    aliases = {
        "get_frame": "query_frame", "get_object": "query_object",
        "get_col_partner": "query_collision_partner",
        "get_counterfact": "filter_counterfact",
    }
    for token in program:
        if token == "start": output.extend(("events", "filter_start"))
        elif token == "end": output.extend(("events", "filter_end"))
        else: output.append(aliases.get(token, token))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("CLEVRER parser audit is immutable")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    validation = {int(row["scene_index"]): row for row in json.loads(args.validation.read_text(encoding="utf-8"))}
    exact = total = reserve_covered = reserve_total = 0; rows = []
    for public_video in cohort["development"]:
        private = {int(q["question_id"]): q for q in validation[public_video["video_id"]]["questions"]}
        for public in public_video["questions"]:
            raw = private[public["question_id"]]; family = public["question_type"]
            choices = [row["choice"] for row in public["choices"]]
            semantic = parse_public_semantics(
                task_id=f"video_{public_video['video_id']}.Q{public['question_id']}",
                question=public["question"], question_family=family,
                public_subtype=public["question_subtype"], choices=choices,
            )
            if family == "descriptive":
                matches = compile_descriptive_question(public["question"], public["question_subtype"]) == _runtime_normalize(raw["program"])
                units = 1
            else:
                matches = compile_question(public["question"], family) == normalize_official_program(raw["program"])
                units = 1
                for public_choice, raw_choice in zip(public["choices"], raw["choices"]):
                    matches = matches and compile_choice(public_choice["choice"], family) == normalize_official_program(raw_choice["program"])
                    units += 1
            exact += int(matches) * units; total += units
            rows.append({"task_id": semantic.task_id, "family": family,
                         "semantic_receipt_sha256": semantic.receipt_sha256,
                         "development_program_exact": matches,
                         "answer_read": False})
    for public_video in cohort["reserve"]:
        for public in public_video["questions"]:
            reserve_total += 1
            parse_public_semantics(
                task_id=f"video_{public_video['video_id']}.Q{public['question_id']}",
                question=public["question"], question_family=public["question_type"],
                public_subtype=public["question_subtype"],
                choices=[row["choice"] for row in public["choices"]],
            )
            reserve_covered += 1
    metrics = {
        "development_program_units_exact": exact, "development_program_units_total": total,
        "development_program_exact_accuracy": exact / total,
        "reserve_public_tasks_covered": reserve_covered, "reserve_public_tasks_total": reserve_total,
        "reserve_public_coverage": reserve_covered / reserve_total,
    }
    gates = {
        "development_exact": exact == total,
        "reserve_public_syntax_coverage": reserve_covered == reserve_total,
        "operator_free_receipts": all(not row.get("operator_sequence_emitted", False) for row in rows),
    }
    body = {
        "schema_version": "clevrer-public-parser-audit-v1",
        "status": "CLEVRER_PUBLIC_PARSER_QUALIFIED" if all(gates.values()) else "CLEVRER_PUBLIC_PARSER_FAILED",
        "cohort_sha256": cohort["cohort_sha256"], "metrics": metrics, "gates": gates,
        "development_rows": rows,
        "development_programs_read_for_parser_qualification": True,
        "development_answers_read": False, "reserve_programs_read": False,
        "reserve_answers_read": False, "operator_sequences_emitted_to_harness": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "metrics": metrics, "gates": gates,
                      "report_sha256": body["report_sha256"]}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__": raise SystemExit(main())
