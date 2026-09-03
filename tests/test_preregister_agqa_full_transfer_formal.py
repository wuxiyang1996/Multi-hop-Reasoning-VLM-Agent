from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from motif_transfer.contracts import stable_hash


def write(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body), encoding="utf-8")


def test_preregistration_accepts_compiled_runtime_receipts(tmp_path: Path) -> None:
    cohort = {
        "status": "FROZEN_BEFORE_OUTCOME_OR_PROGRAM_ACCESS", "manifest_sha256": "cohort-manifest",
        "cohort_sha256": "cohort", "questions": 1,
    }
    controls = {"status": "SIX_ARMS_FROZEN", "formal_outcomes_read": False, "manifest_sha256": "controls"}
    source = {"status": "SOURCE_CAPABILITIES_INDUCED", "artifact_sha256": "source"}
    compiler = {"metrics": {"program_exact_rate": .99, "source_admission_rate": .999}, "report_sha256": "compiler"}
    executor = {"metrics": {"coverage": .90, "conditional_accuracy": .995}, "report_sha256": "executor"}
    compiler_runtime = {
        "cohort_sha256": "cohort", "runtime_sha256": "compiler-runtime",
        "answer_read": False, "oracle_program_read": False,
        "rows": [{"program_admission": {"status": "COMPILED"}}],
    }
    neural_runtime = {
        "cohort_sha256": "cohort", "runtime_sha256": "neural-runtime",
        "answer_read": False, "oracle_program_read": False, "rows": [{}],
    }
    values = [cohort, controls, source, compiler, executor, compiler_runtime, neural_runtime]
    names = ["cohort", "controls", "source", "compiler", "executor", "compiler_runtime", "neural_runtime"]
    paths = []
    for name, value in zip(names, values):
        path = tmp_path / f"{name}.json"; write(path, value); paths.append(path)
    output = tmp_path / "preregistration.json"
    command = [
        sys.executable, "scripts/preregister_agqa_full_transfer_formal.py",
        "--cohort-manifest", str(paths[0]), "--controls-manifest", str(paths[1]),
        "--source-capabilities", str(paths[2]), "--compiler-qualification", str(paths[3]),
        "--executor-development", str(paths[4]), "--compiler-runtime", str(paths[5]),
        "--neural-runtime", str(paths[6]), "--output", str(output),
    ]
    subprocess.run(command, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "FORMAL_AUTHORIZED"
    assert result["qualification_metrics"]["fresh_compiler_admission_rate"] == 1.0
    expected_hash = result.pop("preregistration_sha256")
    assert expected_hash == stable_hash(result)
