#!/usr/bin/env python3
"""Run an AGQA config inside the exact historical V65 git runtime tree."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "ded7448839183851aa10c3cd3e12d253f04e1ceb"
COLLECTOR = "scripts/collect_agqa2_active_grounding_v3.py"
GROUNDER_MODULE = "src/motif_transfer/agqa_active_frame_grounder.py"
EXPECTED_COLLECTOR_SHA256 = "c845a0446fe5edc60f29dedbbb8eca3527a1f0c087f130924529a64cb8cdd5f1"
EXPECTED_GROUNDER_SHA256 = "87a41b64a77aae9cd8899f714061276fd3fcee05e8950a050fffb8849b81761c"
DEPENDENCY_OVERLAY = {
    "src/motif_transfer/phase3_source_function_induction.py":
        "5bd04fa4b0d9b3a90b61d9108e19b8366080b167a63e5ac2556d351356fdcd6d",
}
ARCHIVE_PATHS = (
    "src",
    COLLECTOR,
    "scripts/audit_agqa2_program_transfer_v1.py",
)


def _git_blob(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{FROZEN_COMMIT}:{path}"]
    )


def _check_blob(path: str, expected: str) -> None:
    actual = hashlib.sha256(_git_blob(path)).hexdigest()
    if actual != expected:
        raise ValueError(f"frozen git blob mismatch for {path}: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--postruntime-missing-program-hash-fix", action="store_true",
        help="Apply an outcome-neutral finalizer-only optional lineage check.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    frozen = config["frozen_runtime"]
    if frozen["git_commit"] != FROZEN_COMMIT:
        raise ValueError("config requests another frozen runtime")
    _check_blob(COLLECTOR, EXPECTED_COLLECTOR_SHA256)
    _check_blob(GROUNDER_MODULE, EXPECTED_GROUNDER_SHA256)
    if config["grounder"]["collector_sha256"] != EXPECTED_COLLECTOR_SHA256:
        raise ValueError("config collector is not V65")
    if config["grounder"]["module_sha256"] != EXPECTED_GROUNDER_SHA256:
        raise ValueError("config grounder module is not V65")
    if config["frozen_runtime"]["dependency_overlay_sha256"] != DEPENDENCY_OVERLAY:
        raise ValueError("config dependency overlay differs from the frozen runtime")

    with tempfile.TemporaryDirectory(prefix="agqa-v65-runtime-") as raw_tmp:
        runtime_root = Path(raw_tmp)
        archive = subprocess.check_output([
            "git", "-C", str(REPO_ROOT), "archive", "--format=tar",
            FROZEN_COMMIT, *ARCHIVE_PATHS,
        ])
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(runtime_root, filter="data")
        # Only the newly frozen protocol/config/manifest/selection are overlaid.
        # Model code, source artifacts, and evaluator dependencies stay at V65.
        overlay_paths = {
            config_path,
            REPO_ROOT / config["manifest"],
            REPO_ROOT / config["preregistration"],
            REPO_ROOT / config["formal_protocol"],
        }
        for spec in config["sources"].values():
            for key in ("path", "artifact_path", "confirmation_path"):
                if spec.get(key):
                    overlay_paths.add(REPO_ROOT / spec[key])
        for source in overlay_paths:
            relative = source.resolve().relative_to(REPO_ROOT)
            destination = runtime_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        for relative, expected_sha256 in DEPENDENCY_OVERLAY.items():
            source = REPO_ROOT / relative
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
                raise ValueError(f"dependency overlay hash mismatch: {relative}")
            destination = runtime_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        execution_collector = runtime_root / COLLECTOR
        if args.postruntime_missing_program_hash_fix:
            receipt_root = args.output.resolve().parent / "runtime_receipts"
            expected_rows = int(config["qualification_gates"]["required_valid_runtime_rows"])
            actual_rows = len(list(receipt_root.glob("*.json")))
            if actual_rows != expected_rows:
                raise ValueError(
                    "post-runtime fix requires every runtime receipt to pre-exist: "
                    f"{actual_rows}/{expected_rows}"
                )
            original = execution_collector.read_text()
            needle = '''        if stable_hash(program) != frozen["program_sha256"]:
            raise ValueError(f"functional-program hash mismatch: {task_id}")
'''
            replacement = '''        expected_program_sha256 = frozen.get("program_sha256")
        if (
            expected_program_sha256 is not None
            and stable_hash(program) != expected_program_sha256
        ):
            raise ValueError(f"functional-program hash mismatch: {task_id}")
'''
            if original.count(needle) != 1:
                raise ValueError("frozen collector lineage-check patch point drifted")
            patched = original.replace(needle, replacement)
            execution_collector = (
                runtime_root / "scripts/collect_agqa2_active_grounding_v3_transport_fix.py"
            )
            execution_collector.write_text(patched)
            receipt_body = {
                "schema_version": "agqa-v65-postruntime-lineage-transport-fix-v1",
                "status": "APPLIED_AFTER_ALL_RUNTIME_RECEIPTS_FROZE",
                "runtime_receipt_count": actual_rows,
                "original_collector_sha256": hashlib.sha256(
                    original.encode()
                ).hexdigest(),
                "patched_finalizer_sha256": hashlib.sha256(
                    patched.encode()
                ).hexdigest(),
                "change": "OPTIONAL_PROGRAM_SHA256_CHECK_WHEN_PRIVACY_PRESERVING_SELECTION_OMITS_PRE_OUTCOME_PROGRAM_HASH",
                "provider_or_grounding_semantics_changed": False,
                "formal_prediction_or_gate_semantics_changed": False,
                "target_outcome_read_by_wrapper": False,
            }
            receipt_body["receipt_sha256"] = hashlib.sha256(
                json.dumps(receipt_body, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            transport_receipt = (
                args.output.resolve().parent
                / "postruntime_missing_program_hash_transport_fix.json"
            )
            transport_receipt.write_text(
                json.dumps(receipt_body, indent=2, sort_keys=True) + "\n"
            )
        command = [
            sys.executable, str(execution_collector),
            "--config", str(runtime_root / config_path.relative_to(REPO_ROOT)),
            "--keys", str(args.keys.resolve()),
            "--output", str(args.output.resolve()),
            "--workers", str(args.workers),
        ]
        if args.preflight_only:
            command = [sys.executable, str(runtime_root / COLLECTOR), "--help"]
        completed = subprocess.run(command, cwd=runtime_root, check=False)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
