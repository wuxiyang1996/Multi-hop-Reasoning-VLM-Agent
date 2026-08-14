#!/usr/bin/env python3
"""Apply frozen DiscoveryWorld gates with replication-specific disclosure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.discoveryworld_env import stable_hash  # noqa: E402
from scripts.summarize_discoveryworld_v23_formal import (  # noqa: E402
    file_sha256,
    read,
    summarize,
)


def frozen_runtime_hashes_match(protocol: dict, results: dict) -> bool:
    expected_runtime = {
        "runner": protocol["integrity"]["matched_runner_sha256"],
        "environment": protocol["integrity"]["environment_wrapper_sha256"],
        "target_policy": protocol["integrity"]["target_policy_sha256"],
        "transfer_selector": protocol["integrity"]["transfer_selector_sha256"],
    }
    return all(
        all(row.get("runtime_hashes", {}).get(key) == value
            for key, value in expected_runtime.items())
        for row in results.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--fork-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    freeze_path = args.fork_dir / "fork_freeze_receipt.json"
    freeze = read(freeze_path)
    results = {}
    for generated in freeze["generated_configs"]:
        task_id = Path(str(generated)).stem
        path = args.result_dir / f"{task_id}.json"
        if not path.is_file():
            path = args.result_dir / task_id / f"{task_id}.json"
        value = read(path)
        value["_path"] = str(path)
        results[task_id] = value
    report = summarize(
        protocol=protocol, freeze=freeze, results=results,
        protocol_file_sha256=file_sha256(args.protocol),
        freeze_file_sha256=file_sha256(freeze_path),
    )
    runtime_hashes_match = frozen_runtime_hashes_match(protocol, results)
    report["gates"]["frozen_runtime_hashes_match_protocol"] = runtime_hashes_match
    report["all_predeclared_gates_passed"] = all(report["gates"].values())
    report["status"] = (
        "FRESH_FORMAL_TRANSFER_VALIDATED"
        if report["all_predeclared_gates_passed"]
        else "FRESH_FORMAL_TRANSFER_FAILED"
    )
    report["operational_disclosure"] = {
        "target_baseline_dependency_recovery": (
            "Some scheduler attempts failed before reset while the pinned official "
            "environment dependencies were absent; they issued zero target decisions "
            "and were rerun under the unchanged config after dependency installation."
        ),
        "matched_execution": (
            "The first matched scheduler used a Python environment missing a "
            "dependency and failed before reset. A later scheduler produced six "
            "results with a post-freeze parser file hash; a lineage audit excluded "
            "that entire directory. The included matrix is a complete all-16 restart "
            "from detached commit e075f4e, whose runner, environment, target policy, "
            "and transfer-selector hashes exactly match the frozen protocol. No "
            "threshold, prompt, source artifact, target config, task, or condition "
            "changed. During that exact-code restart, the first parallel Space Sick "
            "launch exposed the known shared PNG-frame race before producing any "
            "Space Sick result. All six Space Sick configs were then run with "
            "task-isolated output/frame directories; the ten already-complete "
            "Proteomics results were not rerun."
        ),
        "frozen_text_typo": (
            "The inherited conclusion_rule says seeds5-10, but the frozen task_ids, "
            "manifest, and claim_boundary unambiguously define replication seeds11-20."
        ),
        "excluded_postfreeze_code_results": 6,
        "excluded_shared_frame_race_results": 0,
        "selective_scientific_retry": False,
        "scientific_configuration_changed_in_included_matrix": False,
    }
    report.pop("report_sha256", None)
    report["schema_version"] = "discoveryworld-sokoban-easy-replication-summary-v1"
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "eligible_forks": report["eligible_forks"],
        "success_counts": report["success_counts"], "gates": report["gates"],
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
