#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.typed_source_tasks import (
    TypedEffect,
    TypedSourceTask,
    collect_suite,
    summarize_edge_replication_gate,
    summarize_effect_value_gate,
)


def main(
    default_config: Path = Path("configs/typed_multisource_v3.json"),
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--groups-per-effect", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tasks = tuple(
        TypedSourceTask(
            task_id=str(item["task_id"]),
            environment_id=str(item["environment_id"]),
            required_effects=tuple(
                TypedEffect(str(effect)) for effect in item["required_effects"]
            ),
            max_depth=int(item["max_depth"]),
            max_states=int(item["max_states"]),
            collector_kind=str(item.get("collector_kind", "minigrid_bfs")),
        )
        for item in config["tasks"]
    )
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    groups_per_effect = (
        args.groups_per_effect
        if args.groups_per_effect is not None
        else int(config["groups_per_effect"])
    )
    report = collect_suite(
        tasks,
        seeds=seeds,
        namespace=str(config["namespace"]),
        groups_per_effect=groups_per_effect,
    )
    edge_requirements = config.get("source_gate", {}).get(
        "required_replicated_edges", []
    )
    task_families = {
        str(item["task_id"]): str(item["simulator_family"])
        for item in config["tasks"]
        if "simulator_family" in item
    }
    if edge_requirements:
        edge_gate = summarize_edge_replication_gate(
            report.get("effect_ir", {}),
            edge_requirements,
            task_families=task_families,
        )
        report["edge_replication_gate"] = edge_gate
    value_requirements = config.get("source_gate", {}).get(
        "required_officially_valued_effects", []
    )
    if value_requirements:
        report["effect_value_gate"] = summarize_effect_value_gate(
            report["collections"],
            value_requirements,
            task_families=task_families,
        )
    subordinate_statuses = [report["gate"]["status"]]
    if "edge_replication_gate" in report:
        subordinate_statuses.append(report["edge_replication_gate"]["status"])
    if "effect_value_gate" in report:
        subordinate_statuses.append(report["effect_value_gate"]["status"])
    if len(subordinate_statuses) > 1:
        report["overall_status"] = (
            "SOURCE_TYPED_GATE_PASSED"
            if all(status.endswith("_PASSED") for status in subordinate_statuses)
            else "SOURCE_TYPED_GATE_FAILED"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report.get("overall_status", report["gate"]["status"]),
        "config": str(args.config),
        "output": str(args.output),
        "effect_ir_sha256": report.get("effect_ir", {}).get("ir_sha256"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
