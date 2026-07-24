#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.target_feasibility import CellAudit, write_summary


DEFAULT_ROOT = Path("/fs/gamma-projects/vlm-robot")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _browser_stats(root: Path, prefix: str) -> tuple[int, int, int]:
    task_dirs = sorted(path for path in root.glob(f"{prefix}*") if path.is_dir())
    episode_files = [path for task in task_dirs for path in task.glob("episode_*.json")]
    reward_rows = 0
    for path in episode_files:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        values = []
        for key in ("reward", "final_reward", "total_reward"):
            if isinstance(row.get(key), (int, float)):
                values.append(float(row[key]))
        for step in row.get("experiences", row.get("steps", [])) or []:
            if isinstance(step, dict) and isinstance(step.get("reward"), (int, float)):
                values.append(float(step["reward"]))
        reward_rows += bool(values)
    return len(task_dirs), len(episode_files), reward_rows


def _source_audit(legacy_repo: Path) -> dict[str, object]:
    phase1 = {
        "candy_crush",
        "tetris",
        "gymv_columns",
        "gymv_streets_of_rage_2",
        "gymv_strider",
        "gymv_thunder_force_iii",
    }
    mega_path = legacy_repo / "frontier_data/output/megaskills_all_stages/mega_skills.jsonl"
    aligned_path = legacy_repo / "frontier_data/output/reasoning_aligned_mega_skills.json"
    mega_rows = [json.loads(line) for line in mega_path.read_text().splitlines() if line.strip()]
    all_members = [member for row in mega_rows for member in row.get("members", [])]
    strict_members = [member for member in all_members if member.get("task") in phase1]
    non_game_members = [member for member in all_members if member.get("domain") != "GAME"]
    aligned = json.loads(aligned_path.read_text())
    aligned_rows = aligned if isinstance(aligned, list) else aligned.get("mega_skills", aligned.get("skills", []))
    aligned_target_members = 0
    for row in aligned_rows:
        by_domain = row.get("members_by_domain", {})
        aligned_target_members += sum(
            len(members) for domain, members in by_domain.items() if domain in {"WEB", "VIDEO", "VISUAL", "ALFWORLD"}
        )
    return {
        "authority": "LINEAGE_RETRIEVAL_ONLY",
        "phase1_allowlist": sorted(phase1),
        "legacy_mega_families": len(mega_rows),
        "legacy_mega_members": len(all_members),
        "strict_phase1_members": len(strict_members),
        "non_game_members_in_all_stages": len(non_game_members),
        "reasoning_aligned_rows": len(aligned_rows),
        "target_members_in_reasoning_aligned": aligned_target_members,
        "raw_reasoning_aligned_allowed_as_source": False,
        "game_only_filter_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=REPO / "docs/results/target_feasibility_audit_v1.json")
    parser.add_argument("--verify-vtb-sha256", action="store_true")
    args = parser.parse_args()
    root = args.workspace_root
    datasets = root / "datasets"
    browser_root = root / "emnlp2026_download/workspace/main_project/Cold-start-out-browsergym"
    legacy = root / "Multi-hop-Reasoning-VLM-Agent-github-main"

    vtb = datasets / "VisualToolBench/test.parquet"
    vtb_manifest = datasets / "VisualToolBench/manifest.json"
    vtb_count = None
    if vtb_manifest.exists():
        vtb_count = json.loads(vtb_manifest.read_text()).get("counts", {}).get("total")
    vtb_evidence = [f"manifest_rows={vtb_count}", f"parquet_bytes={vtb.stat().st_size if vtb.exists() else 0}"]
    if args.verify_vtb_sha256 and vtb.exists():
        vtb_evidence.append(f"sha256={_sha256(vtb)}")

    tir_root = datasets / "TIR-Bench"
    tir_json = tir_root / "TIR-Bench.json"
    tir_rows = json.loads(tir_json.read_text()) if tir_json.exists() else []
    tir_refs = [row[key] for row in tir_rows for key in ("image_1", "image_2") if row.get(key)]
    tir_missing = [ref for ref in tir_refs if not (tir_root / ref).exists()]

    vh_root = datasets / "Video-Holmes/Benchmark"
    videos = {path.stem for path in vh_root.rglob("*.mp4")}
    vh_rows = []
    for split in ("train", "test"):
        path = vh_root / f"{split}_Video-Holmes.json"
        if path.exists():
            vh_rows.extend(json.loads(path.read_text()))
    vh_ids = {row["video ID"] for row in vh_rows}

    mw_tasks, mw_episodes, mw_rewards = _browser_stats(browser_root, "miniwob.")
    ws_runs = sorted(browser_root.glob("webshop_50task_*"))
    ws_task_keys = set()
    ws_episode_count = 0
    ws_reward_count = 0
    for run in ws_runs:
        tasks, episodes, rewards = _browser_stats(run, "webshop.")
        ws_task_keys.update(path.name for path in run.glob("webshop.*") if path.is_dir())
        ws_episode_count += episodes
        ws_reward_count += rewards

    alfworld_data = legacy / ".cache/alfworld_data"
    alfworld_adapter = REPO / "src/motif_transfer/alfworld_env.py"
    alfworld_adaptation = legacy / "artifacts/admission_demos/alfworld/pick_and_place/train_seed42_v3_shot0.json"
    frozen_manifest = REPO / "configs/target_manifests_v1.json"
    frozen_cells = {}
    if frozen_manifest.exists():
        frozen_cells = json.loads(frozen_manifest.read_text()).get("cells", {})
    seen_adaptation = REPO / "runs/target_feasibility_v1/adaptation/alfworld_valid_seen.json"
    unseen_adaptation = REPO / "runs/target_feasibility_v1/adaptation/alfworld_valid_unseen.json"
    miniwob_html = root / "miniwob-plusplus/miniwob/html/miniwob/click-button.html"
    vtb_runner = REPO / "scripts/run_vtb_real_baseline.py"
    tir_runner = REPO / "scripts/run_tir_real_baseline.py"
    video_runner = REPO / "scripts/run_video_holmes_real_baseline.py"
    browser_runner = REPO / "scripts/run_browsergym_real_baseline.py"
    webshop_vendor = root / "emnlp2026_download/workspace/vendor/WebShop"
    webshop_wrapper = root / "emnlp2026_download/workspace/main_project_src/webshop_wrapper"

    # These fail-closed target-only runners never invoke the permissive legacy
    # dispatchers. A cell can be mechanically runnable even when its smoke
    # policy fails; task outcome and receipt integrity are reported separately.
    rows = (
        CellAudit("visual_reasoning", "visual_toolbench", vtb.exists() and vtb_count == 1204,
                  False, vtb_runner.exists(), "visual_toolbench" in frozen_cells,
                  "visual_toolbench" in frozen_cells, False, False,
                  tuple(vtb_evidence + [
                      "official judge exists in xi1ngang/VisualToolBench@d4f200a but is not integrated",
                      "official APR uses weight>=4, not the dataset critical field",
                  ])),
        CellAudit("visual_reasoning", "tir_bench", bool(tir_rows) and not tir_missing,
                  True, tir_runner.exists(), "tir_bench" in frozen_cells,
                  "tir_bench" in frozen_cells, False, False,
                  (f"rows={len(tir_rows)}", f"image_refs={len(tir_refs)}", f"missing_images={len(tir_missing)}",
                   "real smoke exposed fabricated evidence and coordinate grounding failure")),
        CellAudit("video", "video_holmes", bool(vh_rows) and vh_ids <= videos,
                  True, video_runner.exists(), "video_holmes" in frozen_cells,
                  "video_holmes" in frozen_cells, False, False,
                  (f"questions={len(vh_rows)}", f"unique_video_ids={len(vh_ids)}", f"available_videos={len(videos)}",
                   "real smoke produced 34 tool receipts but exhausted budget without final answer")),
        CellAudit("browser", "miniwob", mw_tasks > 0 and mw_episodes > 0,
                  mw_rewards > 0, browser_runner.exists() and miniwob_html.exists(),
                  "miniwob" in frozen_cells, "miniwob" in frozen_cells, False, False,
                  (f"tasks={mw_tasks}", f"episodes={mw_episodes}", f"episodes_with_official_reward={mw_rewards}",
                   "live frozen smoke succeeded in one step with official reward=1")),
        CellAudit("browser", "webshop", bool(ws_task_keys) and ws_episode_count > 0,
                  ws_reward_count > 0,
                  browser_runner.exists() and (webshop_vendor / "web_agent_site/app.py").exists()
                  and (webshop_wrapper / "task.py").exists(),
                  "webshop" in frozen_cells, "webshop" in frozen_cells, False, False,
                  (f"historical_runs={len(ws_runs)}", f"unique_tasks={len(ws_task_keys)}",
                   f"episodes={ws_episode_count}", f"episodes_with_official_reward={ws_reward_count}",
                   "full 1k-product server and bridge restored; frozen live smoke official reward=0.6667")),
        CellAudit("alfworld", "alfworld_valid_seen", alfworld_data.exists(), True,
                  alfworld_adapter.exists(), seen_adaptation.exists(), "alfworld_valid_seen" in frozen_cells, False, False,
                  ("real text environment and official won signal are implemented",
                   f"frozen_adaptation={seen_adaptation}", f"frozen_manifest={frozen_manifest}")),
        CellAudit("alfworld", "alfworld_valid_unseen", alfworld_data.exists(), True,
                  alfworld_adapter.exists(), unseen_adaptation.exists(), "alfworld_valid_unseen" in frozen_cells, False, False,
                  ("real OOD runs and frozen binding artifacts exist", f"frozen_adaptation={unseen_adaptation}",
                   f"legacy_adaptation_reference={alfworld_adaptation}")),
    )
    payload = write_summary(args.output, rows)
    payload["source_treatment_audit"] = _source_audit(legacy)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
