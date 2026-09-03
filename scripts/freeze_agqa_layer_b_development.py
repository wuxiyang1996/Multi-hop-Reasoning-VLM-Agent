#!/usr/bin/env python3
"""Freeze a balanced, already-video-consumed Layer-B development cohort."""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def prior_videos(repo: Path) -> set[str]:
    values=set()
    for raw in glob.glob(str(repo/"runs/agqa2*/runtime_receipts/*.json")):
        try: row=json.loads(Path(raw).read_text())
        except Exception: continue
        if isinstance(row.get("video_id"),str): values.add(row["video_id"])
    return values


def rank(task_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{task_id}".encode()).hexdigest()


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--semantic-data",type=Path,required=True)
    parser.add_argument("--video-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--per-root",type=int,default=4)
    parser.add_argument("--rank-salt",default="agqa-layer-b-balanced-development-v1")
    parser.add_argument("--exclude-cohort",type=Path,nargs="+")
    parser.add_argument(
        "--balanced-matching", action="store_true",
        help=("Use deterministic bipartite capacity matching instead of the "
              "legacy root-ordered greedy selector."),
    )
    args=parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError("Layer-B development freeze is immutable")
    repo=Path(__file__).resolve().parents[1]; exposed=prior_videos(repo); candidates=defaultdict(list)
    excluded_tasks: set[str]=set(); excluded_videos: set[str]=set()
    for excluded_path in args.exclude_cohort or ():
        excluded=json.loads(excluded_path.read_text())
        excluded_tasks.update(str(row["task_id"]) for row in excluded["rows"])
        excluded_videos.update(str(row["video_id"]) for row in excluded["rows"])
    with args.semantic_data.open() as handle:
        for line in handle:
            row=json.loads(line); video=str(row["video_id"]); path=args.video_root/f"{video}.mp4"
            if (video not in exposed or not path.exists() or str(row["task_id"]) in excluded_tasks
                    or video in excluded_videos): continue
            root=str(row["target"]).split("(",1)[0]
            candidates[root].append((rank(str(row["task_id"]),args.rank_salt),row,path))
    selected=[]; used_videos=set()
    if args.balanced_matching:
        # Collapse multiple questions for the same (root, video) to the
        # lowest salted task rank.  Assignment then operates only on public
        # identifiers and never reads answers or semantic targets beyond the
        # already-declared root stratum.
        best_by_root_video: dict[str, dict[str, tuple]] = {}
        for root, rows in candidates.items():
            by_video = {}
            for item in sorted(rows):
                by_video.setdefault(str(item[1]["video_id"]), item)
            best_by_root_video[root] = by_video

        # Deterministic maximum bipartite matching with one node per root
        # capacity slot.  Scarcer roots are visited first; augmenting paths
        # prevent early roots from unnecessarily consuming shared videos.
        slots = [
            (root, index)
            for root in sorted(best_by_root_video,
                               key=lambda x: (len(best_by_root_video[x]), x))
            for index in range(args.per_root)
        ]
        choices = {
            root: [video for video, _ in sorted(
                best_by_root_video[root].items(), key=lambda item: item[1][0])]
            for root in best_by_root_video
        }
        video_to_slot: dict[str, tuple[str, int]] = {}

        def assign(slot: tuple[str, int], seen: set[str]) -> bool:
            root, _ = slot
            for video in choices[root]:
                if video in seen:
                    continue
                seen.add(video)
                previous = video_to_slot.get(video)
                if previous is None or assign(previous, seen):
                    video_to_slot[video] = slot
                    return True
            return False

        for slot in slots:
            if not assign(slot, set()):
                raise RuntimeError(
                    f"insufficient disjoint videos for balanced slot {slot}; "
                    "no answers or outcomes were read"
                )
        slot_to_video = {slot: video for video, slot in video_to_slot.items()}
        for slot in sorted(slots):
            root, _ = slot
            video = slot_to_video[slot]
            digest, row, path = best_by_root_video[root][video]
            selected.append((root, digest, row, path))
            used_videos.add(video)
    else:
        for root in sorted(candidates):
            kept=0
            for digest,row,path in sorted(candidates[root]):
                if row["video_id"] in used_videos: continue
                selected.append((root,digest,row,path)); used_videos.add(row["video_id"]); kept+=1
                if kept==args.per_root: break
            if kept!=args.per_root: raise RuntimeError(f"insufficient already-consumed local videos for root {root}")
    args.output_dir.mkdir(parents=True)
    public={"schema_version":"agqa-layer-b-development-public-v1","status":"FROZEN_DEVELOPMENT_PUBLIC",
            "rows":[{"task_id":r["task_id"],"video_id":r["video_id"],
                     "question":str(r["input"]).removeprefix("parse AGQA semantics: "),
                     "question_sha256":stable_hash(str(r["input"]).removeprefix("parse AGQA semantics: ")),
                     "video_path":str(path),"rank_sha256":digest}
                    for _,digest,r,path in selected],
            "answers_read":False,"scene_graphs_read":False,"functional_program_visible_at_runtime":False}
    public["cohort_sha256"]=stable_hash(public)
    private={"schema_version":"agqa-layer-b-development-private-audit-v1",
             "public_cohort_sha256":public["cohort_sha256"],
             "rows":[{"task_id":r["task_id"],"semantic_root":root,
                      "gold_semantic_target_sha256":stable_hash(r["target"])} for root,_,r,_ in selected],
             "answers_read":False,"scene_graphs_read":False}
    private["audit_sha256"]=stable_hash(private)
    manifest={"schema_version":"agqa-layer-b-development-freeze-v1","status":"FROZEN_BEFORE_PARSER_GROUNDER_OR_OUTCOME",
              "rows":len(selected),"videos":len(used_videos),"per_semantic_root":args.per_root,
              "semantic_roots":sorted(candidates),"all_videos_historically_raw_exposed":True,
              "rank_salt":args.rank_salt,"excluded_cohorts":[str(x) for x in (args.exclude_cohort or ())],
              "selection_algorithm":("DETERMINISTIC_BIPARTITE_CAPACITY_MATCHING_V1"
                                     if args.balanced_matching else "LEGACY_ROOT_ORDERED_GREEDY"),
              "excluded_task_count":len(excluded_tasks),"excluded_video_count":len(excluded_videos),
              "public_cohort_sha256":public["cohort_sha256"],"private_audit_sha256":private["audit_sha256"],
              "answers_read":False,"scene_graphs_read":False,"provider_calls":0}
    manifest["manifest_sha256"]=stable_hash(manifest)
    (args.output_dir/"public_cohort.json").write_text(json.dumps(public,indent=2,sort_keys=True)+"\n")
    (args.output_dir/"private_semantic_audit.json").write_text(json.dumps(private,indent=2,sort_keys=True)+"\n")
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(manifest,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
