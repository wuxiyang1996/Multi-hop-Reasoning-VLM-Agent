#!/usr/bin/env bash
# ==============================================================================
# grade_ab_test161_all.sh — grade claude/gemini/qwen rollouts under /tmp/ab_test161
#
# Walks each model's run-dir, computes type-aware AB rewards, writes:
#   <run_dir>/grading_summary.json
#   <run_dir>/grading_summary.csv
#   <run_dir>/assistantbench_validation_score.json
#   <run_dir>/assistantbench_test_predictions.jsonl   (AB-server upload format)
#   <run_dir>/assistantbench_test_predictions_human.json
#
# Also tees a single combined snapshot to
#   ${REPO_ROOT}/cold_start/_ab_test161_grade.snapshot
# so the agent on the other host can read it from NFS.
#
# Safe to run mid-run: any not-yet-completed task is silently skipped.
#
# Usage:
#     bash cold_start/grade_ab_test161_all.sh [OUT_BASE]
# ==============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_BASE="${1:-/tmp/ab_test161}"
SNAPSHOT="${REPO_ROOT}/cold_start/_ab_test161_grade.snapshot"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"

set +u
source /fs/gamma-projects/vlm-robot/conda/etc/profile.d/conda.sh
conda activate "${BROWSERGYM_CONDA_ENV:-browsergym}"
set -u

# Tee everything to NFS snapshot for cross-host visibility.
exec > >(tee "$SNAPSHOT") 2>&1

echo "================================================================"
echo "  AssistantBench grading sweep"
echo "  OUT_BASE = $OUT_BASE"
echo "  Time     = $(date +%Y-%m-%dT%H:%M:%S)"
echo "================================================================"
echo

declare -a MODELS=(claude gemini qwen)

for tag in "${MODELS[@]}"; do
    run_dir="${OUT_BASE}/${tag}"
    if [[ ! -d "$run_dir" ]]; then
        echo "[skip] $tag  ($run_dir does not exist)"
        echo
        continue
    fi
    n_eps=$(find "$run_dir" -maxdepth 2 -name "episode_000.json" 2>/dev/null | wc -l)
    echo "================================================================"
    echo "[grade] $tag    run_dir=$run_dir    episodes_found=$n_eps"
    echo "================================================================"
    if [[ "$n_eps" -eq 0 ]]; then
        echo "  (no completed episodes yet — skipping)"
        echo
        continue
    fi
    python -u "${REPO_ROOT}/cold_start/grade_assistantbench_eval.py" \
        --run_dir "$run_dir" || echo "[warn] grading failed for $tag (exit $?)"
    echo
done

echo "================================================================"
echo "  ONE-LINE SUMMARY  (parsed from each model's grading_summary.json)"
echo "================================================================"
echo
printf "  %-8s  %-6s  %-16s  %-10s  %-10s  %-10s  %-8s\n" \
    "model" "n_test" "answered" "trunc" "infeasible" "mean_steps" "preds_jsonl"
echo "  ----------------------------------------------------------------------------------"

for tag in "${MODELS[@]}"; do
    run_dir="${OUT_BASE}/${tag}"
    gs="${run_dir}/grading_summary.json"
    pj="${run_dir}/assistantbench_test_predictions.jsonl"
    if [[ ! -f "$gs" ]]; then
        printf "  %-8s  %-6s  %-16s  %-10s  %-10s  %-10s  %-8s\n" \
            "$tag" "-" "-" "-" "-" "-" "(missing)"
        continue
    fi
    python - <<PY "$tag" "$gs" "$pj"
import json, sys, os
tag, gs, pj = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(gs))
ts = d["test_summary"]
n = ts.get("n", 0)
ans = ts.get("answered_count", 0)
trunc = ts.get("truncated_count", 0)
infe = ts.get("infeasible_count", 0)
ar = ts.get("answered_rate", 0.0)
# mean steps over all test rows
rows = [r for r in d["per_task"] if r["split"] == "test"]
mean_steps = sum(r["n_steps"] for r in rows) / max(len(rows), 1)
n_pred_lines = sum(1 for _ in open(pj)) if os.path.isfile(pj) else 0
print(f"  {tag:<8}  {n:<6}  {ans}/{n} ({100*ar:5.1f}%)   {trunc:<10}  {infe:<10}  {mean_steps:<10.1f}  {n_pred_lines}")
PY
done

echo
echo "================================================================"
echo "  AB leaderboard JSONLs written:"
for tag in "${MODELS[@]}"; do
    pj="${OUT_BASE}/${tag}/assistantbench_test_predictions.jsonl"
    if [[ -f "$pj" ]]; then
        echo "    ${tag}: $pj  ($(wc -l < "$pj") preds)"
    fi
done
echo "================================================================"
echo "  Snapshot: $SNAPSHOT"
echo "  Done."
