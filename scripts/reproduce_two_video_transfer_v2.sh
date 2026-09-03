#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$(pwd)}"
repo_root="$(cd "$repo_root" && pwd)"
cd "$repo_root"

agqa_root="$repo_root"
audit_tmp=""
source_inputs_extracted=0
agqa_archive="$repo_root/artifacts/video_transfer_v2/agqa2_full_train_broad_powered_v4_audit.tar.gz"
agqa_archive_sha256="ff5929bb549e9bffc778818acf7cf7d3acd4e0bcb3aee7f828ffcc8fc648aed5"
source_archive="$repo_root/artifacts/video_transfer_v2/source_induction_audit_inputs_v1.tar.gz"
source_archive_sha256="27176a4833de1a4b0ca9b599d95fbc07e9ea3e9c2f07b1506a0cdf45c2af3720"
cleanup() {
  if [[ -n "$audit_tmp" ]]; then
    rm -rf -- "$audit_tmp"
  fi
  if [[ "$source_inputs_extracted" == 1 ]]; then
    rm -rf -- \
      "$repo_root/runs/phase3_source_function_v4_reserve" \
      "$repo_root/runs/agqa2_full_operator_transfer_v1"
  fi
}
trap cleanup EXIT
if [[ ! -f "$repo_root/runs/agqa2_full_train_broad_powered_v4/formal_evaluation.json" ]]; then
  if [[ ! -f "$agqa_archive" ]]; then
    echo "missing portable AGQA audit archive: $agqa_archive" >&2
    exit 1
  fi
  printf '%s  %s\n' "$agqa_archive_sha256" "$agqa_archive" | sha256sum --check --status
  audit_tmp="$(mktemp -d /tmp/two-video-transfer-v2-XXXXXX)"
  tar -xzf "$agqa_archive" -C "$audit_tmp"
  agqa_root="$audit_tmp"
fi

if [[ ! -f "$repo_root/runs/phase3_source_function_v4_reserve/report.json" ]]; then
  if [[ ! -f "$source_archive" ]]; then
    echo "missing portable source-induction archive: $source_archive" >&2
    exit 1
  fi
  for path in \
    "$repo_root/runs/phase3_source_function_v4_reserve" \
    "$repo_root/runs/agqa2_full_operator_transfer_v1"; do
    if [[ -e "$path" ]]; then
      echo "refusing partial portable extraction over existing path: $path" >&2
      exit 1
    fi
  done
  printf '%s  %s\n' "$source_archive_sha256" "$source_archive" | sha256sum --check --status
  source_inputs_extracted=1
  tar -xzf "$source_archive" -C "$repo_root"
fi

python scripts/build_two_video_transfer_bundle_v2.py \
  --clevrer-formal "$repo_root/runs/clevrer_full_raw_video_v2/formal_report.json" \
  --clevrer-substitution "$repo_root/runs/clevrer_full_raw_video_v2/anonymous_harness_substitution_v1.json" \
  --clevrer-taxonomy "$repo_root/runs/clevrer_full_raw_video_v2/failure_taxonomy_v1.json" \
  --agqa-formal "$agqa_root/runs/agqa2_full_train_broad_powered_v4/formal_evaluation.json" \
  --agqa-cohort "$agqa_root/runs/agqa2_full_train_broad_powered_v4/parser_qualified_reserve/public_cohort.json" \
  --agqa-manifest "$agqa_root/runs/agqa2_full_train_broad_powered_v4/parser_qualified_reserve/manifest.json" \
  --agqa-grounding "$agqa_root/runs/agqa2_full_train_broad_powered_v4/qwen32_grounding_full1790.json" \
  --agqa-claims "$agqa_root/runs/agqa2_full_train_broad_powered_v4/atomic_claims_full1790.json" \
  --agqa-fallback "$agqa_root/runs/agqa2_full_train_broad_powered_v4/shared_fallback_full1790.json" \
  --agqa-preoutcome "$agqa_root/runs/agqa2_full_train_broad_powered_v4/preoutcome_receipt.json" \
  --agqa-runtime-freeze "$repo_root/configs/agqa2_full_train_broad_powered_v4_runtime_freeze.json" \
  --anonymous-controller "$repo_root/runs/anonymous_video_harness_v1/controller.json" \
  --output "$repo_root/docs/results/two_video_transfer_bundle_v2.json" \
  --verify-existing

PYTHONPATH="$repo_root/src:$repo_root" pytest -q \
  tests/test_source_video_operator_algebra.py \
  tests/test_video_target_signature_binding.py \
  tests/test_anonymous_video_harness.py \
  tests/test_agqa_layer_b_epistemic.py \
  tests/test_agqa_layer_b_executor.py \
  tests/test_agqa_layer_b_harness.py
