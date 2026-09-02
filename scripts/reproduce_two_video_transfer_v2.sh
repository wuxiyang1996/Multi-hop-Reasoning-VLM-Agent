#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$(pwd)}"
cd "$repo_root"

python scripts/build_two_video_transfer_bundle_v2.py \
  --clevrer-formal runs/clevrer_full_raw_video_v2/formal_report.json \
  --clevrer-substitution runs/clevrer_full_raw_video_v2/anonymous_harness_substitution_v1.json \
  --clevrer-taxonomy runs/clevrer_full_raw_video_v2/failure_taxonomy_v1.json \
  --agqa-formal runs/agqa2_full_train_broad_powered_v4/formal_evaluation.json \
  --agqa-cohort runs/agqa2_full_train_broad_powered_v4/parser_qualified_reserve/public_cohort.json \
  --agqa-manifest runs/agqa2_full_train_broad_powered_v4/parser_qualified_reserve/manifest.json \
  --agqa-grounding runs/agqa2_full_train_broad_powered_v4/qwen32_grounding_full1790.json \
  --agqa-claims runs/agqa2_full_train_broad_powered_v4/atomic_claims_full1790.json \
  --agqa-fallback runs/agqa2_full_train_broad_powered_v4/shared_fallback_full1790.json \
  --agqa-preoutcome runs/agqa2_full_train_broad_powered_v4/preoutcome_receipt.json \
  --agqa-runtime-freeze configs/agqa2_full_train_broad_powered_v4_runtime_freeze.json \
  --anonymous-controller runs/anonymous_video_harness_v1/controller.json \
  --output docs/results/two_video_transfer_bundle_v2.json \
  --verify-existing

pytest -q \
  tests/test_source_video_operator_algebra.py \
  tests/test_video_target_signature_binding.py \
  tests/test_anonymous_video_harness.py \
  tests/test_agqa_layer_b_epistemic.py \
  tests/test_agqa_layer_b_executor.py \
  tests/test_agqa_layer_b_harness.py
