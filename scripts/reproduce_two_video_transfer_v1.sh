#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$(pwd)}"
clevrer_annotations="${2:-/fs/gamma-projects/vlm-robot/datasets/CLEVRER-official/executor/data/validation.json}"

cd "$repo_root"

python scripts/induce_source_video_operator_algebra_v1.py \
  --root "$repo_root" \
  --catalog configs/full_video_source_catalog_v1.json \
  --output runs/full_video_source_algebra_v1/source_algebra.json

python scripts/compile_anonymous_video_harness_v1.py --root "$repo_root"
python scripts/audit_clevrer_anonymous_harness_substitution_v1.py --root "$repo_root"
python scripts/audit_agqa_anonymous_harness_substitution_v1.py --root "$repo_root"

python scripts/evaluate_clevrer_five_arm_predictions_v1.py \
  --predictions runs/clevrer_full_raw_video_v2/five_arm_predictions.json \
  --preregistration configs/clevrer_full_raw_video_v2_preregistration.json \
  --annotations "$clevrer_annotations" \
  --output runs/clevrer_full_raw_video_v2/formal_report.json

python scripts/analyze_clevrer_v2_failure_taxonomy.py \
  --predictions runs/clevrer_full_raw_video_v2/five_arm_predictions.json \
  --annotations "$clevrer_annotations" \
  --actor runs/clevrer_full_raw_video_v2/qwen9b_graph_actor.json \
  --output runs/clevrer_full_raw_video_v2/failure_taxonomy_v1.json

python scripts/audit_two_video_transfer_bundle_v1.py --root "$repo_root"

pytest -q \
  tests/test_source_video_operator_algebra.py \
  tests/test_video_target_signature_binding.py \
  tests/test_anonymous_video_harness.py \
  tests/test_audit_clevrer_anonymous_harness_substitution.py \
  tests/test_audit_agqa_anonymous_harness_substitution.py
