#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${1:-${CODE_ROOT}}"
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}/src:${CODE_ROOT}"

python scripts/build_agqa_offtheshelf_qwen32_formal_v17_paper_bundle.py \
  --formal-evaluation "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_formal_evaluation_v17b.json" \
  --cohort "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_qwen32_formal_v17b/public_cohort.json" \
  --manifest "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_qwen32_formal_v17b/manifest.json" \
  --protocol "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_qwen32_formal_protocol_v17b/protocol.json" \
  --semantic-runtime "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_qwen32_formal_v17b/semantic_runtime.json" \
  --download-receipt "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_qwen32_formal_v17b/download_receipt.json" \
  --grounding-view "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_48f_full_v17b.json" \
  --grounding-view "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_96f_full_v17b.json" \
  --routed-grounding "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_48f_96f_routed_v17b.json" \
  --fallback "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_shared_fallback_qwen9b_v17b.json" \
  --preoutcome "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_five_arm_preoutcome_v17b.json" \
  --source-capabilities "${ARTIFACT_ROOT}/runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json" \
  --anonymous-controller "${ARTIFACT_ROOT}/runs/anonymous_video_harness_v1/controller.json" \
  --development-evaluation "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_compositional_development_v15/qwen32_development_evaluation_v18b.json" \
  --slowfast-development-evaluation "${ARTIFACT_ROOT}/runs/agqa2_offtheshelf_compositional_development_v15/development_evaluation_v15c.json" \
  --artifact-label-root "${ARTIFACT_ROOT}" \
  --output docs/results/agqa_qwen32_compositional_formal_v17b.json \
  --verify-existing

python scripts/build_two_video_transfer_bundle_v3.py \
  --clevrer-formal "${ARTIFACT_ROOT}/runs/clevrer_full_raw_video_v2/formal_report.json" \
  --clevrer-substitution "${ARTIFACT_ROOT}/runs/clevrer_full_raw_video_v2/anonymous_harness_substitution_v1.json" \
  --agqa-bundle docs/results/agqa_qwen32_compositional_formal_v17b.json \
  --anonymous-controller "${ARTIFACT_ROOT}/runs/anonymous_video_harness_v1/controller.json" \
  --artifact-label-root "${ARTIFACT_ROOT}" \
  --artifact-label-root "${CODE_ROOT}" \
  --output docs/results/two_video_transfer_bundle_v3.json \
  --verify-existing

pytest -q \
  tests/test_agqa_layer_b_executor_v3.py \
  tests/test_agqa_layer_b_executor.py \
  tests/test_agqa_offtheshelf_compositional_grounder_v15.py \
  tests/test_evaluate_agqa_query_grounder_v2_fresh_formal.py
