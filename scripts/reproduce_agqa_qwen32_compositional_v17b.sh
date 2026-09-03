#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-$(pwd)}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}"

python scripts/build_agqa_offtheshelf_qwen32_formal_v17_paper_bundle.py \
  --formal-evaluation runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_formal_evaluation_v17b.json \
  --cohort runs/agqa2_offtheshelf_qwen32_formal_v17b/public_cohort.json \
  --manifest runs/agqa2_offtheshelf_qwen32_formal_v17b/manifest.json \
  --protocol runs/agqa2_offtheshelf_qwen32_formal_protocol_v17b/protocol.json \
  --semantic-runtime runs/agqa2_offtheshelf_qwen32_formal_v17b/semantic_runtime.json \
  --download-receipt runs/agqa2_offtheshelf_qwen32_formal_v17b/download_receipt.json \
  --grounding-view runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_48f_full_v17b.json \
  --grounding-view runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_96f_full_v17b.json \
  --routed-grounding runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_48f_96f_routed_v17b.json \
  --fallback runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_shared_fallback_qwen9b_v17b.json \
  --preoutcome runs/agqa2_offtheshelf_qwen32_formal_v17b/qwen32_five_arm_preoutcome_v17b.json \
  --source-capabilities runs/agqa2_full_operator_transfer_v1/source_capabilities_v2.json \
  --anonymous-controller runs/anonymous_video_harness_v1/controller.json \
  --development-evaluation runs/agqa2_offtheshelf_compositional_development_v15/qwen32_development_evaluation_v18b.json \
  --slowfast-development-evaluation runs/agqa2_offtheshelf_compositional_development_v15/development_evaluation_v15c.json \
  --output docs/results/agqa_qwen32_compositional_formal_v17b.json \
  --verify-existing

python scripts/build_two_video_transfer_bundle_v3.py \
  --clevrer-formal runs/clevrer_full_raw_video_v2/formal_report.json \
  --clevrer-substitution runs/clevrer_full_raw_video_v2/anonymous_harness_substitution_v1.json \
  --agqa-bundle docs/results/agqa_qwen32_compositional_formal_v17b.json \
  --anonymous-controller runs/anonymous_video_harness_v1/controller.json \
  --output docs/results/two_video_transfer_bundle_v3.json \
  --verify-existing

pytest -q \
  tests/test_agqa_layer_b_executor_v3.py \
  tests/test_agqa_layer_b_executor.py \
  tests/test_agqa_offtheshelf_compositional_grounder_v15.py \
  tests/test_agqa_action_genome_query_compiler.py \
  tests/test_agqa_query_conditioned_typed_binding_v3.py \
  tests/test_agqa_strict_temporal_projection.py \
  tests/test_agqa_query_grounder_v2.py \
  tests/test_evaluate_agqa_query_grounder_v2_fresh_formal.py
