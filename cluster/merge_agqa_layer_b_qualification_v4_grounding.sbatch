#!/usr/bin/env bash
#SBATCH --job-name=agqa-lb-v4-merge
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/agqa2_layer_b_raw_video_v1/qualification_v4/merge-%j.out

set -Eeuo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosplay-candy-a100
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"

Q48=()
Q96=()
SF=()
for SHARD in $(seq 0 7); do
  Q48+=(--shard "runs/agqa2_layer_b_raw_video_v1/qualification_v4/qwen32_grounding_48f_shard${SHARD}_v4.json")
  Q96+=(--shard "runs/agqa2_layer_b_raw_video_v1/qualification_v4/qwen32_grounding_96f_shard${SHARD}_v4.json")
  SF+=(--shard "runs/agqa2_layer_b_raw_video_v1/qualification_v4/slowfast_dense_shard${SHARD}_v4.json")
done

python scripts/merge_agqa_layer_b_grounding_shards.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/qualification_v4/public_cohort.json \
  "${Q48[@]}" \
  --output runs/agqa2_layer_b_raw_video_v1/qualification_v4/qwen32_grounding_48f_full512_v4.json
python scripts/merge_agqa_layer_b_grounding_shards.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/qualification_v4/public_cohort.json \
  "${Q96[@]}" \
  --output runs/agqa2_layer_b_raw_video_v1/qualification_v4/qwen32_grounding_96f_full512_v4.json
python scripts/merge_agqa_layer_b_action_probe_shards.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/qualification_v4/public_cohort.json \
  "${SF[@]}" \
  --output runs/agqa2_layer_b_raw_video_v1/qualification_v4/slowfast_dense_full512_v4.json

python scripts/merge_agqa_layer_b_charades_actions.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/qualification_v4/public_cohort.json \
  --base-grounding runs/agqa2_layer_b_raw_video_v1/qualification_v4/qwen32_grounding_48f_full512_v4.json \
  --action-probe runs/agqa2_layer_b_raw_video_v1/qualification_v4/slowfast_dense_full512_v4.json \
  --threshold 0.1 \
  --output runs/agqa2_layer_b_raw_video_v1/qualification_v4/composite_qwen32_48f_slowfast_all_full512_v4.json
python scripts/rebase_agqa_layer_b_charades_actions.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/qualification_v4/public_cohort.json \
  --base-grounding runs/agqa2_layer_b_raw_video_v1/qualification_v4/qwen32_grounding_48f_full512_v4.json \
  --frozen-composite runs/agqa2_layer_b_raw_video_v1/qualification_v4/composite_qwen32_48f_slowfast_all_full512_v4.json \
  --slowfast-policy fill_missing \
  --output runs/agqa2_layer_b_raw_video_v1/qualification_v4/composite_qwen32_48f_slowfast_fill_missing_full512_v4.json

python scripts/merge_agqa_layer_b_charades_actions.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/qualification_v4/public_cohort.json \
  --base-grounding runs/agqa2_layer_b_raw_video_v1/qualification_v4/qwen32_grounding_96f_full512_v4.json \
  --action-probe runs/agqa2_layer_b_raw_video_v1/qualification_v4/slowfast_dense_full512_v4.json \
  --threshold 0.1 \
  --output runs/agqa2_layer_b_raw_video_v1/qualification_v4/composite_qwen32_96f_slowfast_all_full512_v4.json
python scripts/rebase_agqa_layer_b_charades_actions.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/qualification_v4/public_cohort.json \
  --base-grounding runs/agqa2_layer_b_raw_video_v1/qualification_v4/qwen32_grounding_96f_full512_v4.json \
  --frozen-composite runs/agqa2_layer_b_raw_video_v1/qualification_v4/composite_qwen32_96f_slowfast_all_full512_v4.json \
  --slowfast-policy fill_missing \
  --output runs/agqa2_layer_b_raw_video_v1/qualification_v4/composite_qwen32_96f_slowfast_fill_missing_full512_v4.json

python scripts/route_agqa_layer_b_shared_grounding.py \
  --cohort runs/agqa2_layer_b_raw_video_v1/qualification_v4/public_cohort.json \
  --semantic-runtime runs/agqa2_layer_b_raw_video_v1/qualification_v4/semantic_runtime_repaired.json \
  --candidate runs/agqa2_layer_b_raw_video_v1/qualification_v4/composite_qwen32_48f_slowfast_fill_missing_full512_v4.json \
  --candidate runs/agqa2_layer_b_raw_video_v1/qualification_v4/composite_qwen32_96f_slowfast_fill_missing_full512_v4.json \
  --shared-action-frame-budget 320 \
  --output runs/agqa2_layer_b_raw_video_v1/qualification_v4/routed_qwen32_48f_96f_slowfast_fill_missing_full512_v4.json
