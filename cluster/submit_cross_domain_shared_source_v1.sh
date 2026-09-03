#!/usr/bin/env bash
# Submit the shared cross-domain source collection declared in
# configs/cross_domain_shared_source_v1.json.
#
#   bash cluster/submit_cross_domain_shared_source_v1.sh smoke
#   bash cluster/submit_cross_domain_shared_source_v1.sh full
#
# smoke: one game, one episode, to prove the vLLM + multi-LoRA + game stack works.
# full:  six games x 16 episodes, %2 concurrency, resumable at game granularity.

set -Eeuo pipefail

CLEAN_REPO="${CLEAN_REPO:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean}"
MODE="${1:-}"
GRES="${GRES:-gpu:rtxa6000:1}"

# Frozen in configs/cross_domain_shared_source_v1.json; do not drift.
SEED_BASE_ROOT=920001
SEED_BASE_STRIDE=1000

cd "${CLEAN_REPO}"

case "${MODE}" in
  smoke)
    # candy_crush is game index 0 of GAME_SET=six and needs no ROM or gymv
    # subprocess, so a failure here is a real stack failure and not a ROM issue.
    sbatch --array=0 \
      --gres="${GRES}" \
      --time=02:00:00 \
      --job-name=xd-src-smoke \
      --export=ALL,GAME_SET=six,EPISODES=1,MAX_STEPS=50,\
SEED_BASE_ROOT=${SEED_BASE_ROOT},SEED_BASE_STRIDE=${SEED_BASE_STRIDE},\
ALLOW_RESUME=1,PORT_BASE=31000,\
OUTPUT_ROOT=${CLEAN_REPO}/runs/cross_domain_shared_source_v1_smoke \
      cluster/collect_phase1_complete.sbatch
    ;;
  full)
    # 6 games x 16 episodes, split into 4-episode chunks = 24 tasks.  Measured
    # cost is 1-2 min vLLM startup against ~4 min per episode, so chunking adds
    # roughly 8% GPU time and makes a lost task cost 16 minutes instead of 64.
    sbatch --array=0-23%4 \
      --gres="${GRES}" \
      --time=02:00:00 \
      --job-name=xd-src-full \
      --export=ALL,GAME_SET=six,EPISODES_PER_GAME=16,CHUNK_SIZE=4,MAX_STEPS=50,\
SEED_BASE_ROOT=${SEED_BASE_ROOT},SEED_BASE_STRIDE=${SEED_BASE_STRIDE},\
ALLOW_RESUME=1,PORT_BASE=32000,\
OUTPUT_ROOT=${CLEAN_REPO}/runs/cross_domain_shared_source_v1 \
      cluster/collect_phase1_complete.sbatch
    ;;
  *)
    echo "usage: $0 {smoke|full}" >&2
    exit 2
    ;;
esac
