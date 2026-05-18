#!/usr/bin/env bash
# =======================================================================
# frontier_data/scripts/run_full_pipeline.sh
#
# Master pipeline: extract all skills → fill missing parts → build
# shared bank → bind mega-skills to per-task skills via harness+crafter.
#
# Prerequisite: emnlp2026_download must be accessible.  Set
# DOWNLOAD_ROOT to its workspace/main_project if not at the default.
#
# Usage:
#   bash frontier_data/scripts/run_full_pipeline.sh          # full run
#   STAGE=3 bash frontier_data/scripts/run_full_pipeline.sh  # resume from stage 3
#   DRY_RUN=1 bash frontier_data/scripts/run_full_pipeline.sh  # print commands only
# =======================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/main_project}"
OUTPUT_ROOT="${REPO_ROOT}/frontier_data/output"
STAGE="${STAGE:-1}"
DRY_RUN="${DRY_RUN:-0}"
MODEL="${MODEL:-gpt-5.4}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-35B-A3B}"
WORKERS="${WORKERS:-4}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

# ── Game + non-game task lists ──────────────────────────────────────
GYMV_GAMES=(
    Temporal_Airstriker-v0
    Temporal_AlteredBeast-v0
    Temporal_CastleOfIllusion-v0
    Temporal_CastlevaniaBloodlines-v0
    Temporal_Columns-v0
    Temporal_DynamiteHeaddy-v0
    Temporal_GoldenAxe-v0
    Temporal_KidChameleon-v0
    Temporal_MortalKombatII-v0
    Temporal_SpaceHarrierII-v0
    Temporal_StreetsOfRage2-v0
    Temporal_Strider-v0
    Temporal_ThunderForceIII-v0
)
ENVW_GAMES=(tetris super_mario candy_crush twenty_forty_eight)
VR_IMAGE_TASKS=(tir_bench visual_toolbench)
VR_VIDEO_TASKS=(siv_bench video_holmes)
WEB_TASKS=(miniwob webshop)

ALL_GAMES=("${GYMV_GAMES[@]}" "${ENVW_GAMES[@]}")
ALL_NONGAME=("${VR_IMAGE_TASKS[@]}" "${VR_VIDEO_TASKS[@]}" "${WEB_TASKS[@]}")

mkdir -p "${OUTPUT_ROOT}"

run_cmd() {
    local desc="$1"; shift
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ${desc}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  CMD: $*"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [DRY_RUN — skipped]"
        return 0
    fi
    "$@"
}

log() { echo "[pipeline $(date +%H:%M:%S)] $*"; }

# ======================================================================
# STAGE 1: Extract skill banks from all 18 tasks
# ======================================================================
if [ "${STAGE}" -le 1 ]; then
    log "STAGE 1: Skill extraction from cold-start rollouts"

    SKILL_BANK_OUT="${OUTPUT_ROOT}/skill_bank_out/run_${TIMESTAMP}"
    mkdir -p "${SKILL_BANK_OUT}"

    # 1a. gym_v games (13 games via SkillBankAgent)
    run_cmd "Stage 1a: Extract skill banks for gym_v games" \
        python labeling/extract_skillbank_gymv_gpt54.py \
            --coldstart-root "${DOWNLOAD_ROOT}/Cold-start-out-gymv" \
            --output-dir "${SKILL_BANK_OUT}/gym_v" \
            --model "${MODEL}" \
            --workers "${WORKERS}"

    # 1b. env_wrapper games (4 games)
    run_cmd "Stage 1b: Extract skill banks for env_wrapper games" \
        python labeling/extract_skillbank_gpt54.py \
            --coldstart-root "${DOWNLOAD_ROOT}/Cold-start-out" \
            --output-dir "${SKILL_BANK_OUT}/env_wrappers" \
            --corpus env_wrappers \
            --model "${MODEL}" \
            --workers "${WORKERS}"

    # 1c. Non-game: browsergym (miniwob + webshop)
    run_cmd "Stage 1c: Extract skill banks for browsergym (miniwob)" \
        python labeling/build_skillbank_qa_gpt54.py \
            --coldstart-root "${DOWNLOAD_ROOT}/Cold-start-out-browsergym" \
            --corpus browsergym \
            --output-dir "${SKILL_BANK_OUT}/browsergym" \
            --model "${MODEL}" \
            --workers "${WORKERS}"

    # 1d. Non-game: visual reasoning (image: tir_bench, visual_toolbench)
    run_cmd "Stage 1d: Extract skill banks for VR image benchmarks" \
        python labeling/build_skillbank_qa_gpt54.py \
            --coldstart-root "${DOWNLOAD_ROOT}/Cold-start-out-visual-reasoning" \
            --corpus vr_image \
            --output-dir "${SKILL_BANK_OUT}/vr_image" \
            --model "${MODEL}" \
            --workers "${WORKERS}"

    # 1e. Non-game: visual reasoning (video: siv_bench, video_holmes)
    run_cmd "Stage 1e: Extract skill banks for VR video benchmarks" \
        python labeling/build_skillbank_qa_gpt54.py \
            --coldstart-root "${DOWNLOAD_ROOT}/Cold-start-out-visual-reasoning-video" \
            --corpus vr_video \
            --output-dir "${SKILL_BANK_OUT}/vr_video" \
            --model "${MODEL}" \
            --workers "${WORKERS}"

    # 1f. LLM-free cross-corpus lift (skill_transfer_test/extract)
    run_cmd "Stage 1f: LLM-free cross-corpus skill lift (full coverage)" \
        python -m skill_transfer_test.extract.runner \
            --output-dir "${SKILL_BANK_OUT}/cross_corpus" \
            --include-incorrect

    log "STAGE 1 complete. Output: ${SKILL_BANK_OUT}"
    echo "${SKILL_BANK_OUT}" > "${OUTPUT_ROOT}/.latest_skill_bank_out"
fi

# ======================================================================
# STAGE 2: Label episodes with skills + build decision SFT
# ======================================================================
if [ "${STAGE}" -le 2 ]; then
    log "STAGE 2: Label episodes with skills → decision SFT JSONL"

    SKILL_BANK_OUT="$(cat "${OUTPUT_ROOT}/.latest_skill_bank_out" 2>/dev/null || echo "${OUTPUT_ROOT}/skill_bank_out/run_${TIMESTAMP}")"
    LABEL_OUT="${OUTPUT_ROOT}/labeled/run_${TIMESTAMP}"
    DECISION_SFT_OUT="${OUTPUT_ROOT}/decision_sft_jsonl/run_${TIMESTAMP}"
    mkdir -p "${LABEL_OUT}" "${DECISION_SFT_OUT}"

    # 2a. Label intentions (OPERATOR/SUBGOAL) on all episodes
    run_cmd "Stage 2a: Label intentions on gym_v + env_wrapper episodes" \
        python labeling/label_skill_actions_gpt54.py \
            --skill-bank-root "${SKILL_BANK_OUT}" \
            --coldstart-root "${DOWNLOAD_ROOT}" \
            --output-dir "${LABEL_OUT}" \
            --model "${MODEL}" \
            --workers "${WORKERS}"

    # 2b. Label skill actions on QA / web tasks
    run_cmd "Stage 2b: Label skill actions on non-game tasks" \
        python labeling/label_skill_actions_qa_gpt54.py \
            --skill-bank-root "${SKILL_BANK_OUT}" \
            --coldstart-root "${DOWNLOAD_ROOT}" \
            --output-dir "${LABEL_OUT}" \
            --model "${MODEL}" \
            --workers "${WORKERS}"

    # 2c. Build decision SFT JSONL (action_taking + skill_selection)
    run_cmd "Stage 2c: Build decision SFT JSONL for all tasks" \
        python labeling/build_decision_sft_jsonl.py \
            --skill-actions-run "${LABEL_OUT}" \
            --output-dir "${DECISION_SFT_OUT}"

    # 2d. Build multimodal decision SFT for webshop (the missing piece)
    if [ -d "${DOWNLOAD_ROOT}/Cold-start-out-browsergym/webshop_50task_low" ]; then
        run_cmd "Stage 2d: Build webshop decision SFT (filling gap)" \
            python scripts/build_multimodal_decision_sft.py \
                --labeled-root "${LABEL_OUT}" \
                --output-dir "${DECISION_SFT_OUT}/webshop" \
                --task webshop 2>/dev/null || log "WARN: webshop decision SFT script not found, skipping"
    fi

    log "STAGE 2 complete. Decision SFT: ${DECISION_SFT_OUT}"
    echo "${DECISION_SFT_OUT}" > "${OUTPUT_ROOT}/.latest_decision_sft"
    echo "${LABEL_OUT}" > "${OUTPUT_ROOT}/.latest_labeled"
fi

# ======================================================================
# STAGE 3: Build frontier_distill_jsonl (the missing validation corpus)
# ======================================================================
if [ "${STAGE}" -le 3 ]; then
    log "STAGE 3: Build frontier_distill_jsonl from decision SFT"

    DECISION_SFT_OUT="$(cat "${OUTPUT_ROOT}/.latest_decision_sft" 2>/dev/null || echo "${OUTPUT_ROOT}/decision_sft_jsonl/run_${TIMESTAMP}")"
    DISTILL_OUT="${OUTPUT_ROOT}/frontier_distill_jsonl/run_${TIMESTAMP}_with_labeled"
    mkdir -p "${DISTILL_OUT}"

    # The frontier_distill_jsonl is the decision SFT corpus reorganized
    # for Phase B Crafter validation — skill_selection.jsonl per corpus
    # with <state> blocks parseable into SegmentRecord instances.
    for corpus_dir in "${DECISION_SFT_OUT}"/*/; do
        task="$(basename "${corpus_dir}")"
        if [ -f "${corpus_dir}/skill_selection.jsonl" ]; then
            mkdir -p "${DISTILL_OUT}/${task}"
            cp "${corpus_dir}/skill_selection.jsonl" "${DISTILL_OUT}/${task}/"
            cp "${corpus_dir}/action_taking.jsonl" "${DISTILL_OUT}/${task}/" 2>/dev/null || true
            log "  distill: ${task} → $(wc -l < "${corpus_dir}/skill_selection.jsonl") rows"
        fi
    done

    # Symlink for the trainer's COLD_START_VALIDATION_ROOT
    ln -sfn "${DISTILL_OUT}" "${REPO_ROOT}/labeling/frontier_distill_jsonl/run_${TIMESTAMP}_with_labeled" 2>/dev/null || true

    log "STAGE 3 complete. Distill corpus: ${DISTILL_OUT}"
    echo "${DISTILL_OUT}" > "${OUTPUT_ROOT}/.latest_distill"
fi

# ======================================================================
# STAGE 4: Lift procedural templates (Layer C) for all skills
# ======================================================================
if [ "${STAGE}" -le 4 ]; then
    log "STAGE 4: Lift procedural templates (Layer C) via GPT-5.4"

    TEMPLATE_OUT="${OUTPUT_ROOT}/skill_templates/run_${TIMESTAMP}"
    mkdir -p "${TEMPLATE_OUT}"

    run_cmd "Stage 4: Lift Layer-C templates for all 18 tasks" \
        python scripts/lift_skill_templates_gpt54.py \
            --output-dir "${TEMPLATE_OUT}" \
            --model "${MODEL}" \
            --workers "${WORKERS}" \
            -v

    log "STAGE 4 complete. Templates: ${TEMPLATE_OUT}"
    echo "${TEMPLATE_OUT}" > "${OUTPUT_ROOT}/.latest_templates"
fi

# ======================================================================
# STAGE 5: Build the shared abstract bank (mega-skills)
# ======================================================================
if [ "${STAGE}" -le 5 ]; then
    log "STAGE 5: Build shared abstract bank (TwoLayerSkillStore)"

    TEMPLATE_OUT="$(cat "${OUTPUT_ROOT}/.latest_templates" 2>/dev/null || echo "${OUTPUT_ROOT}/skill_templates/run_${TIMESTAMP}")"
    SHARED_BANK_OUT="${OUTPUT_ROOT}/shared_skill_bank"
    PLAN_JUDGMENTS="${OUTPUT_ROOT}/plan_level_similarity_judgments.json"
    SIG_JUDGMENTS="${OUTPUT_ROOT}/plan_similarity_judgments.json"
    CLUSTER_METHOD="${CLUSTER_METHOD:-plan_judge}"

    if [ "${CLUSTER_METHOD}" = "plan_judge" ] && [ -f "${PLAN_JUDGMENTS}" ]; then
        # DEFAULT: plan-level LLM judge clustering
        # Clusters skills by shared reasoning procedure (judge score >= 4)
        run_cmd "Stage 5: Build shared bank via plan-level judge clustering" \
            python frontier_data/scripts/build_plan_clustered_bank.py \
                --plan-judgments "${PLAN_JUDGMENTS}" \
                --sig-judgments "${SIG_JUDGMENTS}" \
                --per-task-root "${OUTPUT_ROOT}/per_task_banks" \
                --out "${SHARED_BANK_OUT}" \
                --threshold "${JUDGE_THRESHOLD:-4}"
    else
        # FALLBACK: name-based clustering (if no judge results available)
        log "WARN: No plan-level judge results at ${PLAN_JUDGMENTS}, falling back to name-based clustering"
        run_cmd "Stage 5: Build shared bank from mining + templates (name-based)" \
            python scripts/build_shared_skill_bank.py \
                --out "${SHARED_BANK_OUT}" \
                --template-run "${TEMPLATE_OUT}"
    fi

    log "STAGE 5 complete. Shared bank: ${SHARED_BANK_OUT}"
    echo "${SHARED_BANK_OUT}" > "${OUTPUT_ROOT}/.latest_shared_bank"
fi

# ======================================================================
# STAGE 6: Discover per-task skills into the shared bank (BACKWARD flow)
# ======================================================================
if [ "${STAGE}" -le 6 ]; then
    log "STAGE 6: Discover per-task skills → shared bank (backward lift)"

    SHARED_BANK_OUT="$(cat "${OUTPUT_ROOT}/.latest_shared_bank" 2>/dev/null || echo "${OUTPUT_ROOT}/shared_skill_bank/run_${TIMESTAMP}")"
    SKILL_BANK_OUT="$(cat "${OUTPUT_ROOT}/.latest_skill_bank_out" 2>/dev/null || echo "${OUTPUT_ROOT}/skill_bank_out/run_${TIMESTAMP}")"

    # Discover skills from every per-task bank into the shared bank
    for corpus in gym_v env_wrappers browsergym vr_image vr_video cross_corpus; do
        corpus_dir="${SKILL_BANK_OUT}/${corpus}"
        [ -d "${corpus_dir}" ] || continue
        for task_dir in "${corpus_dir}"/*/; do
            task="$(basename "${task_dir}")"
            [ -f "${task_dir}/skill_bank.jsonl" ] || continue
            run_cmd "Stage 6: Discover ${corpus}/${task} → shared bank" \
                python scripts/discover_skill_to_shared_bank.py \
                    --bank-root "${SHARED_BANK_OUT}" \
                    --task "${task}" \
                    --from-skill-bank "${task_dir}/skill_bank.jsonl" \
                    --model "${MODEL}"
        done
    done

    log "STAGE 6 complete."
fi

# ======================================================================
# STAGE 7: Bind shared mega-skills to per-task skills (FORWARD flow)
# ======================================================================
if [ "${STAGE}" -le 7 ]; then
    log "STAGE 7: Bind mega-skills → per-task concrete skills (forward)"

    SHARED_BANK_OUT="$(cat "${OUTPUT_ROOT}/.latest_shared_bank" 2>/dev/null || echo "${OUTPUT_ROOT}/shared_skill_bank/run_${TIMESTAMP}")"
    BIND_REPORT="${OUTPUT_ROOT}/bind_reports/run_${TIMESTAMP}"
    mkdir -p "${BIND_REPORT}"

    ALL_TASKS=("${ALL_GAMES[@]}" "${ALL_NONGAME[@]}")
    for task in "${ALL_TASKS[@]}"; do
        run_cmd "Stage 7: Bind abstracts → ${task}" \
            python scripts/bind_abstract_to_task.py \
                --bank-root "${SHARED_BANK_OUT}" \
                --target-task "${task}" \
                --batch-strong-candidates \
                --harness-validate \
                --model "${MODEL}" \
                --out-report "${BIND_REPORT}/${task}.json"
    done

    log "STAGE 7 complete. Bind reports: ${BIND_REPORT}"
fi

# ======================================================================
# STAGE 8: Crafter v2 — refine and compose skills
# ======================================================================
if [ "${STAGE}" -le 8 ]; then
    log "STAGE 8: Crafter v2 — refine, patch, compose skills"

    SHARED_BANK_OUT="$(cat "${OUTPUT_ROOT}/.latest_shared_bank" 2>/dev/null || echo "${OUTPUT_ROOT}/shared_skill_bank/run_${TIMESTAMP}")"
    CRAFTER_OUT="${OUTPUT_ROOT}/crafter_v2/run_${TIMESTAMP}"
    mkdir -p "${CRAFTER_OUT}"

    for task in "${ALL_GAMES[@]}"; do
        run_cmd "Stage 8: Crafter v2 for ${task}" \
            python scripts/crafter_v2_batch_pipeline.py \
                --run-dir "${CRAFTER_OUT}" \
                --game "${task}" \
                --judge-url "${JUDGE_URL:-http://localhost:8001/v1}"
    done

    log "STAGE 8 complete. Crafter output: ${CRAFTER_OUT}"
fi

# ======================================================================
# STAGE 9: Final inventory build + symlinks
# ======================================================================
if [ "${STAGE}" -le 9 ]; then
    log "STAGE 9: Build SFT data inventory"

    run_cmd "Stage 9: Rebuild sft_data_inventory" \
        python sft_data_inventory/build_inventory.py

    # Symlink latest outputs for easy access
    ln -sfn "${OUTPUT_ROOT}" "${REPO_ROOT}/frontier_data/latest_output" 2>/dev/null || true

    log "STAGE 9 complete."
fi

# ======================================================================
echo ""
log "╔═══════════════════════════════════════════════════════════════╗"
log "║           FULL PIPELINE COMPLETE                             ║"
log "║  Output root: ${OUTPUT_ROOT}"
log "╚═══════════════════════════════════════════════════════════════╝"
