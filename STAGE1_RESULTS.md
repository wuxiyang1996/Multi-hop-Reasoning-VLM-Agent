# Stage 1 Results

Best co-evolution runs per game, with peak reward and corresponding step.

| Game | Run Path | Best Step | Reward |
|------|----------|-----------|--------|
| Candy Crush | `runs/candy_crush_coevo_v4_20260519_093912` | 6 | 620.0 |
| Columns | `runs/gymv_columns_coevo_v4_20260519_001840` | 3 | 155.98 |
| Strider | `runs/gymv_strider_coevo_v5_20260519_184613` | 9 | 79.17 |
| ThunderForce III | `runs/gymv_thunder_force_iii_coevo_v9_grpo_unclip` | 3 | 800.0 |
| StreetsOfRage2 | `runs/gymv_streets_of_rage_2_coevo_v5_20260520_010806` | 4 | 360.83 |

---

## Stage 2 — Cross-Game Skill Transfer (Auto-Clustering)

### Pipeline

1. **Pass 1**: `extract_mega_skills.py` — LLM free-form labeling (no pre-defined categories).  
   324 skills → 190 auto-discovered mega-skill labels.
2. **Clustering**: `cluster_mega_skills.py` — iterative LLM semantic merging (3 rounds).  
   190 labels → **16 canonical families** (4 three-way GAME+WEB+VR, 6 two-way, 6 single-domain).  
   94.1% of non-game skills land in cross-domain families.
3. **Pass 2**: `extract_mega_skills.py --relabel` — re-labels all skills using the codebook for consistency.
4. **Seed generator**: `stage2_seed_from_clusters.py` — transfers skills to Phase 2 holdout games via genre-matched source mapping + mega-skill family alignment.

### 16 Canonical Mega-Skill Families

| Family | Skills | Domains | Cross-domain? |
|--------|--------|---------|---------------|
| perceive_decide_act_confirm | 199 | GAME+WEB+UNKNOWN | ★ 3-way |
| evaluate_select_confirm | 63 | GAME+VR+WEB | ★ 3-way |
| identify_improve_confirm | 14 | GAME+WEB+UNKNOWN | ★ 3-way |
| sequence_adjust_confirm | 3 | GAME+WEB+UNKNOWN | ★ 3-way |
| explore_update_confirm | 10 | GAME+UNKNOWN | ● 2-way |
| avoid_navigate_confirm | 8 | GAME+UNKNOWN | ● 2-way |
| filter_count_report | 6 | VR+WEB | ● 2-way |
| initialize_setup | 5 | GAME+UNKNOWN | ● 2-way |
| sustain_action_under_threat | 4 | GAME+UNKNOWN | ● 2-way |
| plan_execute_confirm | 3 | GAME+UNKNOWN | ● 2-way |
| perceive_identify_confirm | 3 | VR | single |
| copy_confirm_action | 2 | WEB | single |
| solve_respond | 1 | WEB | single |
| trace_identify_repair | 1 | GAME | single |
| combine_simplify_verify | 1 | VR | single |
| detect_time_execute | 1 | GAME | single |

### Phase 2 Seed Banks

| Holdout Game | Sources | Seeds | Families |
|---|---|---|---|
| SpaceHarrierII | ThunderForceIII | 16 | 7 |
| Airstriker | ThunderForceIII, Strider | 30 | 10 |
| AlteredBeast | StreetsOfRage2, Strider | 32 | 11 |
| DynamiteHeaddy | Strider, ThunderForceIII, Columns | 36 | 11 |
| 2048 | Columns, CandyCrush | 19 | 8 |
| SuperMario | Strider, StreetsOfRage2 | 31 | 11 |

Seed banks: `frontier_data/output/stage2_seed_banks/<game>/skill_bank.jsonl`
