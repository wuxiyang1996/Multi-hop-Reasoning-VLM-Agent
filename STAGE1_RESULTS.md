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

## Stage 2 — Cross-Domain Skill Transfer (Cognitive Signature Clustering)

### Method

Cognitive-signature-based clustering (`cluster_cognitive_signatures.py`): extracts the
cognitive verb sequence from each skill's protocol steps, maps to 7 primitives
(P=Perceive, R=Retrieve, E=Evaluate, S=Select, X=Execute, C=Confirm, T=Transform),
then clusters by signature pattern.

Three domains have fundamentally different cognitive loops:

```
GAME:  X→C→P→T  (reactive — act first, then observe)
WEB:   R→P→S→X  (deliberate — recall instruction, match, then act)
VR:    P→E→S→C  (inferential — observe, reason, then conclude)
```

### 7 Cognitive Loop Families

| Family | Skills | Domains | Transfer targets |
|--------|--------|---------|-----------------|
| reactive_execute | 117 | GAME | GAME |
| deliberate_select | 88 | GAME+VR+WEB ★ | GAME, WEB |
| sequence_chain | 36 | GAME+VR | GAME |
| inferential_reason | 27 | GAME+VR+WEB ★ | VR, WEB |
| retrieve_match_act | 25 | VR+WEB ● | WEB, VR |
| plan_transform | 21 | GAME+WEB ● | GAME, WEB |
| explore_monitor | 10 | GAME | GAME |

### Cross-Domain Transfer Paths

```
GAME action   ──(reactive_execute)───→ GAME action     (same cognitive loop)
GAME deliberate ─(deliberate_select)──→ WEB interaction (PSXC signature bridge)
WEB reasoning ──(inferential_reason)──→ VR inference    (PES signature bridge)
WEB instruction ─(retrieve_match_act)─→ VR tasks        (RPES signature bridge)
GAME state ─────(plan_transform)──────→ WEB form-fill   (PXC/PT signature bridge)
```

### Phase 2 Seed Banks (Cognitive + Genre-Aware)

| Target | Genre | Top Source (count) | Seeds | Bridge |
|--------|-------|-------------------|-------|--------|
| SpaceHarrierII | shooter | TF3(19) | 40 | 6 |
| Airstriker | shooter | TF3(19) | 40 | 6 |
| AlteredBeast | brawler | SoR2(15), Strider(10) | 40 | 6 |
| DynamiteHeaddy | platformer | Strider(17) | 40 | 5 |
| 2048 | puzzle | Columns(14), Strider(11) | 40 | 7 |
| SuperMario | platformer | Strider(17) | 40 | 5 |
| webshop_new | web | miniwob(40) | 40 | 40 |
| miniwob_unseen | web | miniwob(40) | 40 | 40 |
| vr_new_bench | vr | miniwob(16), tir(9), siv(8) | 40 | 40 |

Scripts: `cluster_cognitive_signatures.py`, `stage2_seed_from_cognitive.py`  
Seed banks: `frontier_data/output/stage2_seed_banks/<task>/skill_bank.jsonl`
