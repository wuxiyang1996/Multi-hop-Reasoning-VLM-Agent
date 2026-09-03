# Phase 1：六游戏到四领域的 neural-symbolic transfer 审计

## 结论

截至 2026-08-15，严格审计结果为：

- **6/6 source games 正式通过**独立、target-blind 的 common symbolic IR qualification。
- **4/4 target programs 通过**对新六源 artifact 的 exhaustive route relineage。
- **24/24 source-target mechanism cells 验证通过**。
- **0/24 cells 是新的 joint prospective execution**。source evidence 是本轮新 formal；target success outcome 来自此前已验证运行，并通过逐 route 程序等价重绑定继承。

因此，现在支持的精确命题是：

> 六个不同游戏都独立支持同一个 intervention-grounded symbolic search structure；这个 structure 与 WebShop、ALFWorld、DiscoveryWorld、TIRBench 已验证的 target-native neural grounder/executor 程序完全兼容。

现在**不支持**的更强命题是：“六个 source × 四个 target 已各自新跑过一次 prospective end-to-end experiment。”这一项仍为 0/24。

机器审计见 [`results/phase1_six_game_four_target_transfer_audit_v2.json`](results/phase1_six_game_four_target_transfer_audit_v2.json)，可视化状态页见 [`PHASE1_SIX_GAME_FOUR_TARGET_STATUS_V2.html`](PHASE1_SIX_GAME_FOUR_TARGET_STATUS_V2.html)。

## 到底迁移了什么

迁移对象不是游戏动作、对象名字、候选顺序或 source trajectory，而是以下三状态 symbolic controller：

```text
NO_ACTIVE_CANDIDATE_AND_UNTRIED_REMAIN -> EXPLORE_UNTRIED
ACTIVE_CANDIDATE_REFUTED             -> BACKTRACK_REPLAN
ACTIVE_CANDIDATE_VERIFIED            -> COMMIT_VERIFY
```

canonical policy SHA-256：

```text
02d3adc83616688ae0b51b152bae3ce8ab468f0fc3a99ac0f27b7d399e696fdf
```

transfer contract 是：

```text
source formal interventions
  -> abstract event + attempt ledger
  -> common symbolic routing
  -> target-native neural event/candidate grounding
  -> target-native action executor
  -> observed effect 后才更新 ledger
```

source native action token、candidate identity 和 candidate order 都禁止进入 transfer artifact；unknown/conflicting target event 必须 `ABSTAIN`。这使它区别于 semantic skill retrieval 或复制 source action sequence。

## 六个 source 的正式结果

所有正式 config 和 gate 在读取 formal intervention outcome 之前冻结。每个候选都从完全相同的 seed + observed prefix fork 两次；snapshot/prefix 选择 reward-blind；最终按 multi-step official cumulative return 判定 unique best。冻结 gate 包括：

- fresh eligible states ≥ 8；每个 selected symbolic action 的 fresh support ≥ 8；
- authentic success rate = 1.0；
- authentic 相对每个 destructive control 的 margin ≥ 0.40；
- 每个 symbolic branch 的 mean matched advantage ≥ 1.0；
- 每个 split eligible fraction ≥ 0.40；
- formal intervention failure rows = 0；
- canonical policy hash 必须一致，IR 中不得出现 native action token。

| Source game | Native option grounding | Horizon | Eligible | Discovery / Qualification / Fresh | Authentic fresh | Ledger-blind | Infra failures |
|---|---|---:|---:|---:|---:|---:|---:|
| Tetris | state-conditioned first action + common continuation | 16 | 24/36 | 7 / 9 / 8 | 8/8 | 3/8 | 0 |
| Candy Crush | state-conditioned first action + common continuation | 16 | 29/30 | 10 / 9 / 10 | 10/10 | 5/10 | 0 |
| Columns | discovery-only native primitive vocabulary | 50 | 30/30 | 10 / 10 / 10 | 10/10 | 0/10 | 0 |
| Streets of Rage 2 | discovery-only full execution + frozen destructive controls | 50 | 27/36 | 6 / 10 / 11 | 11/11 | 2/11 | 0 |
| Thunder Force III | discovery-only native primitive vocabulary | 23 | 31/36 | 12 / 10 / 9 | 9/9 | 1/9 | 0 |
| Strider | discovery-only native primitive vocabulary | 22 | 30/30 | 10 / 10 / 10 | 10/10 | 0/10 | 0 |

这里的 Discovery / Qualification / Fresh 对应 report 中的 discovery/calibration/fresh-confirmation eligible states；原始 ledger 字段名分别是 development/qualification/heldout。

六源共识 artifact：

```text
artifact_sha256 = 36bec02285d4b2aca98f0a938447085545b24aff8edd23e2bc7485260da27f58
status          = SOURCE_SEARCH_AUTOMATON_FROZEN
target_authorized = true
```

artifact builder 会重新计算六个 report、验证 config/plan/rows/audit/template hashes、检查 checkpoint 完整性和 outcome-independent retry，再授权 target 使用。任何一个 source gate 失败都会停止构建。

## 四个 target 的程序等价与历史 outcome

新 artifact 的 symbolic policy 与此前 target 验证所绑定的 V16 policy 完全相同。relineage 对历史 evidence 中每一个 `RoutedDecision` 做 self-hash 验证，只替换 source artifact hash 与由此产生的 receipt hash；target event、admission、native action 和 target evidence hash 必须逐字段不变。

共验证 **14,727** 条 route receipt：

| Target | 历史 evidence tier | Route receipts | 历史 raw → authentic success | 说明 |
|---|---|---:|---:|---|
| WebShop | prospectively frozen fresh formal | 781 | strict 7 → 18/32 | 11W/0L；authentic 等于 target-native search ceiling |
| ALFWorld | consumed development reexecution | 1,806 | 4 → 7/8 | 3W/0L；不是新的 prospective reserve |
| DiscoveryWorld | retrospective equivalence | 44 | 10 → 13/16 | 1 次 source abstention；不是新的 prospective confirmation |
| TIRBench | consumed fresh-formal reanalysis | 12,096 | 14 → 23/24 | 9W/0L；不是新的 V2 joint run |

四域的 historical gates、三种 symbolic actions、破坏性 controls 和 target-native ceiling 检查都通过。relineage report SHA-256 为：

```text
b000bb149e849963e4f81c5e7a7e3d5cb7c39ea0db4855f830957f3ea1b0dc78
```

## 6×4 audit 的含义

最终 audit 的七个 gate 全部通过：

- six formal source lineages；
- four validated target programs；
- complete 6×4 matrix；
- all 24 mechanism cells validated；
- common artifact target-authorized；
- source native tokens absent；
- target relineage report passed。

最终 report SHA-256：

```text
01ec85d78ccf7f02e00f1ac2ba9767697529e8efd0320de5a8fadfad4c080358
```

24 个 cell 共享一个 common artifact，并分别携带 source formal report hash、target domain report hash、historical target outcome hash 和 evidence tier。它证明的是**可组合机制**，不是把同一个 target outcome 重复计成六次独立 target experiment。

## Bitter lessons

1. **高层 skill label 不够。** `COMMIT/ATTACK` 之类名字不能证明迁移；必须还原为可干预、可反驳、可重复的 state/action/effect ledger。
2. **复制 raw sequence 不是 transfer。** Strider 的 random option pilot 得到 0 eligible，复制完整 action string 只有 3 eligible；使用 discovery-native primitive vocabulary 并在 target snapshot 上重新 grounded 后，formal 才达到 30/30。
3. **source-native grounding 与 target-native grounding 缺一不可。** common IR 只能控制 explore/backtrack/commit；具体可执行动作必须由当前 domain 产生。
4. **sparse reward 需要正确 horizon。** Streets 的单 primitive 重复在 pilot 中只有 1/9 eligible；discovery-authorized 的 50-step execution 达到 8/9 pilot，并在新 formal seeds 上达到 27/36。
5. **失败结果必须保留。** Streets formal V1 authentic 8/8 且 controls/hash/infra 都通过，但 qualification split 6/16 = 0.375，低于预冻结 0.40，所以整体 FAIL。V2 在读取新 outcome 前冻结新的 source-only option generator 与新 seeds，随后通过；没有降低 gate。
6. **target outcome 的证据等级不能被 relineage 提升。** WebShop 仍是 fresh prospective；ALFWorld/TIRBench 是 consumed；DiscoveryWorld 是 retrospective。程序等价只允许继承已有结论，不会把它变成新实验。

## 可复现命令

以下命令假设位于 repository root，并使用项目现有 Python environment：

```bash
export PYTHONPATH=src
PY=/fs/gamma-projects/vlm-robot/conda/envs/cosplay-candy-a100/bin/python

# 单个 source：先冻结 plan，再完整采集，最后分析
$PY scripts/run_phase1_common_search_ir.py prepare \
  --config configs/phase1_common_search_ir_formal_v1/gymv_strider.json \
  --output runs/phase1_common_search_ir_formal_v1/gymv_strider/plan.json
$PY scripts/run_phase1_common_search_ir.py collect \
  --config configs/phase1_common_search_ir_formal_v1/gymv_strider.json \
  --plan runs/phase1_common_search_ir_formal_v1/gymv_strider/plan.json \
  --output runs/phase1_common_search_ir_formal_v1/gymv_strider/rows.jsonl \
  --workers 2 --incremental
$PY scripts/run_phase1_common_search_ir.py analyze \
  --config configs/phase1_common_search_ir_formal_v1/gymv_strider.json \
  --rows runs/phase1_common_search_ir_formal_v1/gymv_strider/rows.jsonl \
  --output runs/phase1_common_search_ir_formal_v1/gymv_strider/report.json

# 六源 artifact、四 target relineage、最终 6×4 audit
$PY scripts/build_phase1_six_game_search_automaton_artifact.py \
  --manifest configs/phase1_common_search_ir_combined_v2/manifest.json
$PY scripts/relineage_phase1_six_game_four_targets.py
$PY scripts/audit_phase1_six_game_four_target_transfer_v2.py
```

## 下一项最强实验

若要把 claim 从“mechanism transfer validated”升级为“24/24 direct prospective transfer”，下一步不是继续增加 source game，而是冻结四域新的 untouched reserve，然后对统一 artifact 做新的 target execution。最低成本顺序建议：

1. WebShop 新 reserve（已有最成熟的 fresh harness）；
2. TIRBench 新 reserve；
3. ALFWorld 新 held-out tasks；
4. DiscoveryWorld 新 prospective task split。

在此之前，当前结果应始终报告为 **24/24 mechanism cells, 0/24 new joint prospective cells**。
