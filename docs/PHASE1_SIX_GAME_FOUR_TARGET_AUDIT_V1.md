# Phase-1 六游戏 × 四目标迁移审计 V1

日期：2026-08-15

## 结论

当前严格结果是：`0 / 24` 个 **Phase-1 game → target** cell 得到验证。

这不等于 target harness 不工作。四个 target 都已经能执行并受益于同一个 Sokoban V16
search automaton；但该 artifact 不是从 Phase-1 六游戏中的任何一个独立学习出来的。因此这些
target 结果不能重新标成 Tetris、Candy Crush、Streets of Rage 2、Strider、Columns 或
Thunder Force III 的迁移结果。

机器可审计结果：
`docs/results/phase1_six_game_four_target_transfer_audit_v1.json`。

## 6 × 4 矩阵

`SB` 表示 `BLOCKED_AT_SOURCE_GATE_TARGET_NOT_EXECUTED`。

| Phase-1 source | WebShop | ALFWorld | DiscoveryWorld | TIRBench |
|---|---:|---:|---:|---:|
| Tetris | SB | SB | SB | SB |
| Candy Crush | SB | SB | SB | SB |
| Streets of Rage 2 | SB | SB | SB | SB |
| Strider | SB | SB | SB | SB |
| Columns | SB | SB | SB | SB |
| Thunder Force III | SB | SB | SB | SB |

每个 cell 都是 source-blocked，而不是 target-blocked。V16 的四个 target mechanism gate
都通过，因此在“Phase-1 source 先产生相同且通过验证的 IR”这个条件下，24 个 target route
均有现成实现路径。

## 我们实际拥有什么

Phase-1 inventory 是真实的：6 个游戏、13 个 recorded skills、131 个 maximal executions、
1769 个 transitions，其中 11 个 skills 有 three-way execution evidence。但这些只证明
skill 被选择和执行过，不证明抽象 controller 有 causal value，更不证明它能迁移。

逐 source 的最强证据如下：

| Source | 已有证据 | 当前阻塞 |
|---|---|---|
| Tetris | 6 episodes / 292 steps / `COMMIT/POSITION` | 没有 matched intervention source gate；旧 seed receipt 未验证 hidden-state replay |
| Candy Crush | 新鲜同-runtime 960/960 matched action forks 有效 | neural grounder heldout 输给 within-state shuffled control；target 0 calls |
| Streets of Rage 2 | 4 snapshots / 40 matched treatment trajectories | joint `PROGRESS/STALL → SWITCH/PERSIST` controller 未胜 static/permuted controls |
| Strider | 4 snapshots / 40 matched treatment trajectories | 同上；旧 execution motif 也被 recurrence/shortcut gate 拒绝 |
| Columns | 4 snapshots / 40 matched treatment trajectories | 同上；reward 较密，但现有 absolute progress/stall abstraction 不成立 |
| Thunder Force III | 4 snapshots / 40 matched treatment trajectories | micro-controller、visual-effect qualification 和 h8 skill-context smoke 均失败 |

六游戏联合产生的旧 observational micro-controller 也没有通过 matched source gate。因此当前
既不是“每个游戏独立通过”，也不是“六游戏 pooled source 通过”。

## 为什么不能直接复用 V16 的四目标结果

V16 的 canonical source artifact SHA-256 是
`ba848d54a10c785912f2dbe29c9ea8e35ef8eec3daf2cea4e5d303919b6cd75f`，由 fresh Sokoban
states 上的 matched candidate corruptions 支持。它包含：

- `NO_ACTIVE_CANDIDATE_AND_UNTRIED_REMAIN → EXPLORE_UNTRIED`
- `ACTIVE_CANDIDATE_REFUTED → BACKTRACK_REPLAN`
- `ACTIVE_CANDIDATE_VERIFIED → COMMIT_VERIFY`

Phase-1 labels 如 `COMMIT/CLEAR`、`COMMIT/ATTACK`、`COMMIT/POSITION` 或 `__EXPLORE__`
没有形成这个三分支 automaton 的完整、independent、blind-value-qualified lineage。仅把名字映射
到三个 V16 action 会成为人工 ontology transfer，而不是 neural-symbolic skill transfer。

## 最省成本而且严格的下一步

不需要机械地重跑 24 组昂贵 target experiments。可以用组合证明：

1. 对六个游戏分别在 target-unread 条件下产生 matched forks；每个游戏独立诱导
   `EXPLORE_UNTRIED / BACKTRACK_REPLAN / COMMIT_VERIFY` 或选择 abstain。
2. 每个游戏的 candidate 必须在 blind qualification 和 held-out source states 上胜过
   event-binding permuted、ledger-blind、static 和 hash-random controls。
3. 将六个通过的 IR alpha-canonicalize；要求 routing truth table 和 canonical hash 与 target
   harness 使用的 frozen IR 一致。
4. 四个 target 各验证一次 common IR + target-native neural grounding。只有当 6 个 source
   proofs、1 个 linkage proof、4 个 target proofs 都成立，才可组合声明 24/24。

目前 source proofs 是 `0/6`，target mechanism proofs 是 `4/4`。所以下一轮预算应全部放在
source qualification，而不是继续调用 WebShop/ALFWorld/DiscoveryWorld/TIRBench provider。

优先级建议：先做 **Columns**，其次 Candy。Columns 已有 exact replay 且 reward 较密，适合把
atomic button fork 升级为候选级多步 option fork；Candy 的 matched action effect 已成立，但必须
解决 observable grounding 与 cascade 隐变量问题。Tetris 需先补同-runtime replay；三个动作游戏
需要多步 option，而不是单按钮 effect。

## 复现

```bash
PYTHONPATH=src python scripts/audit_phase1_six_game_four_target.py \
  --output docs/results/phase1_six_game_four_target_transfer_audit_v1.json

PYTHONPATH=src pytest -q tests/test_phase1_six_game_transfer_audit.py
```

