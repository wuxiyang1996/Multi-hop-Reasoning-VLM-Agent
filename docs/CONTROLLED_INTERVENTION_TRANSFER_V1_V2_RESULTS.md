# 受控的 game → diagnosis intervention skill transfer：v1/v2 实验记录

日期：2026-08-10

## 先说结论

目前真正被数据支持的是一个窄但可运行的结论：

> 从 intervention-rich hidden-rule game 中学到的、以 belief/action value 表示的 `TEST / COMMIT / ABSTAIN` 结构，可以 zero-shot 迁移到 raw token 完全不相交的 diagnosis targets；target 只使用自己的 observation grounder。

这不是 arcade → ALFWorld 的证据，也还不是严格意义上的 neural grounding。当前 grounder 是 target-native 的统计 likelihood estimator；它的目的，是先验证“迁移 symbolic control structure、重新学习 target grounding”这一机制能否工作。

在观察完 v1/v2 的混合 gate 后，我们重新冻结了一个只检验 zero-shot claim、使用全新 15xxx target seeds 的独立 formal config。它的总 gate 为 `SUPPORTED`；authentic 相对 target-only 的 paired net-return 提升为：

| split | authentic | target-only | paired delta | 95% CI |
|---|---:|---:|---:|---:|
| qualification | 0.6816 | 0.6205 | +0.0611 | [0.0430, 0.0794] |
| held-out | 0.6860 | 0.6188 | +0.0672 | [0.0494, 0.0855] |

同一 zero-shot cell 中，authentic 也显著优于 state-shuffled source 和 source-marginal controls，六个预声明比较全部通过；source/target raw action tokens 的交集为空。完整报告：[`independent zero-shot frozen report`](../runs/controlled_exploration_transfer_zero_shot_frozen/report.json)。

尚未被支持的是 robust few-shot fusion。v2 的 target residual 在 development seeds 上有效，但在新的 qualification split 上没有稳定复现。不能据此声称“有两个或四个 target support states 后仍稳定优于 target-only”。

## 迁移的不是高层自然语言 skill

每个 source receipt 被压成一个可以直接执行和打分的低层表示：

- symbolic action：`TEST(i)`、`COMMIT(h)`、`ABSTAIN`；
- state：当前 posterior entropy、MAP confidence、剩余 budget、重复 intervention 次数；
- intervention features：expected information gain、expected confidence gain、predicted outcome balance；
- label：同一 state 下每个 intervention 的 exact finite-horizon value；
- policy：只有 source value ensemble 的 TEST-vs-COMMIT margin 足够大时才执行，否则交回 target-native fallback。

source surface 使用 `pull_rune_* / unlock_vault_*`，target surface 使用 `order_assay_* / diagnose_syndrome_*`。模型看不到共享 action token，也不复制 source likelihood；target likelihood 从 target calibration receipts 单独估计。

## 对照

- `target_only`：没有 source value examples；zero-shot 时只能使用 target-native fallback，few-shot 时只拟合 target examples。
- `authentic_source_plus_target`：保留 source state-action-value 对应关系。
- `shuffled_source_plus_target`：只在同一个 source state 内打乱 action values，保持 state、action 数量和值分布不变。
- `source_marginal_plus_target`：只保留 TEST/COMMIT 两类的边际均值，删除 state-dependent intervention structure。

所有 condition 使用相同 target domains、hidden hypotheses 和 matched stochastic observations。置信区间来自 episode-level paired bootstrap。

## v1：zero-shot 成立，joint few-shot mixing 失败

最初 `target_mass=1.0` 时，仅两个 target states 就与全部 source 数据获得相同总质量，导致 negative transfer。development 中把 target mass 收缩到 0.01 后，k=0/2/4 gate 均通过，但 authentic 几乎不再吸收 target support。

冻结后的 v1 在新的 qualification/held-out domains 上得到：

- k=0 相对 target-only：`+0.0395 [0.0204, 0.0597]`、`+0.0654 [0.0456, 0.0860]`；
- k=2 仍通过；
- k=4 两个 split 都失败：`-0.0285`、`-0.0221`。

因此 v1 支持 zero-shot transfer，但不支持“source + target joint fit 能可靠融合 few-shot target evidence”。完整报告：[`v1 frozen report`](../runs/controlled_exploration_transfer_v1_frozen/report.json)。

## v2：source prior + target residual 仍不稳

v2 固定 source value prior，只让 target support 拟合 residual；residual strength 在 k=0/2/4 分别为 0/0.5/1.0。development 的 absolute net returns 为：

| k | authentic | target-only | paired delta |
|---:|---:|---:|---:|
| 0 | 0.6745 | 0.6085 | +0.0661 |
| 2 | 0.6718 | 0.6112 | +0.0606 |
| 4 | 0.6787 | 0.6812 | -0.0025 |

这看起来像合理的 low-data advantage → target saturation 曲线，但冻结后没有稳定复现：

- qualification k=2：authentic vs target-only `-0.0047 [-0.0223, 0.0127]`；
- held-out k=2：authentic vs target-only `+0.0412 [0.0195, 0.0624]`；
- qualification k=4：`-0.0260 [-0.0477, -0.0044]`，未通过预先声明的 0.03 non-inferiority margin；
- residual 也会快速修复 shuffled/marginal controls，所以 k=2 时 authentic 并不稳定优于这两个 controls。

完整报告：[`v2 frozen report`](../runs/controlled_exploration_transfer_v2_frozen/report.json)。总 gate 为 `NOT_SUPPORTED`，这一状态应保留。

## 可支持与不可支持的 claim

可支持：

1. operational intervention-value structure 能跨两个语义表面 zero-shot 迁移；
2. 迁移增益依赖正确的 state-action-value correspondence，不只是 TEST/COMMIT 频率或 value marginal；
3. target-native grounding 足以让 source symbolic policy 在新 target tokens 上执行；
4. zero-shot 结果在 discovery、v1 两个新 splits、v2 两个新 splits 和独立 zero-shot formal 的两个新 splits 上方向一致；最后一轮六个预声明比较的 CI 都严格大于 0。

不可支持：

1. robust few-shot residual/joint fusion；
2. 从现有 Atari/game logs 到 ALFWorld 的真实迁移；
3. VLM/neural grounder 已经训练成功；
4. 仅凭这组 synthetic family 宣称 domain-invariant skill transfer 已普遍解决。

## 复现

核心实现：[`controlled_exploration_transfer.py`](../src/motif_transfer/controlled_exploration_transfer.py)
运行入口：[`run_controlled_exploration_transfer_v1.py`](../scripts/run_controlled_exploration_transfer_v1.py)
单测：[`test_controlled_exploration_transfer.py`](../tests/test_controlled_exploration_transfer.py)

```bash
cd /fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean

pytest -q tests/test_controlled_exploration_transfer.py

python scripts/run_controlled_exploration_transfer_v1.py \
  --config configs/controlled_exploration_transfer_zero_shot_frozen.json \
  --output-dir runs/controlled_exploration_transfer_zero_shot_frozen_reproduction
```

独立 zero-shot 冻结配置：[`zero-shot frozen config`](../configs/controlled_exploration_transfer_zero_shot_frozen.json)，首次运行前记录的 SHA-256 为 `dfa19d792343293eb47cf7a78a678329b4e0efae2633825fb4394dba3b5fedc6`。few-shot 失败配置仍保留为 [`v2 frozen config`](../configs/controlled_exploration_transfer_v2_frozen.json)，不可用前一个 `SUPPORTED` 状态覆盖后一个 `NOT_SUPPORTED` 状态。

## 下一步进入真实环境的硬门槛

不要直接搬运高层 skill 文本。下一步应在 source game log 中抽取可验证的 matched intervention tuples：

```text
(belief/uncertainty state,
 reversible probe action,
 observed state delta,
 remaining budget,
 estimated progress value)
```

然后让 ALFWorld-native grounder 把 admissible actions 映射到 `TEST / COMMIT / ABSTAIN`，只迁移 TEST-vs-COMMIT value comparator。真实实验必须至少保留四个条件：target-only、authentic、within-state shuffled、source-marginal；如果 authentic 不能在固定 canonical task list 上提高 paired success/progress，就停止，不把 synthetic zero-shot 结果外推过去。
