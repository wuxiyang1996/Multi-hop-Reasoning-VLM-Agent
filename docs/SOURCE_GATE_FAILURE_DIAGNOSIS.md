# Source gate 失败诊断

本诊断回答的不是“如何把当前结果调成正结果”，而是当前 Strider Phase 7 究竟在哪一层失败，以及现有六游戏证据能否支持下一轮实验。机器可读报告为
[`results/source_gate_failure_diagnosis_v1.json`](results/source_gate_failure_diagnosis_v1.json)，生成器为
[`scripts/diagnose_source_gate_failure.py`](../scripts/diagnose_source_gate_failure.py)。

## 结论

当前失败分类是：

```text
MEASUREMENT_AND_REPRESENTATION_FAILURE_BEFORE_TRANSFER
```

更具体地说：

```text
source weights / context influence  SUPPORTED
one-step source value               NOT SUPPORTED
delayed treatment value             NOT IDENTIFIED
explicit reasoning motif            NOT SUPPORTED
far-domain transfer                 NOT AUTHORIZED
```

因此现在不能得出“游戏训练没有价值”，也不能得出“存在 transferable reasoning backbone”。已经证明的只有：LoRA weights、authentic skill context 和 random skill context 会在相同 snapshot 上产生不同 action/next-state。

## 为什么 one-step value gate 失败尚不等于训练无价值

正式 `COMMIT/POSITION` run 有 155 个 matched snapshots 和 620/620 observed one-step replays，但六个 live authentic episodes 的 300 步里只有一个正 reward event。该 reward 出现在 step 38，而 step 38 并未选择 `POSITION`。

沿 live authentic path，`POSITION` 的机械 reward coverage 是：

| horizon | 155 次选择中之后出现正奖励的次数 |
|---:|---:|
| 1 | 0 |
| 2 | 0 |
| 4 | 1 |
| 8 | 4 |

当前 replay receipt 只保存 alternative action 的一步 reward、termination 和 next-state，没有 `return_h2/h4/h8`。所以它严格证明 `POSITION` 没有 immediate value advantage，但无法识别 delayed value。完整 episode return 只有 live `G+S` trajectory，不能拿来构造另外三个 treatment 的 counterfactual return。

下一轮必须预注册两个不同 estimand，不能混为一谈：

1. treatment-specific first action，随后由同一个 frozen continuation policy rollout `1/2/4/8` 步，用来估计 proposal 的直接长程效应；
2. 四个 treatment 各自继续 rollout 相同 horizon，用来估计完整 policy-regime effect。

## 为什么当前 graph 还不是 reasoning backbone

当前 graph node 是六个 action/state separation boolean 的 hash。由于环境 transition 对 action 是确定性的，`*_action` 与 `*_state` 在本批数据中总是成对出现，实际只有三个二值 treatment-effect 维度。这是有用的 causal diagnostic，但不是从 Agent reasoning 中抽出的高层程序。

Discovery 有 6 种 effect node；qualification 和 held-out 各出现 2 种 discovery 未见 node。Exact edge recurrence 分别只有 `20/36` 和 `19/37`。此外，旧 88-call skill-internal matrix 中 authentic condition 仍是 0 backbone-eligible candidate。因此失败不是简单地“样本少一点”：当前表示既没有 exact recurrence，也没有 authentic reasoning-program evidence。

另一个必须修正的问题是，`strider_v3_structural_skill_ranking` 用三个 split 的 minimum continuous edges 选择 `POSITION`。它没有读取 skill 名称或动作语义，但读取了 qualification/held-out lineage，因此存在 selection leakage。该结果只能作 protocol diagnosis。新候选必须只从 discovery episodes 选择，冻结后 qualification/held-out 才能保持盲测。

## 旧六游戏 reward 结果为什么不能升级为主证据

旧 matched-by-seed skill-on/off episode 的均值差为：

| game | authentic − skill-off | 正/零/负 seed 数 | 正 reward density |
|---|---:|---:|---:|
| Tetris | -8.33 | 1/2/3 | 1.000 |
| Candy Crush | +67.33 | 5/0/1 | 1.000 |
| Streets of Rage 2 | +28.33 | 4/0/2 | 0.050 |
| Strider | +33.33 | 4/2/0 | 0.053 |
| Columns | +15.50 | 5/1/0 | 0.336 |
| Thunder Force III | +283.33 | 4/1/1 | 0.057 |

这些是独立采样 trajectory；相同 environment seed 不会让模型 sampling 和后续状态保持一致，因此只是 discovery diagnostic，不是 causal estimate。更严重的是，六组旧 run 都没有 no-human-hints exclusion receipt；Streets、Strider、Thunder 的 Agent response 分别有 49、33、44 次直接出现 `critical action`。旧 Strider authentic mean 为 183.33，而 fresh no-hint matched run 只有 8.33，不能把差异归因于 skill 或训练。

目前真正带 exclusion receipt 的可用 discovery evidence 只有：

| game | episodes | mean return | reward density | 主要连续 source ID | steps / edges / h1 support |
|---|---:|---:|---:|---|---:|
| Columns | 2 | 25.0 | 0.292 | `__EXPLORE__` | 24 / 22 / 7 |
| Thunder Force III | 2 | 600.0 | 0.250 | `early:EXPLORE` | 24 / 22 / 6 |
| Strider | 6 | 8.33 | 0.003 | `COMMIT/POSITION` | 155 / 114 / 0 |

Columns 和 Thunder 是更适合检验 value-aware multi-step gate 的 smoke candidates，但各只有两个 episode，不能直接用于 discovery/qualification/held-out claim。

## 冻结后的下一最小实验

1. 六游戏全部重新采集 fresh no-human-hints trajectories；同 checkpoint、seed schedule、budget 和 receipt schema。
2. 只在 discovery split 内按机械证据选候选：official reward support、连续 lineage、跨 discovery episode recurrence。不得读取 skill 名称、action 名称或自然语言描述。
3. 候选与 selection rule 一起冻结；qualification/held-out 不参与选择。
4. 对冻结候选运行 `B/G−S/G+S/G+Rand` 的 `1/2/4/8-step` 两种 estimand replay，并保存每步 official transition receipt。
5. authentic/random context 需要报告逐 snapshot token delta；当前 155 个 snapshot 没有一个 exact-token match，均值差 21.03 tokens，仍是明确 confound。
6. 只有 held-out value、blind recurrence 和 controls 同时通过，才允许 Motif/Harness Agent提出 source-grounding-free graph，并进入远域 one-shot binding。

停止规则：如果 fresh no-hint 六游戏中没有任何 discovery-frozen candidate 同时通过 held-out value 与 blind graph recurrence，就停止“显式 transferable backbone”主张。之后只允许转向两个可区分的问题：直接测 `game-trained − base` 的 weight-level adaptation prior，或研究 fail-closed negative-transfer detection。
