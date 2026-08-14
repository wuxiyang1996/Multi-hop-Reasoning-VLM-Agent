# 真实游戏到视觉、视频与 WebShop 的 Neural-Symbolic V5 协议

## 决定

V5 不再把 target 限定为 ALFWorld。冻结三个互补 target：

1. **WebShop** 是主实验。它与游戏一样是有真实 intervention、延迟 outcome 和失败恢复的 MDP，
   但动作、实体和任务语义完全不同。
2. **Video-Holmes** 测试时间事件结构能否迁移到 target-native 的 frame/evidence 操作。
3. **TIR-Bench** 测试空间关系和约束结构能否迁移到 target-native 的 crop/detect/answer 操作。

VisualToolBench 暂不进入 V5 主矩阵。它的 single-turn runner 和官方 rubric judge 已存在，但完整六工具
runtime 仍依赖缺失的外部 key；用降级工具集运行不能当作正式结论。

## 什么被迁移

迁移对象不是旧 skill 名称，也不是自然语言的游戏总结。真实 source trajectory 只暴露：

```text
(state, available actions, chosen action, next state, reward, termination)
```

编译器不预先接受任何有语义的 option 名称。它从 episode-disjoint 的 source discovery split 读取
低层 effect vector：

```text
state-change
controllable-entity-change
object/relation-change
available-action-set-change
immediate/delayed progress at h=1/2/4/8
repeat/no-op
termination
```

然后无监督归纳匿名的 `z0...zk-1` temporal options，并学习它们的 precondition、transition effect、
termination 与 h=1/2/4/8 value。`k` 只能根据 source qualification 选择。任何
`reposition/enable/commit/recover` 名称都只能在冻结以后作为展示 alias，不能进入模型。旧数据中的
`intentions`、`skill_reasoning`、skill name/description 和 summary 一律禁止进入编译器。

Target 使用自己的 neural grounder，把原生 observation 和 candidate action/tool 映射为 option 概率、
低层 effect、前置条件、即时/延迟 progress、no-op、不可逆性和 termination 概率。Source model 只能在匿名
option 层改变
选择，不能输出 WebShop BID、图像坐标、视频时间戳或最终答案。

这产生清楚的归因边界：

```text
real-game receipts -> symbolic option/value model
target adaptation -> target-native neural grounding
两者组合 -> target option arbitration
```

所有条件共享完全相同的 target grounder。因此 authentic 的增量不能来自额外的 target 标注或手写
source-to-target action mapping。

## Source 与 target 配对

| Target | Authentic source | 主要结构 | Other-game control |
|---|---|---|---|
| WebShop | Sokoban + Tetris + 2048 | enable-before-commit、延迟 progress、recover | Candy + Mario |
| Video-Holmes | Thunder + Mario | temporal change、maintain/reposition、budgeted commit | Tetris + 2048 |
| TIR-Bench | Sokoban + 2048 | spatial relation、constraint satisfaction、check | Candy + Mario |

Thunder 的 matched replay 只用于 source causal gate，不能与 observational source episode 混为同一种证据。

## 冻结 split

`scripts/freeze_real_game_multitarget_v5.py` 只根据 salted sample-ID hash 选择 split，不读取 task outcome。
生成的 `configs/real_game_multitarget_v5_manifest.json` 包含每个 source 文件 hash、target dataset hash、
discovery/adaptation、qualification、held-out 和 reserve IDs。

先前已查看或执行的样本被排除：

- TIR-Bench: `668`, `1001`；
- Video-Holmes: `nT7w-T2aBOo.Q770`；
- WebShop: `webshop.32`, `webshop.40`。

Source discovery、qualification 和 held-out 按 episode 分割。旧的 56 个 one-step program 是跨全部
episode 编译的，因此不能直接进入 V5；V5 必须从新 discovery split 重新编译。

## Controls 与成立标准

冻结条件包括 target-only、authentic multi-game、single-game、leave-one-game-out、alpha-renamed、
phase-permuted、within-state value shuffle、other-game 和 marginal prior。

Source gate 要求 authentic 在 source-held-out 的多 horizon value prediction 上相对每个 destructive
control 至少降低 10% MSE。Target held-out 之前还要求 qualification 中 source 实际改变至少 8% 的
option selection。

最终成立需要：

1. authentic 提高 official target outcome 或在相同 outcome 下减少 steps/tools；
2. authentic 优于 phase-permuted、within-state shuffle、other-game 和 marginal；
3. alpha-renamed 与 authentic 等价，证明结果不依赖游戏词汇；
4. target-native grounder、Decision model、seed、budget、初始资产完全 matched；
5. 无效 receipt、伪造 evidence 或 binding 不稳定时 source-off，而不是继续产生建议。

WebShop 首先运行，因为它提供最干净的 intervention-level 对照。Video-Holmes 随后测试时间证据结构；
TIR-Bench 最后运行，并按 task family 单独报告，避免把不具空间/关系结构的题型混入主 estimand。
TIR 的冻结 allowlist 为 `refcoco`、`maze`、`jigsaw`、`visual_search`、`word_search`、
`spot_difference` 和 `rotation_game`；family 选择不读取答案或模型 outcome。

## Source development probe

`scripts/run_real_game_source_option_probe_v5.py` 已用 Sokoban、Tetris、2048 的 discovery split
拟合匿名 effect clusters，并只用 qualification split 在 `k=3...8` 中选择 `k=7`。它没有读取
skill、intention、reasoning 或 summary 字段。episode-held-out 的 h=1/2/4/8 综合 MSE 为：

| Condition | Aggregate MSE | Authentic relative improvement |
|---|---:|---:|
| authentic anonymous options | 3.8749 | — |
| phase-permuted | 5.8028 | 33.22% |
| within-episode option shift | 5.2549 | 26.26% |
| marginal context | 4.6094 | 15.94% |
| value shuffle | 4.6512 | 16.69% |

因此真实 source 里存在不能完全由 episode progress、边际 reward 或打乱时间顺序解释的低层 value
signal，可以进入 target qualification diagnostic。结果和冻结参数位于
`runs/real_game_multitarget_neurosymbolic_v5/source_development/`。

证据边界：精确 compiler 是第一次查看这组 episode-held-out 指标之后才固化，因此状态是
`DEVELOPMENT_PASS_FRESH_CONFIRMATION_REQUIRED`，不是 confirmatory source gate。它不能单独授权
positive-transfer claim；正式主张还需要新的 source causal/fresh confirmation，以及未读取 target 上的
matched controls。序列化参数固定到 12 位小数；连续两次运行的 candidate/report 文件 SHA-256 完全一致。

## V6 fresh cross-game confirmation

V5 的 `k=7` candidate 随后在打开确认结果前被锁定，并原样应用到两个未参与
candidate 拟合或选型的游戏：Candy Crush 与 Super Mario。两个确认游戏的
discovery episodes 只用于各自 reward 的 location/scale；cluster centers 与 value
coefficients 均不更新。正式角色包含 24 episodes、1,207 transitions。

预注册确认门没有通过，因此 learned universal ontology 的主张当前不成立：

| Metric | Authentic | Control | Relative change |
|---|---:|---:|---:|
| pooled aggregate MSE vs phase-permuted | 8.5700 | 10.5650 | +18.88% |
| pooled aggregate MSE vs within-episode shift | 8.5700 | 9.0862 | +5.68% |
| pooled aggregate MSE vs marginal context | 8.5700 | 8.2226 | **-4.22%** |
| Candy Crush vs marginal | 2.7960 | 3.2947 | +15.14% |
| Super Mario vs marginal | 14.2774 | 13.0936 | **-9.04%** |

这不是“完全没有 structure”：authentic 明显优于 phase-permuted，并在 Candy
Crush 上优于 marginal。但它不足以证明一个可跨游戏复用的统一 latent option
vocabulary。结果更支持较窄的设计：从 source interventions 学 executable program
或 relation，并在每个 target 使用 native neural grounding；不要让一个 pooled
KMeans ontology 直接承担跨域语义对齐。正式状态为
`FRESH_CROSS_GAME_LATENT_ONTOLOGY_NOT_CONFIRMED`，完整 compact receipt 位于
`docs/results/real_game_latent_options_v6_formal_summary.json`。
