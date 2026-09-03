# Harness Retargeting：跨版本实验审计与 Bitter Lessons

日期：2026-08-11

## 结论

本审计覆盖以下五个工作目录中的旧文档、实验摘要、raw episode reports 与
Slurm logs：

```text
Multi-hop-Reasoning-VLM-Agent
Multi-hop-Reasoning-VLM-Agent-experiment-clean
Multi-hop-Reasoning-VLM-Agent-github-main
Multi-hop-Reasoning-VLM-Agent-source-fresh-v1
Multi-hop-Reasoning-VLM-Agent-two-agent-clean
```

这些目录是同一项目的连续快照，不是五次独立 replication。多个核心文档具有相同
SHA-256，实验结果也沿分支继承。因此证据应按时间与机制演化解释，不能把目录数量当作
独立样本数。

跨版本最稳定的结论是：

> 旧系统多次证明了 source context 能改变行为、Harness 能执行和 veto、target adapter
> 能提高安全性；但这些都不等于 source skill 具有增量价值。正迁移只在 source 与 target
> 共享可执行的中层 option structure、且 target 保留 native grounding 时出现。

这支持继续研究 frozen skill + target harness retargeting，但需要严格限制 Harness 的
authority。否则 Harness 会包含 target policy，成功无法归因于 source skill。

## 证据演化

| 阶段 | 观测结果 | 正确解释 |
|---|---|---|
| 早期 task-axis Harness | 2048→Tetris 有 2/3 skills 被 promotion；Tetris→2048 有 4/6 | 多个 PASS 来自 observational hops、unresolved-action soft skip 或 undecidable predicates；eligibility widening 不是 task transfer |
| 旧 mega-skill | 180 members 被压成 20 families、14 signatures；`thr4` 把 82 skills 合成一个 `ACT → VERIFY` family | 高层 event templates 发生 representation collapse；统一闭环是 interface protocol，不是 transferable skill |
| Skill-internal graph | 88 次模型调用、约 485 万 tokens；authentic 条件得到 0 个 backbone-eligible candidate，controls 得到 6 个 | LLM 可以生成合法 topology，但合法 topology 不证明 source-specific structure |
| Source matched interventions | 155 matched snapshots、620 treatment receipts 证明 authentic context 会改变 action；one-step value 和 blind recurrence 未通过 | `policy influence` 已建立，`source value` 与 transferable motif 未建立 |
| Principled Game-SFT→ALFWorld | 合并全部 shards 后，base 为 7/35 ID、6/24 OOD；Game-SFT 为 1/35、0/24；Base+Harness 与 Game-SFT+Harness 均为 0 | 原域 SFT 和 generic Harness 都可能造成强负迁移；训练或 interposition 本身不是 transfer |
| Harness V3 target pilot | source-conditioned 与 target-only 都是 2/8；source cost 约 9.8 倍。Online Harness、naive source 与 rotated control 后续均为 1/4 | source treatment 真实接入但没有增量价值；changed-action、receipt completeness 和 online verification 都只是机制指标 |
| Weak-knowledge ALFWorld pilot | 初次出现一个 authentic-only success；冻结 10 seeds 后 target-only 0/10、authentic 1/10、generic 2/10 | 单条漂亮 trajectory 是 replication candidate；未复制时应降级为 sampling variance |
| Controlled hidden-rule→diagnosis | zero-shot qualification net return `+0.0611`，held-out `+0.0672`，且胜过 shuffled/marginal | 当 `TEST / COMMIT / ABSTAIN` 的 belief/action-value structure 真正同构时，target-native grounding 足以执行 source symbolic policy |
| Hierarchical synthetic→ALFWorld V4 | authentic 19/24，neural-only 14/24，shuffled 12/24，phase-permuted 2/24 | 目前最强正证据；有效 factorization 是 source 选择 option、target 在 option 内选择 concrete action |
| Real Candy→ALFWorld | fresh source forks 全部可重放，但 held-out normalized regret authentic `0.634`、shuffled `0.584` | 真实 intervention 存在不等于 observable state 能预测 value；source grounder 不合格时必须停止 target transfer |
| Real-game→WebShop V7/V8 | V7 strict `1/8→0/8`、mean reward `0.573→0.531`；V8 恢复安全但零 intervention | source action arbitration 会放大 target-grounder 错误；安全 abstention 不是 transfer benefit |
| Real Sokoban→ALFWorld V1 | authentic `14/24` 高于 target-only `11/24`，但 phase-permuted 为 `18/24` | source POSITION-heavy occupancy 与错误 feature semantics 被迁移；表面 success 增益未通过因果 control |
| Real Sokoban→ALFWorld V2 | source effect guard 在新 boards 为 `95/96`，target masked grounder 通过；但 target authentic `10/24`、target-only `16/24` | effect direction 有 source 内容，但 binary COMMIT 折叠 ACQUIRE/TRANSFORM/PLACE，产生 typed-effect negative transfer |

主要本地证据：

- [`OLD_RESULTS.md`](OLD_RESULTS.md)
- [`SKILL_INTERNAL_BACKBONE.md`](SKILL_INTERNAL_BACKBONE.md)
- [`SOURCE_GATE_FAILURE_DIAGNOSIS.md`](SOURCE_GATE_FAILURE_DIAGNOSIS.md)
- [`GAME_TRAINING_REASONING_TRANSFER.md`](GAME_TRAINING_REASONING_TRANSFER.md)
- [`ALFWORLD_WEAK_KNOWLEDGE_TEACHER_PILOT.md`](ALFWORLD_WEAK_KNOWLEDGE_TEACHER_PILOT.md)

以下较新证据在审计时存在于 workspace，但仍是本分支基线上的未跟踪实验产物，因此本次
文档提交只记录其路径，不把它们纳入提交：

```text
docs/CONTROLLED_INTERVENTION_TRANSFER_V1_V2_RESULTS.md
docs/MULTISOURCE_ALFWORLD_NEUROSYMBOLIC_V2_V4_RESULTS.md
docs/REAL_GAME_TO_ALFWORLD_INTERVENTION_V1_RESULTS.md
docs/WEBSHOP_NEUROSYMBOLIC_CAUSAL_V7_RESULTS.md
docs/WEBSHOP_NEUROSYMBOLIC_APPLICABILITY_V8_RESULTS.md
```

## Bitter lessons

### 1. Rollout 数量不等于 skill evidence 数量

旧正式六游戏 rollout 没有完整 proposal set、post-transition verdict、explicit replan 或
abstain。部分长轨迹整局只选择一个 skill：Tetris 292 步均为 `COMMIT/POSITION`，Candy
300 步均为 `COMMIT/CLEAR`。这种数据可以证明 checkpoint policy 和静态 context 的原域行为，
不能事后恢复 closed-loop procedural skill。

后续 source 数据必须原生记录 decision cycle，或在同一 snapshot 上执行 matched
interventions。不能让 Agent 给旧 trajectory 补写没有发生过的 `VERIFY/RECOVER/BRANCH`。

### 2. 通用 event loop 不是 skill

```text
observe → act → verify → recover
```

可以作为 Harness ISA，但没有足够内容成为 transfer object。Frozen skill 至少需要：

```text
typed precondition
relational roles
choice rule
expected transition
failure-specific recovery
termination condition
source intervention lineage
```

如果一个表示在 authentic、receipt-only、renamed、shuffled 或 skill-off 条件中同样容易出现，
它应标记为 `GENERIC_INTERFACE_STRUCTURE`，不能晋升为 source skill。

### 3. Behavior change 不等于 value

以下指标只能证明 treatment active：

- changed-action/changed-option rate；
- source admission rate；
- graph recurrence；
- Harness replan 数；
- receipt completeness；
- 更短但 reward 更低的 trajectory。

Source 必须在 matched source forks 上超过 shuffled/random/marginal controls，并在 target 上
改善 official outcome 或在相同 outcome 下减少 steps/tools/cost，才能支持 transfer。

### 4. Abstraction level 比 surface semantics 更重要

ALFWorld V1 把 `take/cool/put` 等中间阶段都压成 terminal `COMMIT`，四个 target 条件均为
0/8。V4 使用 `SEARCH → ACQUIRE → TRANSFORM → PLACE → VERIFY` 后才获得正结果。

因此 source 与 target 不需要共享实体名或 action token，但必须能在相同控制层级上表示：

- option precondition；
- option completion；
- option transition；
- delayed option value。

增加更多 source domains 不会自动修复层级不匹配。

### 5. Source 应选择 option，不应选择 target action

V2 允许 source values 重排 concrete ALFWorld actions，导致错误 completion prediction 被放大；
V3/V4 改为 source 只选择 abstract option，target neural policy 在该 option 内选择原生 action，
结果明显改善。

新的 Harness contract 应保持同一边界：

```text
source skill: choose canonical option / branch
target harness: ground state, realize selected option, verify effect
target Decision model: choose concrete native action inside the option
```

### 6. Harness 可能偷带 target policy

V4 是重要 mechanism evidence，但其 ALFWorld grounder 显式定义
`SEARCH/ACQUIRE/TRANSFORM/PLACE/VERIFY`，并包含 action→option 与 required-phase parsing。
因此它不能单独证明 ontology 是从真实 game 自动迁移的。

Harness 可以：

- 把 raw target observation 映射为 canonical typed state；
- 把已选 canonical action 编译为合法 target action；
- 返回真实 transition/effect receipt；
- 在无法 grounding 时拒绝执行。

Harness 不可以：

- 选择 option 或 policy branch；
- 依据 target reward 调整 frozen source skill；
- 在多个 canonical actions 中决定哪一个更有价值；
- 为某个 task/sample 手写解决规则。

### 7. Safety、applicability 和 utility 是三个独立 gate

V8 的 conservative shield 防止了 V7 的 premature commit，但通过零 intervention 达成。
这建立 safety，不建立 utility。正式报告必须分别给出：

```text
safety: harmful intervention / strict-success regression
applicability: admitted interventions that produce predicted target effect
utility: paired official outcome or equal-outcome efficiency delta
```

### 8. Target capability 必须先单独成立

旧 visual/video smokes 暴露过 fabricated evidence、坐标 grounding failure、重复无信息图像变换、
budget exhausted 和缺少 final answer。若 target-only 的 tool/evaluator loop 尚不可靠，source
treatment 的失败不能解释成 skill mismatch。

Target-native observation/action/effect grounder 应先在独立 adaptation/validation split 通过，
但 grounder pass 本身仍不能授权 positive-transfer claim。

### 9. Identity 与 reproducibility bugs 会制造科学结论

旧运行先后暴露：

- `seed` 被传入 reset，但历史 observable state 无法重新构造；
- TextWorld reset 顺序与 input-order task ID 不一致；
- WebShop goal 跨 restart 不稳定；
- provider 在 temperature 0 下仍产生不同 completion；
- other-game condition 使用了错误的 recurrent source state；
- retry 发送相同 memoized request，无法修复 invalid JSON；
- broad glob 混入 failed/debug adaptation artifact。

因此正式 run 必须冻结 task identity、goal/state hash、runner/config/model hash、exact-request
cache、artifact allowlist 和 environment fingerprint。修复 runner 后所有 conditions 必须同轮重跑，
不能拼接旧 condition 结果。

### 10. 单个正例必须复制，held-out 只能消费一次

一个 task 或一个 seed 上的 authentic-only success 只能标记为
`REPLICATION_CANDIDATE`。冻结 replication 未复现时，应降级而不是继续 outcome-driven prompt
调整。被读取的 qualification/held-out states 必须进入 consumed manifest；后续只能用于机制诊断，
不能再次承担 confirmatory evidence。

## 对 Harness-retargeting proposal 的约束

建议形式化为：

```text
z_t = f_obs^B(o_<=t)
u_t = S(z_<=t)
a_t = f_act^B(u_t, z_t)
e_t = f_feedback^B(o_t, a_t, o_{t+1})
```

其中：

- `S` 是 source-only discovery/qualification 后冻结并 hash 的 skill；
- `f_obs^B` 只做 target→canonical grounding；
- `f_act^B` 只 realization 已由 `S` 选择的 canonical action；
- `f_feedback^B` 只报告 target-native observable effect；
- target test receipt 不更新 `S` 或 frozen Harness artifact。

必须保持同一个 target Harness，比较：

| Condition | 识别目标 |
|---|---|
| raw target-only | 原始 target policy 能力 |
| null skill + retargeted Harness | Harness 自身的贡献 |
| shuffled/wrong source skill + same Harness | 任意程序/context 是否都有效 |
| authentic frozen source skill + same Harness | source-specific skill contribution |
| target-written skill + same Harness | target oracle upper bound |

核心 estimand 是：

```text
(authentic frozen source skill, same Harness)
    - (null/shuffled source skill, same Harness)
```

而不是只比较 `skill+Harness` 与 raw target agent。

## 当前项目应采取的下一步

1. 不先建覆盖所有 domain 的大 Agent VM。
2. 只在 source data 中寻找一个低层、typed、intervention-supported relational skill。
3. 要求该 skill 在 source qualification/held-out forks 上胜过 shuffled、marginal 和 wrong-source controls。
4. 冻结 skill schema、parameters、lineage 与 content hash。
5. 在 disjoint target adaptation split 训练 Harness grounding；不得使用 confirmation/held-out outcome。
6. 冻结同一个 Harness，运行 null/shuffled/authentic/oracle 四条件 paired experiment。
7. Authentic 未超过 null/shuffled 时，结论是 Harness-only recovery，不称为 game-to-target transfer。

如果现有 game rollouts 无法提供第 2–3 步所需的 intervention evidence，正确动作是重新采集
专门暴露该 skill 的 source trajectories，而不是继续提高自然语言 abstraction 或在 target log
上反向设计一个 source skill。

## Claim boundary

现有证据支持：

- Harness-mediated neural-symbolic factorization 技术上可行；
- target-native grounding 与 source option control 的分工是合理的；
- controlled hierarchical source→ALFWorld 已出现受限正迁移；
- real-game source action/value 直接进入 WebShop 尚未产生 success-rate improvement。

现有证据不支持：

- 已经从旧 arcade/game rollouts 抽取到可冻结的跨域 skill；
- 当前 WebShop paired-label recovery 可归因于 game source；
- generic `observe/act/verify/recover` loop 是 non-trivial transferable skill；
- 五个仓库目录构成五次独立 replication。

最新 V1/V2 的完整结果与 held-out 边界见
[`SOKOBAN_ALFWORLD_NEUROSYMBOLIC_TRANSFER_V1_V2_RESULTS.md`](SOKOBAN_ALFWORLD_NEUROSYMBOLIC_TRANSFER_V1_V2_RESULTS.md)。

## V20–V24 与 controlled V3 的新增结论

V24 给出一个更尖锐的 bitter lesson：预测“这个 target state 最终会成功”不等于预测
“source intervention 相对 target abstain 会导致成功”。V24 neural risk 在 271 个已打开 forks 的
OOF 中是 `6W/0L`，但 sealed confirmation 的 7 个 admissions 全部为 ties。与此同时，
always-source 在完整 200-task split 上得到 `5W/0L`、`+2.5pp`，却与 lexical control 完全相同。
因此 relation intervention 有机制信号，但 non-trivial selector 未验证。完整审计见
[`REAL_SOURCE_RELATION_V20_V24_RESULTS.md`](REAL_SOURCE_RELATION_V20_V24_RESULTS.md)。

受控 V3 则第一次在真正的 target-native MLP grounding 下通过完整 cross-domain gate：source
迁移 state-dependent `TEST/COMMIT` matched intervention values；target MLP 只从自己的 anonymous
binary calibration outcomes 学 grounding。两个新 target splits 上，success 相对 target-only
分别提高 `+2.00pp` 与 `+1.95pp`，且 paired CI 下界均严格大于零；authentic 同时超过
within-state shuffled 与 source-marginal controls。完整结果见
[`CONTROLLED_NEURAL_SYMBOLIC_TRANSFER_V3_RESULTS.md`](CONTROLLED_NEURAL_SYMBOLIC_TRANSFER_V3_RESULTS.md)。

这使 claim boundary 变为：

- controlled neural-symbolic mechanism transfer：已验证；
- real-game relation intervention：sealed split 上有正机制效应；
- real-game non-trivial neural selector：V24 未验证；
- real arcade → ALFWorld/WebShop：仍未验证。

## V13 更新：真实 Sokoban effect program → WebShop 已验证

上面的最后一条已被 V13 的新证据部分取代。V13 没有重新使用 ALFWorld 的 binary
POSITION/COMMIT value prior，而是迁移 fresh-confirmed Sokoban intervention-effect relation：
只有 target-native neural grounder 预测到正向、可验证的不可逆效果时才 COMMIT，否则保持可逆
POSITION/PREPARE 或 abstain。

在运行前冻结的 WebShop goals `114–145` 上，authentic 为 `18/32` strict success，target-only
为 `11/32`；paired `7W/0L/25T`、`+21.875pp`、双侧 exact `p=.015625`。Authentic 同时严格
超过 target-native myopic、commit-availability、inverted-effect 和 POSITION-prior controls；
192 个 receipts 完整、hash 有效、initial states matched、零 failure、零 retry。

因此截至 V13，更新后的边界是：

- controlled neural-symbolic mechanism transfer：已验证；
- real Sokoban intervention-effect program → local WebShop：已验证；
- source/target typed correspondence 自动发现：未验证，当前仍由研究者定义；
- Thunder/任意 arcade → 任意 target：未验证；
- TIR/Video-Holmes external validity：未验证，且现有 adaptation preflight 尚未通过。

完整 protocol、结果、V12 pre-action abort 和限制见
[`SOKOBAN_WEBSHOP_NEUROSYMBOLIC_TRANSFER_V12_V13_RESULTS.md`](SOKOBAN_WEBSHOP_NEUROSYMBOLIC_TRANSFER_V12_V13_RESULTS.md)。
