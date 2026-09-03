# Phase 11：ALFWorld program-driven neural-symbolic transfer

## 结论

现在可以回答之前最关键的质疑：**ALFWorld 的成功率提升不只是“harness 允许了一个
skill”，source-induced symbolic IR 确实介入了 option/policy selection，并在自然 headroom
上提高了最终 success rate。**

独立的 V14 prospective replication 使用 V13 后剩余的全部 21 个 execution-untouched
`valid_train` multiplicity tasks，结果为：

| Arm | Success | Mean steps |
|---|---:|---:|
| neural-only | 11/21 | 42.71 |
| **source-induced program** | **18/21** | **28.48** |
| source cardinality control | 11/21 | 42.71 |
| source effect-permuted | 11/21 | 42.71 |
| generic scaffold | 11/21 | 42.71 |
| target-native ceiling | **18/21** | **28.48** |

Source-induced 相对 neural-only 和每个非因果 control 均为 `7W/0L/14T`，exact two-sided
sign test `p=0.015625`。Success rate 从 52.4% 提升到 85.7%，增加 33.3 percentage points；
没有 negative transfer，并与 target-native ceiling 完全相同。

这不是人为削弱 baseline：neural-only 是原来冻结的 target policy；V11 action runtime、
V13 `valid_train` transport、source artifacts、target neural grounder、thresholds、60-step
horizon 和六臂 controls 全部保持不变。

## 这里的 neural-symbolic transfer 到底做了什么

实际 authority chain 是：

```text
source (state, action, effect, next_state)
  -> source-only typed acquisition/relation IR
  -> anonymous BIND / RELATE obligation and SEARCH option
  -> target-native neural ranking over admissible ALFWorld actions
  -> target-native executor
  -> concrete ALFWorld action
```

Source 端归纳的是 intervention-grounded structure：当第一个 relation 已观察到、下一 slot
仍缺少 positive binding 时，进入匿名 acquisition update；binding 建立后再执行 recurrent
relation update。无法满足 learned cardinality/handle contract 时 abstain。它不携带 Sokoban
action names、坐标、ALFWorld object names 或 target outcome，也没有手工提供
`EXPLORE/BACKTRACK/COMMIT` controller 模板。

Target-native neural grounder 只负责在 source 选定的 option family 内绑定当前 admissible
action；source selector 本身不能输出 target action。V14 的 280 个 program-active decisions
都有一一对应的 authority receipt：`source_selector_action_emitted=false`、target executor
恰好调用一次、selection 时不读取 formal outcome。

因此这里“统一”的是 IR schema、applicability/utility authorization 和 action-authority
protocol；不是一个共享手写 controller。Transferred symbolic function 与 concrete target
grounding 的职责是分开的。

## 为什么可以说 source IR 驱动了 policy

### V13 retrospective contribution audit

V13 原来已经是 prospectively frozen 24-task run（20/24 vs 13/24），但旧 report 主要证明
success 和 authority，没有把 7 个 rescues 的动作因果链单独列成 gate。新的独立审计不改
任何 rollout，重新验证全部 report/episode/record/receipt hashes，并发现：

- 75 个 source-admitted action 与同状态 raw fallback 不同；
- 7/7 rescues 都先出现 source acquisition option 导致的动作分歧；
- 随后都出现 source relation transition，把 completed slots 从 1 推到 2 并达到 success；
- 291/291 program-active actions 均由 target-native executor 发出；
- 三个 controls 的 24/24 action traces 与 neural-only 完全相同；
- authentic 的 24/24 traces 与 target-native ceiling 完全相同。

这是对已有 prospective outcomes 的 retrospective mechanism audit，不把 V13 伪装成新实验。
机器可读结果：
[`results/alfworld_policy_contribution_v13_audit.json`](results/alfworld_policy_contribution_v13_audit.json)，
audit SHA-256 `f285aa43efd3c69285a8b3b8be46906af6f4273451d8b824607944a8668a056f`。

### V14 prospective contribution gate

V14 在任何新 task outcome 前把 causal-policy criterion 写入 runner：每个 rescue 必须满足

```text
first observed relation
  < source-admitted acquisition action that differs from raw fallback
  < source-advanced terminal relation transition
  = official task success
```

正式结果中：

- 91 个 source-admitted policy divergences，其中 76 个是 acquisition divergences；
- 7/7 rescues 都满足上面的严格时序；
- 18 个 source terminal transitions，0 regression；
- 三个 controls 的 21/21 exact action traces 等于 neural-only；
- authentic 的 21/21 exact action traces 等于 ceiling；
- 126/126 episode hashes、4,784/4,784 record hashes、280/280 authority receipt hashes
  经独立脚本重新验证。

所以提升不能由 source identity、generic recurrence、错误 cardinality、effect permutation，
或单纯 harness scaffolding 解释。有效信息是 source-induced acquisition/relation structure；
具体 target action 仍来自本域 neural grounding。

## Reserve 与 preregistration

Packaged `valid_train` multiplicity pool 固定为 45 tasks。V13 执行 24 个后，五个历史仓库的
run-JSON exposure scan 仍找到 21 个从未执行的 tasks。V14 不抽取“看起来容易”的子集，而是
冻结并执行全部 21 个：

- frozen config self-hash：
  `f0af81719bbcb9fa40f5711b45beca70faf8cbbc98281aa643b30001313968d6`；
- identity-only selection hash：
  `b795b1d30f56d8b640a55d2f8b7f3f23e2acff855c8a273837e5a567f139a4fc`；
- final report self-hash：
  `de5710f5cf305a64a0365f3c94ea550c1fd43c0e61fd888a6f70ea65abf0dc51`；
- independent audit self-hash：
  `7af5f51410a3195b1aae422fdb7688b075ad78cf243e5f328f6e7612ea4b118b`。

第一次启动误用了 Python 3.13；旧 TextWorld grammar 在 `reset()` 生成 observation 前因
`eval` scope 变化报 `NameError: r`。同一 task 在 ALFWorld-compatible Python 3.11 下通过。
随后使用完全相同的 frozen config 重跑；没有在失败尝试中产生 action、reward 或 outcome，
也没有修改 task set、policy、IR、grounder、threshold 或 gate。这是 interpreter compatibility
修正，不是 selective retry。

## 三个剩余 failure

V14 authentic 的三个 failure 与 ceiling 完全相同：

- Cup→Shelf 和 ToiletPaper→Drawer 在 60 步内没有完成第一个 relation，因此 source recurrence
  从未获得合法 activation；
- CellPhone→Dresser 完成了第一 slot，source program active 46 steps，但 target-native
  acquisition candidates 最终耗尽，仍未找到第二 instance。

这说明 18/21 是当前 target-native grounding/execution ceiling，不是 harness abstention 造成的
额外损失。若继续提升 success rate，正确方向是 development-only 改善第一 relation acquisition
和 exhausted-search recovery，然后重新冻结新的 target-native grounder；不能在这 45 个已全部
消费的 tasks 上调参。

## Claim boundary

现在成立：

- source-only、intervention-grounded symbolic program 能跨 game→ALFWorld 迁移；
- transferred IR 会改变真实 option/action trajectory，并显著提高 success rate；
- concrete action 始终由 target-native neural grounding/execution 完成；
- 两个独立 populations（V13 24 tasks、V14 21 tasks）都得到 7W/0L；合计覆盖完整 45-task
  `valid_train` multiplicity pool，但统计结论分别报告，没有事后合并 p-value；
- wrong-source / permuted-effect / generic-scaffold controls 不能解释提升。

仍不成立：

- 任意 game skill 可自动迁移到任意 ALFWorld family；
- target-native grounding 不再需要；
- ALFWorld 全任务或 100% success；
- source provenance 对同构 target-written program 是必要条件；
- video MDP claim。视频按既定决定继续暂停。

## 复现

旧 TextWorld 需要 Python 3.11 或更早；不要使用 Python 3.13：

```bash
PY=/fs/gamma-projects/vlm-robot/conda/envs/browsergym/bin/python

$PY scripts/run_alfworld_program_driven_policy_v14.py \
  --config configs/alfworld_program_driven_policy_v14_formal.json

/fs/gamma-projects/vlm-robot/conda/bin/python \
  scripts/audit_alfworld_program_driven_policy_v14.py

/fs/gamma-projects/vlm-robot/conda/bin/python -m pytest -q \
  tests/test_alfworld_policy_contribution.py \
  tests/test_alfworld_goal_relation_macro.py
```

关键文件：

- policy-contribution gate：
  `src/motif_transfer/alfworld_policy_contribution.py`；
- V13 independent audit：
  `scripts/audit_alfworld_policy_contribution_v13.py`；
- V14 freezer / runner / independent audit：
  `scripts/freeze_alfworld_program_driven_policy_v14.py`、
  `scripts/run_alfworld_program_driven_policy_v14.py`、
  `scripts/audit_alfworld_program_driven_policy_v14.py`；
- compact result：
  [`results/alfworld_program_driven_policy_v14_summary.json`](results/alfworld_program_driven_policy_v14_summary.json)。
