# Phase 13：ALFWorld source acquisition value 与 program-content control

## 最终结论

本轮把 Phase 12 的可识别性问题真正闭合了：

> **ALFWorld execution gain 的直接原因是正确的 symbolic program content，不是 source
> provenance 本身。Source intervention 的可测价值在 program acquisition：它用 0 条完整
> target trajectory 得到该 content；source-blind target learner 在 `K=0` 必须 abstain，看到
> 1 条完整、有序、成功的 target trajectory 后才恢复同构 program。**

所以两个极端说法都不正确：

- “只要写任何同构大小的 target controller 都行”不成立。generic、错误 cardinality、effect
  permutation 和 binding/relation permutation 全部回到 raw 的 `4/14`；
- “program 一旦完全相同，source 出身仍有额外执行魔力”也不成立。source-induced 与 target-only
  `K=1` 在 fresh tasks 上都是 `9/14`，14/14 完整 action/state/effect traces 逐步相同。

精确的 claim 是：

> Source interventions can automatically acquire a useful symbolic program without complete target
> demonstrations. Correct program content transfers through a target-native neural grounder; once
> extensionally identical content is otherwise supplied, provenance has no additional execution
> effect.

## Fresh matched result

V23 从官方 ALFWorld train multiplicity trajectories 中冻结一个与既有 execution pools 不重叠的
identity rank slice，先对 32 个 identities 做 TextWorld compiler audit，再按照预先冻结的
`SOLVABLE_TRUE_ONLY` eligibility rule 保留全部 14 个 compiler-valid tasks，不使用 policy outcome、
observation walkthrough 或人工替换。

八个条件共享相同的 task identities、60-step horizon、target neural grounder、causal-effect head、
admissible-action interface 和 target executor：

| Condition | Complete target demos | Success | 相对 raw |
|---|---:|---:|---:|
| neural-only / raw | 0 | 4/14 | — |
| **source-induced content** | **0** | **9/14** | **5W/0L/9T** |
| source cardinality control | 0 | 4/14 | 0W/0L/14T |
| source effect-permuted control | 0 | 4/14 | 0W/0L/14T |
| generic single-relation scaffold | 0 | 4/14 | 0W/0L/14T |
| target-native recurrent ceiling | specification | 9/14 | 5W/0L/9T |
| **target-only `K=1` induced content** | **1** | **9/14** | **5W/0L/9T** |
| target `K=1` binding/relation-permuted | 1 | 4/14 | 0W/0L/14T |

Source-induced 与 target-only `K=1` 不只是 aggregate outcome 相同：

- 14/14 success/failure outcomes 相同；
- 14/14 完整 action traces 相同；
- 14/14 before/after state hashes、effect receipts 和 steps 相同；
- target-only arm 的 source admissions 为 0；
- source 和 target-only 相对 raw 都是 `5W/0L`，没有 negative transfer；
- authentic source condition 有 120 次 harness-authorized source admissions，并产生 76 次
  post-first-relation action divergence；三个 source controls 都是 0 admission。

paired exact two-sided sign test 为 `p=0.0625`。样本量 14 不足以作为 powered split-level
population claim；它是完整 matched matrix 的 fresh mechanism replication。此前 V13/V14 对完整
45-task packaged `valid_train` multiplicity population 的 performance evidence 保持不变。

机器可读 compact audit 位于
[`results/alfworld_source_acquisition_value_v24_summary.json`](results/alfworld_source_acquisition_value_v24_summary.json)，
summary SHA-256 为 `e6d9a866f7153bfe4015537c743ce2567b10c3258e7e64e3b3baa51fe791fc16`。

## Target-only acquisition curve

为了不把人工 target controller 当成零成本 baseline，V16 新增了 source-blind learner。输入仅为完整
target trajectories 的 `(state, action, observed effect, next_state)` tuples；它不读取 source artifact、
source identity、source operator ID，也没有命名的 `EXPLORE/BACKTRACK/COMMIT` 模板。

Learner 在共享的匿名 typed-operator ontology 上归纳：

```text
activation: one positive relation observed and one relation remains
CONTROL --IGNORE--> CONTROL
CONTROL --BIND_INSTANCE--> RELATION
RELATION --IGNORE--> RELATION
RELATION --RELATE_SLOT_CLOSED--> TERMINAL
argument rule: preserve first positive relation handle
abstain: missing/ambiguous binding, handle conflict, nonconforming effect
```

Retrospective chronological protocol 使用 V13 的 9 条 eligible raw successes 作 development，使用更晚
V14 的 11 条 eligible raw successes 作 held-out qualification：

| Target complete-demo budget | Program status | Match source normal form | Held-out support | Shuffled | Permuted |
|---:|---|---:|---:|---:|---:|
| 0 | `ABSTAIN_NO_COMPLETE_TARGET_TRAJECTORY` | no | 0/11 | 0/11 | 0/11 |
| 1 | `TARGET_ONLY_RECURRENT_PROGRAM_INDUCED` | yes | 11/11 | 0/11 | 0/11 |
| 2/4/8 | qualified | yes | 11/11 | 0/11 | 0/11 |

V17 又在已经消费的 V14 21-task population 上做 execution diagnostic：raw `11/21`、source
`18/21`、target-only `K=1` `18/21`，21/21 action/state/effect traces 与 source 完全相同。V17
没有新增 prospective success claim；它只验证 learned target artifact 确实控制 policy，而不只是
静态 support checker。

V16 report SHA-256：
`c04be83dbf50fd348bbe41738022399295f2026d2ed760f1c77a05ea1e05f5ee`；V17 report
SHA-256：`20757c00c5049c33e13b333b0cae36f96596b1292ef78614487aca7cd86b45ed`。

## Source-only induction 仍然成立

Fresh result 没有把 source program 换成人工模板。执行使用的 source acquisition/relation artifacts
仍满足：

- induction authority 为 `SOURCE_STATE_ACTION_EFFECT_NEXT_STATE_ONLY`；
- `target_data_read=false`；
- `named_controller_template_used=false`；
- typed operators、transition graph、positive-effect cardinality、terminal predicate 和 abstention rule
  都由 source intervention tuples 归纳；
- held-out source transition/program conformance 通过；
- shuffled-effect binding 被拒绝；
- target action 只能由 frozen target-native neural grounder/executor 发出。

统一 neural-symbolic authority chain 没有改变：

```text
source intervention tuples
  -> source-only typed symbolic program
  -> unified applicability/authorization harness
  -> target-native neural grounding over current admissible actions
  -> target-native executor
  -> ALFWorld transition/effect receipt
```

Target-only control 只替换最上游 program acquisition 路径，复用完全相同的 normal form、grounder 和
executor。因此 source 与 target-only 的 exact trace equality 是 acquisition-cost comparison，而不是
两个不受控系统的粗粒度 success-rate 对比。

## Fresh-pool 与 failure audit

这部分有几个值得保留的 bitter lessons：

1. `valid_seen` 的 8 个 identity-only reserve candidates 中只有 3 个能由当前 TextWorld compiler
   产生 solvable game，不能在看到 compiler result 后偷偷补齐；V18 因最小 task gate fail closed。
2. train missing-game pool 有 228 个候选。V20 冻结 16 个后只有 5 个 compiler-valid；把 11 个
   compiler-rejected identities 强行放进 ITT policy matrix 会在 `reset()` 前失败，不能当作正常 policy
   failure。
3. V21 对下一个 32-task slice 预先冻结 compiler-valid eligibility，得到 13 tasks，但使用 Python
   3.13 时旧 TextWorld `EvalSymbol.derive` 因 `eval` local-scope semantics 抛出 `NameError: r`。
   Phase 11 已记录同一兼容性问题；本轮重复踩到它，说明 interpreter contract 必须在 freeze 阶段成为
   mandatory dependency，而不能靠运行者记忆。
4. V23 在另一个 disjoint 32-task slice 冻结 Python 3.11 executable/hash，得到 14 个
   compiler-valid tasks，并成功完成全部 112 个 condition-task episodes。
5. 原 V23 execution 在所有 episodes 结束后的 report assembly 中，因为 selection schema 从
   `selection_used_observation_walkthrough_or_policy_outcome` 改名为更严格的新字段而触发 `KeyError`。
   Frozen runner、tasks、policy、program、thresholds 均未改。V24 只提供 read-time metadata alias，
   对已消费的 deterministic tasks 重建 report；freshness 只属于第一次 execution，reconstruction
   不伪装成新的 fresh trial。完整 reconstruction log 与 report 均保留。

V24 reconstructed report SHA-256：
`dd8786e782b029365f8bc2747bbe7ffe5f847d2d4a552337f8c1b3a37054cc91`。全部 19 个
fresh/acquisition/mechanism gates 为 true。

为避免提交 8.8 MiB raw JSON，lossless deterministic archives 保存在：

- `runs/alfworld_target_induced_policy_v17_consumed/report.json.gz`，archive SHA-256
  `8efe743a68231245dfa9a333e80b3e6d463b0554c930b5595dfd067eaf3c8db8`；
- `runs/alfworld_target_acquisition_py311_v23/report_deterministic_reconstruction.json.gz`，archive
  SHA-256 `1f900dfcc4ae2202cbb7321f21fa60033409ba3e013432508f72dde5d0d2bc7c`；
- `runs/alfworld_target_acquisition_py311_v23/deterministic_reconstruction.log.gz`，archive SHA-256
  `80d40bb18d80dc9858a4b079a114b39bfc2785dfc2662675c6fd3fb2fc6102e2`。

## 这回答了什么、还没回答什么

现在可以回答：

- **是不是任何 target controller 都有效？** 不是。错误但同规模的 source/target programs 与 generic
  scaffold 都是 `4/14`；learned transition/effect/argument structure 是必要内容。
- **正确的 target-written/isomorphic controller 能不能匹配？** 能，而且必然应该能。相同函数的
  provenance 无法从 target behavior 识别。
- **source intervention 的增量在哪里？** 在 acquisition：source 需要 0 条完整 target demo；当前
  source-blind target learner 需要 1 条。
- **这还是 neural-symbolic transfer 吗？** 是。symbolic program 从 source intervention tuples 归纳并
  决定 option/obligation；target-native neural grounder 绑定 concrete action；harness 保持权限与
  applicability gate。

仍不能声称：

- source 优于一个带正确 domain prior 的人类或 LLM 手写 controller；本实验没有给 authoring effort
  定价；
- 一条 target trajectory 是所有 target-side learners 的 information-theoretic lower bound；
- 14-task fresh slice 给出了 powered ALFWorld population effect；
- 该 recurrence 自动覆盖 ALFWorld 非-multiplicity families 或 video MDP；
- source provenance 本身有独立 causal effect。

下一项真正有价值的实验不再是增加“同 program、不同标签”的 tasks，而是扩大 acquisition curve：
比较 source interventions 与 matched target exploration budget、target LLM synthesis、retrieval/memory
和 human specification cost，并在第二个需要不同 program content 的 target family 上重复同一 IR/grounder
separation。

## 关键文件与复现

- target-only inducer：`src/motif_transfer/alfworld_target_recurrent_induction.py`；
- V16 acquisition curve：`scripts/qualify_alfworld_target_acquisition_value_v16.py`；
- V17 consumed execution diagnostic：`scripts/run_alfworld_target_induced_policy_v17.py`；
- V23 freezer/compiler audit：`scripts/prepare_alfworld_target_acquisition_py311_v23.py`；
- frozen historical runner：`scripts/run_alfworld_target_acquisition_fresh_v19.py`；
- V24 schema-only recovery：`scripts/recover_alfworld_target_acquisition_py311_v24.py`；
- compact summarizer：`scripts/summarize_alfworld_source_acquisition_value_v24.py`。

Python 3.11 reproduction commands：

```bash
PY=/fs/gamma-projects/vlm-robot/conda/envs/cosplay-candy-a100/bin/python

$PY scripts/qualify_alfworld_target_acquisition_value_v16.py
$PY scripts/run_alfworld_target_induced_policy_v17.py
$PY scripts/recover_alfworld_target_acquisition_py311_v24.py
$PY scripts/summarize_alfworld_source_acquisition_value_v24.py

$PY -m pytest -q \
  tests/test_alfworld_target_recurrent_induction.py \
  tests/test_recover_alfworld_target_acquisition_py311_v24.py
```

其中 V16/V17/V24 policy commands 需要本 workspace 的 historical raw reports、generated games 和
ALFWorld cache。Portable checkout 可直接使用已提交的 gzip evidence 重建 compact audit：

```bash
$PY scripts/summarize_alfworld_source_acquisition_value_v24.py
```
