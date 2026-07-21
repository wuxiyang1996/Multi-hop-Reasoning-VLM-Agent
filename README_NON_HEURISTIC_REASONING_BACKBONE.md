# 非启发式多步 Reasoning Backbone 与 Harness 设计

本文统一此前关于 skill clustering、one-shot transfer、Game SFT、Agent proposer、
predicate 主观性和 Harness 信任边界的讨论，并给出当前实现与下一阶段协议。

## 1. 我们最终要迁移什么

不迁移人工命名的 `NAVIGATE/ACQUIRE/COMMIT`，也不声称 `COLLECT == TAKE`。
要检验的是：

> source 中可重放的多步执行结构或 Agent control hypothesis，经过一条固定 target
> demonstration 的原生证据过滤后，能否成为 target Actor 的可靠 control prior，
> 并在 frozen held-out evaluation 上优于同一条 target demo 单独构造的程序。

通用 backbone 是 event protocol，而不是 domain ontology：

```text
OBSERVATION_RECEIPT
    → AGENT_PROPOSAL_SET
    → NATIVE_ADMISSIBILITY_CHECK
    → ENVIRONMENT_STEP
    → NATIVE_DELTA_RECEIPT
    → AGENT_CONTINUE / ABSTAIN / STOP
    → OFFICIAL_STOP_CHECK
```

它适用于游戏、ALFWorld、Browser、Visual Reasoning 和 Video，因为每个节点只说明
“谁提出了什么、环境允许什么、实际执行了什么、原生状态如何变化”，不预定义
跨域语义。

## 2. “完全不 predefined”为什么不可能

可验证系统至少必须预先固定：

- target/source adapter 的 `reset/step/admissible/evaluator` 接口；
- immutable receipt/hash 规则；
- Agent closed JSON schema；
- 哪些 proof obligations 是可判定的；
- test/adaptation 数据边界。

否则 Harness 无法解析、更无法拒绝 Agent 输出。我们要消除的是：

- domain-semantic predicate ontology；
- 人工 source→target 映射；
- embedding/text similarity；
- threshold、score、phase/tag 和 fixed-length segmentation；
- majority vote 或 LLM-as-judge。

形式接口与 proof obligation 不是迁移 heuristic；它们是让“Agent 只提议、Harness
能证伪”成为可能的最小信任边界。

## 3. 四层证据，禁止跨层表述

| 层级 | 可以声称什么 | 不可以声称什么 |
|---|---|---|
| `OBSERVED_TRACE` | 原生 action trace 完整、连续、可重放 | skill boundary、branch、guard、reasoning |
| `AGENT_HYPOTHESIS` | hypothesis 引用的 receipts 存在且与 observed path 不冲突 | hypothesis 因果正确或穷尽所有可能程序 |
| `TARGET_DEMO_VERIFIED` | binding 与固定 target demo 的原生执行约束一致 | 未覆盖状态、对象、branch 上必然正确 |
| `HELD_OUT_GENERALIZATION` | frozen system 在预注册 held-out 数据上的经验表现 | 普遍语义等价或完整识别 |

只有 snapshot reset 后执行 alternative action、counterfactual 或专门 instrumentation，
才能把 guard/branch/retry/termination claim 升级为 `INTERVENTION_VERIFIED`。

## 4. 当前 source 数据能证明什么

全量审计结果：

```text
299 episodes
23,072 transitions
299/299 full-episode replay pass
all executed actions are native-admissible
all adjacent raw_next_state → raw_state chains are intact
```

但现有 56 个 `CanonicalSkillProgram` 全部只有一个 `COMMIT` step。旧 skill labels、
intentions 和 chosen skills 来自 retrieval/LLM，不是 ground-truth segmentation。
`outcome=true` 也不是可靠 official success receipt；数据中存在 truncated/unsolved
episode，因此不能拿它证明 termination correctness。

所以当前实现新增的 `TraceProgram` 使用唯一不需要 segmentation 的边界：

```text
environment reset → complete episode trace → environment stop/truncation
```

它保留每个 transition 的：

- source file hash；
- state/next-state hash；
- exact available actions hash；
- exact executed action；
- reward/done；
- exact observed temporal edge。

它明确记录：

```text
segmentation = none
official_success_verified = false
agent_proposal_receipted = false
continuation_decision_receipted = false
official_stop_receipted = false
```

因此这是“真正的多步 observed execution program”，不是已经恢复出的 reasoning
program。

## 5. Agent 如何提出多步结构

Agent 可以自由提出：

- 任意 evidence-referenced continuous span；
- action/data identity links；
- candidate boundary；
- candidate guard/branch/loop/retry/verify/termination；
- source trace 到 target demo trace 的 node/slot/control-edge binding。

每个 claim 必须携带 immutable transition IDs 和 proposal receipt。三个角色独立运行：

```text
proposer_a ∪ proposer_b ∪ challenger/skeptic
```

合并器只做：

```text
closed-schema validation
reference validation
content-hash deduplication
set union
```

不做排序、投票或“最佳 program”选择。所有非同构且与 receipts 一致的候选都保留。

需要准确使用术语：如果不预定义有限 grammar 并穷举，就不能声称构造了“所有与
证据一致的 version space”。当前只能称为：

> budget-limited, Agent-proposed, evidence-qualified candidate set

遗漏 hypothesis 不能被解释为已排除。

## 6. Source Harness 的机械检查

Source Harness 只检查：

1. source file、episode、step 和所有 hash 存在且一致；
2. step indices 与 receipt 引用一致；
3. executed action 是该 step native available actions 的 exact member；
4. `raw_next_state[t] == raw_state[t+1]`；
5. observed temporal edge 确实存在；
6. Agent 引用的原生 byte/token span equality 是否精确成立；
7. 未观察 branch/guard/retry 没有 intervention receipt 时保持 hypothesis；
8. legacy `outcome` 不能冒充 official success。

Replay consistency 只能证明“这条 path 出现过”，不能证明 action 导致 reward、guard
必要、branch 完整或 termination 正确。

## 7. Target one-shot binding

```text
source OBSERVED_TRACE + Agent control hypotheses
        ↓
independent target binding Agents
        ↓
one fixed target-native demonstration
        ↓
exact action/admissibility/entity/ordering/delta constraints
        ↓
evidence-qualified surviving candidate set
```

Harness 不比较 semantic predicate，只验证 target-native receipts。运行时：

- 为每个 surviving candidate 机械计算当前 cursor 支持的原生 exact-action set；
- 所有 set 的交集非空：Actor 只在交集上调用一次并选择动作；
- source 条件把所有活动候选 node 的原始 transition receipts 与未验证 edge claims 一并
  提供给 Actor；target-only 对照中该字段为空；
- 交集为空：`NO_COMMON_EXACT_COMMAND`，立即 abstain；
- demo 未覆盖某个 edge：`INCONCLUSIVE`；
- receipt/schema 矛盾：`REJECTED`；
- runtime native delta 偏离 frozen pattern：当前 episode suspend，不能修改 artifact。

这叫 candidate-set exact intersection，不叫投票，也不叫完整 version-space consensus。
对 provenance 不同但结构相同的候选重复调用随机 Actor 会测到采样噪声，因此被明确禁止。
Source receipts 只影响交集内的 Agent 选择，不可扩大 operator/action scope；Harness 不把
Agent 是否使用这些 receipts 判为语义正确。

## 8. 真正 reasoning backbone 需要的新数据

旧 episode 没有保存 Agent 内部 proposal、rejected candidate、verification、retry 或
abstention。下一轮必须统一 event sourcing：

```text
observation receipt
candidate set + Agent/role/model/prompt hash
admissibility decision
selected action
environment transition
native delta
continue/replan/abstain decision
official evaluator result
```

对于 branch/guard，必须从同一 snapshot reset 后执行有区分性的 alternative；对于
retry/recovery，必须保留第一次失败、诊断、第二次执行与 state chain。否则这些概念
只能是 `AGENT_HYPOTHESIS`。

## 9. 决定 source 是否真的有贡献的对照

至少报告：

1. Base；
2. Target one-shot only；
3. Target-native trace program + same one-shot；
4. Source observed trace + same one-shot；
5. Source Agent hypothesis + same one-shot；
6. Randomized/renamed source trace + same one-shot；
7. Core without abstention；
8. Target SFT/GRPO，作为独立非 one-shot baseline。

核心比较：

```text
source reasoning prior + same target demo
vs.
target-native program + same target demo
```

若无提升，就不能声称 game reasoning backbone transfer。

## 10. OpenRouter 35B smoke（2026-07-20）

模型：`qwen/qwen3.5-35b-a3b`。固定 sorted-prefix 4 个 source programs，仅作工程
smoke，不用于选方法或报告 performance。

```text
12 Agent calls = 4 programs × 3 roles
11 valid closed-schema responses
1 hallucinated program ID → rejected
4 explicit Agent abstentions
7 executable candidates
0 endpoint failures
19,032 prompt tokens
1,217 completion tokens
OpenRouter reported cost: $0.00450075
```

Harness admission：

```text
2 CONDITIONAL
2 INCONCLUSIVE (NON_EQUIVALENT_BINDING_AMBIGUITY)
0 semantic-predicate verdicts
```

这说明真实 Agent 可以进入 proposal path，且 proposer disagreement 没有被投票掩盖。
它没有证明 source skill transfer；当前单步 source programs 仍缺少 reasoning structure。

## 11. 当前实现

- `skill_bank/trace_program_ir.py`：evidence/status 分层的多步 IR；
- `skill_bank/trace_program_validator.py`：full-episode compiler 与 replay validator；
- `scripts/build_observed_trace_programs.py`：全量构建 299 个 observed programs；
- `harness/skill_admission.py`：target-native one-shot admission；
- `scripts/propose_alfworld_bindings_35b.py`：OpenRouter/local 多角色 candidates；
- `harness/frozen_transfer_policy.py`：frozen runtime 与 native-pattern check。
- `harness/capability_gaps.py`：选择结果、已实现能力与未实现 gap 的 claim-boundary
  manifest；它不能改变 admission verdict；

全量构建命令：

```bash
python scripts/build_observed_trace_programs.py \
  --source-root labeling/gpt54_skill_labeled \
  --output runs/observed_trace_programs/programs.jsonl \
  --report runs/observed_trace_programs/report.json
```

## 12. 下一实现阶段

可执行、带验收门槛的 v3 顺序见
[`README_HARNESS_V3_EXECUTION_PLAN.md`](README_HARNESS_V3_EXECUTION_PLAN.md)。

1. 为 Agent control hypotheses 定义 evidence-reference-only JSON schema；
2. 实现多 Agent source hypothesis proposer，不提供 legacy skill labels；
3. 实现 proposal-set union、非同构保留与 exact reference validator；
4. 新增 instrumented rollout recorder；（已接入，4 个真实 2048 episodes；显式
   REPLAN/ABSTAIN 仍不受当前 Actor 协议支持）
5. 对 guard/branch/retry 做 snapshot interventions；（已有 13 条 2048 observable
   fork receipts，但尚不足以验证一般 guard/branch/retry）
6. 实现 multi-step target span binding 与 per-state exact action-set intersection；（已完成）
7. 加入 target-native same-demo baseline；（已完成）
8. 小规模 matched pilot；（已完成，但 source 0/8、target-only 1/8，未通过放大门槛）
9. 仅用 source-side 证据补齐 instrumented/replay-qualified 多步候选，再重复冻结 pilot；
10. 只有出现预先定义的正向 source signal 后，才注册 2×4 正式实验。

2026-07-21 的首批 source-only 闭环已经得到 6 个 full-path multi-node 2048 candidates；
它们来自 4 条新 instrumented episodes、12 条 observable fork receipts 和 12 次独立
35B proposals。Harness 拒绝 4 个 single-node 输出及 2 个 receipt attachment 错位输出，
没有进行 ranking、vote 或 semantic clustering。下一步是 matched target-binding smoke，
不是直接放大到 held-out 或 2×4。

在第 4–5 步完成前，论文应使用“observed execution traces + control hypotheses”，
不能使用“verified transferable reasoning backbone”。
