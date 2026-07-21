# Agent-proposed、Target-native Skill Harness

本文取代“人工 source predicate 对齐 target operator”的实验路径。核心问题不再是
`COLLECT 是否等于 TAKE`，而是：一个从游戏轨迹得到的程序骨架，经独立 Agent
提出 target 实例候选后，能否通过一条固定 target demonstration 的原生执行证据，
并在 frozen held-out evaluation 中带来收益。

## 1. 信任边界

系统只预先信任每个环境无法省略的原生接口：

```text
reset / step
official admissible actions
native action parser
official success evaluator
immutable transition receipts
```

这些接口定义“环境实际允许和发生了什么”，不定义跨域技能语义。禁止在新路径中
使用：

- 手写 `source effect → target operator` 表；
- 人工共享的 `NAVIGATE/ACQUIRE/POSSESSION/COMMIT` ontology 作为准入证据；
- skill 名称、embedding、文本相似度或阈值作为 verdict；
- Agent confidence、rationale、投票或 majority vote 作为 verdict；
- target test trajectory 对 admission artifact 的更新。

完全无预定义接口是不可能的：没有 target parser 和 official evaluator，系统甚至无法
判断命令是否合法或任务是否完成。我们消除的是跨域语义预定义，不是环境 API。

## 2. Agent 端

对每个 source program，三个相互独立的角色读取同一份 source receipts 和同一条固定
target demonstration：

1. `proposer_a`：提出最强的可执行结构实例；
2. `proposer_b`：独立寻找另一种实例；
3. `skeptic`：主动寻找会暴露歧义的竞争实例，没有证据时输出 `ABSTAIN`。

每个角色最多输出一个 closed JSON candidate：

```json
{
  "source_program_id": "...",
  "source_step_id": "...",
  "operator": "...",
  "argument_types": {},
  "rationale": "untrusted"
}
```

正式 prompt 会把 game name、skill name、source operator name 和真实 step ID 替换为
opaque content-addressed references。Agent 可以读取真实 source action receipts，但不能
依赖 `Sokoban/NAVIGATE/COLLECT` 等人工标签完成匹配。

合并器只进行 schema validation、引用检查、去重和集合并集。它不评分、不投票，
skeptic 也没有 veto 权。Agent 幻觉只会形成无效 candidate，不能修改真实 schema、
demo 或 verifier。

## 3. Target-native demonstration receipt

每一步保存内容而不仅是 hash：

- exact action 和 native parsed arguments；
- before/after exact admissible action arrays；
- before/after state receipts；
- reward、terminated、truncated；
- 该步后的 official success；
- 每项内容的完整性 hash。

Harness 机械检查 action 是 before-admissible list 的 exact member、receipt hash 一致、
相邻 state/action-affordance receipts 构成连续链，并且整条 demo 的 official evaluator
成功。自然语言 rationale 不参与任何检查。

## 4. Admission

```text
source program receipts
        +
independent Agent candidate set
        +
one fixed target-native demo
        ↓
reference/hash/schema checks
        ↓
candidate target action was really executed in the demo?
        ↓
equivalent candidates canonicalize
non-equivalent candidates remain ambiguous
        ↓
CONDITIONAL / INCONCLUSIVE / REJECTED
```

当前 source compiler 只有单个 `commit-observed-action`，没有非平凡控制拓扑。因此
即使 candidate 在 target demo 中真实执行过，也最多得到 `CONDITIONAL`，artifact
必须写入：

```text
semantic_alignment_claimed = false
required_conditions = exact current environment admissibility
```

这不是完整 program transfer 的证据。只有未来 source program 含多个 evidence-backed
steps，且其 control/entity-dependency graph 被完整覆盖时，才可能讨论更强的 admission。

## 5. Runtime

Artifact 自包含 source skill identity、target-native scope、proposal provenance 和 demo
transition IDs；v2 runtime 不再读取人工 binding config。

- `ADMITTED/CONDITIONAL` 且当前 exact action admissible：可以交给 Actor 选择；
- 没有可用 artifact：Harness 条件 abstain，Base 条件仍正常运行；
- scope 外 operator：abstain；
- held-out episode 不更新 artifact；
- success 只读取 ALFWorld official `won`。

运行时会把 `state_changed`、`admissible_set_changed` 和“执行动作是否仍可用”与固定
demo 中同一 target operator 的原生 pattern 做 exact matching；不一致时当前 episode
立即停止并记录 `false_admission`。这仍只能声称 target-native pattern verification，
不能声称跨域 effect equivalence proof。

## 6. 必做对照

| Condition | 目的 |
|---|---|
| Base | 原始能力 |
| Target one-shot only | 测量示范本身的收益 |
| Source program only | zero-shot source prior |
| Random/renamed source program + one-shot | 排除额外 context 和名称泄漏 |
| Source program + Agent candidates + Harness | 核心条件 |
| Target-native program + same one-shot | 检验 source game 是否真的贡献 |
| Core without abstention | 检验 selective gate 的作用 |

最关键的比较是：

```text
source program + same one-shot
vs.
target-native program + same one-shot
```

如果没有显著差异，就不能声称游戏技能发生了迁移。

## 7. 不变量测试

正式结果必须加入：

- `no_cross_domain_effect_table`；
- `predicate_name_invariance`；
- `source_skill_name_randomization_invariance`；
- `proposal_order_invariance`；
- `agent_vote_is_not_verdict`；
- `hallucinated_reference_rejects`；
- `ambiguous_candidate_set_abstains`；
- `missing_native_receipt_rejects`；
- `target_demo_chain_is_contiguous`；
- `test_failure_cannot_repair_artifact`；
- `official_success_only`。

## 8. 当前实现与运行顺序

当前 v2 已完成：

- 删除 principled admission 中的 `TARGET_OPERATOR_EFFECTS`；
- `BindingCandidate` 不再携带 `source_effect`；
- v3 ALFWorld demo 保存 target-native before/after action arrays 和 outcome；
- 35B 以两个 proposer 加一个 skeptic 的方式产生候选；
- admission 默认拒绝人工 binding JSON；
- artifact 自包含 source skill name 和 Agent provenance；
- 单步 source program 只能 `CONDITIONAL`。
- held-out 每步执行后进行 target-native pattern check，失败即停止且不更新 artifact。

2×4 运行顺序为：

```text
freeze source receipts + v3 target demo
→ start source-only SFT and 35B service
→ three Agent roles generate candidate set
→ Harness freezes run-local admission manifest
→ Base/Base+Harness paired evaluation
→ frozen Game-SFT/Game-SFT+Harness paired evaluation
→ official aggregation
```

旧 v1 admission artifacts 和旧 binding config 只保留作历史 smoke test，不能与 v2
结果混合。
