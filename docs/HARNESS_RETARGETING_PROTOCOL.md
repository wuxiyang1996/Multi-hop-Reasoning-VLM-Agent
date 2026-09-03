# Permission-Bounded Harness Retargeting Protocol

日期：2026-08-11

## 目标

本协议实现并验证以下受限命题：

> 一个在 source intervention forks 上通过资格门、随后被冻结的 symbolic skill，能否只选择
> canonical option；同一个 target-native Harness 只负责 grounding、option 内 action realization
> 和 observable-effect verification，并使 authentic skill 在 paired target episodes 上超过 null
> 与 shuffled skill controls。

这不是“把 source memo 放进 target prompt”，也不是允许 Harness 自己决定 target workflow。
历史失败与设计依据见
[`HARNESS_RETARGETING_BITTER_LESSONS.md`](HARNESS_RETARGETING_BITTER_LESSONS.md)。

## Authority contract

| Component | 可以读取 | 可以输出 | 明确禁止 |
|---|---|---|---|
| Frozen source skill | canonical facts、role slots、前一步 observable-effect receipt | 一个 canonical option、ABSTAIN 或 TERMINATE | raw target action、target reward/success、修改自身参数 |
| Target fallback policy | target Harness frame | null/abstain 时的 canonical option；raw baseline 的 native action | 修改 source skill 或 Harness artifact |
| Target-native Harness | raw target observation、native action set、外部已选 option | canonical facts、该 option 内 native action、transition effect receipt | 跨 option 排序、选择 workflow branch、读取 official outcome |
| Evaluator | immutable episode receipts、official outcome | paired metrics 与 verdict | 把 outcome 反馈给当轮 skill/Harness |

Harness API 故意没有 `choose_option()`。它只能：

```text
ground(observation) -> facts + {option: native actions}
realize(externally_selected_option, frame) -> native action in that option
verify(before_frame, action, after_frame) -> observed fact delta
```

所有 frame、decision、transition 和 outcome 均带 content hash。Target observation 传给 Harness
时不包含 reward、score 或 official success。

## Frozen source skill contract

每个可执行 rule 至少包含：

```text
typed precondition (requires / forbids)
canonical option
expected added/removed effects
failure-specific recovery option（可选）
termination facts
source intervention lineage
source qualification receipt
```

Authentic artifact 必须在 source qualification 中严格超过 shuffled 与 marginal controls；否则
runner 拒绝把它作为 authentic treatment。Frozen payload 的任何字段改变都会导致 artifact hash
失配。

Shuffled control 保留同一组 guards、effects、lineage 和 rule order，只对 option labels 做固定的
derangement。这样它控制“额外程序/context”，而不是换成一个更短或更弱的对象。

## Target experiment matrix

每个 frozen target episode 必须运行：

| Condition | Source selector | Harness |
|---|---|---|
| `raw_target_only` | 无 | bypass；target policy 直接选 native action |
| `null_skill_same_harness` | 永远 ABSTAIN | 与 authentic 完全相同 |
| `shuffled_source_skill` | shuffled frozen program | 与 authentic 完全相同 |
| `authentic_source_skill` | authentic frozen program | 与 controls 完全相同 |
| `target_oracle_skill` | target-written upper-bound program | 与 authentic 完全相同 |

核心 estimand 是：

```text
authentic_source_skill - max(null_skill_same_harness, shuffled_source_skill)
```

`raw_target_only` 只用于量化 Harness/fallback 自身贡献；`target_oracle_skill` 是上界和 target
可解性检查。不能用 `authentic > raw` 替代同 Harness 对照。

## Fail-closed invariants

Runner 在执行 native action 前检查：

1. frozen skill、Harness 和 fallback policy 的 content hash；
2. frame 只使用声明过的 predicates、roles 和 options；
3. grounded actions 是当前 native action set 的精确成员；
4. 一个 native action 不跨 option 重复；
5. skill 只输出 option，且 option 在当前 frame 可用；
6. Harness realization 必须属于外部选择的同一个 option；
7. paired conditions 的 initial-state、environment、budget、Harness 和 fallback hashes 一致。

任一检查失败，该 episode 标记为 invalid/rejected，不允许悄悄 fallback 后继续计入正迁移。
Skill 明确 ABSTAIN 才可调用相同的 target fallback policy。

## Reported gates

正式 report 分开报告：

- safety：authentic 相对 null 的 paired regressions；
- applicability：authentic intervention 后 expected observable effect 的支持率；
- utility：paired official success、score 和 equal-success efficiency；
- non-triviality：authentic 改变 fallback option 的比例；
- oracle sanity：oracle 不得低于 authentic。

最小 mechanism-positive verdict 要求：

```text
authentic successes > null successes
authentic successes > shuffled successes
oracle successes >= authentic successes
authentic intervention rate >= frozen threshold
all core conditions share exactly one Harness hash
all paired identity checks pass
```

这个 verdict 只证明 Harness-mediated mechanism 在当前 target suite 中工作。只有 authentic artifact
来自真实、held-out source intervention evidence，且 target confirmation split 未被调参消费时，才可
进一步声称 real-game cross-domain transfer。

## Implementation and smoke validation

实现由以下独立文件组成，避免修改当前 workspace 中尚未提交的 V4/V5 实验代码：

- `src/motif_transfer/retargeting_harness.py`：frozen artifact、authority-enforcing runner、paired evaluator；
- `scripts/run_harness_retargeting_smoke.py`：语义不同 target 上的确定性跨域 mechanism smoke；
- `configs/harness_retargeting_smoke_v1.json`：冻结条件、seeds、budget 和 gates；
- `tests/test_retargeting_harness.py`：权限、hash、control、paired identity 与端到端回归测试。

Smoke 使用 controlled source program，因此只能验证代码路径、权限边界和对照能否识别 authentic
structure；它不会被报告为真实游戏到 ALFWorld/WebShop 的新结果。
