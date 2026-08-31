# Multi-source → ALFWorld Neural-Symbolic Transfer V1

日期：2026-08-10

## 结论

本轮实现了一个实际执行、非平凡，但 target 效果失败的 neural-symbolic transfer：

- source 扩大为五类语义 surface 和多种结构参数；
- source state-action-value correspondence 在独立 source domains 上胜过 shuffled/marginal；
- ALFWorld-native neural grounder 只使用 adaptation expert receipts；
- authentic controller 在 qualification 中确实改变了 22.75% 的 `TEST/COMMIT` role；
- 但 target-only、authentic、within-state shuffled、source-marginal 全部为 `0/8` success。

因此状态是 `QUALIFICATION_CANDIDATE_FAILED`。heldout 没有执行，不能声称真实跨域正迁移。

本轮回答了两个不同问题：

1. source domain 能否扩大：可以，而且 source-side value structure 变得稳定；
2. 扩大 source 是否自动产生有效 ALFWorld transfer：不能。迁移结构必须与 target 的决策层级匹配。

## Source 不再是单一游戏

Source family 包含：

- `game`：hidden vault / rune tests；
- `software`：fault isolation / traces；
- `science`：catalyst identification / assays；
- `forensics`：case attribution / clues；
- `network`：fault localization / link probes。

不同 family 改变：

- hypothesis count：3–6；
- test count：4–7；
- test budget：3–5；
- intervention cost：0.01–0.08。

总数据：40 个 train domains、10 个 source evaluation domains、5,654 个 matched train
action-value examples、1,811 个 evaluation examples。

Source held-out value MSE：

| condition | MSE ↓ |
|---|---:|
| authentic | **0.00690** |
| source marginal | 0.02414 |
| within-state shuffled | 0.05145 |

因此 source-side state-conditioned action binding gate 通过。Source raw action tokens 没有进入
ALFWorld；传递的字段只有 entropy、confidence、estimated information gain、remaining budget、
action repetition 和 TEST/COMMIT role 等九个数值 symbolic features。

## Target-native neural grounding

从 frozen manifest 的八个 adaptation tasks 收集官方 expert receipts。官方 expert 在四个任务
成功、四个任务达到 adaptation cap；模型只使用四个成功 episodes，共 180 个 states。

Grounder 是一个 target-native one-hidden-layer MLP。输入由 ALFWorld goal、observation、native
action 和 target action history 构造；source tokens 和 source labels 不进入 grounder。

Leave-one-task-out：

- top-1 expert-action agreement：0.100；
- random expectation：0.0583；
- ratio：1.71x；
- frozen minimum：1.50x。

该 gate 只授权 qualification。0.100 的绝对准确率仍然很弱，不能单独视为可靠 target policy。

## Qualification 四条件结果

八个 qualification tasks 使用相同 frozen manifest、budget 和 target grounder。Heldout 未读。

| condition | success | changed action | changed TEST/COMMIT role | source admission |
|---|---:|---:|---:|---:|
| target-only | 0/8 | 0% | 0% | 0% |
| authentic | 0/8 | 26.5% | **22.75%** | 57.0% |
| shuffled | 0/8 | 78.75% | 16.25% | 78.75% |
| marginal | 0/8 | 65.25% | 65.25% | 81.25% |

Non-triviality gate 要求 authentic role-change rate 至少 5%；观测值 22.75%，通过。也就是说
source controller 不是名义存在或与 target-only 等价。Success superiority gate 失败。

## 为什么失败

Authentic 的 400 个 qualification decisions 中：

- TEST：385；
- COMMIT：15；
- `look`：160；
- `SOURCE_SELECTED_TEST`：216。

Source family 学的是 active identification：`TEST → posterior update → terminal COMMIT`。
但 ALFWorld 的有效行为不是一次 terminal COMMIT，而是层级 workflow：

```text
SEARCH / INSPECT
  → ACQUIRE target object
  → optional TRANSFORM (clean / heat / cool / light)
  → NAVIGATE to destination
  → PLACE
  → VERIFY official progress
```

当前二元 abstraction 把 `take`、`cool`、`move/put` 等不同阶段压成同一个 COMMIT。Source value
model 因此把继续 TEST 估得过高；target grounder 也无法向它表达“这是一个必要的中间 commit，
不是终止猜测”。这属于抽象层级不匹配。

Controls 进一步支持该诊断：shuffled 和 marginal 更频繁地干预，但仍为零 success；仅增加 source
authority 或 TEST 频率没有帮助。

## 是否 non-trivial

本轮满足 operational non-triviality：

- source program 从 matched intervention values 学得；
- authentic 必须胜过保留边际分布的 controls 才能通过 source gate；
- target grounder 独立训练；
- authentic 实际改变 target action role；
- transfer 不是 raw-token reuse、static prompt、action-frequency prior 或通用 argmax。

它没有满足 effectiveness：没有 target success/progress improvement。因此正确表述是
“non-trivial transfer mechanism executed, but positive transfer was not observed”。

## 下一版必须改变什么

继续增加相同 active-identification source domains 不会解决层级错配。V2 source 应成为
hierarchical intervention workflow family，并学习 option-level values：

```text
state = (belief, workflow_phase, acquired?, transformed?, destination_known?, budget)
option = SEARCH | ACQUIRE | TRANSFORM | PLACE | VERIFY | ABSTAIN
receipt = option precondition + observed delta + option completion + downstream value
```

建议 source surfaces 仍可覆盖 game、software、science、network 和 embodied simulation，但每个
surface 必须包含相同的多阶段依赖图，而不是只更换自然语言。Target-native neural grounder负责把
ALFWorld actions 绑定到 options；source 只提供 phase transition/value program。

V2 的 non-triviality controls 至少包括：

1. within-state option-value shuffle；
2. option marginal；
3. phase-permuted program；
4. target-only；
5. authentic 的 option-role change rate；
6. authentic 的 receipt-verified option completion rate。

只有 authentic 在 qualification 同时胜过所有 controls，才允许读取 heldout。

## Artifacts

- Config: `configs/multisource_alfworld_neurosymbolic_v1.json`
- Adaptation receipts: `runs/multisource_alfworld_neurosymbolic_v1/adaptation_expert_receipts.json`
- Frozen candidate artifact: `runs/multisource_alfworld_neurosymbolic_v1/frozen_candidate_artifact.json`
- Qualification report: `runs/multisource_alfworld_neurosymbolic_v1/qualification_report.json`
- Target grounder: `src/motif_transfer/alfworld_neural_grounder.py`
- Qualification runner: `scripts/run_multisource_alfworld_qualification.py`
