# 4 个目标域、8 个评测单元的训练、One-shot Adaptation 与测试协议

> 本文定义 game-to-target skill transfer 的数据边界、one-shot admission
> 单位、held-out 测试矩阵和统计口径。目标是在不把 target test reward、轨迹
> 或答案写回模型和技能库的前提下，判断游戏技能经过一条目标域示例验证后是否
> 真正可用。

> 本文与 [`README_PRINCIPLED_SKILL_TRANSFER.md`](README_PRINCIPLED_SKILL_TRANSFER.md)
> 配套使用。后者定义技能表示、binding、Harness 和 admission 语义；本文定义
> 8 个评测单元上的实验协议。

## 1. 评测范围

当前仓库对应以下 4 个目标域和 8 个结果单元：

| Domain | Evaluation cell A | Evaluation cell B |
|---|---|---|
| Visual Reasoning | VisualToolBench | TIR-Bench |
| Video | Video-Holmes | SIV-Bench |
| Browser | MiniWoB | WebShop |
| ALFWorld | `valid_seen`（IID） | `valid_unseen`（OOD） |

前 6 个是命名 benchmark。ALFWorld 的两个单元是同一 benchmark 的官方 IID/OOD
评测划分，不应被错误描述为两个独立数据集。最终结果表可以报告 8 个 cell，但
论文和 README 必须保留这个区别。

## 2. 必须先定义 one-shot 的单位

主实验采用：

> **one-shot per transferred skill per benchmark**：每个 source skill 在每个
> 目标 benchmark 上最多使用一条预先固定的成功 demonstration 进行 binding
> verification/admission。

这是 skill-level verification，不是从一个 target example 训练整个 domain，也
不是每个 benchmark 使用 20% 数据训练。

可额外报告更严格的 ablation：

> **one-shot per benchmark**：整个 benchmark 仅提供一条 demonstration，所有
> 候选技能共享它。

在后一条件中，大部分未被该 demonstration 覆盖的技能应得到
`INCONCLUSIVE`，而不是由模型补全或默认接纳。

如果同一 demonstration 的 typed proof trace 完整覆盖多个技能，它可以验证多个
技能，但实验必须报告 unique target demonstrations 数量、每个技能使用的 demo ID
以及 operator coverage，避免用“每技能一条”掩盖实际 target supervision 总量。

## 3. 三层数据边界

### 3.1 Source training set

```text
D_source = verified GymV trajectories
```

只用经过真实环境验证的游戏数据：

- 归纳 `CanonicalSkillProgram`；
- 训练 game-to-game binding proposal model；
- 训练 source actor 或 source LoRA；
- 学习 candidate ordering；
- 选择全局超参数和训练预算。

Source training 不得读取 8 个 target cell 的答案、reward、成功轨迹或 test
statistics。9B/35B 输出始终是 untrusted proposal，不能产生 transfer verdict。

### 3.2 Target adaptation pool

每个命名 benchmark 定义固定 adaptation pool：

```text
A_b ⊂ official target train split
```

它仅用于：

- 按预注册规则选择 one-shot verification demonstration；
- 构造 target binding candidates；
- 运行 one-shot skill admission；
- 生成 immutable admission artifact、verified scope 和 proof trace。

纯 one-shot 条件禁止：

- target gradient update；
- target LoRA 或 GRPO；
- 根据 adaptation reward 调 threshold 或反复选最好 demo；
- 把失败轨迹持续写回跨 episode skill bank；
- 使用 adaptation pool 估计最终 test performance。

现有 `cold_start/task_samples/stage3_splits/*_train.txt` 可以作为候选 adaptation
pool，但核心实验只能按每个 skill 一条的规则消费数据，不能把整个 20% split
都用于训练。

ALFWorld 只能从官方 `train` split 选择 adaptation demonstration，随后同一个
admission artifact 同时用于 `valid_seen` 和 `valid_unseen`。

### 3.3 Held-out test set

必须满足：

```text
A_b ∩ T_b = ∅
```

进入 test 前冻结：

- 9B actor；
- 35B proposer；
- source LoRA；
- Binder/constraint solver 配置；
- transferred skill library；
- admission artifact 和 verified scope；
- 所有全局阈值、预算和 prompt template。

每个 test episode 独立运行。前一个 test episode 的 observation、错误、reward、
成功轨迹或 suspension 状态不能改变后一个 episode 的模型、binding、技能排序或
admission artifact。

## 4. One-shot demonstration 的选择

对 source skill `s` 和 benchmark `b`，先用 typed execution trace 定义：

```text
Eligible(s, b) = {
    d ∈ A_b :
    d is officially successful
    and d covers the required operators of s
}
```

是否覆盖由 target action schema、真实 state delta 和 operator alignment 机械判断，
不能由 embedding、文本相似度或 35B judge 决定。

从 `Eligible(s,b)` 选 demo 必须使用预先固定的稳定规则，例如：

```text
demo(s, b, seed) =
    first item after stable-hash ordering of Eligible(s, b)
```

禁止执行多个 demos 后挑选 admission/test 表现最好的一个。若不存在完整覆盖该技能
的单条 demo：

```yaml
admission_status: INCONCLUSIVE
failure_code: INSUFFICIENT_COVERAGE
```

纯 one-shot 主实验在这里 abstain。它可以输出需要什么额外证据，但不能读取第二条
示例。真正取得额外示例属于独立的 adaptive evidence acquisition 条件。

## 5. One-shot admission

每个 `(skill, benchmark, adaptation_seed)` 产生独立 artifact：

```text
source verified skill
        ↓
untrusted target binding proposal
        ↓
candidate target skill
        ↓
one fixed target demonstration
        ↓
typed static checks
        ↓
real target execution/replay
        ↓
official evaluator
        ↓
ADMITTED | CONDITIONAL | INCONCLUSIVE | REJECTED
```

Artifact 至少记录：

```yaml
source_skill_id: program.acquire_transform_deliver
target_domain: alfworld
target_benchmark: alfworld
adaptation_demo_id: train/demo_0001
adaptation_seed: 0
binding_status: unique
admission_status: CONDITIONAL
verified_scope:
  task_family: pick_and_place
  object_type: food
  destination_type: openable_receptacle
verified_operators: [NAVIGATE, ACQUIRE, OPEN, COMMIT]
unverified_operators: []
required_preconditions:
  - holding(object) before COMMIT
  - open(destination) before COMMIT
proof_trace: [...]
```

一次成功只能证明候选技能在 artifact 声明范围内局部兼容，不能自动推广到未观测
entity types、operators、task families 或另一个 benchmark。

## 6. 五次独立 one-shot trials

单条 demonstration 具有较大偶然性。主实验建议预注册：

```text
adaptation_seed ∈ {0, 1, 2, 3, 4}
```

每个 seed 都从未合并的状态独立运行：

```text
one demo → admission → freeze → full held-out test
```

五条 demo 不能合并为一个 skill artifact，因此这是 5 次独立 one-shot trial，
不是 five-shot。最终报告 adaptation-seed 均值、标准差或 bootstrap confidence
interval，并保存每次 trial 的 demo manifest。

## 7. 测试矩阵

### 7.1 Within-benchmark transfer

每个命名 benchmark 独立执行：

```text
game skill
    ↓
one demo from benchmark A adaptation pool
    ↓
immutable admission artifact
    ↓
benchmark A held-out test
```

需要覆盖：

```text
Game → VisualToolBench admission → VisualToolBench test
Game → TIR-Bench admission        → TIR-Bench test
Game → Video-Holmes admission     → Video-Holmes test
Game → SIV-Bench admission        → SIV-Bench test
Game → MiniWoB admission          → MiniWoB test
Game → WebShop admission          → WebShop test
Game → ALFWorld train admission   → valid_seen test
Game → ALFWorld train admission   → valid_unseen test
```

### 7.2 Cross-benchmark within-domain transfer

为了区分 domain-level transfer 与 benchmark-specific adaptation，还应运行：

```text
VisualToolBench admission → TIR-Bench test
TIR-Bench admission       → VisualToolBench test

Video-Holmes admission    → SIV-Bench test
SIV-Bench admission       → Video-Holmes test

MiniWoB admission         → WebShop test
WebShop admission         → MiniWoB test
```

测试 paired benchmark 时不能提供新 demonstration 或重新 binding。跨 benchmark
时如果 verified scope 不适用，正确结果是 abstain/不接纳，而不是强行执行。

ALFWorld 使用一个 `train` admission artifact 同时测试 `valid_seen` 和
`valid_unseen`，不在两个 test split 上分别重新 admission。

### 7.3 可选 leave-one-benchmark-out meta-adaptation

如果未来要使用 target benchmarks 训练 learned Binder，必须建立独立条件：在 7 个
evaluation cells 上训练/选择方法，对第 8 个完全冻结测试并轮换。该条件不再是
game-only transfer，必须标记为 target meta-adaptation，不能与核心 claim 合并。

## 8. Test-time Harness 规则

每个 held-out episode：

```text
load frozen admission artifact
        ↓
task/state satisfies verified scope and preconditions?
        ├─ no  → abstain or fall back to Base actor
        └─ yes → Harness-authorized skill execution
                    ↓
              verify every real transition
                    ↓
              official success evaluator
```

若 test episode 中发生 contract violation：

- 当前 episode 内立即停止该技能；
- 记录 `false_admission` 和结构化 failure code；
- 不修复或更新 artifact；
- 不把 test transition 写入 training/adaptation memory；
- 下一个 test episode 从相同 frozen artifact 和干净状态开始。

部署时可以使用 `SUSPENDED → REVERIFY`；正式 benchmark 测试不能跨 test episodes
持久化 suspension 或 repair，否则结果依赖 episode 顺序并发生 test leakage。

## 9. 主实验条件

| Condition | Target demonstrations | Target gradient | 作用 |
|---|---:|---:|---|
| Base 9B | 0 | 否 | 原始模型基线 |
| Game-trained 9B | 0 | 否 | 游戏训练本身的迁移 |
| Game skill, unverified | 0 | 否 | 直接注入未验证技能的风险 |
| Random game skill | 0 | 否 | 排除额外 prompt/context 增益 |
| Game skill + one-shot admission | 每技能 1 条 | 否 | 核心方法 |
| One-shot admission without abstention | 每技能 1 条 | 否 | 验证 selective admission 的作用 |
| Target LoRA/GRPO | 多条 target train samples | 是 | 非 one-shot adaptation baseline/upper bound |
| Target-domain skill bank | 多条 | 可选 | target supervision upper bound |

所有条件必须使用相同 base checkpoint、test IDs、episode budgets、temperature、
decoding policy 和官方 evaluator。除明确的 target adaptation baseline 外，不允许
target gradient update。

## 10. Target LoRA/GRPO 是独立实验线

当前 Stage 3 的协议是：

```text
20% target train → GRPO/LoRA
80% held-out target test → evaluation
```

这个协议可以保留，但只能称为 target-domain adaptation。它不能称为 one-shot
skill transfer，也不能与 one-shot admission artifact 混合后只报告一个结果。

两条实验线应严格分开：

```text
Core:
    game training
    + one target demo per skill
    + no target gradient
    + frozen held-out evaluation

Auxiliary:
    game initialization
    + full target training split
    + target rollout reward / GRPO / LoRA
    + held-out evaluation
```

如果 auxiliary 条件用 one-shot artifact 初始化 LoRA，也必须命名为
`one-demo initialization + online target adaptation`，不能继续使用纯 one-shot claim。

## 11. Group-aware split 与泄漏防护

正式结果前必须重新审计现有 manifests。随机 sample-level split 不足以阻止内容泄漏：

- **Video-Holmes/SIV-Bench**：同一 `video_id`、source clip 或派生 clip 必须完整位于一个 split；
- **VisualToolBench/TIR-Bench**：同一 image、tool trace、问题模板或近重复样本不能跨 adaptation/test；
- **MiniWoB**：同一 task template 的不同随机 seed 需要分别定义 template-IID 和 template-OOD，不能全当独立任务；
- **WebShop**：尽可能按 goal template、product/category 或共享状态分组，不能只随机 numeric task ID；
- **ALFWorld**：使用官方 `train / valid_seen / valid_unseen` 边界，不自行混合。

Split builder 必须保存：

```text
dataset version
source manifest hash
group key
split seed
adaptation IDs
test IDs
overlap audit
near-duplicate audit
```

当前 `stage3_splits` 中的 20/80 manifests 适合现有 GRPO baseline，但不能在未经
group-aware audit 的情况下直接作为最终 one-shot transfer 结果。

## 12. 指标与聚合

每个 evaluation cell 至少报告：

- official task success / exact accuracy；
- 相对 Base 的 transfer lift；
- negative-transfer rate；
- `ADMITTED / CONDITIONAL / INCONCLUSIVE / REJECTED` 分布；
- admission coverage 和 verified-scope coverage；
- unconditional success；
- conditional-on-coverage success；
- abstention rate；
- false-admission rate；
- invalid-action rate；
- contract/effect violation rate；
- unsupported operator/predicate rate；
- environment steps、tokens 和 wall time。

不能只报告 conditional success；否则系统可能通过拒绝几乎所有任务得到虚高结果。
必须同时报告 coverage-risk curve。

由于各 benchmark 样本量差异很大，最终至少提供：

```text
per-evaluation-cell metrics
→ two-cell macro average per domain
→ four-domain macro average
→ eight-cell macro average
```

Micro average 可以作为补充，但不能作为唯一总体数字。SIV/TIR 等大 benchmark
不能淹没 WebShop 或 ALFWorld。

统计检验使用相同 test IDs 上的 paired bootstrap：

- Browser/ALFWorld 按 task/episode 分组；
- Visual QA 按独立 sample 或 source image 分组；
- Video 按 `video_id/source_clip_id` 分组；
- 结果同时跨 test groups 和 5 个 adaptation seeds 汇总。

## 13. 必须执行的对照和不变量

1. `adaptation_test_disjoint`：adaptation/test 的 group IDs 不相交；
2. `one_demo_per_skill`：每个独立主实验 artifact 至多读取一条 target demo；
3. `no_best_of_n_demo_selection`：不能按 admission/test 表现挑最好 demonstration；
4. `target_gradient_disabled`：核心条件没有 target LoRA/GRPO update；
5. `artifact_frozen_before_test`：test 前 artifact hash 固定；
6. `test_episode_isolation`：测试 episode 之间没有 persistent learning/memory；
7. `official_evaluator_only`：成功只来自环境或 benchmark 官方 checker；
8. `scope_violation_abstains`：verified scope 外不能强制调用技能；
9. `test_failure_does_not_repair`：test failure 不能修复 binding；
10. `alfworld_shared_artifact`：seen/unseen 使用同一个 train-derived artifact；
11. `cross_benchmark_no_readmission`：paired benchmark test 不读取新 demo；
12. `llm_vote_is_not_verdict`：9B/35B 自评或投票不能改变 admission/success。

## 14. 推荐执行顺序

```text
1. 用 game-only verified trajectories 训练 source actor/proposer
2. 归纳并冻结 source CanonicalSkillPrograms
3. 重建并审计 6 个命名 benchmark 的 group-aware manifests
4. 固定 ALFWorld train/valid_seen/valid_unseen 配置与版本
5. 预注册 5 个 adaptation seeds 和 stable demo selection rule
6. 为每个 (skill, benchmark, seed) 运行 one-shot admission
7. 冻结并 hash 所有 admission artifacts
8. 运行 8 个 within-benchmark evaluation cells
9. 运行 6 个 cross-benchmark within-domain directional cells
10. 聚合 per-cell、per-domain、8-cell macro 和 risk-coverage
11. 最后单独运行使用完整 target train split 的 LoRA/GRPO baseline
```

## 15. 完成定义

只有满足以下条件，才能把主结果称为 4-domain/8-cell principled one-shot skill
transfer：

- source program、actor 和 proposal model 的训练仅使用 game data；
- one-shot 的单位和 target supervision 总量完整披露；
- 每个核心 artifact 至多读取一条预先固定的 target demonstration；
- demonstration 由 typed coverage 和稳定规则选择，不做 best-of-N；
- admission verdict 来自真实 Harness execution 和官方 evaluator；
- test 前模型、skill bank、binding 和 artifact 全部冻结；
- test episodes 之间不持续学习；
- ALFWorld seen/unseen 共享同一个 train-derived artifact；
- target LoRA/GRPO 作为独立非 one-shot 条件报告；
- 所有 8 个 evaluation cells 同时报告 coverage、risk、false admission 和
  unconditional performance，而不只报告成功样本上的条件指标。
