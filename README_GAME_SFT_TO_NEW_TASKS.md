# 用 Game-only SFT 处理新领域任务

> 本文定义如何只使用游戏数据训练通用 Actor，再把它与 transferred skills、
> one-shot admission 和 target Harness 组合，用于 ALFWorld、Browser、Visual
> Reasoning 与 Video，而不默认依赖 target-domain SFT。

> 本文与 [`README_PRINCIPLED_SKILL_TRANSFER.md`](README_PRINCIPLED_SKILL_TRANSFER.md)
> 和 [`README_8_BENCHMARK_ADAPTATION_EVAL.md`](README_8_BENCHMARK_ADAPTATION_EVAL.md)
> 配套使用。

> 当前 ALFWorld 实现实验见
> [`README_ALFWORLD_2X4_L40S_EXPERIMENT.md`](README_ALFWORLD_2X4_L40S_EXPERIMENT.md)。
> 其中 Game SFT 只克隆 source 中精确执行的编号 skill/action，不使用 target sample，
> 也不把 teacher rationale 或 skill 名称当作语义 proof。它是实验条件，不是默认有效的
> transfer mechanism。

## 1. 核心假设

可以使用 Game SFT 模型处理新任务，但训练目标不能是：

```text
raw game frame → game-specific button
```

它必须学习领域无关的技能调用协议：

```text
typed state
+ goal
+ available admitted skills
+ legal entities/actions
        ↓
skill selection
+ typed slot binding
+ verify / abstain decision
```

目标研究假设是：

> Game-only SFT 可以学习选择、实例化、验证和拒绝技能的通用控制协议；新领域只需
> Adapter 提供统一状态/动作 schema，并用一条目标示例验证 transferred skill，
> 不必默认进行 target-domain gradient update。

该假设不声称游戏数据能让模型自动获得新领域知识、视觉识别能力或目标环境特有的
动作语法。

## 2. Game SFT 应输出什么

每条训练样本把真实游戏轨迹编译为 canonical decision example。输入示例：

```json
{
  "goal": ["inside(target_object, target_container)"],
  "state": {
    "entities": [
      {"id": "object_1", "type": "movable_object"},
      {"id": "container_1", "type": "openable_receptacle"}
    ],
    "facts": [
      "visible(object_1)",
      "reachable(container_1)",
      "closed(container_1)"
    ]
  },
  "eligible_skills": [
    {
      "skill_id": "acquire_then_deliver",
      "input_slots": {
        "object": "movable_object",
        "destination": "receptacle"
      },
      "postconditions": ["inside(object, destination)"]
    }
  ]
}
```

期望输出不是自由文本计划，而是严格 typed invocation：

```json
{
  "decision": "invoke_skill",
  "skill_id": "acquire_then_deliver",
  "slots": {
    "object": "object_1",
    "destination": "container_1"
  },
  "expected_effects": [
    "inside(object_1, container_1)"
  ]
}
```

其他合法输出包括：

```json
{"decision": "abstain", "reason": "NO_APPLICABLE_SKILL"}
```

```json
{
  "decision": "request_evidence",
  "reason": "AMBIGUOUS_BINDING",
  "required_fact": "type(object_1)"
}
```

```json
{
  "decision": "verify",
  "claim": "inside(object_1, container_1)",
  "evidence_locator": "transition/step_004/state_delta"
}
```

所有输出必须经过 schema parser。缺字段、引用不存在实体或类型冲突时必须拒绝，不能
从自由文本猜测并补齐。

## 3. 从游戏轨迹构造 SFT 数据

只使用经过真实游戏环境验证的轨迹：

```text
raw verified game trajectory
        ↓
typed state/action/effect compiler
        ↓
CanonicalSkillProgram alignment
        ↓
decision-level SFT examples
```

建议从每个成功 episode 生成多类 supervision：

1. 当前 goal 应选择哪个 skill；
2. abstract slots 应绑定到哪些 typed entities；
3. skill 的 preconditions 是否满足；
4. 下一步应执行、验证、恢复还是 abstain；
5. observed state delta 是否支持 expected effect；
6. termination condition 是否真正成立。

SFT label 必须来自可重放 trajectory、typed compiler 和环境事实，不能由 35B 根据
自然语言轨迹自行生成“正确答案”。9B/35B 可以提出 segmentation 或 program
candidate，但只有通过 replay 和 effect checks 的结果才能进入训练集。

## 4. 必须包含的 negative examples

如果 Game SFT 只包含成功 skill invocation，模型很容易在新领域强行调用任意技能。
训练集必须系统加入可机械验证的负例：

| Negative case | 正确 label |
|---|---|
| skill postcondition 与 goal 不统一 | `NO_APPLICABLE_SKILL` |
| 缺少必要 precondition | `MISSING_PRECONDITION` 或先调用 prerequisite skill |
| slot entity 不存在 | `UNKNOWN_ENTITY` |
| entity type 与 slot type 不一致 | `TYPE_MISMATCH` |
| action 不在当前 schema/admissible set | `UNSUPPORTED_ACTION` |
| 多个 binding 会产生不同动作 | `AMBIGUOUS_BINDING` |
| effect 没有出现在真实 state delta | `EFFECT_NOT_VERIFIED` |
| termination 未成立但轨迹文字声称成功 | `SUCCESS_NOT_VERIFIED` |
| task/state 超出 verified scope | `OUT_OF_SCOPE` |
| evidence locator 不存在或无法重放 | `INVALID_EVIDENCE` |

负例优先通过反事实扰动生成，例如交换 slot、删除 precondition、替换不存在实体或改变
state delta。每个扰动必须由 deterministic checker 确认确实无效，不能让 LLM
judge 决定 negative label。

## 5. 避免学习游戏特有 shortcut

Game SFT 的主要输入不能依赖：

- 游戏名称；
- ROM-specific action token；
- 固定按钮组合；
- HUD 的固定坐标；
- 某个 game ID 与 skill ID 的共现；
- 手写 target predicate translation；
- benchmark-specific preferred operator。

主要决策依据应是：

- typed entities 与 relations；
- goal predicates；
- skill preconditions/postconditions；
- legal action schemas；
- observed state delta；
- evidence status；
- verified scope；
- failure code。

可以保留原始图像作为辅助输入，但 SFT supervision 必须落在 canonical skill-call
层，而不是游戏按键层。训练时应随机化无语义 ID、game labels 和 serialization
顺序，并做 entity renaming/counterfactual tests，检查模型没有使用表面 token shortcut。

## 6. 三种 target 部署方式

### 6.1 Raw direct transfer

```text
Game-SFT model → raw target observation → raw target action
```

这是最弱且预期最不可靠的 baseline。视觉格式、语言分布和动作空间同时变化，Game
SFT 很可能生成非法或游戏化动作。

### 6.2 Game SFT + Target Adapter

```text
target observation
        ↓
Target Adapter → canonical typed state/action schema
        ↓
Game-SFT Actor → typed skill invocation
        ↓
Harness → legal target action
```

这测试 zero-gradient protocol transfer：模型没有见过 target demonstration，
但通过统一接口尝试使用 source skill。

### 6.3 Game SFT + Transferred Skill + One-shot Admission

```text
Game-SFT Actor
+ game-derived CanonicalSkillPrograms
+ one fixed target verification demo per skill
+ frozen admission artifact
+ Target Adapter/Harness
```

这是核心条件。目标 demonstration 只用于 binding verification/admission，不用于
target gradient update。测试前模型、skill library、binding 和 admission artifact
全部冻结。

## 7. 新任务上的系统分工

```text
Goal + canonical target state
        ↓
logical postcondition unification
        ↓
eligible admitted skills
        ↓
Game-SFT Actor chooses skill + fills typed slots
        ↓
Harness checks scope, preconditions and legal arguments
        ↓
Target Adapter executes concrete actions
        ↓
Harness verifies state delta/evidence/termination
        ↓
official environment evaluator
```

Game-SFT Actor 不能绕过 Harness 直接调用环境。Actor 的职责应尽可能收窄为：

- 在已经机械过滤的候选中选 skill；
- 从合法 entity candidates 中填 slots；
- 对 `VERIFIED/REFUTED/UNKNOWN` evidence 作结构化决策；
- 在无合法候选、scope 外或歧义时 abstain。

具体动作序列越能由 admitted executable program 和 Harness 直接执行，就越有可能
不需要 target SFT。若 skill 仍只是自然语言说明书、每一步都由 9B 自由生成，则
Game SFT 很难可靠替代 target adaptation。

## 8. Game-to-game held-out 验证

进入四个目标域前，必须先证明 Game SFT 能迁移到未参与训练的游戏：

```text
for each game g:
    train SFT on all games except g
    freeze model
    optionally run one-shot skill admission on g
    evaluate held-out episodes from g
```

这是 leave-one-game-out protocol。至少报告：

- skill selection accuracy；
- typed slot grounding accuracy；
- invalid invocation rate；
- abstention/coverage；
- false-admission rate；
- official held-out-game task success；
- negative transfer relative to Base。

如果模型只能在训练过的 game ID 上工作，或 canonical entity renaming 后性能显著下降，
说明它学到的是游戏 shortcut，不能进入 game-to-domain 主实验。

## 9. 对四个目标域的预期边界

| Domain | Game SFT 可能迁移的能力 | Game SFT 很难单独提供的能力 |
|---|---|---|
| ALFWorld | prerequisite、skill selection、slot binding、effect verification | household entity semantics、特有 command grammar |
| MiniWoB | UI 操作流程、precondition、termination verification | DOM/AXTree grounding、具体 widget semantics |
| WebShop | search/filter/compare/commit 控制结构 | 商品知识、长程页面语义和目标解析 |
| Visual Reasoning | evidence gathering、tool sequencing、answer verification | 新视觉对象识别和 benchmark-specific perception |
| Video | temporal query decomposition、frame evidence checks | motion/event perception 和精确 temporal grounding |

Target Adapter、base VLM 和专用工具负责目标感知与动作接口；transferred skill 负责
控制结构；Game-SFT Actor 负责选择与实例化。不能把感知能力缺失错误归因于 skill
admission。

## 10. 与 target SFT/LoRA 的关系

核心 claim 应是：

> 在 admitted skill 的 verified scope 内，Game-only SFT 与可执行技能迁移可以减少
> 或消除 target-domain gradient adaptation。

不应预先声称所有四个域都完全不需要 SFT。target SFT 仍可能帮助：

- Actor 经常忽略 eligible skill；
- typed slot grounding 持续失败；
- target perception/tool grounding 是主要瓶颈；
- skill 无法由 Harness 直接执行，只能作为模型提示；
- base model 不理解 target observation schema。

如果只需提高 schema 遵循，可以增加 **source-only protocol SFT** 或从公开 action
schema 自动生成的 synthetic protocol SFT，但必须保证不使用 target benchmark
answers/rewards，并单独披露这部分 supervision。

## 11. 实验条件

| Condition | Game SFT | Transferred skills | Target demo | Target SFT/GRPO |
|---|---:|---:|---:|---:|
| Base | 否 | 否 | 0 | 否 |
| Game SFT only | 是 | 否 | 0 | 否 |
| Game skills only | 否 | 是 | 1/skill | 否 |
| Game SFT + unverified skills | 是 | 是 | 0 | 否 |
| Game SFT + one-shot admission | 是 | 是 | 1/skill | 否 |
| Game SFT + admission, no abstention | 是 | 是 | 1/skill | 否 |
| Target SFT only | 否 | 否 | 多条 | 是 |
| Game SFT + admission + target SFT | 是 | 是 | 多条 | 是 |

最重要的比较是：

```text
Game SFT + one-shot admission
    vs.
Game SFT + one-shot admission + target SFT
```

以及：

```text
Game SFT + one-shot admission
    vs.
Target SFT only
```

Target SFT/GRPO 使用现有完整 target train split 时属于独立的 target adaptation
condition，不能继续使用纯 one-shot claim。

## 12. “可以跳过 target SFT”的判定

不能因为无 SFT 条件偶然略高就宣布成功。应预注册 non-inferiority margin `δ`：

```text
Success(game_sft + admission)
≥ Success(game_sft + admission + target_sft) - δ
```

同时必须满足：

- false-admission rate 不显著上升；
- invalid-action rate 不显著上升；
- unconditional success 达到 non-inferiority；
- coverage 不能通过大量 abstain 人为压低风险；
- 每个 domain 和 8 个 evaluation cells 分别报告，不能只用总体平均掩盖失败域。

`δ` 必须在运行 target test 前固定，并结合 benchmark 原有方差和业务可接受误差设置，
不能看到结果后选择。

## 13. 推荐训练与执行顺序

```text
1. 收集并验证 game trajectories
2. 编译 typed state/action/effect 与 CanonicalSkillPrograms
3. 生成 positive、negative、abstain、verify SFT examples
4. 运行 leave-one-game-out，排除 game/token shortcuts
5. 用全部允许的 game-only data 训练最终 Game-SFT Actor
6. 冻结 Actor、source skills 和所有全局超参数
7. Target Adapter 发布 canonical state/action schemas
8. 每个 transferred skill 使用一条固定 target demo 运行 admission
9. 冻结并 hash admission artifacts
10. 在 4 domains / 8 cells 上运行独立 held-out evaluation
11. 最后单独运行 target SFT/LoRA/GRPO baselines
```

## 14. 必须具备的不变量测试

1. `game_only_sft_lineage`：Game SFT 样本不能引用 target benchmark data；
2. `label_has_execution_proof`：每个正例 label 都能追溯到真实游戏 transition；
3. `counterfactual_negative_is_invalid`：每个负例由 deterministic checker 确认；
4. `entity_renaming_invariant`：无语义 entity ID 重命名不改变决策；
5. `game_id_not_required`：移除 game name 后 canonical decision 仍可执行；
6. `typed_output_only`：自由文本、缺字段和非法 slot 必须拒绝；
7. `out_of_scope_abstains`：verified scope 外不能强行调用技能；
8. `actor_cannot_bypass_harness`：Actor 不能直接调用 target environment；
9. `target_demo_no_gradient`：核心 one-shot condition 中没有 target gradient update；
10. `test_episode_isolation`：target test episodes 不共享学习状态；
11. `official_success_only`：模型自评不能替代官方 evaluator；
12. `target_sft_condition_is_separate`：使用 target train data 的结果不能标为纯 transfer。

## 15. 完成定义

只有同时满足以下条件，才能声称 Game SFT 成功迁移到新任务：

- SFT 数据和 labels 只来自可验证 game trajectories；
- 模型学习的是 canonical skill-call protocol，而非 game-specific buttons；
- leave-one-game-out 验证通过，并排除 game ID/entity token shortcuts；
- target domain 通过 typed Adapter/Harness 接入；
- transferred skill 在测试前经过一条固定 target demo admission；
- target test 前模型、binding 和 admission artifacts 全部冻结；
- 核心条件不使用 target SFT、LoRA 或 GRPO；
- target test reward/trajectory 不写回模型和技能库；
- 8 个 evaluation cells 同时报告 success、coverage、abstention、negative transfer 和
  false admission；
- 与 target SFT baseline 使用预注册 non-inferiority protocol 比较。
