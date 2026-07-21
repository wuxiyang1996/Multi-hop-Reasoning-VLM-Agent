# 基于可执行语义的跨域技能迁移

> **2026-07 v2 修订：** 本文中任何依赖手写共享 predicate 或
> `source effect → target operator` 表的段落，不再作为正式实验协议。当前协议以
> [`README_AGENT_NATIVE_SKILL_HARNESS.md`](README_AGENT_NATIVE_SKILL_HARNESS.md)
> 为准：Agent 只提出候选，Harness 只使用 target-native execution receipts 和官方
> evaluator，且当前单步 source program 最多只能获得 `CONDITIONAL`。

> 设计目标：把游戏中发现的技能迁移到 ALFWorld、Browser、Video 和 Visual Reasoning，同时消除文本相似度、人工权重、手写 predicate 映射、模糊动作匹配和 LLM-as-judge 等启发式裁决。

> **状态（2026-07-20）：ALFWorld v1 executable vertical slice 已实现，完整研究路线仍未完成。**
> 当前已有 immutable source receipts、typed source-effect parser、exact ALFWorld
> grammar、真实 one-shot demo、frozen admission、source-only Game SFT 与 2×4 L40S
> paired evaluator。具体冻结协议见
> [`README_ALFWORLD_2X4_L40S_EXPERIMENT.md`](README_ALFWORLD_2X4_L40S_EXPERIMENT.md)。
> 当前 source program 仍是 observed-action v1/v2，而非本文后半部分设想的完整
> precondition/protocol/termination program induction；因此不能把 vertical slice
> 的结果扩大为本文全部 principled claim。

## 1. 核心研究问题

本项目希望验证：

> 游戏环境中发现并由真实交互验证的抽象技能，能否先被迁移为目标域候选技能，再仅使用目标域的一条成功示例，通过领域执行 Harness 判断该候选应被接纳、条件接纳、请求更多证据还是拒绝，并改善 held-out 任务表现？

这里的一条示例主要用于 **验证与准入（one-shot verification/admission）**，而不是声称从一个示例中完整恢复目标域的真实技能语义。核心对象是一个已经迁移完成、但尚未被信任的候选技能；核心问题是它在什么已验证范围内可用，以及证据不足或结构不兼容时系统能否正确拒绝使用。

迁移对象不是游戏按键或具体动作序列，而是带有可执行语义的控制与推理程序，例如：

```text
OBSERVE(target)
→ NAVIGATE(target)
→ INTERACT(target)
→ VERIFY(expected_state_change)
→ COMMIT
```

目标域只负责把抽象实体、动作和效果绑定到自己的 schema。游戏技能本身的控制结构不应被目标域重新定义。

## 2. 设计原则

### 2.1 行为等价优先于文本相似

两个技能是否相同，应该由它们在环境中的 precondition、transition、effect、evidence 和 termination 行为决定，而不是由技能名称、自然语言描述或 protocol 文本的字符串相似度决定。

### 2.2 LLM 只提出候选，不能裁决

9B 和 35B 都容易 hallucinate。参数更多不代表模型能可靠地理解环境状态、生成合法 predicate、选择正确实体、判断动作是否真正执行，或验证任务是否完成。因此，所有模型输出都必须默认视为 **untrusted proposal**。

9B 或 35B 模型可以：

- 把原始轨迹解析成 typed program 候选；
- 提出 slot/action/predicate binding 候选；
- 在大搜索空间中提供 proposal ordering。

LLM 不可以：

- 宣布两个技能功能等价；
- 决定 predicate translation 是否正确；
- 决定迁移是否通过；
- 替代 benchmark evaluator 或真实环境 reward。

所有候选必须经过可重复的约束检查和真实执行验证。

### 2.3 9B/35B 输出的信任边界

模型生成结果在进入系统其他组件前必须满足以下要求：

- 通过严格 schema parsing；不能从自由文本中猜测缺失字段；
- 引用的 entity、action、predicate 必须存在于当前环境 schema；
- 生成的 action 必须通过 adapter 的 typed admissibility check；
- 声称的 effect 必须由执行前后 state delta 证明；
- 声称的 evidence 必须带真实 locator，不能是模型自述；
- 声称的 success 必须由环境或 benchmark evaluator 确认；
- 无法验证、解析失败或引用不存在对象时必须拒绝，不能自动修补后继续；
- 原始输出、解析结果、拒绝原因和验证 proof 必须同时记录。

让 35B 检查 9B，或让 9B/35B 多次投票，不能构成 proof；这些方法最多用于生成候选或安排搜索顺序。模型之间可能共享相同 hallucination，因此最终裁决必须来自模型之外。

### 2.4 不确定时保留歧义或拒绝

一条示例通常不足以识别目标域中的所有 slot、operator 和 predicate。系统不能用最高相似度候选掩盖这种不可识别性：

- 唯一合法 binding：接受；
- 多个合法但行为等价的 binding：保留等价类；
- 多个非等价 binding：标记 `ambiguous`；
- 没有合法 binding：拒绝迁移；
- 测试状态要求未识别的 operator：`abstain`。

### 2.5 Harness 必须 fail closed

缺少 schema、adapter、evidence、effect verifier 或官方 evaluator 时，实验必须失败或跳过该样本，不能静默使用 stub、identity mapping、substring match 或默认部分奖励。

## 3. 需要消除的启发式机制

| 当前机制 | 主要风险 | 原则性替代 |
|---|---|---|
| `SequenceMatcher + 人工权重 + threshold` | 文本相似不代表功能相同 | typed behavioral equivalence + MDL program induction |
| `DOMAIN_PREFERRED_OPS` | 人工预设目标域需要哪些技能 | goal 与 skill postcondition 的逻辑统一 |
| 手写 `PREDICATE_TRANSLATIONS` | 人工声明跨域 predicate 等价，甚至静默 drop | 从 one-shot transition 合成并验证 predicate homomorphism |
| substring/partial action matching | 可能执行语义错误但文本相似的动作 | action grammar parser + typed exact unification |
| `pass_rate >= 0.5` promotion | 阈值没有可识别性含义 | 全部约束满足、保持 version space 或拒绝 |
| LLM/Judge 相似度 verdict | 不稳定、不可复现、可能引入目标域先验 | proposal-only model + executable verifier |
| fuzzy answer matching | 空答案或 substring 可能得到非零奖励 | 官方 evaluator、exact/typed checker |
| 目标域 per-task bank 混入 seeds | 破坏 game-to-target transfer claim | source-only lineage enforcement |

旧机制可以暂时保留用于复现实验，但不能进入新的 principled transfer 主结果。

## 4. Canonical Skill Program

每个游戏技能首先编译为领域无关的 typed transition program：

```text
CanonicalSkillProgram = {
    input_slots,
    preconditions,
    protocol,
    expected_effects,
    evidence_obligations,
    termination_condition,
    failure_conditions,
    source_lineage
}
```

示例：

```yaml
input_slots:
  object: movable_entity
  container: container_entity

preconditions:
  - reachable(container)
  - contains(container, object)

protocol:
  - OBSERVE(container)
  - NAVIGATE(container)
  - ACQUIRE(object, container)
  - VERIFY(holding(object))

expected_effects:
  add:
    - holding(object)
  delete:
    - contains(container, object)

evidence_obligations:
  - GATHER
  - VERIFY

termination_condition:
  - holding(object)
```

typed IR 是明确、可审计的接口规范，不是按目标 benchmark 调整的相似度规则。

## 5. 从 clustering 改为程序归纳

### 5.1 功能等价

两个 concrete skills 只有在存在变量重命名后同时满足以下条件，才允许共享一个 abstract program：

1. input slot types 可统一；
2. preconditions 同构；
3. protocol operator 与依赖关系同构；
4. 在 replay/counterfactual states 中产生一致的 typed effects；
5. evidence obligations 和 termination semantics 一致。

自然语言名称和描述不参与最终等价判定。

### 5.2 MDL/program grammar induction

不预先指定 cluster 数量，也不使用手工 merge threshold。选择使下式最小的共享程序库：

```text
L(shared_program_library)
+ Σ L(task_specific_residuals)
+ L(heldout_transition_prediction_errors)
```

只有当抽取一个 shared program 能减少总描述长度，并保持 held-out transition prediction，才合并 source skills。

最终的 mega-skill cluster 是程序等价类的视图，而不是文本聚类结果。

### 5.3 Source-only lineage

程序归纳阶段只允许读取 `gymv` 游戏数据。每个 abstract program 必须保存完整 lineage，并机械检查：

```text
all(member.domain == "gymv" for member in abstract_skill.lineage)
```

任何包含 ALFWorld、Browser、Video、Visual Reasoning 或测试样本的程序都不得进入 game-to-target 条件。

## 6. One-shot transfer verification 与准入

整个过程先迁移、后验证：

```text
source verified skill
        ↓
target binding proposal（untrusted）
        ↓
candidate target skill
        ↓
one target demonstration + Harness verification
        ↓
ADMITTED | CONDITIONAL | INCONCLUSIVE | REJECTED
```

Binder 仍然是约束求解器，但它的任务不是从一条示例宣布唯一的普遍语义，而是构造候选 target binding、验证示例覆盖到的部分，并保留尚未被证据消除的歧义。

给定一个 abstract program 和一条目标域成功轨迹：

```text
D = (s0, a0, s1, a1, ..., sn)
```

Binder 求解三个映射：

```text
φslot       abstract slots      → target entity types/instances
φaction     abstract operators  → target action schemas
φpredicate  abstract predicates → target observable predicates
```

合法 binding 必须保证 abstract transition 和 target transition 交换：

```text
abstract_transition(z, op)
          │                 │
       φstate            φaction
          │                 │
          ▼                 ▼
target_transition(φstate(z), φaction(op))
```

换句话说，绑定后的程序必须解释 demo 中真实发生的 action 和 state delta。

### 6.1 Binding artifact

每次 one-shot adaptation 输出持久化、可复现的 artifact：

```json
{
  "abstract_skill_id": "program.acquire_transform_deliver",
  "target_domain": "alfworld",
  "target_task": "pick_and_place",
  "demo_id": "train/demo_0001",
  "slot_map": {
    "object": "movable_object",
    "container": "receptacle"
  },
  "action_map": {
    "NAVIGATE": "GoTo(receptacle)",
    "ACQUIRE": "Take(object, receptacle)",
    "COMMIT": "Put(object, receptacle)"
  },
  "predicate_map": {
    "holding(object)": "inventory_contains(object)",
    "goal_completed": "environment_reward_positive"
  },
  "observed_coverage": [
    "NAVIGATE",
    "ACQUIRE",
    "COMMIT"
  ],
  "binding_status": "unique",
  "admission_status": "pending",
  "proof_trace": []
}
```

### 6.2 Binding version space

接口不返回单个最高分候选，而返回所有满足约束的候选：

```python
BindingVersionSpace(
    valid_bindings=[...],
    equivalent_classes=[...],
    unsupported_constraints=[...],
    observed_coverage={...},
    status="unique | equivalent | ambiguous | unsat",
)
```

Harness 只能执行所有剩余候选一致支持的动作。任何未由 one-shot demo 或 target schema 识别的行为都必须 abstain。

### 6.3 One-shot skill admission

Harness 使用一条预先固定的目标域成功示例重放或执行候选技能，并检查：

1. 每个 target action 是否存在于环境 action schema；
2. action arguments 与当前实体类型是否合法；
3. 每一步是否产生候选技能声明的真实 state delta；
4. precondition、expected effect 和 termination condition 是否有环境证据；
5. 最终结果是否通过官方 success evaluator。

验证结果不能压缩为单一 pass rate，而必须是以下四类之一：

| Admission status | 含义 | 后续行为 |
|---|---|---|
| `ADMITTED` | 示例覆盖的完整候选程序通过静态与动态验证 | 仅在 artifact 记录的 verified scope 内使用 |
| `CONDITIONAL` | 技能可用，但依赖已识别且可检查的额外 preconditions | 满足 contract 时使用，否则 abstain |
| `INCONCLUSIVE` | 示例未覆盖全部 operator，或多个非等价 binding 仍无法区分 | 接纳已验证子技能，或请求有区分性的额外示例 |
| `REJECTED` | 存在已证实的语义冲突、目标能力缺失或不可修复 contract violation | 禁止使用该候选，并保存结构化拒绝原因 |

一次成功只能证明技能在已观测条件和已覆盖操作上的局部兼容性，不能推出“任意对象、任意状态和任意任务都可用”。因此 admission artifact 必须携带适用范围：

```yaml
admission_status: CONDITIONAL
verified_scope:
  task_family: pick_and_place
  object_type: food
  destination_type: openable_receptacle
verified_operators: [NAVIGATE, ACQUIRE, OPEN, COMMIT]
required_preconditions:
  - holding(object) before COMMIT
  - open(destination) before COMMIT
unverified_operators: []
proof_trace: [...]
```

`ADMITTED` 不等于永久有效。运行时一旦观察到 effect、evidence 或 termination violation，状态必须从 `ADMITTED/CONDITIONAL` 转为 `SUSPENDED`，停止调用并进入重新验证。

### 6.4 失败诊断与是否请求更多示例

“Try and see”必须由真实 target Adapter、环境 transition、确定性 step checker 和官方 evaluator 共同实现，不能让 9B/35B 根据自由文本轨迹自行宣布成功。一次失败后，Harness 根据结构化 failure code 决定后续路径：

| Failure class | 判定 | 处理 |
|---|---|---|
| `BINDING_ERROR` | action 名称、参数或 predicate 映射与 schema 冲突 | 在同一示例上修复候选并重新验证 |
| `MISSING_PRECONDITION` | 目标动作需要候选 contract 未声明的条件 | 增补可验证 precondition，标记 `CONDITIONAL` 后重验 |
| `INSUFFICIENT_COVERAGE` | 示例没有执行候选技能的部分 operator | 标记 `INCONCLUSIVE`，请求针对未覆盖 operator 的示例 |
| `AMBIGUOUS_BINDING` | 多个与示例一致的 binding 会在 held-out 状态产生不同动作 | 请求能区分这些候选的示例；获得证据前 abstain |
| `TARGET_CAPABILITY_MISSING` | 目标环境没有实现技能必需的 action/effect | `REJECTED`，更多同类示例不能解决 |
| `STRUCTURAL_MISMATCH` | 源程序的控制或因果结构不能在目标域保持 | `REJECTED` |
| `TRANSIENT_ENV_FAILURE` | timeout、reset failure 或无法归因于技能的环境异常 | 标记 `UNKNOWN` 并按固定预算重试，不能计作通过或语义拒绝 |
| `OFFICIAL_EVAL_FAILURE` | 执行完成但官方目标条件不成立 | 不得接纳；定位为可修复 binding/contract 问题，否则拒绝 |

额外示例不是默认 fallback。只有当新增示例能够覆盖一个未验证 operator 或区分仍存活的 binding hypotheses 时才请求；如果目标域缺少必要能力或已经观察到结构矛盾，应直接拒绝该技能。

在严格的纯 one-shot 主实验中，`REQUEST_EVIDENCE` 是一个终止 verdict：系统记录自己需要什么证据并 abstain，但不能实际读取第二条示例。实际取得额外示例并再次验证属于单独的 **adaptive evidence acquisition** 扩展条件，必须单独报告示例数量、选择策略和结果，不能计入纯 one-shot success。

### 6.5 技能状态机

```text
PROPOSED → BOUND → ONE_SHOT_VALIDATING
                         ├→ ADMITTED ─────┐
                         ├→ CONDITIONAL ──┤→ SUSPENDED → REVERIFY
                         ├→ INCONCLUSIVE → REQUEST_EVIDENCE
                         └→ REJECTED
```

Binding version space 是 admission 的内部机制，而不是要求一个示例恢复完整语义。当多个候选对当前状态产生相同的归一化 action 时，Harness 可以执行该共同动作；一旦候选行为分歧，必须 abstain 或请求额外证据，不能使用 LLM 投票选一个。

## 7. Source-domain meta-learning

约束求解可以保证正确性，但候选空间可能较大。可在游戏域之间训练一个 frozen proposal model：

```text
game-A abstract skill + game-B schema + game-B one-shot demo
                              ↓
                  qθ(binding hypotheses)
```

训练过程使用 game-to-game episodic meta-learning：

1. 一个游戏作为 source；
2. 另一个游戏模拟 unseen target；
3. Binder 只看到 target 的一条 demonstration；
4. 使用 target-game held-out replay success 训练 proposal model；
5. 真正迁移到 ALFWorld/Web/VR/Video 时冻结全部参数。

最终系统仍然是：

```text
Frozen proposal model
        ↓
small binding candidate set
        ↓
constraint solver
        ↓
real Harness execution proof
```

Proposal model 的置信度不能覆盖约束求解器的否决。

## 8. Harness 作为 proof checker

Harness 不再为技能计算人工 fit score。它只检查可判定命题：

1. 所有 slot type 是否统一；
2. preconditions 在当前 `StateSchema` 中是否成立；
3. 目标 adapter 是否提供对应 typed action schema；
4. 执行前后的 state delta 是否满足 expected effects；
5. evidence 引用是否指向真实 observation、frame、DOM、command result 或 state transition；
6. termination condition 是否成立；
7. benchmark 官方 success evaluator 是否通过。

推荐执行顺序：

```text
goal/intent
    ↓
contract unification / backward chaining
    ↓
runnable abstract programs
    ↓
one-shot binding version space
    ↓
static Harness checks
    ↓
Actor choice among valid programs
    ↓
real target Adapter execution
    ↓
dynamic effect/evidence verification
```

检索主路径应使用 goal 与 postcondition 的逻辑统一。Embedding 只能作为搜索加速器，不能决定 eligibility 或 transfer verdict。

## 9. Adapter 接口

每个目标域 Adapter 应公开以下结构化接口：

```python
class PrincipledSkillAdapter:
    def action_schemas(self) -> list[ActionSchema]: ...
    def predicate_schemas(self) -> list[PredicateSchema]: ...
    def parse_state(self, observation) -> StateSchema: ...
    def observe_delta(self, before, after) -> StateDelta: ...
    def execute(self, grounded_action: GroundedAction) -> StepResult: ...
    def success_spec(self) -> SuccessSpec: ...
```

不允许 adapter 通过未知字符串透传、identity fallback 或静默 predicate drop 来制造成功。

## 10. ALFWorld vertical slice

ALFWorld 是第一优先级，因为 admissible commands 具有明确 grammar，环境还提供原生任务 reward。

### 10.1 Structured action grammar

例如：

```text
take apple 1 from fridge 1
put apple 1 in/on microwave 1
```

必须解析为：

```python
Take(object="apple 1", source="fridge 1")
Put(object="apple 1", destination="microwave 1")
```

之后用 action type 和 entity type 做 exact unification，不能使用 substring 或 unique-partial matching。

### 10.2 ALFWorld one-shot protocol

1. 从 ALFWorld train split 固定选择一条成功轨迹；
2. 记录完整 observation、admissible actions、selected command、state delta 和 reward；
3. 先生成 candidate target skill 和 binding version space；
4. 使用该轨迹执行 one-shot admission，保存 verified scope、未覆盖 operator、failure code 和 proof trace；
5. 只有 `ADMITTED` 或满足已验证 preconditions 的 `CONDITIONAL` 技能可以进入 held-out evaluation；
6. 冻结 binder、actor、skill library、admission artifact 和所有模型；
7. 在 held-out IID/OOD tasks 上执行，运行时 contract violation 会暂停技能但不能更新 binding；
8. Harness 使用 typed state delta 和原生 reward 验证；
9. 不允许使用 target per-task skill bank 或 test trajectories 构建、修复或重新接纳 binding。

### 10.3 ALFWorld 验收条件

- admissible command 被 grammar parser 完整解析；
- 所有执行动作来自 exact typed unification；
- 任何歧义都会产生 abstention；
- 每个候选技能获得 `ADMITTED/CONDITIONAL/INCONCLUSIVE/REJECTED` verdict；
- admission artifact 明确记录 verified scope，不能把一次成功外推到未覆盖对象、operator 或 task family；
- success 只由 ALFWorld 原生 completion reward 决定；
- game-only lineage audit 通过；
- one-shot demo 与 evaluation episodes 无交叉；
- 真实环境测试通过，不能只依赖 FakeALFWorldEnv。

## 11. 实验条件

| 条件 | 技能来源 | Target supervision | 是否训练 Target LoRA | 用途 |
|---|---|---:|---:|---|
| Base | 无 | 0 | 否 | 无技能基线 |
| Random game program | 游戏 | 0 | 否 | 排除普通 prompt/context 增益 |
| Retrieved game program | 游戏 | 0 | 否 | zero-shot program transfer |
| Game program + one-shot admission | 游戏 | 1 条 demo | 否 | 核心实验 |
| Admission + adaptive evidence | 游戏 | 1 条 demo 起，按需增加 | 否 | 非纯 one-shot 的证据获取扩展 |
| One-shot binding + online GRPO | 游戏 | 1 条 demo + rollout reward | 是 | RL adaptation 扩展条件 |
| Target-domain skill bank | 目标域 | 多条 | 可选 | Upper bound |

必须把以下两种 claim 分开：

- **纯 one-shot transfer**：一条固定的有标签 target demo 用于 binding verification/admission；进入 held-out test 前冻结 artifact 和所有参数；没有 target rollout learning。
- **one-demo initialization + online RL adaptation**：一条 demo 初始化，但继续使用 target rollout reward 更新模型；这不是严格的纯 one-shot。

## 12. 评价指标

最终 reward 之外，必须报告：

- held-out task success rate；
- 相对 Base 的 transfer lift；
- negative-transfer rate；
- binding coverage；
- binding ambiguity rate；
- admission rate，以及 `ADMITTED/CONDITIONAL/INCONCLUSIVE/REJECTED` 分布；
- verified-scope coverage 和 operator coverage；
- false-admission rate：被接纳但在声明 scope 内违反 contract 的比例；
- evidence-request rate；在独立的 adaptive evidence 条件中，另报告每条额外示例实际消除的 binding hypotheses 或新增覆盖的 operator；
- abstention rate；
- contract satisfaction rate；
- evidence completeness；
- unsupported operator/predicate rate；
- action-schema execution success；
- environment steps、tokens 和 wall time。

建议同时报告 unconditional success 和 conditional-on-coverage success，避免系统通过大量 abstain 得到虚高的条件成功率。

## 13. 2 × 4 L40S 资源布局

对于纯 one-shot transfer：

- 4 GPU：并行执行 binding hypotheses 或 held-out rollout；
- 2 GPU：用于 source-domain binder/meta-learning，target evaluation 时冻结；
- 2 GPU：35B untrusted proposal model，只生成 program/binding candidates，不产生最终 verdict。

对于独立的 `one-shot + GRPO` 条件：

- 4 GPU：rollout；
- 2 GPU：LoRA/GRPO；
- 2 GPU：35B proposal 服务。即使保留 `judge` 这个进程名称，其输出也不能作为成功或迁移 verdict。

9B 与 35B 输出必须经过相同的 executable verifier，不能直接计入成功率。若 verifier 不可用，该 rollout 必须标记为 invalid，而不是采用模型自评。

## 14. 实现顺序

### P0：修复实验闭环

1. Stage 3 JSONL seed 必须加载为真实 Bank + `SkillQueryEngine`；
2. 修复最终 `info`、VR `AdapterRunContext`、空答案奖励和 BrowserGym task ID；
3. 修复 GRPO `train_step` 参数及 evaluation adapter loading；
4. evaluation 禁止静默退回 stub；
5. 禁止 target per-task bank 和 mixed-domain mega-skill 数据泄漏。

### P1：Typed semantics

1. 新增 `CanonicalSkillProgram`；
2. 为 action、predicate、state delta 和 evidence 定义 typed schema；
3. 游戏轨迹编译到 typed IR；
4. 添加 source-only lineage audit。

### P2：Program induction

1. 删除 `SequenceMatcher` 主路径；
2. 实现 functional equivalence replay；
3. 实现 MDL/shared-program induction；
4. 输出 shared program 与 task-specific residual。

### P3：One-shot binding 与 admission

1. 新增 `BindingHypothesis` 和 `BindingVersionSpace`；
2. 实现 slot/action/predicate constraint generation；
3. 实现 satisfiability、equivalence、ambiguity 和 coverage 检查；
4. 保存 proof trace；
5. 实现 one-shot admission verdict、verified scope 和结构化 failure taxonomy；
6. 实现 `ADMITTED/CONDITIONAL/INCONCLUSIVE/REJECTED/SUSPENDED` 状态机；
7. 只有 coverage/ambiguity failure 可以请求有区分性的额外示例；
8. Harness 遇到未知绑定或 scope 外调用时必须 abstain。

### P4：ALFWorld

1. 实现 admissible-command grammar parser；
2. 删除 partial string action resolution；
3. 记录一条真实 train demonstration；
4. 对 candidate target skill 运行 one-shot admission；
5. 接入 Stage 3 registry、runner 和 evaluator；
6. 完成 Base/Random/Zero-shot/One-shot 四组实验。

### P5：扩展目标域

ALFWorld 跑通后按以下顺序扩展：

1. MiniWoB/Browser；
2. Visual Reasoning；
3. Video；
4. 最后运行完整跨域矩阵和可选 GRPO 条件。

## 15. 必须具备的不变量测试

新增实现必须覆盖：

1. `game_only_lineage`：target-domain 数据不能进入 source program induction；
2. `binding_preserves_effects`：映射后的 target transition 满足 abstract effects；
3. `ambiguous_binding_abstains`：非等价多解不能任意选一个；
4. `unsupported_predicate_rejects`：未知 predicate 不能 identity-pass；
5. `unsupported_action_rejects`：未知 action 不能 substring-match；
6. `llm_cannot_override_verifier`：proposal model 的文字判断不能改变 verifier 结果；
7. `no_silent_stub_in_eval`：真实评估缺少 executor 时必须失败；
8. `one_shot_split_isolation`：demo、validation 和 test 样本严格隔离；
9. `official_reward_only`：成功率只读取环境或 benchmark 官方 evaluator；
10. `proof_trace_replayable`：每个接受的 binding 都能重放其 proof trace。
11. `model_output_is_untrusted`：9B/35B 的自评、置信度或多数投票不能直接改变 verdict；
12. `hallucinated_reference_rejects`：不存在的 entity/action/predicate 必须拒绝并记录，不能静默修补。
13. `one_success_is_scope_bounded`：一条成功示例不能自动扩大到未观测 entity type、operator 或 task family；
14. `inconclusive_is_not_rejected`：覆盖不足必须请求针对性证据或接纳子技能，不能伪装成验证通过或结构性拒绝；
15. `missing_capability_does_not_request_demo`：目标能力缺失时直接拒绝，不能用更多示例掩盖不可迁移性；
16. `runtime_violation_suspends_skill`：已接纳技能发生 contract violation 后必须暂停并重新验证；
17. `test_rollout_cannot_update_admission`：held-out test transition/reward 不得修改 binding 或 admission artifact。

## 16. 完成定义

只有同时满足以下条件，项目才能声称实现了 principled game-to-domain one-shot skill transfer：

- shared skill library 仅由游戏数据归纳；
- 技能等价不由文本相似度或 LLM verdict 决定；
- 9B/35B 只提供 untrusted proposals，不能产生最终事实或成功判断；
- target candidate skill 的 binding 与首次 admission 只使用一条预先固定的成功 demo；
- binding 通过 typed constraints 和真实 transition 验证，并获得四类 admission verdict；
- 每个接纳技能都声明 verified scope、covered operators 和运行时 preconditions；
- 只有确实能减少歧义或增加 operator coverage 时才生成额外证据请求；纯 one-shot 主实验不得读取第二条示例，结构不兼容或能力缺失必须拒绝；
- 歧义被显式保留或 abstain；
- evaluation 使用 held-out target tasks；
- 所有参数在纯 one-shot evaluation 中冻结；
- 真实 Harness/Adapter 被调用；
- reward 来自官方环境/evaluator；
- 每次接受、拒绝和 abstain 都有可重放 proof trace。
