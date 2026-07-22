# 双 Agent 架构

## 定位：适应助手，不是替代推理

模型本身已经具有通用推理能力。Motif/Harness Agent 的作用不是向 Decision Agent 注入一条完整 reasoning chain，也不是把游戏 action 翻译成 target action；它把历史 source experience 转化成可被在线证伪的结构化建议，减少新任务中的探索和适应成本。

它可以建议：

- 当前处于 evidence acquisition、verification、recovery 或 commitment 等临时角色；
- 当前仍存在哪些候选假设；
- 缺少哪类信息；
- 下一次 transition 应观察到什么变化；
- 无变化或证据冲突时应 replan 还是 abstain；
- 哪个 termination condition 仍未被满足。

这些字段都是 untrusted adaptation proposals，不是预定义 ontology，也不能包含 target action。Decision Agent 自行理解 target observation，并把适用建议 grounding 成原生动作。

## 权限表

| 能力 | Decision Agent | Motif/Harness Agent | Deterministic Harness |
|---|---:|---:|---:|
| 读取 observation / native actions | 是 | 是 | 是 |
| 提出目标原生 action | **唯一有权** | 否 | 否 |
| 与环境交互 | **唯一经 runtime 执行** | 否 | 否 |
| 提议 rollout 分段和 motif | 否 | 是 | 否 |
| 提议 one-shot binding | 否 | 是 | 否 |
| 建议 information need、expected transition、admit/replan/abstain | 否 | 是 | 否 |
| 判断 receipt/action/实验身份是否合法 | 否 | 否 | **唯一有权** |
| 判断任务成功 | 否 | 否 | 环境 official outcome |

Harness 是普通确定性代码，不是第三个 Agent。任何 LLM，包括 GPT-5-mini、Qwen 或 35B/9B 本地模型，都只能实现上面两个接口之一，不能改变权限。

## 数据流

```text
observation + native actions
          │
          ▼
    Decision Agent ─── proposal(action) ──► deterministic validation
          ▲                                      │
          │ advisory only                        ▼
 Motif/Harness Agent ◄── proposal + history ─ environment.step(action)
          │                                      │
          └──────────── receipt ◄────────────────┘
```

`PROPOSE / EXECUTE / OBSERVE / UPDATE / BRANCH / TERMINATE` 只是已有事件的统一记录字母表，不构成一个可迁移 skill。Motif 必须在该字母表上显示非平凡拓扑，并且每个 node/edge 都引用真实 receipt。

旧游戏模型的 `segment` LoRA 不是 Motif Agent 本身。它在旧 co-evolution 中学到的是“给定 trajectory segment，对当前游戏 skill bank 排序”。因此它只能作为 source evidence encoder 使用：输入保持训练时的单段 ranking prompt，输出必须是全部候选的严格排列。随后由独立 Motif/Harness Agent 跨 episode 提议控制图。把该 LoRA 直接要求成新 schema、直接生成图或直接控制 target action，均属于接口外推，不能作为迁移证据。

## Fail-closed 规则

- action 不是原生 action set 的精确成员：拒绝。
- receipt hash 不一致：拒绝。
- motif 引用未知 receipt/node 或重复消费 transition：拒绝。
- 只有自然语言描述，没有 replay evidence：不晋升。
- one-shot binding 只能是 provisional，不能成为真值。
- online review 可以要求 Decision Agent replan 或 abstain，但不得替换 action。
- source 内容价值只能由 matched official outcomes 确认。

## 我们声称与不声称的内容

我们尝试提取 **skill-conditioned policy 在交互中表现出的控制 motif**，再把它作为固定 Decision Agent 的 online adaptation assistant。这可能利用一部分外显 reasoning structure，但不是对模型内部 latent reasoning 的可识别恢复，也不声称模型原本没有这种推理能力。

Motif 只有在相同模型和预算下，稳定降低 target supervision 或交互成本，并超过 generic prompt、raw trajectory、shuffled topology 和 other-source controls 时才有增量意义。自然语言 skill 名称、游戏动作名称、embedding proximity 和 Agent 自评均不是迁移证据。
