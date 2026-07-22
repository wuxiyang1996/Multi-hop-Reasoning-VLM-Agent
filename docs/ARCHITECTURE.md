# 双 Agent 架构

## 权限表

| 能力 | Decision Agent | Motif/Harness Agent | Deterministic Harness |
|---|---:|---:|---:|
| 读取 observation / native actions | 是 | 是 | 是 |
| 提出目标原生 action | **唯一有权** | 否 | 否 |
| 与环境交互 | **唯一经 runtime 执行** | 否 | 否 |
| 提议 rollout 分段和 motif | 否 | 是 | 否 |
| 提议 one-shot binding | 否 | 是 | 否 |
| 建议 admit/replan/abstain | 否 | 是 | 否 |
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

## Fail-closed 规则

- action 不是原生 action set 的精确成员：拒绝。
- receipt hash 不一致：拒绝。
- motif 引用未知 receipt/node 或重复消费 transition：拒绝。
- 只有自然语言描述，没有 replay evidence：不晋升。
- one-shot binding 只能是 provisional，不能成为真值。
- online review 可以要求 Decision Agent replan 或 abstain，但不得替换 action。
- source 内容价值只能由 matched official outcomes 确认。

## 我们声称与不声称的内容

我们尝试提取并迁移 **skill-conditioned policy 在交互中表现出的控制 motif**。这可能近似 reasoning backbone，但不是对模型内部 latent reasoning 的可识别恢复。自然语言 skill 名称、游戏动作名称、embedding proximity 和 Agent 自评均不是迁移证据。
