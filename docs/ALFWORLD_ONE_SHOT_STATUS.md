# ALFWorld one-shot transfer：当前状态与下一步

## 已经跑到哪里

2026-07-22 已把 clean two-agent implementation 接到真实 ALFWorld OOD text environment：

```text
一个成功的 ALFWorld adaptation example
  → GPT-5-mini 提议 provisional binding
  → Qwen3.5-35B Decision Agent 读取 target-native interface
  → Motif/Harness Agent 只能 ADMIT / REPLAN / ABSTAIN
  → exact native action membership
  → 真实 transition receipts
  → ALFWorld official won
```

这已经是 one-shot online execution，不再只是离线 admission。但当前 source candidate 在 source
qualification 中仍是 `GENERIC_ONLY`，所以本轮是机制与负迁移诊断，不是已证明的 reasoning-backbone
transfer。

## 本轮发现并修复的错误

1. binding prompt 原来只有 node/edge 数量，没有真实匿名 topology、decision signatures 或 source
   lineage。已补齐 receipt-grounded motif view。
2. GPT-5-mini 的区域 endpoint 必须是 `us.api.openai.com`；原 hostname 返回 401。
3. GPT-5-mini 不接受该调用中的固定 `temperature=0`。backend 现在允许不发送 temperature。
4. Qwen3.5-35B 默认隐藏 reasoning 会耗尽 completion budget并返回空/截断 JSON。恢复旧 baseline
   已验证的 `reasoning.effort=none`，加入同请求、固定次数 schema retry。
5. Motif review 原来只看到 observation/action-set hash，无法在线检查 binding。现在可读取已经由
   Decision Agent 选好的 proposal、真实 observation 和 live receipts，但输出 schema 仍无 action。
6. runtime 异常原来丢失 partial episode。现在 exception 携带已完成 receipts，诊断不会消失。
7. 新 actor 一度只收到 receipt hashes，又被额外 post-assessment 的 REPLAN 自激，导致强 baseline
   从 5 步成功退化为 12 步失败。现在恢复旧 closed planner 的短
   `state_summary + next_subgoal + action_number` schema，并保存完整 target-native history；post-step
   只记录 official terminal/success，语义 advisory 仍属于 Motif Agent。
8. transient 429/5xx/timeout/incomplete-read 只允许同请求固定两次 transport retry，不按 outcome 重抽。

39 个测试全部通过。

## matched one-episode smoke

固定 OOD seed 47、game index 0、12-step cap、相同 Qwen actor、相同 target example。五组
initial-state hash 都是
`3b94c09825e873d2fbc61df89d20277607f7617ac561d32694370debc9b2381a`。

| condition | official success | steps | repeated | Harness replan |
|---|---:|---:|---:|---:|
| target-only | 1 | 5 | 0 | 0 |
| authentic Strider | 1 | 6 | 0 | 1 |
| generic protocol | 1 | 7 | 0 | 1 |
| shuffled topology | 1 | 5 | 0 | 0 |
| other-source Thunder | 1 | 5 | 0 | 0 |

target-only、shuffled 和 other-source 的五个 native actions 完全相同。authentic 先多执行一次
`look`；generic 多走一次无关 shelf。当前结论是：

冻结的小型机械汇总位于
[`results/alfworld_one_shot_summary.json`](results/alfworld_one_shot_summary.json)；完整逐步 receipts 保留在
本地 ignored `runs/alfworld_one_shot_actor_recovery_v1/`，不向 Git 提交大日志。

> one-shot Harness 已可在线运行并 fail closed，但这个 source family 没有显示 attributable positive
> transfer；authentic 有 +1 step 的负效率信号，样本量不足以做统计结论。

## 下一步 concrete plan

### Gate 1：停止把 generic source graph 当 authentic backbone

当前 Strider/Thunder graph 主要编码 skill-ID run change、reward sign 和 topology，authentic 并未超过
receipt-only/renamed/shuffled。下一轮只允许 `source_agent_v2` 数据进入 authentic treatment：每个 source
step 必须有 Agent proposal set、selected proposal、predicted observable delta、真实 transition、post-verdict
和 continue/replan/abstain receipt。六游戏各至少两个 episode，并做 correct/reasoning-null/randomized/
target-only matched source qualification。未过 gate 仍可跑工程 control，但状态必须是 `GENERIC_ONLY`。

### Gate 2：one-shot 初始化 version space，而不是生成一个过拟合故事

当前 GPT-5-mini 把一个 demo 的具体 `look/go to/move` 序列写进 binding，容易把 example 记忆成 motif。
改成一次生成多个匿名 binding hypotheses；Harness 只检查引用和 schema，不选择“最合理”者。后续 live
receipt 由 Agent verifier 给出 untrusted supported/refuted/inconclusive，外部 version space 只淘汰被
注册 verifier refute 的候选。若候选无法区分，返回 `NEED_MORE_EXAMPLE`；若全部消失，立即 target-only
fallback。不得增加 COLLECT→TAKE、predicate ontology 或 embedding alignment。

### Gate 3：预注册 bounded negative-transfer budget

source advisory 的额外 replans、model calls 和 wall time必须计费。pilot 预注册最多一次 source-induced
replan；超过 budget 或 Agent 明确 abstain 时关闭 source intervention，后续使用同一个 target-only actor。
这是资源/伤害上限，不是人工判断 motif 语义。

### Gate 4：扩成可证伪 pilot

先固定一个 ALFWorld task family，使用 adaptation-set 中一个 example；在至少 8 个 held-out instance 上
跑五条件 paired evaluation。主指标依次为 official success、steps、invalid/repeated/no-progress、Decision
calls、Harness calls、tokens 和 wall time。只有 authentic 同时超过 target-only、generic、shuffled 和
other-source，才称为 positive pilot；否则报告 generic-only、negative、not-applicable 或 inconclusive。

### Gate 5：再决定是否扩 GPU/四领域

ALFWorld pilot 出现可归因 separation 后，才扩 task families、adaptation seeds 和其它远域。GPU/SFT
不能修复 source evidence 不可归因或 one-shot binding 过拟合，因此当前不启动 2×4 大规模训练。
