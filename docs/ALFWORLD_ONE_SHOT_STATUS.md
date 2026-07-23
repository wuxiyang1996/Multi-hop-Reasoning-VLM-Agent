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

51 个测试全部通过。

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

## version-space 与负迁移修复（2026-07-22）

旧的单一自由文本 binding 已替换为最多四个结构化候选。每个候选必须：

- 用 source-node ordinal 对 adaptation example 的全部 transition 做无重叠、连续、完整 partition；
- 为每条 source edge 引用一个顺序一致的 target transition boundary；
- 在真实 demo 的两次独立 induction 中都重现，不能只靠一次有利采样 admission；
- full-action alpha-renamed demo 不再是硬 admission gate，而是区分 content-free structure 与
  target-grounded one-shot binding 的 attribution control。

在线 runtime 现在维护 `BindingVersionSpace`。Motif Agent 在真实 transition 后只能给绑定 ID、当前
receipt ID 和 `SUPPORTED / REFUTED / INCONCLUSIVE`；Harness 检查引用，`REFUTED` 淘汰该候选并按
binding ID 的确定性顺序试下一个。Agent `ABSTAIN`、候选耗尽或达到一次 source-induced replan budget
时，不再终止 ALFWorld episode，而是关闭 source intervention 并继续同一个 target-only actor。

真实默认 smoke 的两次 induction 为：

| repetition | raw candidates | alpha-renamed | within-repeat intersection |
|---|---:|---:|---:|
| 0 | 1 | 2 | 1 |
| 1 | 0 | 0 | 0 |

旧实现把 raw/alpha 交集当作硬门；该轮跨 repetition 交集为 0，因此从 step 0 fail closed，target actor 随后用与 baseline 完全相同的五个
actions official success。冻结摘要见
[`results/alfworld_binding_stability_summary.json`](results/alfworld_binding_stability_summary.json)。

另一个单 repetition 复跑曾得到 raw=4、renamed=4、intersection=4，证明同 prompt 的 binding
generation 本身不稳定；这正是 repeated intersection 必须存在的原因，不能事后挑选有 binding 的 run。

2026-07-22 后续修正：硬门应当是 repeated **raw** stability，而不是 raw/alpha stability。alpha 删除
target action content 后结构变化，只说明候选利用了 one-shot target grounding，不能说明它无效。新 artifact
把候选分别标记为 `GENERIC_STRUCTURAL` 与 `TARGET_GROUNDED_PROVISIONAL`；两者都必须接受后续 live
transition 检验。正式 eval 默认要求预先冻结 artifact，禁止 test-time rebinding。

同日生成的 Strider→ALFWorld frozen artifact（hash
`020feba4870f93f56f58ee69f633a63cdfd502d8f723adfe29f19c9fafbee432`）在两次 raw induction 中
稳定保留 3 个候选：1 个 `GENERIC_STRUCTURAL`、2 个 `TARGET_GROUNDED_PROVISIONAL`。这只证明 binding
机制可重复，不证明 Strider content 有增量价值。

固定 `PYTHONHASHSEED=0` 后，OOD game index 0 的 matched initial state 上 target-only 与 authentic 都在
12-step cap 失败；authentic 使用一次 replan 后更晚才探索 cabinet，属于负效率信号。game index 11 的
target-only 在 5 步官方成功。新的 source-grounded review 首轮又发现接口 bug：motif view 只公开 receipt
count，却要求 Agent 返回未公开的 hash；现已改为 `(source_node_ordinal, receipt_ordinal)`，Harness 再解析
真实 receipt。该失败 run 不计入 transfer 结果。单候选旧 runtime 曾以相同 5 个 actions 成功，但它按 hash
任意选择一个 binding，现已废弃。新 runtime 对完整 version space 做一次 aggregate review：只有所有候选
一致 `ADMIT` 或一致 `REPLAN` 才干预，分歧或 schema/evidence 缺失立即 fallback。OpenRouter 对相同 prompt
即使给 seed 仍可能返回不同 completion，因此 matched conditions 还共享 exact-request memo；只有 history
真的分叉才重新调用。最新 smoke 的 aggregate response 未返回完整 binding ID 集，step 0 fail closed，随后
authentic 精确复现 target-only 的 5 actions 并官方成功。它证明 fallback 无损，但仍无迁移价值证据。
机械汇总见 [`results/alfworld_frozen_binding_grounded_summary.json`](results/alfworld_frozen_binding_grounded_summary.json)。

## 下一步 concrete plan

> 2026-07-23 更新：本文件前半记录旧 one-shot vertical slice。更新的 weak-knowledge
> strong-teacher feasibility pilot 已不再使用 exact-topology binding，详见
> [`ALFWORLD_WEAK_KNOWLEDGE_TEACHER_PILOT.md`](ALFWORLD_WEAK_KNOWLEDGE_TEACHER_PILOT.md)。
> 新 pilot 使用四个 canonical adaptation artifacts，因此不是新的 one-shot 结果。它发现过
> task 0 的一个 authentic-only 16-step success，但冻结 10-actor-seed 复制未复现：
> target-only `0/10`、generic `2/10`、authentic `1/10`。因此不执行本文件 Gate 4，不把
> adaptation 降为 one-shot，也不训练 Harness LoRA；若继续，必须先预注册新的 target-side
> predictive Harness 机制。

### Gate 1：停止把 generic source graph 当 authentic backbone

当前 Strider/Thunder graph 主要编码 skill-ID run change、reward sign 和 topology，authentic 并未超过
receipt-only/renamed/shuffled。下一轮只允许 `source_agent_v2` 数据进入 authentic treatment：每个 source
step 必须有 Agent proposal set、selected proposal、predicted observable delta、真实 transition、post-verdict
和 continue/replan/abstain receipt。六游戏各至少两个 episode，并做 correct/reasoning-null/randomized/
target-only matched source qualification。未过 gate 仍可跑工程 control，但状态必须是 `GENERIC_ONLY`。

### Gate 2：one-shot version space（已实现，等待更强 source evidence）

当前 GPT-5-mini 把一个 demo 的具体 `look/go to/move` 序列写进 binding，容易把 example 记忆成 motif。
现已生成多个匿名 binding hypotheses；Harness 只检查引用和 schema，不选择“最合理”者。后续 live
receipt 由 Agent verifier 给出 untrusted supported/refuted/inconclusive，外部 version space 只淘汰被
注册 verifier refute 的候选。若候选无法区分，返回 `NEED_MORE_EXAMPLE`；若全部消失，立即 target-only
fallback。不得增加 COLLECT→TAKE、predicate ontology 或 embedding alignment。

### Gate 3：bounded negative-transfer budget（已实现）

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
