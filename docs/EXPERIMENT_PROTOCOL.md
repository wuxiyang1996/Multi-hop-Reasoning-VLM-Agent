# 六游戏到远域的实验协议

## Phase A：source motif discovery

对六个游戏分别采集多个 episode，并做 matched skill-on / skill-off intervention：相同 checkpoint、seed、初始状态、step/token budget。记录完整 observation、native actions、Decision Agent proposal、选择、验证、replan、abstain 和 official outcome。

Motif Agent 在跨 episode 数据上提出分段和图结构。Harness 只接受 receipt 完整、至少两个 node、拓扑非空且可 replay 的候选。候选需要在 held-out source episodes 上重现，才成为 `SOURCE_SUPPORTED`。不按 skill 文本、action 名称或 embedding 聚类；“合并”仅表示 verified structural fingerprint 与干预效应均一致。

## Phase B：one-shot online grounding

每个 target task family 只给一个 adaptation example：

1. Motif Agent 为 source motif 提议 target binding hypothesis。
2. Harness 标记为 `TARGET_PROVISIONAL`，不认可其语义。
3. Decision Agent 仍在 target-native interface 中独立选 action。
4. Motif Agent 每一步只能建议 admit/replan/abstain。
5. live receipt 与 official feedback 持续淘汰不一致 binding。
6. 无可行 binding 时回退 target-only；若需要区分多个 binding，则请求额外 example，并记录额外监督成本。

如果 motif 没帮助，应区分：`NOT_APPLICABLE`、`NEGATIVE_TRANSFER`、`GENERIC_ONLY` 和 `INCONCLUSIVE`，而不是强制映射或继续消耗 rollout。

## Phase C：matched target evaluation

每个样本至少运行：

- `authentic`：真实 source motif；
- `target_only`：不提供 source；
- `generic_protocol`：等长但无 source 内容的通用控制提示；
- `shuffled_topology`：receipt 数量相同、图结构破坏；
- `other_source`：来自不匹配游戏/episode 的 motif。

五组必须共享 initial state、prefix、policy、model、sampling、tool/token/step budget。主结论只使用环境 official success/score；Agent judge 只用于诊断。

## 最小判据

- `authentic > target_only` 且优于 generic、shuffled、other-source：正迁移 pilot evidence。
- generic 与 authentic 相当：只能称 generic scaffold effect。
- target-only 优于 authentic：负迁移。
- 缺少 identity 或完整 controls：inconclusive。

先在两个 target benchmark 做小规模可证伪 pilot；只有观察到 attributable separation 才扩到四领域、八 benchmark。Game SFT 可以作为 source policy 初始化或统一 base-policy control，但不能替代 target grounding，也不能默认免除 SFT；“可以跳过 target SFT”必须由 target-only matched sample-efficiency 曲线证明。
