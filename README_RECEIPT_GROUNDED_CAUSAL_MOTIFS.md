# Receipt-Grounded Causal Motifs：避免 reasoning backbone 坍缩

## 结论

统一事件协议不是 transferable skill。`PROPOSE → EXECUTE → OBSERVE → UPDATE`
只说明系统记录了闭环，不包含 source skill 的增量信息。真正的 transfer object 是由真实
skill-loaded rollout 支持的、保留 decision structure 的 causal motif。

旧 mega-skill 继续保留，但权限固定为 `LINEAGE_RETRIEVAL_ONLY`：它可以提供 source skill
索引、成员 lineage、检索候选和 negative control，不能作为可执行语义或 source→target
binding 真值。

## 1. 旧 mega-skill 的实证警告

对 `frontier_data/output/megaskills_all_stages/mega_skills.jsonl` 的机械审计得到：

- 180 个成员被压缩成 20 个 families；
- 只有 14 个不同 template signatures；
- 整个 event vocabulary 只有 `ACT/DECIDE/EVALUATE/NAVIGATE/PERCEIVE/VERIFY`；
- 最大 family 包含 38 个 skills（21.1%）；
- 20/20 families 都没有 replay-receipted branch/replan/control-flow；
- `thr4` 版本甚至把 82 个 skills 合为一个 `ACT → VERIFY` family。

旧流程还显式要求 LLM `aggressively merge` 并把 family 数量压到 15–20。因此这些产物适合
做历史 lineage 和 semantic-clustering baseline，不足以证明 shared reasoning program。

复现实验：

```bash
python scripts/audit_causal_motif_collapse.py \
  --legacy-megaskills \
    frontier_data/output/megaskills_all_stages/mega_skills.jsonl \
  --source-hypotheses \
    runs/reasoning_backbone_v2_smoke_7118703/source_hypotheses_qwen3max.json \
  --output runs/causal_motif_audit/report.json
```

## 2. 三层表示

```text
统一日志协议（不是 skill）
  PROPOSE / EXECUTE / OBSERVE / UPDATE / BRANCH / TERMINATE
                         ↓
source-specific causal decision motif（transfer object）
  node partition + topology + proposal-set shape + update/replan path
  + exact source receipts + replay-fork receipts
                         ↓
one-shot target binding（Agent hypothesis）
  target policy 仍然只生成 target-native actions
```

### 2.1 不迁移的 interface capsule

- 游戏 observation 文本或图像语义；
- controller buttons/native actions；
- entity/predicate/reward 名称；
- `COLLECT→TAKE` 等人工映射；
- mega-skill 名称和自然语言 description。

### 2.2 迁移的非语义 decision structure

- 节点数量、节点中的有序 transition spans；
- 每步 proposal-set cardinality；
- selected proposal 在候选集合中的位置；
- observed `SUPPORTED/REFUTED/INCONCLUSIVE`；
- observed `CONTINUE/REPLAN/ABSTAIN`；
- node/edge topology；
- edge 引用的 replay-fork receipt；
- source program、reasoning log 和 transition content hashes。

Agent-authored prediction、rationale、edge kind 和 causal prose 全部保留用于审计，但不进入
verified causal fingerprint。改变 `BRANCH` 为 `LOOP` 或改写理由，不得改变 verified
fingerprint；只有 receipt/topology/decision events 改变才会改变它。

## 3. Atomic skill 与 motif 的边界

借鉴 `Video_Skills` 的 atomic-node/composed-motif 分层，但不复制视频 ontology。

- atomic unit：一个 receipt-bounded decision segment；
- composed motif：多个 segments 的可验证控制图；
- event alphabet：日志虚拟机指令集，不是 atomic skill ontology；
- legacy mega-skill：检索/lineage prior，不是 motif；
- target binding：一次示范初始化的临时假设，不是语义事实。

不进行文本 clustering，不要求预定 family 数，不因两个 skills 都写成 `ACT→VERIFY` 就合并。
两个 motifs 只有在 fingerprints 相同且后续 source substitution/target matched forks 不可区分时，
才有资格被报告为 behaviorally equivalent；在此之前保持 version space。

## 4. Anti-collapse gates

结构候选必须满足：

1. 至少两个非空 receipt-bounded nodes；
2. 每个 transition 都有完整 proposal 和 post-transition receipt；
3. 至少一条 control edge 引用真实 replay-fork receipt；
4. 存在可观察 control variation（候选集、selection、verdict 或 replan/abstain）；
5. topology 不只是未分段的统一 event chain。

通过这五项只能标记为 `STRUCTURALLY_SPECIFIC_CANDIDATE`。下面两项永远不能由结构审计、
人工或 GPT verdict 自动置真：

- `source_attribution_requires_matched_intervention`；
- `target_incremental_value_requires_matched_forks`。

## 5. Source attribution

相同 source initial state、prefix、policy identity、seed 和 budget 下运行：

```text
authentic skill-loaded
skill-disabled
generic protocol
shuffled topology
other-source motif
```

只使用 official/native environment outcome 和 exact execution receipt。GPT‑5‑mini 可以提出
segmentation、binding 或诊断，但不能决定 source attribution。

## 6. Target transfer

在相同 target state/prefix、target policy、token/call/action budget 下运行：

```text
authentic source motif
target-only
generic protocol
shuffled topology
other-source motif
```

每个 treatment 的环境动作都必须来自同一个 target policy；Harness Agent 不能替换动作。
`SUPPORTED/REFUTED` 只能是诊断文本，不能激活或关闭 binding。binding 的经验状态只能由
matched official environment outcomes 更新。

当前实现的 paired evaluator 只给 `PILOT_SUPPORTED/PILOT_CONTRADICTED/INCONCLUSIVE`，不把
小样本 pilot 冒充统计结论。正式论文结果仍需预注册样本数、置信区间和多 benchmark 分析。

## 7. 已实现

- `harness/causal_reasoning_motif.py`
  - legacy mega-skill 的只读 lineage contract；
  - receipt-grounded causal motif schema；
  - verified causal fingerprint；
  - anti-collapse audit；
  - authentic/generic/shuffled/receipt-null conditioning views；
  - official-outcome matched contrast evaluator。
- `scripts/audit_causal_motif_collapse.py`
  - 审计旧 mega-skill uniformity；
  - 从已验证 source graph artifact 编译 causal motif candidates。
- `scripts/eval_advisory_reasoning_backbone.py`
  - GPT post-transition verdict 已改为 diagnostic-only；
  - GPT 不能确认、否定或关闭 binding；
  - binding authority 显式记录为 matched official environment outcome。

2026-07-21 的真实 Candy v2 artifact 编译出 3 个 structurally-specific candidates，其中两个
拥有相同 causal fingerprint。这是可验证结构层面的重复，不是文本语义聚类；目前仍然没有
source attribution 或 target incremental-value 证据。

## 8. 下一步实验

1. 六个 Phase‑1 游戏分别编译 causal motif candidates；
2. 按 causal fingerprint 去除完全重复表示，但保留 lineage/version space；
3. 每个 motif 跑 skill-loaded/disabled/generic/shuffled/other-source matched source forks；
4. 只让通过 source attribution 的 motifs 进入 one-shot target binding；
5. 在两个 target benchmark 先完成五条件 paired pilot；
6. 有稳定增量信号后再扩到 4 domains × 8 benchmarks。

如果 authentic 与 generic/shuffled/other-source 不可区分，应将该 motif 标记为
`GENERIC_OR_NOT_USEFUL_UNDER_CURRENT_EVIDENCE`，而不是增加手写 ontology、重新命名或让
GPT 强行解释它为何可迁移。
