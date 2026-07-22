# 旧结果如何进入新 Harness

旧结果不会丢弃，但按证据强度分层使用。

## 可以直接使用

- 六游戏 best checkpoint：生成 skill-on / skill-off matched rollout。
- 完整 observation、native actions、proposal、choice、verification、replan、abstain、official outcome：构造 transition receipt。
- 原始 episode 和 checkpoint identity：保证干预对照可复现。

## 只能作为 lineage 或 baseline

- 旧 skill bank、mega-skill family、自然语言描述和聚类标签。
- 旧 source→target proposal。
- 任何没有真实 before/after 或 replay receipt 的 branch/replan 声称。

它们通过 `legacy_import.py` 加载时固定标记为 `LINEAGE_RETRIEVAL_ONLY`，不能进入 action authority，也不能直接晋升为 `SOURCE_SUPPORTED`。

历史审计曾发现 mega-skill 约有 180 个 members、20 个 families、14 个 signatures，而且大族会坍缩为类似 `ACT → VERIFY` 的 generic protocol。这些数字是诊断线索，不是本仓库硬编码的结论；必须对实际导入文件重新运行审计。

## 旧结果的四种用途

1. **数据源**：从原始 rollout 重新构造 receipt，而不是信任旧标签。
2. **候选 prior**：告诉 Motif Agent 哪些 episode 值得比较，但不决定语义。
3. **对照组**：比较 raw trajectory retrieval、verbal skill、mega-skill 和新 motif graph。
4. **失败回归**：确保新方法不会把统一事件协议、三步单 trace 或无 receipt 的文本重新包装成 backbone。

## 迁移方式

新分支不复制大型资产。运行时显式传入旧 JSONL/rollout 路径；推荐把不可变 manifest 和内容 hash 写进实验输出。原始仓库、B2 bucket 和 parent commit `948f64a` 是 lineage 来源，不是隐式运行依赖。
