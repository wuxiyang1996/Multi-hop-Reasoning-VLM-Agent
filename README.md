# Two-Agent Motif Transfer Harness

这是从历史仓库拆出的最小研究核心。仓库只保留两个有模型判断能力的角色：

1. **Decision Agent**：读取目标环境的原生 observation/action space，并且是唯一可以选择环境动作的 Agent。
2. **Motif/Harness Agent**：从真实 rollout receipt 提议行为 motif，在一个 adaptation example 上初始化跨域 binding，并在在线交互中给出 `ADMIT / REPLAN / ABSTAIN` 建议。它不能生成、替换或执行目标动作。

确定性 Harness 不是第三个 Agent。它只验证结构、receipt、原生动作成员关系、实验身份和 official outcome。Agent 的自然语言解释从不被当作真值。

## 研究目标

从六种游戏的 skill-conditioned / skill-disabled rollout 中提取可复现的控制结构，再迁移到异构领域。这里迁移的是 receipt-grounded 的行为图，而不是游戏 action 名称、手写 ontology 或通用提示词。

一个 target example 只会创建 `TARGET_PROVISIONAL` binding。后续每一步都在线验证；证据不足时 abstain 或退回 target-only。只有 matched controls 的 official outcome 能把候选更新为正迁移、负迁移、generic-only 或 inconclusive。

## 快速运行

```bash
python -m pip install -e '.[test]'
pytest -q
python examples/smoke_two_agent.py
```

## 目录

- `src/motif_transfer/decision_agent.py`：唯一的动作 authority。
- `src/motif_transfer/motif_harness_agent.py`：motif、binding 和在线审查接口。
- `src/motif_transfer/harness.py`：fail-closed 验证与 matched evaluation。
- `src/motif_transfer/runtime.py`：双 Agent 交互循环。
- `src/motif_transfer/legacy_import.py`：把旧 mega-skill 读成只读 lineage/baseline。
- `docs/ARCHITECTURE.md`：权限边界和数据流。
- `docs/EXPERIMENT_PROTOCOL.md`：六游戏到远域的实验设计。
- `docs/OLD_RESULTS.md`：旧 checkpoint、rollout 和 mega-skill 的保留方式。

历史实现仍完整保存在 Git parent `948f64a` 以及原始工作副本中；本分支不复制 checkpoint、rollout、Slurm log 或生成结果。
