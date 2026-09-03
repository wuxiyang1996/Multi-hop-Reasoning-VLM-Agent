# STAR V38：annotation-side goal-relation preflight

## 结论

按成本边界完成了最后一个 STAR、零新增 API 调用的 annotation-side preflight，并在这里停止。
结果不是自然视频迁移验证：**STAR symbolic dynamics 与备选 policy headroom 确实存在，但当前
neural grounding 的选择信号和 source-specific value 都没有通过 qualification。**

本实验重用已经消费过的 V27 development 数据：128 个 Interaction questions、64 个 video
clusters；direct 与 typed-proof 使用同一个 `openai/gpt-4.1-mini` 和完全相同的 24 帧。没有打开
fresh split，没有重新调用 OpenRouter/Gemini/Claude。

| Condition | Correct | 相对 neural-only |
|---|---:|---:|
| Neural-only uniform direct | 60/128 | — |
| Unified fail-closed source-induced | 60/128 | `0W/0L` |
| Source relation semantics（仅反事实诊断） | 63/128 | `11W/8L`, `p=.648` |
| Target-native identical relation rule | 63/128 | 与 source 完全相同 |
| Generic always-use-proof | 62/128 | `11W/9L`, `p=.824` |
| Source-permuted binding | 60/128 | `0W/0L` vs neural |
| Direct/proof oracle ceiling | 71/128 | `+11` |
| Four-policy oracle ceiling | 83/128 | `+23` |
| Official STAR situation-graph executor | 128/128 | evaluator-only ceiling |

Source-semantics 分支的 `+3` 不能作为成功证据：paired test 不显著，而且 target-native rule
逐题完全相同。这说明当前增益由 target proof relation rule 解释，不能归因于游戏 source。

## 这次实际迁移了什么

Source 没有换回旧的手写 `COMMIT/VERIFY/REPLAN` 模板。它仍然是 CLEVRER V15 使用的同一份、
从 Sokoban `(state, action, effect, next_state)` 自动归纳且 fresh-confirmed 的匿名 program：

```text
UPDATE ENTITY_GOAL_RELATION / RELATION_COVERAGE
  -- observed positive relation delta -->
UPDATE ENTITY_GOAL_RELATION / RELATION_COVERAGE  [ONE_OR_MORE]

terminal: ENTITY_GOAL_RELATION coverage == 1
otherwise: abstain
```

STAR adapter 只做 target-native binding：

1. public functional program 确认问题属于 situation → action → entity/verb/action relation query；
2. frozen neural typed proof 提供 `ENTITY_GROUNDING`、`EVENT_OCCURRENCE`、
   `ANSWER_ENTAILMENT` 状态；
3. proof 相对 direct 新覆盖至少一个 relation、coverage 正增长且 terminal relations 全部
   `SUPPORTED` 时，才形成结构适用 receipt；
4. selector 只能授权，只有 STAR executor 可以选择 `uniform_typed_proof`；否则保留 direct。

128 题中有 28 题满足结构条件，且所有 grounding/authorization receipt 都标记
`formal_outcome_read=false`。Source artifact hash、target grounder、target state 和 authorization
全部 content-addressed。

## 为什么 unified condition 仍然是 60/128

统一 harness 没有为了得到正数而绕过 utility gate。此前唯一可用的 matched STAR calibration
只有 `9W/7L/112T`；其 directional posterior 下界不足以超过中性点。因此：

```text
100 rows -> structural applicability failed
 28 rows -> directional utility not calibrated
  0 rows -> executor called
```

这正是 fail-closed harness 应有的行为。`63/128` 只报告为“若绕过 utility gate 执行 source
relation semantics 会发生什么”的反事实诊断，不是实际 unified transfer score。

## Annotation-side 诊断回答了什么

它把两个问题分开了：

- **Dynamics/executor 是否存在？存在。** 官方 situation hypergraph + functional executor 对这
  128 题达到 128/128；只在已有四个 neural policies 中选择，oracle ceiling 也能从 60 提到 83。
- **当前 oracle-free neural grounder 能否可靠选中有益操作？没有证据。** 关系规则只有
  `11W/8L`，且无法区别 source 与 target-written equivalent；这才是自然视频 extension 的 blocker。

所以增加帧数或再换大模型不是当前最有信息量的下一步。若未来重开 STAR，应先在已消费的
development videos 上训练并锁定一个不读取 functional program/situation graph 的 target-native
grounder，并同时满足：显著正向 paired utility、优于 source-permuted、优于等价 target-only rule。

## 成本与停止线

- 本轮 external provider calls：`0`；
- 本轮新增成本：`$0.00`；
- 被重放的旧 V27 receipts 历史成本约 `$0.856376`，本轮没有再次计费；
- 不打开 fresh STAR，不恢复 NExT-QA/Video-Holmes/AGQA spend；
- 状态固定为 `STAR_V38_NOT_QUALIFIED_STOP`。

## 复现与证据

```bash
PYTHONPATH=src:. python scripts/run_star_annotation_goal_relation_v38.py \
  --config configs/star_annotation_goal_relation_v38_development.json \
  --output docs/results/star_annotation_goal_relation_v38_preflight.json

PYTHONPATH=src pytest -q \
  tests/test_star_annotation_goal_relation.py \
  tests/test_star_annotation_goal_relation_v38.py
```

- 机器结果：
  [`results/star_annotation_goal_relation_v38_preflight.json`](results/star_annotation_goal_relation_v38_preflight.json)
- frozen consumed-development config：
  [`../configs/star_annotation_goal_relation_v38_development.json`](../configs/star_annotation_goal_relation_v38_development.json)
- target adapter：
  [`../src/motif_transfer/star_annotation_goal_relation.py`](../src/motif_transfer/star_annotation_goal_relation.py)
