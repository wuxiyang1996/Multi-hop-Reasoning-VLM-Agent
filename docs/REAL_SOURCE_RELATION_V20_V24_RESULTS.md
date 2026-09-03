# Real-game relation transfer V20–V24

## 结论

V20–V24 完成了真实 source game 到 ALFWorld 的冻结 matched-fork 链路，但截至 V24，
**非平凡 neural-symbolic selector 未通过独立确认**。

源端不是自然语言 skill。它来自 MiniGrid `put_near` 与 MiniWorld `putnext_3d` 的真实
intervention receipts，并冻结为：

```text
BIND --CARRIER_BOUND--> RELATE
```

target 端只使用 ALFWorld-native action/effect features。V24 的小型 MLP 在 271 个已打开且
task-disjoint 的 matched forks 上用 5-fold task-hash OOF 训练 paired utility；开发期选择结果为
`27 selected, 6 wins, 0 losses, p=0.015625`。然而唯一一次 sealed confirmation 中，7 个
admissions 全部为 success ties：

| policy | selected | wins | losses | net | full success | target baseline | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| V24 neural risk | 7 | 0 | 0 | 0 | 151/200 | 151/200 | 1.0 |
| V20 selective | 12 | 2 | 0 | +2 | 153/200 | 151/200 | 0.25 |
| late-step | 28 | 4 | 0 | +4 | 155/200 | 151/200 | 0.0625 |
| causal-only / always-source | 52 | 5 | 0 | +5 | 156/200 | 151/200 | 0.03125 |
| lexical move-relation | 52 | 5 | 0 | +5 | 156/200 | 151/200 | 0.03125 |

因此 V24 的正式状态是 `V24_SEALED_CROSS_DOMAIN_TRANSFER_NOT_VALIDATED`。另一方面，
`causal-only/always-source` 在同一 sealed split 上得到 `+2.5pp`，证明 relation intervention
本身能改变成功率；但它与 lexical control 在这个 target family 上 admissions 完全相同，不能作为
non-trivial neural-symbolic selection 的证据。

## 各版本发生了什么

- V20：收集 147 个真实 matched forks，训练 target-native causal/utility heads；conformal
  selector 最终没有可用 admissions。
- V21：在新的 78 个 opportunities 上，primary selector 为 `3W/1L`、`p=0.3125`，失败；
  always/causal/lexical 为 `8W/3L`，方向为正但 `p=0.113`。
- V22：加入 source `BIND→MUTATE` 与 ALFWorld clean/heat families。170 个 outcome-blind
  calibration tasks 只产生 11 个 action contrasts（MUTATE 2，RELATE 9），在读取 outcome 前
  fail closed。这复现了“source skill 变多不等于 target 有可识别 action contrast”。
- V23：把 V21 的 causal-only secondary signal 作为新 hypothesis；200-task development 为
  `2W/1L`、整体 `+0.5pp`、`p=0.5`，未授权确认。
- V24：271 个已消费 forks 的 OOF neural risk 开发结果很好，但 sealed confirmation 为
  `0W/0L/7 ties`，正式拒绝。

## Bitter lesson

V24 学到的主要是“这个 late target state 本来容易成功”，不是：

```text
E[Y(source edge) - Y(target abstain) | pre-action state]
```

这两者在开发数据上相关，在 sealed split 上分离。下一 selector 必须直接使用 paired uplift / CATE
目标，并让 target domain 包含 lexical/always-source 会犯错的状态；继续调 late-step 阈值、增加
高层 skill 名称或在已消费 confirmation 上重训都不能修复这个问题。

## 审计 artifacts

- V24 candidate：`configs/real_source_relation_neural_risk_candidate_v24.json`
- Outcome-blind enumeration：
  `runs/real_source_relation_causal_v20/confirmation_v24_enumeration.json`
- Frozen confirmation plan：
  `configs/real_source_relation_neural_risk_confirmation_plan_v24.json`
- Authoritative outcome：
  `runs/real_source_relation_causal_v20/confirmation_v24_outcomes.json`

confirmation report SHA-256：
`10c0eaada6ca4998675b812b37abca2b04918b454d5ae3ad27e7e60afe75de14`。
该 200-task split 已消费，不得再次作为 confirmatory evidence。
