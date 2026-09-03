# Real Game → ALFWorld Neural-Symbolic Transfer V1

日期：2026-08-10

## 结论

本轮没有获得真实 game → ALFWorld transfer 的正证据。实验在 source-native neural
grounding 闸门处按协议停止；没有执行 ALFWorld 四条件测试，也没有把 source action token、
Candy board feature 或 reward marginal 注入 target。

最终状态：`TRANSFER_BLOCKED_AT_SOURCE_GROUNDER`。

这不是“跨域 neural-symbolic transfer 不成立”的结论。它说明当前 Candy source receipts
虽然是真实、可复现、action-specific 的 intervention evidence，但从可观察 board/action
不能稳定预测 held-out intervention value，因此还没有形成可以交给 target-native grounder
使用的 source symbolic comparator。

## 为什么旧日志不能直接使用

首先冻结了历史 Candy evidence 中 24 个状态、每状态 8 个动作，共 192 个 forks。
当前 runtime 用 evidence 记录的 `requested_seed + prefix_actions` 重放时：

- `VALID = 0 / 192`；
- `FORK_STATE_MISMATCH = 192 / 192`；
- development、qualification、heldout 的 valid fraction 均为 0。

历史 manifest 只声称 seed 被传给 `env.reset`，没有验证 hidden state。当前 runtime 的重放
在自身内部是确定的，但与旧 evidence 的 observable-state hash 不同。因此旧日志可能存在
runtime/config drift，不能用于 state-matched counterfactual claim。

审计文件：

- `runs/real_game_to_alfworld_intervention_v1/source_plan.json`
- `runs/real_game_to_alfworld_intervention_v1/source_receipts.jsonl`
- `runs/real_game_to_alfworld_intervention_v1/source_gate.json`

## 新鲜同-runtime intervention collection

新 collection 不读取 reward 来选择 seed、snapshot、rollout action 或 fork action：

1. 预先固定 30 个新 seeds；
2. seed 按 namespace hash 分到 development / qualification / heldout，各 10 个；
3. 每 seed 用 hash policy 运行真实 Candy 环境；
4. 每 episode 按 hash 选 4 个状态；
5. 每状态冻结 rollout action 加 7 个 hash-ranked alternatives；
6. 每个 fork 从相同 seed 重放完整 prefix，并要求 observable-state SHA-256 完全一致。

结果：

| split | states | forks | valid | action-dependent states |
|---|---:|---:|---:|---:|
| development | 40 | 320 | 320 | 39 |
| qualification | 40 | 320 | 320 | 40 |
| heldout | 40 | 320 | 320 | 40 |

因此“真实 source 中存在可复现的 state-matched action effect”成立。这只是必要条件，不是
transfer evidence。

## Source-native neural grounding gate

训练输入只含 Candy-native 可观察 board/action feature；模型是固定的
`StandardScaler + MLPRegressor(16 tanh units, LBFGS)`。development 有 320 个 action
receipts。label 是每个 state 内归一化的 immediate reward。

Controls：

- authentic：正确 state-action-value correspondence；
- within-state shuffled：保留每个 state 的 reward multiset，但打乱 action binding；
- source marginal：删除 state-dependent correspondence，只输出 development mean。

预先固定的 gate 是：heldout authentic mean normalized regret 必须严格低于两个 controls。

| split | condition | normalized regret ↓ | top-action tie accuracy ↑ | within-state Pearson ↑ |
|---|---|---:|---:|---:|
| qualification | authentic | 0.639 | 0.100 | 0.079 |
| qualification | shuffled | 0.702 | 0.125 | -0.006 |
| qualification | marginal | 0.728 | 0.100 | N/A |
| heldout | authentic | 0.634 | 0.150 | 0.046 |
| heldout | shuffled | **0.584** | **0.225** | 0.019 |
| heldout | marginal | 0.701 | 0.075 | N/A |

Authentic 在 heldout 优于 marginal，但输给 within-state shuffled，且 within-state
correlation 接近零。因此 `SOURCE_GROUNDER_GATE_FAILED`。

## 对“是否 non-trivial”的回答

如果只迁移“选预测 value 最大的动作”，这是通用 argmax，不足以构成非平凡 skill transfer。
本方案要求 source 先证明 state-dependent action binding 可被 source-native grounder 恢复，
再只迁移 action-token-free comparator，由 ALFWorld-native grounder 重新绑定 target actions。
当前第一部分没有通过，所以强行跑 target 只能测到 target heuristic 或 reward marginal，不能
归因于 source neural-symbolic skill。

## 下一步边界

下一轮不应继续在这批 Candy 数据上调 MLP。更有效的选择是：

1. 使用具有可观察 intervention semantics 的 source task，例如 hidden-rule diagnosis，动作
   明确分成 `TEST(h)` 与 `COMMIT(h)`，test outcome 能更新 belief state；或
2. 在 Candy collection 中记录可重放 simulator/RNG state，并把 label 改成多步可观察
   structural progress，而不是受 cascade 隐变量影响的 immediate score；
3. source gate 通过后，再冻结 ALFWorld canonical task list，执行四个 paired conditions：
   target-only、authentic、within-state shuffled、source-marginal。

在 paired target success/progress 上 authentic 未同时胜过 controls 之前，不能声称真实
game → other-domain transfer。

## 复现

```bash
PYTHONPATH=src /fs/gamma-projects/vlm-robot/conda/envs/cosplay-candy-a100/bin/python \
  scripts/prepare_fresh_real_source_interventions.py \
  --config configs/real_game_to_alfworld_intervention_v1_expanded.json

PYTHONPATH=src /fs/gamma-projects/vlm-robot/conda/envs/cosplay-candy-a100/bin/python \
  scripts/run_real_source_interventions.py \
  --config configs/real_game_to_alfworld_intervention_v1_expanded.json

PYTHONPATH=src python scripts/analyze_real_source_grounder.py \
  --config configs/real_game_to_alfworld_intervention_v1_expanded.json

PYTHONPATH=src python scripts/finalize_real_transfer_gate.py \
  --config configs/real_game_to_alfworld_intervention_v1_expanded.json
```

核心 artifacts：

- `runs/real_game_to_alfworld_intervention_v1_expanded/source_plan.json`
- `runs/real_game_to_alfworld_intervention_v1_expanded/source_receipts.jsonl`
- `runs/real_game_to_alfworld_intervention_v1_expanded/source_gate.json`
- `runs/real_game_to_alfworld_intervention_v1_expanded/source_grounder_gate.json`
- `runs/real_game_to_alfworld_intervention_v1_expanded/transfer_gate.json`
