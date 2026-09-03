# Source-only triage gate V15

日期：2026-08-15

## 结论

我们没有把 WebShop V14 中成功的 target-native backtracking/exploration 重新命名为
“game transfer”。V15 先冻结一个三分支候选控制器：

```text
INFEASIBLE                 -> BACKTRACK / REPLAN
FEASIBLE_AND_UNTRIED       -> EXPLORE
READY_AND_POSITIVE_EFFECT  -> COMMIT -> VERIFY
```

然后只读取 source artifacts 与真实游戏 matched forks，不读取 WebShop outcome。结果为：

```text
SOURCE_TRIAGE_GATE_V15_FAILED_CLOSED
STOP_BEFORE_TARGET_ADAPTER
provider calls = 0
formal reserve opened = false
```

这不是一次 target transfer 负结果，而是一次更早、更便宜的 **source identifiability failure**：现有
source evidence 只能支持二元的 `COMMIT/POSITION`，不能支持把 `POSITION` 进一步拆成两个由 source
决定的 `BACKTRACK` 与 `EXPLORE`。

## 三个分支逐项审计

| candidate branch | source evidence | gate |
|---|---|---:|
| `READY_AND_POSITIVE_EFFECT -> COMMIT -> VERIFY` | Sokoban fresh held-out 96 states；optimal labels 为 24 COMMIT / 72 POSITION；authentic accuracy .9896，最好 control .75 | PASS |
| `INFEASIBLE -> BACKTRACK/REPLAN` | effect program 写了 `EXPECTED_EFFECT_REFUTED -> REPLAN_OR_ABSTAIN`，但 confirmation 中该 action 的 intervention labels 为 0 | FAIL |
| `FEASIBLE_AND_UNTRIED -> EXPLORE` | 4 个真实 replayable GymV games、32 个 matched snapshot/mode cells；比较 `ALWAYS_SWITCH - ALWAYS_PERSIST` 的 h8 return | FAIL |

Sokoban topology executor 额外确认了 sequence recognition/refutation：62 个 fresh examples 上 canonical
executor 为 1.0，direction-permuted 与 sequence-length controls 为 0。但它的 receipt boundary 是
candidate recognition；没有比较执行 `BACKTRACK` 与其他 action 后的环境结果。因此它不能把第二行从
recognition rule 升级成 intervention-grounded procedural skill。

## EXPLORE 的 matched-intervention 结果

主 estimand 是 common continuation：在同一 fork state 只改变第一类 treatment，之后使用共同的
hash continuation。完整 full-treatment regime 作为 robustness diagnostic。

| split / estimand | cells | SWITCH win | tie | PERSIST win | mean Δ return |
|---|---:|---:|---:|---:|---:|
| qualification / common | 8 | 0 | 4 | 4 | -19.875 |
| held-out / common | 8 | 0 | 8 | 0 | 0.000 |
| qualification / full | 8 | 1 | 4 | 3 | -14.875 |
| held-out / full | 8 | 0 | 5 | 3 | -13.625 |

唯一一次 SWITCH win 出现在 qualification/full 的 Streets of Rage 2（+20）；它没有在 blind held-out
复现。主 common estimand 的 16 cells 没有一个 SWITCH win。旧 absolute event controller 同样已经被
matched smoke 拒绝；因此不能用 observational rollout 中“PROGRESS 后 switch rate 高”来补这个缺口。

## 为什么这比直接跑 WebShop 更重要

WebShop V14 的 `go_back()`、跳过 rejected ASIN、打开 untried ASIN 都确实提高了 live success，但这些
行为来自 target-native coverage ledger。若现在把它们写进所谓 source controller，实验条件会变成：

```text
target heuristic + source COMMIT gate
```

而不是：

```text
source-learned triage structure + target-native grounding
```

V14 的 0/2 → 2/2 仍是有价值的 target bridge result；V15 防止我们把这项工程收益错误归因于
cross-domain neural-symbolic transfer。

## 下一项被授权的 source experiment

下一步不是打开 WebShop formal，也不是继续增加 target tasks。需要新的
`RELATIVE_FEASIBILITY_AND_NOVELTY_FORKS_V16`：

1. 在 replayable game states 中建立显式 attempt/branch ledger；
2. 每个相同 state 做 `BACKTRACK`、`EXPLORE_UNTRIED`、`COMMIT` matched action forks；
3. 用相对 intervention effect，而不是 absolute reward/event label，定义最优 branch；
4. discovery 与 qualification 按 game/level 隔离，blind held-out 必须包含三个 action 的非零支持；
5. authentic routing 必须同时超过 always-backtrack、always-explore、always-commit、permuted binding；
6. 只有 source gate 通过后，才允许 target 用 native probes ground `INFEASIBLE/UNTRIED/READY`。

如果真实 source games 不能自然产生三个可比较的 action，就应缩小 claim，保留已经通过的
`COMMIT/POSITION/VERIFY` transfer，而不是创造一个高层三分支 ontology。

## Reproduce

```bash
PYTHONPATH=src python scripts/audit_source_triage_gate_v15.py
PYTHONPATH=src pytest -q tests/test_source_triage_gate_v15.py
```

Machine-readable receipt：
[`results/source_triage_gate_v15.json`](results/source_triage_gate_v15.json)。输入文件 path 与 SHA-256
都记录在 receipt 中；审计代码不 import WebShop runner/controller。
