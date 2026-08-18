# AGQA 2.0 V1：game-induced symbolic program feasibility audit

## 结论

AGQA 2.0 **可以接入现有 unified neural-symbolic harness，但目前只通过了
program-family / typed-applicability mechanism gate，没有验证 video QA success transfer**。

本轮对官方 balanced test metadata 的 669,207 个 functional programs（1,814 个 video IDs）
做了零 provider-cost 流式审计。selector 只接收 `task_id` 和 public functional program；不读取
answer、`sg_grounding`、官方 reasoning labels、video frames、source game identity 或 target action。
原始视频和预计算视觉特征均未下载。

最终状态固定为：

`AGQA2_PROGRAM_TRANSFER_MECHANISM_FEASIBLE_NOT_SUCCESS_VALIDATED`

## 统一 IR 如何映射

这里没有重新手写 `EXPLORE/BACKTRACK/COMMIT` controller。三个候选 program 都来自既有
source intervention rollouts，并通过同一个匿名 structural type checker 选择：

| AGQA program obligation | Source-induced IR | 数量 | 结果 |
|---|---|---:|---|
| recurrent relation scan | Sokoban recurrent goal-relation update | 25,542 | exact match |
| recurrent pairwise temporal comparison | Candy Crush arity-2 temporal effect scorer | 15,936 | exact match |
| non-recurrent duration scoring（含 target-native `Equals` wrapper） | Columns arity-1 temporal effect scorer | 13,255 | exact match |
| temporal + relation composite | 没有 exact composite source IR | 595,013 | abstain |
| 其他 program family | 没有 exact source IR | 19,461 | abstain |

总计选择 54,733 / 669,207 = **8.18%**；其余 91.82% fail closed。这个低 coverage 是设计结果：
atomic source program 不允许对 composite target obligation 做 partial match。

`Equals(Superlative(..., Subtract(...)))` 与直接 `Superlative(...)` 共享同一个内部 arity-1
score obligation；`Equals` 是消费该 score 的 target-native wrapper，并没有引入第二个
source-IR obligation。因此两者都能精确绑定到 Columns contract，而不是用 question label 做
heuristic routing。

## Controls

1. **Wrong-source permutation：54,733 / 54,733 被拒绝。** relation → temporal-pair、
   temporal-pair → temporal-single、temporal-single → relation 的轮换全部因匿名类型不匹配而
   abstain。selector 不使用 game name。
2. **Composite abstention：595,013 / 595,013。** 同时需要 temporal 与 relation operator 的
   program 不会被 atomic source IR 部分覆盖。
3. **Target-written equivalent：3 / 3 仍会匹配。** 这是重要的失败控制：仅凭 extensionally
   identical structural contract，type checker 无法鉴别它来自 source rollouts 还是 target 人工
   编写。因此本实验不能单独证明 source provenance。
4. **既有 arcade source-specific gate 仍失败。** qualified authentic 与 source-permuted 都是
   22 / 27。AGQA audit 没有掩盖或“修复”这个负结果。
5. **成本为零。** 0 model calls、0 provider calls、0 new raw-video bytes、0 new visual-feature
   bytes；只下载了 142,198,618-byte 的官方 compressed balanced metadata。

## 为什么这仍然 non-trivial

这不是“所有游戏套一个 canonical controller”。三个 source programs 的 operator arity、recurrence
和 terminal predicate 不同，并预测了三个不同 AGQA functional subfamilies；错误轮换全部被
类型系统拒绝，大量 composite programs 也被明确拒绝。因此它验证的是：

> source-induced intervention structure 可以通过统一匿名 IR，选择性绑定到语义不同的 target
> program obligations。

但它仍未证明：

> game source knowledge 提高了 AGQA video answer success rate。

## 为什么现在不下载 AGQA 视频

以下 claim gates 全部未通过：

- arcade temporal portfolio 没有优于 source-permuted；
- structural contract 无法区分 source-induced 与 target-written equivalent；
- 没有 frame-only target-native neural grounder；
- 没有测量 answer accuracy；
- 当前 official test metadata 已用于 development audit，不再是 untouched evaluation split。

因此 `raw_video_advancement_authorized = false`。在这些问题解决前，下载 Charades/Action Genome
视频或调用 VLM 只会增加成本，不会修复 transfer attribution。

## 若未来继续 AGQA，正确的下一步

1. 先在新的 source-only held-out rollouts 上让 temporal authentic 显著优于 shuffled-effect 和
   source-permuted；不接触 AGQA。
2. 在单独的 AGQA development videos 上训练并冻结 frame-only target-native grounder，只输出
   `ENTITY_GOAL_RELATION` 或 `TEMPORAL_EFFECT_VECTOR` typed receipts，不读取 functional program
   或 answer。
3. 预留 video-disjoint untouched reserve，比较 neural-only、source-induced、source-permuted、
   target-written equivalent / generic scaffold 和 target-native ceiling。
4. 只有 applicability、negative-transfer、grounding qualification 与 provenance gates 预先通过，
   才允许 raw-video formal run。

## 可复现 artifacts

- Config: `configs/agqa2_program_transfer_v1_development.json`
- Compiler: `src/motif_transfer/agqa_program_transfer.py`
- Audit runner: `scripts/audit_agqa2_program_transfer_v1.py`
- Machine report: `docs/results/agqa2_program_transfer_v1.json`
- Unit tests: `tests/test_agqa_program_transfer.py`

运行：

```bash
pytest -q tests/test_agqa_program_transfer.py
python scripts/audit_agqa2_program_transfer_v1.py
```

数据使用声明：这是已消费的 metadata development audit，不是预注册 formal evaluation。
