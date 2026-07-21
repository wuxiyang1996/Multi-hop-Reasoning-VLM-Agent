# ALFWorld one-shot skill transfer：2×4 L40S 实验

> **历史 v1 说明：** 本文记录的四条 admission 使用过人工
> `source effect → target operator` 假设，只能视为工程 smoke test，不能作为正式
> 跨域语义迁移结果。新的 Agent-proposed、target-native v2 协议见
> [`README_AGENT_NATIVE_SKILL_HARNESS.md`](README_AGENT_NATIVE_SKILL_HARNESS.md)。

本文是当前可执行 vertical slice 的冻结实验说明。它不把 game skill 名称、9B/35B
解释文本或旧 mega-skill cluster 当作语义证据；成功只读取 ALFWorld 官方 `won`。

## 当前结论边界

当前实现足以回答一个较窄但可证伪的问题：

> 只保留 source state delta 真正支持的 effect，并用一条固定 target train demo
> 做 exact operator/type/effect admission 后，剩余技能能否安全地帮助完整
> `pick_and_place` held-out task？

它还不能证明完整的通用 skill-program induction。当前 source program 是按精确
`(game, chosen_skill_id)` 编译的 observed-action program；precondition 与长程控制结构
仍不完整。因此结果必须报告为 v1 executable vertical slice，不能写成已经解决任意
跨域技能迁移。

## 冻结证据

- Source：5 个游戏、299 episodes、23,072 transitions。
- 56 个 exact-identity source programs；不做文本/embedding clustering。
- 所有 23,068 个进入 program 的 transition receipt 都从原 JSON 按 file/state/action
  hash 重放通过。
- Source effect 由环境 parser 从前后状态机械提取，模型 skill label 不构成证据。
- Game SFT 数据为每游戏 250 个精确执行步骤，共 1,250 步/adapter；只做行为克隆，
  不作为 admission proof。
- Target：ALFWorld `train`、task type 1、显式 TextWorld `env.seed(42)` 的第 0 条成功
  expert demo；没有 retry 或 best-of-N。
- Target gradient updates：0。

早期文件 `train_seed42_shot0.json` 只设置了 ALFWorld config，底层 TextWorld 实际仍用
默认 seed 1234。该协议错误在提交大实验前被发现；它不再参与任何冻结输入。有效 demo
是 `train_seed42_v2_shot0.json`，hash 为
`c92a05274bacf88286a05a1be48c8d6bd48da6285038be7f01b1682d014e7cd1`。

## Admission 结果

冻结 manifest：

```text
manifest-c92a...-fbf22197882400f3.json
```

| Source proposal | Target operator | Source typed effect | Verdict |
|---|---|---|---|
| Sokoban NAVIGATE | `GOTO` | `agent_location_changed` | `CONDITIONAL` |
| Sokoban SETUP | `OPEN` | `receptacle_opened` | `REJECTED` |
| Mario COLLECT | `TAKE` | `possession_acquired` | `REJECTED` |
| Sokoban EXECUTE | `MOVE_TO` | `movable_location_changed` | `CONDITIONAL` |

`COLLECT → TAKE` 被拒绝是刻意的安全结果：名为 COLLECT 的 Mario 轨迹没有可观察的
possession fact，viewport 中对象消失也不能被解释为获取物品。两个 conditional binding
只在目标命令精确存在于当步官方 admissible list 时可调用。

因此当前 Harness 缺少 `TAKE`，很可能无法完成完整 pick-and-place。正确行为是 abstain，
不是让 9B/35B 补出一个 TAKE skill。大实验仍保留这些条件，用来同时量化 task success
和安全 coverage，而不是事先删除负结果。

## 预注册评测矩阵

官方 task type 1 held-out pool 一共 59 个唯一任务：

- `eval_in_distribution` / valid_seen：35；
- `eval_out_of_distribution` / valid_unseen：24。

四个条件使用完全相同的 task order 与全局 index，合计 236 episodes：

| Condition | Actor | Frozen Harness | Target update |
|---|---|---:|---:|
| `base` | 9B base | 否 | 0 |
| `game_sft` | source-only action LoRA | 否 | 0 |
| `base_harness` | 9B base | 是 | 0 |
| `game_sft_harness` | source-only skill/action LoRA | 是 | 0 |

模型只能返回严格的 `SKILL: N` 或 `ACTION: N`。格式错误、编号越界、HTTP failure、
unsupported operator 和 scope 外状态全部记为 abstention；没有 substring parser、随机
action fallback 或 LLM judge。

主比较为 `game_sft_harness - game_sft`，并报告 paired bootstrap 95% CI 与 exact
McNemar test。由于 selective Harness 可能主动拒绝，success、abstention、invalid-output、
request latency 与官方 task coverage 必须一起报告。

## 2×4 L40S 布局

阶段 A 同时占满 8 张卡：

```text
node 0 / GPU 0-3  4× Qwen3.5-9B base rollout replicas
node 1 / GPU 0-1  source-only skill/action LoRA（每卡一个）
node 1 / GPU 2-3  Qwen3.5-35B-A3B TP=2 proposal audit
```

阶段 B 重启 node 0 的四个 server 并加载两张 LoRA，运行剩余两个条件。35B 只输出
closed-schema proposal audit，不能修改 frozen admission artifact 或 success verdict。
每阶段最多 16 个 CPU ALFWorld shard clients 并发请求四个 vLLM replicas，以增加
continuous batching 和 GPU utilization。

## 运行与审计

预注册配置：`configs/principled_alfworld_2x4_experiment.json`。

```bash
scripts/submit_principled_alfworld_2x4.sh
```

提交前可执行：

```bash
DRY_RUN=1 scripts/submit_principled_alfworld_2x4.sh
sbatch --test-only cluster/run_principled_alfworld_2x4.sbatch
```

每个 run 写入：

```text
runs/principled_alfworld_2x4_<jobid>/
├── experiment_spec.json
├── source_replay_receipts.jsonl
├── source_only_lora/
├── proposals_35b.json
├── eval/<condition>/<split>/shard_<0..3>.json
├── aggregate.json
├── aggregate.md
├── adapter_sha256.txt
└── slurm/nvidia-smi logs
```

Aggregator 只有在 4×2 cells 的全部 shard、59 个 paired unique task IDs、frozen input
hash 和 `target_gradient_updates=0` 全部通过时才写 `complete=true`。
