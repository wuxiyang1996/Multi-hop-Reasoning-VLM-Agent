# Neural-Symbolic Harness Scientific V4 Update

## 结论先行

对“当前 SFT 是否太 heuristic”的回答是：**five-domain V1 的 target-grounding 部分确实有
明显启发式成分，但整个项目并不需要推倒重来。** 本次更新把两个角色彻底分开：

1. `target_harness_sft_five_domain_v1` 原样保留，冻结为 target-adapted engineering / deployment
   baseline；
2. 已训练的 Qwen3.5-9B V3 adapter 作为 scientific controller，因为它的 V2 初始训练和 V3
   continued SFT 都只使用 source-game supervision；
3. scientific V4 不做任何 target SFT，而是让冻结的 V3 adapter 零样本执行 target-native
   grounder 产生的匿名 symbolic IR；
4. video 的非启发式 V2 grounder 在真实 intervention ledger 到位前 fail closed，不能把已有
   static QA receipts 重新命名为 intervention data。

因此现在检验的是一个清楚的 claim：

> source-game intervention 学出的 program execution 能否被 9B controller 学会，并在不更新
> 权重、不暴露 domain identity/native action 的情况下，执行来自新 domain 的同一个匿名 typed
> IR？

它不是“在五域数据上训练后再验证五域”，也不把 controller execution 等同于 official task
success。

## 为什么旧 V1 仍然保留

V1 的 6,000 条数据、3,000 个 authentic/control matched pairs、训练/验证划分和 exact tokenizer
审计都是真实且可复现的。它适合回答工程问题：如果允许 target development grounding，9B
Harness 能否被 target-adapted，是否比 source-only 更稳。

但它不能承担最强 transfer claim，原因已写进 immutable lock：

| Domain | V1 中的主要启发式/适配 |
|---|---|
| CLEVRER | learned scalar uplift 被 broadcast 到四个 effect heads |
| AGQA2 | multiview evidence 通过手写组合公式变成 typed effects |
| ALFWorld | persistence 使用 exponent-64 transform |
| TIRBench | symbolic program 与 geometry effect feature 是 engineered |
| 全部 target | target development states 出现在 SFT 中，因此训练后的模型不再 zero-shot |

锁文件：`configs/target_harness_sft_five_domain_v1_baseline_lock.json`。它固定 config、builder、
audit、manifest、structured/train/validation/video provenance 的 SHA256。任何改变必须创建新版本，
不得覆盖 V1。

## Scientific controller 的完整 lineage

冻结模型仍是：

```text
Qwen/Qwen3.5-9B
  -> source-only V2 LoRA
  -> source-only V3 continued LoRA
  -> frozen target-IR zero-shot evaluation
```

两阶段 training receipts 都记录 `target_data_used=false`。V3 的冻结 adapter hash 为：

```text
7e66bfec63787141898642fe85134857c53682742cdd10a536f5eed61635b6a1
```

V3 已在 model-unseen Thunder source family 上通过 exact executor gate：21/21 exact JSON，base
为 0/21。这说明 9B 学到的不只是一个输出格式，也能处理 source family 未见的 binding、state
transition、retry ledger 和 abstention。

完整 protocol 位于 `configs/harness_controller_scientific_v4.json`。它同时固定 V2 初始 adapter、
V2/V3 source datasets、V3 adapter、training receipts 和 source-held-out report，避免只审计最后
一次 continued SFT 而漏掉初始权重 lineage。

## 五域零样本 controller evaluation

V4 从冻结的 five-domain validation 中选取全部 1,500 行，**不读取 target train split，不做梯度
更新**。Video 作为一个 domain，但按 benchmark 单独报表，因此共有六个 gate group：

| Evaluation group | Rows | ABSTAIN | EXECUTE_OPERATOR |
|---|---:|---:|---:|
| WebShop | 300 | 100 | 200 |
| ALFWorld | 300 | 97 | 203 |
| DiscoveryWorld | 300 | 110 | 190 |
| TIRBench | 300 | 101 | 199 |
| CLEVRER | 150 | 52 | 98 |
| AGQA2 | 150 | 39 | 111 |
| **Total** | **1,500** | **499** | **1,001** |

Freeze audit 已通过：

- V2、V3 target weight-update examples 均为 0；
- target prompt 与 V2/V3 source train/validation/held-out prompts 交集为 0；
- target example IDs 与全部 source IDs 交集为 0；
- source 与 target 使用完全相同的 anonymous controller instruction schema；
- prompt 中没有 domain/benchmark/source-family identity；
- 全部六组都同时含 EXECUTE 与 ABSTAIN；
- authentic grounding 与三类 matched controls 都存在；
- 1,500/1,500 行全量进入评测，不做结果驱动采样。

冻结 eval artifact：

```text
runs/harness_controller_scientific_v4_zero_shot/zero_shot_eval.jsonl
SHA256 82a4c04b048501b0596e07e5ff6f82f22f0d62ad7a1573a92e4c38ef15acc65c
```

预注册 gate：overall valid JSON ≥98%、decision ≥95%、exact symbolic output ≥90%、相对 base
exact gain ≥20 points；每个 target group decision ≥90%、exact ≥80%；每个 control variant
decision ≥90%。Evaluator 同时报告完整 binding、reason 和 next-state exactness，不能只靠猜
EXECUTE/ABSTAIN 通过。

A6000 launcher：
`cluster/evaluate_harness_controller_qwen35_9b_scientific_v4_zero_shot_a6000.sbatch`。

## Video V2：为什么当前必须停在 readiness

现有 CLEVRER 792 rows 和 AGQA2 40 rows 是有价值的 neural grounding / QA development receipts，
但它们没有同时记录以下结构：

```text
belief_state_before
  -> explicit intervention
  -> observations/effects at horizons 1, 4, 8
  -> executability persistence
  -> belief_state_after / transition receipt
```

所以它们不能无损支持“从 intervention 学到 H1/H4/H8/persistence grounder”的 claim。本次新增
`video-intervention-ledger-v2` contract，要求每条记录具有真实 state/action/effect/next-state、
blindness receipt、measured belief delta、四个 typed effects 和三档 horizon evidence；读取 gold、
formal success、official scene graph、functional program 或 source identity 都会拒绝该记录。

Readiness audit 的诚实结果：

| Benchmark | Existing rows scanned | Eligible intervention tuples |
|---|---:|---:|
| CLEVRER | 792 | 0 |
| AGQA2 | 40 | 0 |

状态为 `BLOCKED_NEEDS_INTERVENTION_LEDGER_COLLECTION`。这只阻止 video V2 grounder induction，
不阻止 source-only controller 的五域 IR execution evaluation。配置、schema 和审计报告分别位于：

- `configs/video_intervention_grounder_v2.json`
- `src/motif_transfer/video_intervention_grounder_v2.py`
- `runs/video_intervention_grounder_v2_readiness/report.json`

## 如何解释接下来的结果

如果 V4 gate 通过，可以说：

> 9B source-only neural controller 在六个 target evaluation groups 上零样本执行同一套
> source-induced anonymous symbolic IR，并对 matched symbolic controls 保持准确。

仍然不能仅凭该结果说：

- official target success 一定提高；
- target grounder 是因果正确的；
- video 四个 effect heads 已从 intervention 非启发式学出；
- five-domain V1 target-adapted SFT 是 cross-domain zero-shot transfer。

如果 V4 gate 失败，合法操作是按 domain/control/field 分解错误，修通用 controller training 或
schema compatibility，然后重新冻结 source-only model；不能把 target validation 加回 SFT 来让同一个
zero-shot test 通过。
