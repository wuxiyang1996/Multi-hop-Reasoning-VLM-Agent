# Five-domain Harness SFT dataset V1

> **Role freeze (2026-08-30):** this artifact is retained byte-for-byte as a
> heuristic, target-adapted engineering baseline. It is not the primary evidence for
> target-data-free transfer. The immutable hashes are in
> `configs/target_harness_sft_five_domain_v1_baseline_lock.json`; the source-only
> scientific evaluation is documented in `docs/HARNESS_SFT_SCIENTIFIC_V4_UPDATE.md`.

## 结论

视频理解已经作为**一个第五 target domain**加入独立的 SFT lineage；现有 four-domain
artifact 没有被覆盖或改写。新数据位于
`runs/target_harness_sft_five_domain_v1/`，包含 6,000 条 examples / 3,000 个
matched authentic-control pairs：

| Target domain | Train authentic | Train control | Validation authentic | Validation control | Total |
|---|---:|---:|---:|---:|---:|
| WebShop | 450 | 450 | 150 | 150 | 1,200 |
| ALFWorld | 450 | 450 | 150 | 150 | 1,200 |
| DiscoveryWorld | 450 | 450 | 150 | 150 | 1,200 |
| TIRBench | 450 | 450 | 150 | 150 | 1,200 |
| Video | 450 | 450 | 150 | 150 | 1,200 |
| **Total** | **2,250** | **2,250** | **750** | **750** | **6,000** |

Video 是一个 domain，CLEVRER 与 AGQA2 是其中两个 benchmark。两者各贡献 225 个
train authentic、75 个 validation authentic，以及一一对应的 executor-derived control。
因此 video mixture 在 benchmark 层面精确平衡。

这次构建没有新 provider/API call，新增 API cost 为 `$0`。AGQA2 receipts 记录的
`$0.361693924` 是此前 development acquisition 的历史成本，不是本次构建成本。

## 神经符号接口

模型仍然只看到同一个匿名接口：

```text
source-only induced anonymous program
  + symbolic controller state
  + opaque candidates C0..Cn
  + four target-native neural effect probabilities
  -> EXECUTE_OPERATOR / ABSTAIN + next symbolic state
```

completion 仍由 frozen source symbolic executor 重新计算，不由视频答案、target reward
或 teacher LLM 自由生成。CLEVRER/AGQA2、视频 ID、source family、native relation/action、
question、answer、functional program、scene graph、raw frame 和 control identity 都只存在于
audit lineage，完全不在 prompt/completion 中。

因此当前 9B Harness 仍是**文本/IR controller**，不是 VLM：

```text
frames / neural event graphs
  -> target-native video grounder
  -> anonymous typed effects
  -> 9B Harness
  -> symbolic verifier
```

## CLEVRER supervision

CLEVRER 使用 792 条已消费 development proof receipts，以及冻结的五头 neural
proof-uplift grounder。每个 receipt 比较两种 target-native neural event-graph
representations；candidate identity 被匿名化，grounder 的 calibrated uplift 经冻结 logistic
adapter 变成候选 effect probability，两个 candidate-order variants 用来训练 permutation
equivariance。

这里必须明确披露：CLEVRER proof grounder 曾用 consumed development 的 success-uplift
labels 训练。因此 five-domain SFT 是 target-adapted deployment mixture，不是 zero-shot
transfer data。SFT controller labels 本身仍只来自 source program executor，且没有读取
formal/reserve outcome。四个 effect heads 当前保守地共享同一 calibrated uplift probability；
这能训练 video-grounded candidate selection，但不能单独证明视频中的四个真实时间尺度都已被
辨识。

## AGQA2 supervision

AGQA2 使用 V59 的 40 个已消费 official-train development videos。每个 runtime receipt 在
evaluator 打开 answer 之前已冻结，并逐行确认：

- runtime answer / functional program / scene graph / source identity 未读取；
- independent VLM view 的 observation confidence、coverage、temporal support 和 evidence
  persistence 被转换为四个 typed effects；
- 多视角 consensus 是额外 target-native candidate；
- 只枚举 2/3-candidate subsets 与 candidate-order reversal；
- 没有使用 qualification、formal 或 reserve rows。

40 个视频通过 candidate subset/order 和四个 qualified source programs 产生足够的 controller
supervision，但这些 augmentation 不是独立视频。AGQA2 的 600 SFT rows 不能被报告为 600 个
视频或 600 次独立 transfer trials。

## 独立审计

真实 cached Qwen3.5-9B tokenizer 审计全部通过：

- 6,000 unique model rows，4,500 train / 1,500 validation；
- 3,000 complete pairs，每个 control 都改变 frozen executor output；
- 每个 domain 保持 450/150 authentic train/validation quota；
- CLEVRER/AGQA2 各自保持 225/75 quota；
- video train/validation 在 video-group 层面完全 disjoint；
- 226 个 train video groups、75 个 validation video groups；
- 543 个 selected grounded-state receipts；
- prompt、target task、state receipt 均 train/validation disjoint；
- 无 domain/benchmark/native action/relation/question/answer/outcome leakage；
- token length：min 678、median 807、p95 1,414、p99 1,523、max 1,534；
- 0/6,000 超过 2,048-token contract；
- 第二次完整 build 的 structured/train/validation/video-provenance 四个文件逐字节相同。

关键 hashes：

- manifest: `9888a9e3a644ca05ccaca96ee55e15780267cab4667659e04169b72538d6a046`
- independent audit: `221d5392e5a9b60a1c596513f7969ab303e75a2adb334fa774342250caccc486`
- structured: `7368d407f1944ae749b283f76328c8c42109557739ca56a444ea6ecaea78d4be`
- train: `f5be7ab178928c225536ee6cecff1032da6b743234d54608b3db19990c4ba62c`
- validation: `30479b181b5dc68fcf4f76fbf3fd3a0337727f1b9444cec4af21580cf8d40f15`
- video provenance: `f202e558e210caf62387bf10476f1478fafc4f7dd53527bbe8328ae1ef51d1fb`

## 正确的训练顺序

不能立即用 five-domain 数据替代 OOD experiment。科学上应保持两条 lineage：

1. 用现有 four-domain V2 训练 adapter，video 仍 untouched；
2. 冻结该 adapter，在未进入 SFT 的 video reserve 上做 zero-shot controller evaluation；
3. 只有保存 zero-shot result 后，再用本 five-domain V1 训练 deployment adapter；
4. 比较 four-domain zero-shot 与 five-domain adapted，并同时跑 source-unseen regression，
   检查 catastrophic forgetting。

five-domain SFT 只能说明“9B 可以学习使用 video-native grounding 后的 transferable symbolic
program”。它本身不能证明 raw-video understanding、video MDP transfer 或 target success-rate
提升；这些结论必须来自未用于这里的数据和 frozen downstream evaluator。

## 复现

```bash
python scripts/build_target_harness_sft_five_domain_v1.py \
  --config configs/target_harness_sft_five_domain_v1.json \
  --output-dir runs/target_harness_sft_five_domain_v1

conda run -n cosplay-candy-a100 python scripts/audit_target_harness_sft.py \
  --dataset-dir runs/target_harness_sft_five_domain_v1 \
  --model Qwen/Qwen3.5-9B \
  --max-length 2048 \
  --output runs/target_harness_sft_five_domain_v1/independent_audit.json
```
