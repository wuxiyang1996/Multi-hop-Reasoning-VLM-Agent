# Qwen3.5-9B Neural-Symbolic Harness Controller V2–V3

## Question

Can a 9B model learn the anonymous typed symbolic controller contract from source-game
evidence, instead of using a handwritten target controller or merely acting as the target
Decision Agent?

This experiment trains only the Harness controller. The model receives an anonymous typed
program, symbolic controller state, candidate effects, and evidence bindings. It must emit one
exact JSON decision:

```text
EXECUTE_OPERATOR / ADVANCE_STATE / ABSTAIN / TERMINATE
```

The target-native neural grounder and official target evaluator remain separate. No WebShop,
ALFWorld, DiscoveryWorld, TIR, or video labels are used in this SFT stage.

## Data and leakage repairs

The first V2 build was rejected before it became authoritative because `control_variant` was
visible in the prompt. A running job was cancelled, the field was moved to audit-only metadata,
and prompt/completion duplicates were removed.

The authoritative V2 model dataset contains 658 unique examples:

```text
train / validation / Thunder held-out = 532 / 64 / 62
```

All frozen data gates pass: no source identity, native action, reward, success, target action
authority, split overlap, ambiguous prompt, or duplicate model pair is exposed.

V3 adds source-only, executor-derived equivariance supervision for the only V2 failure family:

- exact candidate-ID alpha renaming;
- dominated zero-effect candidate extensions at several lengths;
- labels recomputed by the same frozen symbolic executor;
- no target examples and no hand-labelled target action.

The V3 model dataset contains 847 train and 64 validation examples. Before V3 training, 21
prompts were frozen from retrospective source reserves after removing every prompt present in the
V2 held-out set and V3 train/validation sets. The overlap is exactly zero. This is a
fresh-to-model source-family replication, not a new prospective target-transfer result.

## Training

V2 trained from Qwen3.5-9B for 240 steps on one L40S. V3 continued from the V2 adapter for 120
steps on one 48 GB RTX A6000:

```text
train examples       847
validation examples   64
effective epochs     1.132
learning rate        5e-5
LoRA rank / alpha    16 / 32
train runtime        2,016.8 s
final eval loss      0.0007057
Slurm wall time      38m19s, including staging and exact generation
peak CPU RSS         42,168,308 KiB
```

All 12 required Qwen3.5 projection families have LoRA coverage. There was no OOM, NaN,
supervision truncation, or invalid adapter save. This establishes that a single A6000 is adequate
for 9B Harness LoRA idea-level SFT and short continued training; its limitation here is throughput,
not capacity.

## Frozen result

The exact-generation evaluator uses greedy decoding and compares the complete JSON object,
including decision, reason, operator binding, and next symbolic state.

| Regime | Valid JSON | Decision | Exact JSON | Rows |
|---|---:|---:|---:|---:|
| Base Qwen3.5-9B | 19.0% | 0.0% | 0.0% | 21 |
| V3 Harness LoRA | **100%** | **100%** | **100%** | 21 |

Per-decision V3 exact accuracy is also 100%:

```text
ABSTAIN 11/11 · EXECUTE_OPERATOR 4/4 · ADVANCE_STATE 3/3 · TERMINATE 3/3
```

All five preregistered gates pass: valid JSON at least 98%, decision at least 95%, exact output at
least 90%, non-abstain recall at least 90%, and an exact-accuracy margin over base of at least 20
points.

The previously consumed 62-row V2 held-out set is reported only as a regression diagnostic. V3
improves exact accuracy from V2's 55/62 (88.7%) to 60/62 (96.8%). The two remaining errors are the
same retry-ledger family: on a seven-candidate input the model repeats attempted `C0` instead of
selecting `C1`. We do not add a seven-candidate special case after inspecting this consumed set.

## What this proves—and does not prove

This result supports the following narrower claim:

> A Qwen3.5-9B LoRA can learn the source-induced, anonymous typed symbolic execution and
> abstention contract, generalize it to model-unseen source-family structures, and strongly
> outperform the base model.

It does **not** by itself show that the learned 9B controller improves official success in a new
target domain. Scientific V4 now freezes this adapter and evaluates it without target weight
updates on 1,500 target-grounded IR rows spanning WebShop, ALFWorld, DiscoveryWorld, TIRBench,
CLEVRER, and AGQA2. See `docs/HARNESS_SFT_SCIENTIFIC_V4_UPDATE.md`. That controller test remains
separate from official downstream success.

The next causal contrast is:

```text
same target Decision Agent + same target-native grounder + same budgets
  BASE neural-only controller
  V3 source-induced Harness controller
  source-permuted controller
  generic scaffold
  target-native ceiling
```

SFT is sufficient for the exact executor contract. OPD is the preferred next training stage for
learning whether to accept or reject a transferable program from target rollouts. GRPO should be a
later online-reward ablation, not the first replacement for exact SFT.

## Artifacts

- V3 frozen eval manifest: `runs/harness_controller_sft_v3/model_unseen_eval.manifest.json`
- V3 adapter: `runs/harness_controller_qwen35_9b_v3/continued_sft_v1/adapter/`
- training receipt: `runs/harness_controller_qwen35_9b_v3/continued_sft_v1/training_receipt.json`
- frozen report: `runs/harness_controller_qwen35_9b_v3/continued_sft_v1/model_unseen_source_family_report.json`
- consumed diagnostic: `runs/harness_controller_qwen35_9b_v3/continued_sft_v1/v2_consumed_diagnostic_report.json`
- training launcher: `cluster/train_harness_controller_qwen35_9b_v3_a6000.sbatch`

Slurm jobs: V2 training `7382528`, V3 A6000 training/evaluation `7383443`, and consumed-set
diagnostic `7383942`.
