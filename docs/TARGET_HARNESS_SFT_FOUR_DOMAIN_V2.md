# Four-domain Harness SFT dataset V2

## Outcome

The Harness SFT mixture now covers all four registered non-video target domains:
WebShop, ALFWorld, DiscoveryWorld, and TIRBench. The frozen dataset contains
4,800 examples in 2,400 matched authentic/control pairs:

| Target domain | Train authentic | Train control | Validation authentic | Validation control | Total |
|---|---:|---:|---:|---:|---:|
| WebShop | 450 | 450 | 150 | 150 | 1,200 |
| ALFWorld | 450 | 450 | 150 | 150 | 1,200 |
| DiscoveryWorld | 450 | 450 | 150 | 150 | 1,200 |
| TIRBench | 450 | 450 | 150 | 150 | 1,200 |
| **Total** | **1,800** | **1,800** | **600** | **600** | **4,800** |

There are 3,600 train rows and 1,200 validation rows. The four qualified source
program families remain balanced to within 16 examples. Video is deliberately
absent and remains the fifth, out-of-distribution target domain.

No new provider/API call was made. All grounding and SFT construction reused
frozen local development/adaptation receipts, so new external inference cost is
zero.

## Unchanged neural-symbolic contract

Every prompt contains only an anonymous source-induced typed program, its symbolic
state, opaque candidate IDs, and target-native neural probabilities over the same
four effects:

- effect by transition 1;
- effect by transition 4;
- effect by transition 8; and
- executable-transition persistence.

Every completion is recomputed by the unchanged frozen source symbolic executor.
Target domain, source family, native action/tool, reward, official success, target
outcome, and control identity are audit metadata only and do not appear in the
model-visible prompt or completion.

The 2,400 matched controls contain 808 effect-binding permutations, 823 missing
effect schemas, and 769 tied argmax cases. There are 3,208 executable and 1,592
abstaining labels. All 2,400 authentic rows execute; controls are not merely an
abstention shortcut because some controls produce a changed executable binding.

## DiscoveryWorld grounding

DiscoveryWorld uses only the six consumed development tasks from the previously
frozen structured-acquisition protocol: seeds 45--48 for grounder training and
49--50 for task-disjoint validation. The episodes contribute 54 real successful
target transition tuples, zero schema fallbacks, zero invalid native actions, no
evaluator finalization, and no formal-outcome access.

A target-native balanced logistic grounder predicts the four typed effects over
three action options (`PICKUP`, `TELEPORT_TO_OBJECT`, and `USE`) from current facts,
option feasibility, step, and action history. Target success is not a label.
Validation results are:

| Effect head | Task-held-out AUC |
|---|---:|
| H1 | 0.6651 |
| H4 | 0.8849 |
| H8 | 1.0000 |
| persistence | 1.0000 |
| **Macro** | **0.8875** |

The gates were frozen before fitting at per-head AUC >= 0.55 and macro AUC >=
0.65. H1 is the weakest head and should remain a monitored risk in later OOD
evaluation; it was not hidden by the stronger longer-horizon heads.

The 54 base states yield candidate subsets of cardinality 2 and 3. V1 of the
four-domain build correctly stopped before output because only 429 unique train
pairs were executable, below the frozen quota of 450. V2 adds one reversed
anonymous candidate-ID ordering per subset. This is candidate-order equivariance
augmentation: it changes neither target transition nor neural score nor source
program, and labels are recomputed after rebinding. V2 then meets the quota.

## TIRBench grounding

TIR uses exactly the frozen V11 development receipts and V12 four-head target
grounder:

- 16 development-train tasks;
- 8 task-disjoint development-validation tasks;
- 0 qualification tasks read;
- 0 formal tasks read; and
- 0 source identity or formal outcome exposed to neural collection.

Each real receipt contains four target-native eight-step wrapper intervention
programs. The Harness data enumerates their real 2-, 3-, and 4-candidate subsets,
yielding 264 candidate-set variants. Native `zoom_region`/`extract_colors`
programs never enter the Harness prompt; only the frozen V12 typed probabilities
do.

## Independent audit

The independent V2 audit passes every gate:

- exact four-domain coverage and exact per-domain quotas;
- 4,800 unique example IDs and model rows;
- one authentic plus one changed control in every pair;
- train/validation target task, state receipt, and prompt disjointness;
- no model-visible domain, native action/tool, source, control, reward, or success
  identity;
- byte-identical structured, train, validation, DiscoveryWorld grounder, and
  supervision files after a second full build; and
- exact cached Qwen3.5-9B tokenizer compatibility.

Token statistics under the real Qwen3.5-9B tokenizer are: minimum 707, median
833, p95 1,452, p99 1,523, maximum 1,534, and 0/4,800 above the 2,048-token
contract.

Authoritative files are under `runs/target_harness_sft_four_domain_v2/`:

- `manifest.json`
- `independent_audit.json`
- `structured.jsonl`
- `train.jsonl`
- `validation.jsonl`
- `discoveryworld_typed_grounder.json`
- `discoveryworld_grounder_supervision.jsonl`

Key hashes:

- manifest: `c4186e07c0630e28d55c397ae3df7ce562962712e45f484d588df063ca52c1ea`
- independent audit: `d9a261e7b2e16bf6315aae6f9e61040d5ef4a5a13ea8408112ded92fe572a493`
- structured: `98456f54c4300c0cae0fe9bec3a6936c3b08ba3850bcfb68d05434da8162e4b0`
- train: `27dd7734c67599c5bee1f23c258fe8a10cfa13da4c58658330195d1c3338e992`
- validation: `a37075a4181461fcfa217d71de25f619aa4501179cdb502d5463eebaba11a536`
- DiscoveryWorld grounder: `8023d2e39f8324bab92d428c97c09a31dae23ebf5904e92fddd8afbb4f5ee09a`

## Claim boundary and next experiment

The 4,800 rows are SFT examples, not 4,800 independent target trajectories.
DiscoveryWorld has 54 base transitions and TIR has 24 base tasks; candidate subset
and order variants are valid controller augmentation but must not be counted as
evaluation sample size.

Training on all four domains will produce an operational multi-domain neural
Harness. It will not by itself prove weight-level cross-domain transfer. That
claim requires leave-one-domain-out adapters (three target domains for SFT, the
fourth held out) or the untouched fifth video domain. In every case, final target
success must be evaluated on frozen tasks not used here.

The guarded A6000 launcher is
`cluster/train_harness_controller_qwen35_9b_four_domain_v1_a6000.sbatch`. It is
prepared but not submitted. It warm-starts the verified source-only V3 adapter,
runs 450 optimizer steps (one 3,600-row pass at gradient accumulation 8), and then
checks exact execution separately in each of the four development domains plus
the frozen source-unseen regression set.
