# Target-domain Harness SFT data pilot V2

> Superseded as the final training mixture by
> `TARGET_HARNESS_SFT_FOUR_DOMAIN_V2.md`. This two-domain artifact remains the
> immutable WebShop + ALFWorld parent lineage of the four-domain dataset.

## Outcome

The WebShop + ALFWorld target-development SFT dataset is frozen and independently
audited. It contains 2,400 prompt/completion examples grouped into 1,200 matched
pairs. No new API calls were made: the build reuses frozen local adaptation
rollouts and qualified target-native grounders.

This artifact is training data, not transfer evidence. Formal, qualification,
held-out, and reserve target tasks remain excluded.

## What each example teaches

The Harness input contains only:

1. one anonymous source-induced typed symbolic program;
2. its symbolic controller state;
3. opaque candidate IDs such as `C0` and target-native neural probabilities for
   the same four typed effects; and
4. no chain of thought.

The completion is exact JSON produced by the unchanged frozen source symbolic
executor. It either selects an operator/binding and applies a declared transition,
or abstains. Domain names, native actions, reward, official success, source-family
identity, and control identity do not enter the model prompt or completion.

The data therefore trains the 9B Harness to execute the existing transfer
interface. It does not train the Harness to imitate WebShop or ALFWorld actions.

## Composition

| Target | Split | Authentic neural grounding | Matched control | Total |
|---|---:|---:|---:|---:|
| WebShop | train | 450 | 450 | 900 |
| WebShop | validation | 150 | 150 | 300 |
| ALFWorld | train | 450 | 450 | 900 |
| ALFWorld | validation | 150 | 150 | 300 |
| **Total** |  | **1,200** | **1,200** | **2,400** |

The 1,200 controls comprise 419 permuted effect bindings, 389 missing-schema
cases, and 392 tied-argmax cases. The executor labels contain 1,619
`EXECUTE_OPERATOR` and 781 `ABSTAIN` decisions. Authentic rows are all executable;
controls include both safe abstention and changed executable bindings, so the
negative data are not an abstention-only shortcut.

Only four source programs passed source-only qualification and are used:
Candy Crush, Columns, Thunder Force III, and Tetris. Their example counts are
604, 604, 596, and 596. Streets of Rage 2 and Strider remain excluded because
their source-induced programs abstained at qualification.

## Neural grounding

ALFWorld uses the already-qualified V8 four-effect option grounder over 1,297
adaptation transitions from 48 official expert-receipt episodes.

WebShop uses a new four-head MLP grounder trained on six adaptation tasks and
validated on the two frozen adaptation-validation tasks. It sees 704 states. Its
validation AUCs are:

| Typed effect | AUC |
|---|---:|
| effect by transition 1 | 0.9047 |
| effect by transition 4 | 0.9038 |
| effect by transition 8 | 0.9177 |
| executable transition persistence | 0.9708 |
| **Macro** | **0.9243** |

The gates were frozen at per-head AUC >= 0.55 and macro AUC >= 0.65. The four
LBFGS heads reached the configured iteration ceiling, but the independently
repeated build produced byte-identical grounder, supervision, structured, train,
and validation files. The iteration warning is therefore recorded but is not an
uncontrolled source of dataset variation.

## V1 failure and V2 correction

The first frozen export passed its semantic gates but failed the exact
Qwen3.5-9B context audit: 682/2,400 sequences exceeded 2,048 tokens, with a maximum
of 2,213. It must not be used for training.

V2 preserves all candidates, programs, partitions, quotas, grounders, and executor
labelling. It changes only prompt serialization of neural probabilities from raw
floating-point precision to six decimal places. Under the actual cached
Qwen3.5-9B tokenizer, V2 has:

| Statistic | Tokens |
|---|---:|
| minimum | 754 |
| median | 1,053 |
| p95 | 1,509 |
| p99 | 1,528 |
| maximum | 1,534 |
| above 2,048 | 0 |

This avoids truncation while retaining all 12 ALFWorld candidate bindings.

## Independent gates

All independent gates pass:

- file hashes match the frozen manifest;
- 2,400 structured/model IDs are unique and correspond exactly;
- all 1,200 pairs contain one authentic row and one changed matched control;
- all four target/domain/split/kind quotas are exact;
- train and validation target tasks, state receipts, and prompts are disjoint;
- source identity, control identity, target identity, native action, reward, and
  success authority are absent from model-visible text; and
- every sequence fits the 2,048-token training contract.

The authoritative files are under `runs/target_harness_sft_pilot_v2/`:

- `manifest.json`
- `independent_audit.json`
- `structured.jsonl`
- `train.jsonl`
- `validation.jsonl`
- `webshop_typed_grounder.json`
- `webshop_grounder_supervision.jsonl`

Key content hashes are:

- structured: `80c4bbfdaaf57033e4d16fe2c3687b653cfb524baaa84090167f889953c12a24`
- train: `455bf7d1ce460a2f6b75334921f2e6d32bdebbc214fe21b6144b8a3c5443b093`
- validation: `2f9b2a55568a44bb6549886efb297870bee6a4962e67253d55d6ff65adae6cd5`
- WebShop grounder: `dc60a99cc25a25d6e49757d3360e33ea44b8e45d043ae9249b8ab8052643f86b`

## Prepared training step

`cluster/train_harness_controller_qwen35_9b_target_pilot_v1_a6000.sbatch` is
prepared but has not been submitted. It warm-starts the verified source-only V3
adapter, trains for 225 optimizer steps (one pass over 1,800 rows at gradient
accumulation 8) with learning rate `2e-5`, and refuses to launch unless the frozen
manifest and exact-tokenizer audit still match.

After training it runs two distinct diagnostics:

1. exact JSON execution on the target-development validation split; and
2. the frozen model-unseen source-family set to detect catastrophic forgetting.

Passing those diagnostics still does not establish the final claim. The next
scientific step is a preregistered comparison on untouched target tasks, including
source-only V3, target-SFT Harness, source-permuted control, generic scaffold, and
the frozen symbolic executor ceiling.
