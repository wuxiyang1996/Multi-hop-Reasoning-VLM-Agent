# Target rollout collection protocol

## Purpose

Target rollouts are collected to measure whether game-trained Harness weights or an explicit
source motif reduce adaptation cost. They are not allowed to influence source motif discovery.

The frozen manifest is `configs/target_rollout_collection_v1.json`. For both
ALFWorld-valid-unseen and VisualToolBench-single-turn it contains:

- 8 adaptation tasks;
- 8 qualification tasks;
- 24 held-out tasks.

Selection uses only a salted hash of benchmark-native IDs. All tasks previously used by
infrastructure smoke are excluded before ranking.

## Data authority

Adaptation rollouts may be used to:

- instantiate a source binding hypothesis;
- propose and audit target-native execution motifs;
- train an optional lightweight target Harness LoRA;
- estimate examples-to-success for `k=0/1/2/4/8`.

Qualification can select among candidates frozen from adaptation, but cannot update weights.
Held-out is not executed until Harness weights, target motifs, source treatments, prompts and
budgets have all been frozen.

Every rollout must record the full observation, native actions/tools, Decision proposals,
selection, post-transition assessment, advisory decision, real transition receipt, official
outcome, replan/abstain and termination.

## Current execution

ALFWorld target-only adaptation collection ran as Slurm array `7122411`. It used eight new official
train tasks with Qwen3.5-35B-A3B as the Decision Agent, 30 environment steps maximum, one exact
request cache per task, and no Harness or source motif.

The first task-2 attempt stopped after 22 valid transitions because a 256-token JSON response was
truncated. That artifact and cache are retained with the suffix `failed_schema_256tokens`. The
collector was fixed to reserve 512 output tokens and task 2 alone was rerun as `7122461`. The rerun
then produced a genuine Decision failure after 8 valid transitions: it selected an out-of-range
native action. This negative example was retained rather than repeatedly sampled until success.

The final mechanical audit accepted all eight rollout lineages and 152 transition receipts:

- official success: 4/8;
- mean steps: 19.0;
- one fail-closed invalid Decision output;
- qualification and held-out tasks remain unexecuted.

The audit is `docs/results/alfworld_target_adaptation_v1.json`.

## GPT-5-mini Harness oracle diagnosis

Before training a smaller Harness LoRA, GPT-5-mini was tested as an adaptation-only motif proposer.
It received all eight ALFWorld adaptation traces but no qualification or held-out data and had no
action authority.

The first proposal suggested `search -> take -> transport -> recover`, but hallucinated offsets up
to 29 for a four-step episode and assigned one span to multiple nodes. It was rejected.

A protocol-level correction alpha-renamed long episode IDs to `E0...E7`, included exact
`record_count`, and explicitly required one-node-per-span. The second proposal removed all
out-of-range references and proposed a five-node search/observe/move/take/deliver graph. It was
still rejected because spans overlapped, one span was assigned to two nodes, and claimed edges did
not recur across at least two episodes.

These are fail-closed results, not evidence that the motif works. The second graph also mostly
describes ALFWorld action phases rather than a demonstrated transferable reasoning backbone.
Results are:

- `docs/results/alfworld_gpt5mini_target_native_motif_v1.json`;
- `docs/results/alfworld_gpt5mini_target_native_motif_v2.json`.

LoRA job `7122386` is therefore held before GPU allocation. It can be released later as a matched
training ablation, but is not needed to test whether the Harness mechanism is feasible.

VisualToolBench IDs are frozen but not executed. The pinned official runtime currently lacks
`SERP_API_KEY` and `OPENWEATHER_API_KEY`; full-tool preflight therefore fails. We do not run those
items in degraded mode and later describe them as official VTB results.

## Frozen comparisons after adaptation

For the same target IDs, seeds, Decision model and budget:

1. base Decision, no Harness;
2. base Harness with target-native discovery;
3. game-receipt-trained Harness with target-native discovery;
4. base Harness with authentic source motif;
5. game-trained Harness with authentic source motif;
6. game-trained Harness with shuffled source topology.

Primary outcomes are official success/score, examples-to-success AUC, environment/tool steps,
invalid or repeated actions, token/tool cost and negative-transfer recovery latency.
