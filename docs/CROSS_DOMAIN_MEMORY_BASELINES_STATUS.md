# Cross-domain memory baselines: state as of 2026-09-02

Comparing the receipt-grounded method against ExpeL, AWM, and ReasoningBank on
WebShop, ALFWorld, DiscoveryWorld, and TIRBench, with video-game episodes as the
shared source.

Status: machinery complete and tested end to end on real data. No formal result
has been produced. One design question is open and blocks the formal run.

## What is settled

**Every method reads one superset and binds to its hash.** `induce_memory_artifact`
records `source_superset_sha256`, identical across all three baselines, plus a
per-method `source_projection` naming exactly which episodes that method was
allowed to read and which were withheld. Verified on the real artifacts:
superset `fdf65312…` shared by all three, AWM's 12 eligible episodes a strict
subset of ExpeL's 22.

**Outcome routing follows each published algorithm.** AWM reads the positive
class only; ExpeL and ReasoningBank read both. `UNKNOWN` is withheld from every
method, so an unlabelled episode never becomes one method's failure. A method
left with nothing raises `InsufficientEligibleSourceError` rather than silently
producing an empty bank.

**Absent outcomes stay absent.** `official_success` is `bool | None` through the
library and exporter. All 36 real episodes carry no official predicate and
resolve to `UNKNOWN` at that stage; none is rewritten as failure.

**Memory reaches the target through the prompt and nothing else.** The target
policy still proposes every candidate and still picks the action.
`MemoryAugmentedDecisionBackend` wraps the `CompletionBackend` the runners
already call, so no rollout loop changed and the target-only arm is the same
loop unwrapped. Retrieval strips target rewards, and `assert_action_free` fails
closed if retrieved memory ever names an executable target action.

**Domains that call a model client directly are covered too.** TIRBench is
multimodal and legitimately needs the OpenAI SDK's image content blocks, which a
JSON-payload backend cannot carry. `retrieve_target_advisory` is transport
independent and is the same function the wrapper uses, so the two channels
cannot drift apart.

**ALFWorld has a Decision Agent now.** The validated structural runner has no
language model in the loop, so there was nothing for a prompt-injected baseline
to influence. `alfworld_llm_decision` selects an index into the environment's own
admissible commands; an invented or out-of-range action cannot be returned.

## The labelling problem, and how it was resolved

These six games are score-maximisation and survival games with **no win
condition**, so "did this episode succeed" has no answer.

The first attempt (`configs/cross_domain_source_outcome_v1.json`, now
**withdrawn**) asked an LLM judge exactly that question. It failed:

- the verdict matched the `terminated` flag alone in 33/36 episodes (92%);
- it returned 14 SUCCESS, 0 FAILURE, 22 UNKNOWN, leaving ExpeL no contrast and
  ReasoningBank no pitfalls;
- it was **anti-correlated with play quality**: tetris episodes that terminated
  had median score 128 against 220 for those that survived, yet every terminated
  episode was labelled SUCCESS. On `gymv_columns` it called an episode whose
  observation reads `Gameover: 3` with a near-minimum score of 60
  "successfully achieved the game's objective before game over."

The question was wrong, not the judge. What this study transfers is skill, not
victory. `v2` therefore labels **relative skill-demonstration quality**: each
episode is ranked against the same game's own cohort under the same policy, by
the game's own score, into terciles. This is not `reward > 0` — the comparison is
within-game and relative, and no fixed number crosses domains. The middle tercile
abstains, and a game whose scores tie so heavily that the quantiles collapse
abstains entirely.

Result on the 36 pilot episodes: 12 high, 10 low, 14 withheld, decorrelated from
`terminated` and now running in the correct direction (terminated gives 2 high /
6 low). `gymv_thunder_force_iii` abstains entirely. Zero API calls.

`label_semantics` in the config records that the positive class means
`HIGH_QUALITY_SKILL_DEMONSTRATION`, not task success, so no reader mistakes these
for success rates.

## Open question, blocking the formal run

**Most of the induced memory is game-specific despite an explicit instruction to
abstract.** All three induction prompts already say "domain-agnostic". The model
produced this anyway:

| method | items referencing game-specific vocabulary | names a game in the title |
|---|---|---|
| ExpeL | 17/21 (81%) | 3 |
| AWM | 8/13 (62%) | 4 |
| ReasoningBank | 40/79 (51%) | 1 |

Typical entries: "Tetris: Build a Flat Base with No Holes", "Avoid creating holes
when placing pieces". None of that helps buy a blue mug or put one in a
coffeemachine. Genuinely cross-domain items exist but are rare, e.g. "Leveraging
Valid Action Sets for Decision Making".

This decides what the experiment measures. If the baselines' banks are largely
tetromino tactics, they will score at roughly target-only and "the method wins"
becomes true but uninformative — a reviewer will say the baselines were handed a
source from which no success was possible. The options are to add a steelman arm
(induction told the target is a non-game domain, items naming a game rejected)
alongside the current naive arm, or to report the abstraction failure as itself
the finding with the `source_permuted` and `generic_scaffold` controls.

## Artifacts on disk

`runs/cross_domain_memory_pilot_v0_provisional/` — **pilot, not a formal result.**
Built from the 36 existing `phase1_*_long_v1` trajectories, not from the frozen
96-episode shared superset.

| file | note |
|---|---|
| `source_superset.json` | 36 episodes exported from existing evidence |
| `source_labelled_v2.json` | skill-quality labels; the three artifacts derive from this |
| `artifact_{expel,awm,reasoning_bank}.json` | 21 / 13 / 79 items |
| `source_labelled_v1_WITHDRAWN.json`, `judge_cache_v1_WITHDRAWN.json` | evidence for the withdrawal above; do not use |

## Not yet done

- The 96-episode shared collection. Seeds frozen at 920001–925016 in
  `configs/cross_domain_shared_source_v1.json`, verified disjoint from every
  observed family (7301–7801, 91001, 92001, and the 101001–406024 fork families).
  Smoke job `7431973` was queued and had not started.
- Counterfactual forks on those same episodes, without which the proposed method
  and the baselines do not yet share one superset.
- Any live target-domain run. The WebShop runner exists and its CLI and imports
  are verified, but it has never executed against the live wrapper environment.
- DiscoveryWorld role wiring, and the same-domain sanity arm that would show the
  baseline implementations are faithful on their native benchmarks.

## Operational notes

- API keys live in `/fs/gamma-projects/vlm-robot/keys.py`, which is outside any
  git repository. `OPENROUTER_API_KEY` drives labelling, induction, and the
  target runs.
- Source collection itself needs no key; it serves Qwen3.5-9B locally through
  vLLM.
- `cluster/collect_phase1_complete.sbatch` is the only pre-existing tracked file
  this workstream modified. `SEED_BASE_ROOT`/`SEED_BASE_STRIDE`, `CHUNK_SIZE`,
  and `ALLOW_RESUME` were added; every default reproduces the original behaviour.
- `CHUNK_SIZE` splits a game across array tasks so a lost task costs 16 minutes
  instead of 64. Measured on the existing runs: vLLM startup is 1–2 minutes
  against roughly 4 minutes per episode, so 4-episode chunks add about 8% GPU
  time. The upstream writer is already atomic per invocation, so no patch to it
  was needed.
