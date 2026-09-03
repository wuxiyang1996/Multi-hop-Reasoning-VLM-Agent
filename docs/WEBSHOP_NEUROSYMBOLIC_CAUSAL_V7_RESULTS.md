# WebShop neural-symbolic causal transfer V7

## Decision

The current repeat-gated transfer policy is rejected. On all eight consumed
reserve tasks it did not improve pass success, reduced strict success from
`1/8` to `0/8`, and reduced mean official reward from `0.573` to `0.531`.
The untouched held-out partition was not read or run.

The experiment does retain one narrower signal: at the four reconstructed
intervention states, authentic source values outperformed the matched
target-native minimum-repeat heuristic by `+0.083` mean horizon reward. They
still underperformed target rank zero by `-0.292` and produced no strict
horizon successes. Game-derived symbolic values therefore contain some
non-trivial action preference, but the current target applicability gate is
wrong.

## What V7 fixes

V7 closes three confounds found in the earlier WebShop runs.

1. WebShop sampled goal details before seeding its random generator. The same
   task ID could therefore acquire a different price constraint after a server
   restart. The server now seeds before goal construction; a 24-task consumed
   goal manifest was frozen and verified exactly across a restart.
2. Every condition now uses an independent namespaced WebShop session while
   session identity is canonicalized before Decision prompting. This avoids
   shared environment state without changing the common-randomness prompt.
3. Decision uses 3,200 output tokens and up to three schema retries. All five
   conditions completed on all eight tasks with zero infrastructure failures.

Every episode checks its reset instruction against the frozen manifest. Every
causal fork replays the target-only action prefix in a fresh session and checks
that the reconstructed pre-action state hash exactly matches the original
receipt.

## Matched episode results

The tasks were `webshop.1`, `webshop.6`, `webshop.13`, `webshop.24`,
`webshop.28`, `webshop.31`, `webshop.34`, and `webshop.49`.

| condition | strict | pass | mean reward | mean steps | interventions |
|---|---:|---:|---:|---:|---:|
| target only | 1/8 | 6/8 | **0.573** | 7.00 | 0/56 |
| neural minimum-repeat | 0/8 | 6/8 | 0.531 | 6.75 | 4/54 |
| authentic game source | 0/8 | 6/8 | 0.531 | **6.25** | 3/50 |
| phase-permuted source | 1/8 | 6/8 | **0.573** | 7.00 | 0/56 |
| Candy/Mario source | 0/8 | 6/8 | 0.531 | 6.50 | 4/52 |

The authentic source exactly tied the neural-only intervention baseline on
pass success and mean reward. Its shorter trajectories are not evidence of
better transfer: some interventions simply committed to a lower-reward item
earlier.

## Intervention-level causal forks

Four states contained a disagreement between target rank zero and at least one
intervention condition. Each unique first action was executed from the exact
same reconstructed state, followed by target rank zero for a five-action
horizon. All 12 branch reconstructions matched their source state hash and all
completed without infrastructure failure.

| task, step | branch | first action | horizon reward | branch steps |
|---|---|---|---:|---:|
| 1, 6 | target / phase | `click('44')` | **1.000** | 5 |
| 1, 6 | neural minimum | `click('59')` | 0.000 | 5 |
| 1, 6 | authentic / other game | `click('80')` | 0.667 | 1 |
| 1, 7 | target / phase | `click('44')` | **1.000** | 5 |
| 1, 7 | neural minimum | `click('59')` | **1.000** | 5 |
| 1, 7 | authentic / other game | `click('80')` | 0.667 | 1 |
| 28, 5 | target / phase | `click('49')` | **0.750** | 3 |
| 28, 5 | neural / authentic / other | `click('82')` | 0.500 | 1 |
| 49, 6 | target / phase | `click('32')` | **1.000** | 3 |
| 49, 6 | neural minimum | `click('68')` | 0.750 | 1 |
| 49, 6 | authentic | `scroll(0, 100)` | 0.750 | 3 |
| 49, 6 | other game | `click('61')` | 0.750 | 3 |

Aggregate fork outcomes:

| condition | strict horizon | pass horizon | mean horizon reward | mean steps |
|---|---:|---:|---:|---:|
| target only | 3/4 | 4/4 | **0.938** | 4.0 |
| neural minimum-repeat | 1/4 | 3/4 | 0.563 | 3.0 |
| authentic game source | 0/4 | 4/4 | 0.646 | 1.5 |
| phase-permuted source | 3/4 | 4/4 | **0.938** | 4.0 |
| Candy/Mario source | 0/4 | 4/4 | 0.646 | 1.5 |

## Mechanistic diagnosis

The repeat detector conflates two different situations:

- unproductive action cycling, where a recovery intervention may help; and
- repeated selection of required product constraints, where changing course
  prevents the target from completing the request.

Task 28 is the clearest correction to the V6 interpretation. The intervention
finished two steps earlier, but its reward was `0.50` rather than the target
branch's `0.75`: this was premature purchase, not recovery efficiency. On task
1, the authentic source sometimes chose a better recovery than the neural-only
minimum-repeat rule, but both were worse than preserving target rank zero.

This distinguishes representation from applicability. Anonymous game-derived
option values can change target choices in a source-specific way, so the
transfer is not merely a high-level language hint. But a repeat-only gate lacks
the target-native state needed to decide whether those values apply.

## Required next model

Do not add more source games to the current gate. The next experiment should
learn or enforce a target-native applicability shield before source reranking:

1. ground whether every required WebShop constraint has been selected;
2. distinguish a no-state-change repeat from a necessary repeated constraint
   action;
3. estimate intervention utility from matched forks; and
4. admit a source action only when its conservative predicted advantage over
   target rank zero is positive.

The already consumed fork states can train or calibrate this shield. A valid V8
test must compare it against target-only and the neural-only intervention rule,
preserve strict success on consumed tasks, and beat target rank zero on new
matched intervention forks before any held-out execution.

## Artifacts

- Frozen consumed goals: `configs/webshop_consumed_goals_v7.json`
- Protocol: `configs/webshop_neurosymbolic_causal_v7.json`
- Episode receipts: `runs/webshop_neurosymbolic_causal_v7/`
- Causal forks: `runs/webshop_neurosymbolic_causal_v7/intervention_forks/fork_report.json`
- Authoritative aggregate: `runs/webshop_neurosymbolic_causal_v7/causal_report.json`

The aggregate status is `REJECT_CURRENT_GATE`. It records one runner hash for
all episode receipts, one frozen-goal manifest hash, exact fork reconstruction,
zero episode/fork infrastructure failures, and
`held_out_read_or_run: false`.
