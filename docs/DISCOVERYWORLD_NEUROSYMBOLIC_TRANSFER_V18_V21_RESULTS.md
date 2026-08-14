# DiscoveryWorld neural-symbolic transfer: V18--V21 adaptation

## Verdict

V21 passes the **consumed-adaptation mechanism gate**. Across four outcome-blind,
matched forks, authentic source transfer succeeds on 4/4, compared with 3/4 for
target-native myopic selection, 2/4 for the natural recorded continuation, 2/4
for commit availability, and 0/4 for both inverted-effect and always-position
controls. It has zero negative transfer and all audit checks pass.

This is the first DiscoveryWorld run in this line with a source-specific success
rescue. It is not yet held-out validation: V18--V21 use already consumed Easy
seeds2--4. Normal-difficulty instances remain unopened.

The exact record is
`docs/results/discoveryworld_v18_v21_adaptation_summary.json`.

## What changed, and what did not

The transferred source artifact did not change. It is still the
source-qualified Sokoban rule:

`positive commit effect witnessed -> COMMIT and verify; otherwise POSITION and recompute`.

Target-native neural modules bind the scientific hypothesis, object UUID,
required relation, commit action, and candidate native actions. Exact symbolic
predicates decide whether `DROP` or `PUT` has the required direct effect.

V21 adds only a shared target-native realization layer for spatial `POSITION`.
After a successful target-bound teleport, it enumerates every relative vector
consistent with DiscoveryWorld's public direction and Manhattan-distance facts.
It may replace a neural POSITION action with a currently valid cardinal move
only when that move strictly reduces worst-case error to the bound relation. It
cannot select COMMIT and is identical across all five arms.

## Bitter lessons from V18--V20

V18 found zero strict applicability forks within the original 32-step traces
after rejecting invalid neural bindings. V19 then showed that one target success
occurred only after step32 and separated three target failures: horizon cutoff,
exploration loops, and final spatial realization.

V20 extended acquisition to 64 steps and recovered four first-commit forks.
Authentic transfer was faster on two Proteomics cases and beat availability on
one, but tied target-native myopic at 3/4. On Proteomics seed4 the source guard
correctly prevented a premature flag drop, yet the target grounder cycled
`TELEPORT -> west -> TELEPORT`. This was safety without utility.

The lesson is not to transfer the failed game STALL controller or add another
prompt heuristic. The missing component was a target-native executor for an
already bound symbolic relation.

## V21 matched result

| Condition | Successes / 4 |
|---|---:|
| Natural target continuation | 2 |
| Target-native myopic | 3 |
| Authentic Sokoban effect + target grounding | **4** |
| Commit-availability control | 2 |
| Inverted-effect control | 0 |
| Always-position control | 0 |

The decisive fork is Proteomics Easy seed4:

- myopic teleports beside the correct Echojelly statue, drops the flag from the
  wrong side, and cannot recover;
- authentic sees that the exact effect predicate is false, selects POSITION,
  executes west then north through the shared relation realizer, observes the
  statue exactly east at distance one, and only then drops the flag successfully;
- availability commits immediately and fails;
- always-position reaches the relation but never switches to COMMIT and fails.

Thus neither the neural target binding, the spatial executor, availability, nor
generic conservatism alone explains the success. The source-qualified effect
guard supplies the missing POSITION-to-COMMIT switching structure.

## Integrity and claim boundary

All four result self-hashes, policy/audit fork hashes, source-selection receipts,
and spatial-realization receipts validate. There are no runtime errors and no
policy access to official scorecards. V21 seeds every task with the exact V20
neural cache and verifies its SHA-256 and backend identity. Every binder and
every grounding request through the first realized action divergence is a cache
hit; only genuinely new post-intervention states may call the target model.

The valid claim is therefore:

> On four consumed Easy first-commit forks from Space Sick and Proteomics, a
> source-qualified Sokoban positive-effect guard, instantiated by target-native
> neural binding and exact target symbols, causes one matched success rescue and
> no negative transfer.

It would be incorrect to call this held-out DiscoveryWorld generalization. The
next step is to freeze the full V21 candidate and evaluation protocol before
opening Space Sick/Proteomics Normal seeds2--4.
