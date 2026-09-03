# Four-domain neural-symbolic release V1

## Result

This release separates two reproducibility levels:

1. **Portable evidence audit.** A clean checkout can validate all four exact routes, every compact receipt and adapter hash, positive dispatch, negative abstention, and target-native action authority using Python 3.10+ and no third-party packages.
2. **ALFWorld target replay.** The 1.7 MB frozen candidate, 2.2 MB development report, and 7.0 MB final report are committed as deterministic gzip artifacts. A materializer verifies both compressed and uncompressed hashes and replaces only machine-local ALFWorld resource paths.

The release manifest is `configs/four_domain_neurosymbolic_release_v1.json`. Its audit status is:

```text
PORTABLE_FOUR_DOMAIN_AUDIT_AND_ALFWORLD_BUNDLE_VALIDATED
```

## Portable audit

```bash
PYTHONPATH=src:. python scripts/audit_four_domain_release_v1.py \
  --manifest configs/four_domain_neurosymbolic_release_v1.json \
  --output /tmp/four_domain_release_audit.json
```

The committed audit receipt is `docs/results/four_domain_neurosymbolic_release_v1_audit.json`.

## Materialize an ALFWorld replay

Install ALFWorld separately, then set:

```bash
export ALFWORLD_CONFIG=/path/to/alfworld_base_config.yaml
export ALFWORLD_DATA=/path/to/alfworld_data
export ALFWORLD_PYTHON=/path/to/alfworld/python
```

Materialize the exact artifacts and a resource-path-retargeted replay config:

```bash
python scripts/materialize_four_domain_release_v1.py \
  --output-dir /tmp/four_domain_alfworld_replay
```

The command prints the exact target runner invocation. The generated replay config retains the frozen task IDs, source models, target neural grounder, thresholds, seed, policy, runner hash, development evidence hashes, and 70-step budget. It changes only resource and output paths and records that fact under `portable_reproduction`.

## Frozen environments

| Role | Python | NumPy | Additional packages |
|---|---:|---:|---|
| Candidate building / exact candidate reconstruction | 3.13.5 | 2.1.3 | scikit-learn 1.6.1 |
| ALFWorld target execution | 3.9.23 | 1.26.4 | ALFWorld 0.4.2, TextWorld 1.7.0 |
| Four-route evidence audit | 3.10+ | not used | none |

The environment split is intentional: NumPy RNG output differs between the builder and ALFWorld environments. The bundled candidate avoids silently rebuilding different weights in the target environment.

## Claim boundary

The bundle makes the four-route evidence audit portable and packages all large ALFWorld artifacts required for a target replay. It does not redistribute ALFWorld itself.

For WebShop, DiscoveryWorld, and TIR, the committed compact receipts and adapter hashes are portable and sufficient for evidence audit. Their provider-backed execution caches, benchmark installations, and full raw trajectories are not vendored. A full four-target clean-room re-execution therefore remains separate from the portable evidence audit.
