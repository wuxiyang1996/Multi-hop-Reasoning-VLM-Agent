# Five-repository reproducible workspace V1

The five historical directories are retained as exact Git commits, but there
is now one bootstrap and verification entry point.  This avoids merging
incompatible histories or silently selecting whichever dirty checkout happens
to be present on a machine.

## Frozen roles

| Directory | Role | Needed for V3 substitution |
|---|---|---:|
| `Multi-hop-Reasoning-VLM-Agent-two-agent-clean` | Canonical harness, frozen artifacts, evaluation and reports | yes |
| `Multi-hop-Reasoning-VLM-Agent-github-main` | LoRA target-module resolver and source-game runtime | yes |
| `Multi-hop-Reasoning-VLM-Agent-source-fresh-v1` | Optional source rollout regeneration | no |
| `Multi-hop-Reasoning-VLM-Agent-experiment-clean` | Archival experiment/DDP lineage | no |
| `Multi-hop-Reasoning-VLM-Agent` | Optional raw-video grounding tools | no |

The package contains one Git bundle for the canonical harness and one bundle
with the other four frozen branch refs.  Required result archives are unpacked
under the canonical repository.  The 15 GB model cache is optional so a server
can use its shared Hugging Face cache instead.

## Bootstrap on a server

Run these commands from the extracted package directory:

```bash
sha256sum -c SHA256SUMS
git clone -b agent/harness-transfer-bitter-lessons \
  00a-two-agent-clean-repository.bundle bootstrap
python bootstrap/scripts/bootstrap_reproducible_workspace_v1.py \
  --package "$PWD" \
  --workspace "$PWD/workspace"
python workspace/Multi-hop-Reasoning-VLM-Agent-two-agent-clean/scripts/verify_reproducible_workspace_v1.py \
  --workspace "$PWD/workspace" \
  --package "$PWD"
```

Use `--include-model-cache` during bootstrap only if the packaged Qwen3.5-9B
cache is required.  Otherwise set the job's `SHARED_HF_HOME` to a server-local
cache.

## Claim boundary

The verifier reproduces the historical frozen-cohort controller-substitution
and native-action-equivalence evidence for WebShop, ALFWorld, DiscoveryWorld,
TIRBench, CLEVRER and AGQA2.  It does not relabel those cohorts as the official
full benchmark sizes.  The manifest records both this boundary and the intended
full protocol sizes.
