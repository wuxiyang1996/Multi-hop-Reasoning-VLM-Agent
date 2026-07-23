#!/usr/bin/env bash
# Dump 2 representative trajectories per model from /tmp/ab_test161 into a
# NFS-shared snapshot so a remote agent shell on a different node can Read it.
# Usage: bash cold_start/ab_test161_dump_traj.sh [OUT_BASE]
OUT_BASE="${1:-/tmp/ab_test161}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT="${SCRIPT_DIR}/_ab_test161_traj.snapshot"
exec > >(tee "$SNAPSHOT") 2>&1

python3 - <<PY
import json, os, sys
OUT = "${OUT_BASE}"
for tag in ("claude", "gemini", "qwen"):
    root = os.path.join(OUT, tag)
    if not os.path.isdir(root):
        print(f"=== {tag} : no dir ===\n")
        continue
    eps = []
    for d in sorted(os.listdir(root)):
        sub = os.path.join(root, d)
        ep_p = os.path.join(sub, 'episode_000.json')
        sm_p = os.path.join(sub, 'rollout_summary.json')
        if os.path.exists(ep_p) and os.path.exists(sm_p):
            try:
                sm = json.load(open(sm_p))
                steps = sm.get('mean_steps', 0)
                eps.append((sub, ep_p, steps))
            except Exception:
                pass
    eps.sort(key=lambda x: -x[2])
    print(f"=== {tag} (showing 2 of {len(eps)} completed; one longest, one shortest) ===")
    samples = []
    if eps: samples.append(eps[0])
    if len(eps) > 1: samples.append(eps[-1])
    for (sub, ep_p, steps) in samples:
        name = os.path.basename(sub)
        print(f"\n--- {tag} :: {name}  ({int(steps)} steps) ---")
        try:
            ep = json.load(open(ep_p))
        except Exception as e:
            print(f"  (load failed: {e})")
            continue
        for i, e in enumerate(ep.get('experiences', [])):
            a = e.get('action_str') or e.get('action') or ''
            if isinstance(a, str) and len(a) > 110: a = a[:110] + '...'
            r = e.get('reward', 0.0)
            reason = (e.get('reasoning') or '').strip().replace('\n', ' ')[:100]
            print(f"  step {i:2d}: r={r:+.2f}  {a}")
            if reason:
                print(f"          reason: {reason}")
    print()
PY
PY_EXIT=$?
echo
echo "=== snapshot written: $SNAPSHOT ==="
exit $PY_EXIT
