#!/usr/bin/env bash
# Live status of run_ab_test161_3models.sh sweep.
# Usage:  bash cold_start/ab_test161_status.sh [OUT_BASE]
#
# In addition to printing to stdout, a copy is written to the NFS-shared path
# ``cold_start/_ab_test161_status.snapshot`` so a remote agent shell on a
# different node (where /tmp is not shared) can ``Read`` the snapshot file.
OUT_BASE="${1:-/tmp/ab_test161}"
TOTAL_TASKS=161
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT="${SCRIPT_DIR}/_ab_test161_status.snapshot"
exec > >(tee "$SNAPSHOT") 2>&1

echo "================================================================"
echo "  AssistantBench test_feasible (n=${TOTAL_TASKS}) — multi-model"
echo "  OUT_BASE = $OUT_BASE"
echo "  Time     = $(date +%H:%M:%S)"
echo "================================================================"

# Note: ``pgrep -c X || echo 0`` is unsafe — when pgrep finds 0 matches it
# prints "0" *and* exits 1, so the ``|| echo 0`` appends another "0",
# producing "0\n0" which breaks downstream printf %d. Pipe to ``wc -l``
# instead (wc always exits 0 and a no-match pgrep prints nothing → "0").
active_py=$(pgrep -f 'generate_cold_start_actor_browsergym' 2>/dev/null | wc -l)
active_chrome=$(pgrep -x chrome 2>/dev/null | wc -l)
echo "  Active python procs:   ${active_py}"
echo "  Active chrome procs:   ${active_chrome}"
free -h | awk '/^Mem:/ {printf "  Memory:                %s used / %s total (%s avail)\n", $3, $2, $7}'
echo

for tag in claude gemini qwen; do
    out="${OUT_BASE}/${tag}"
    [[ -d "$out" ]] || { printf "  %-7s : (no output dir)\n" "$tag"; continue; }
    done_files=$(find "$out" -maxdepth 1 -name '_shard_*.done' 2>/dev/null | wc -l)
    summaries=$(find "$out" -mindepth 2 -name 'rollout_summary.json' 2>/dev/null)
    # ``grep -c . || echo 0`` is unsafe: when grep finds 0 matches it
    # returns "0" + exit-1, then the `||` appends another "0" → "0\n0" which
    # breaks both [[ -gt ]] and printf %d. Use ``printf … | wc -l`` instead.
    if [[ -z "$summaries" ]]; then
        n_completed=0
    else
        n_completed=$(printf '%s\n' "$summaries" | wc -l)
    fi
    if [[ "$n_completed" -gt 0 ]]; then
        stats=$(printf '%s\n' "$summaries" | python3 -c "
import json, os, re, sys
SEND   = re.compile(r'send_msg_to_user\\(\\s*([\"\\'])(?P<a>.*?)\\1\\s*\\)', re.DOTALL)
INFEAS = re.compile(r'report_infeasible\\(', re.DOTALL)
PLACE  = re.compile(r'<\\s*your\\s+answer\\s+here\\s*>|placeholder|TODO|YOUR ANSWER', re.IGNORECASE)
rs = nz = st = 0
ans = inf = trunc = placeholder = 0
n = 0
samples = []
for p in sys.stdin.read().splitlines():
    if not p: continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    rs += d.get('mean_reward', 0.0)
    if d.get('mean_reward', 0.0) > 0: nz += 1
    st += d.get('mean_steps', 0.0)
    n += 1
    # Look at the per-task episode_000.json next to the summary for answer
    ep_path = os.path.join(os.path.dirname(p), 'episode_000.json')
    pred = None; kind = 'truncated'
    try:
        ep = json.load(open(ep_path))
        for e in reversed(ep.get('experiences', [])):
            a = e.get('action_str') or e.get('action') or ''
            if not isinstance(a, str): continue
            m = SEND.search(a)
            if m:
                pred = m.group('a'); kind = 'send'; break
            if INFEAS.search(a):
                kind = 'infeasible'; break
    except Exception:
        pass
    if kind == 'send': ans += 1
    elif kind == 'infeasible': inf += 1
    else: trunc += 1
    if pred and PLACE.search(pred): placeholder += 1
    if len(samples) < 2 and pred:
        tid = d.get('target_payload', '?').replace('browsergym/assistantbench.', 'ab.')
        snip = pred[:60].replace('\\n', ' ')
        samples.append(f'        e.g. {tid}  ->  \"{snip}\"')
if n:
    print(f'meanR={rs/n:.3f}  steps={st/n:.1f}  '
          f'sent={ans}/{n}  infeasible={inf}  trunc={trunc}  placeholder={placeholder}')
    for s in samples: print(s)
")
    else
        stats="(no summaries yet)"
    fi
    printf "  %-7s : shards_done=%d/4  completed=%-3d\n" "$tag" "$done_files" "$n_completed"
    printf "%s\n" "$stats" | sed 's/^/    /'
done

echo
echo "  Per-shard:"
for tag in claude gemini qwen; do
    for sh in 00 01 02 03; do
        log="${OUT_BASE}/${tag}/_shard_${sh}.log"
        [[ -f "$log" ]] || continue
        sentinel="${OUT_BASE}/${tag}/_shard_${sh}.done"
        if [[ -f "$sentinel" ]]; then
            status="DONE  $(cat "$sentinel")"
        else
            # See note on ``grep -c . || echo 0`` above; same trap here.
            n_targets=$(grep -c '^  TARGET (task):' "$log" 2>/dev/null)
            n_targets="${n_targets:-0}"
            [[ "$n_targets" =~ ^[0-9]+$ ]] || n_targets=0
            last=$(grep '^  TARGET (task):' "$log" 2>/dev/null | tail -1 \
                   | sed 's|^.*: browsergym/||')
            status="task #${n_targets}  cur=${last:-?}"
        fi
        printf "    %-7s sh%s : %s\n" "$tag" "$sh" "$status"
    done
done
