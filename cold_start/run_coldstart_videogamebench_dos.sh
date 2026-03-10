#!/usr/bin/env bash
#
# run_coldstart_videogamebench_dos.sh — Cold-start from VideoGameBench DOS games
#
# Generates 5 episodes per DOS game using GPT-5-mini (dummy language agent).
# Requires the videogamebench repo as a sibling and Playwright for the browser.
#
# Output: cold_start/output/videogamebench_dos/<game_name>/
#   episode_000.json .. episode_004.json, episode_buffer.json, rollouts.jsonl,
#   rollout_summary.json
#
# Usage:
#   bash cold_start/run_coldstart_videogamebench_dos.sh
#   bash cold_start/run_coldstart_videogamebench_dos.sh --episodes 3 --headless
#   bash cold_start/run_coldstart_videogamebench_dos.sh --games doom2 civ
#   bash cold_start/run_coldstart_videogamebench_dos.sh --list-games

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEOGAMEBENCH_ROOT="$(cd "$CODEBASE_ROOT/../videogamebench" 2>/dev/null && pwd || echo "")"

export PYTHONPATH="${CODEBASE_ROOT}:${VIDEOGAMEBENCH_ROOT:-}:${PYTHONPATH:-}"

if [ -z "$VIDEOGAMEBENCH_ROOT" ] || [ ! -d "$VIDEOGAMEBENCH_ROOT" ]; then
    echo "[ERROR] videogamebench repo not found at $CODEBASE_ROOT/../videogamebench"
    echo "        Clone it as a sibling of Game-AI-Agent for DOS cold-start."
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    OPENAI_API_KEY="$(python3 -c "
import sys; sys.path.insert(0, '${CODEBASE_ROOT}')
from api_keys import openai_api_key; print(openai_api_key)
" 2>/dev/null || echo "")"
    export OPENAI_API_KEY
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] OPENAI_API_KEY not set and not found in api_keys.py"
    echo "        export OPENAI_API_KEY='sk-...' and retry"
    exit 1
fi

echo "================================================================"
echo "  Cold-Start: VideoGameBench DOS (GPT-5-mini)"
echo "================================================================"
echo "  Codebase:   $CODEBASE_ROOT"
echo "  VideoGameBench: $VIDEOGAMEBENCH_ROOT"
echo "  API key:    ${OPENAI_API_KEY:0:12}..."
echo "================================================================"
echo ""

# Default: 5 episodes per DOS game, gpt-5-mini
EXTRA_ARGS=("$@")
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--episodes 5 --model gpt-5-mini --no_label --headless)
fi

python3 "${SCRIPT_DIR}/generate_cold_start_videogamebench_dos.py" "${EXTRA_ARGS[@]}"

OUTPUT_DIR="${SCRIPT_DIR}/output"
echo ""
echo "================================================================"
echo "  Post-Run Summary"
echo "================================================================"
DOS_DIR="${OUTPUT_DIR}/videogamebench_dos"
if [ -d "$DOS_DIR" ]; then
    for game_dir in "$DOS_DIR"/*/; do
        [ -d "$game_dir" ] || continue
        game="$(basename "$game_dir")"
        count=$(find "$game_dir" -maxdepth 1 -name 'episode_*.json' ! -name 'episode_buffer.json' 2>/dev/null | wc -l)
        printf "  %-25s %3d episodes\n" "$game" "$count"
    done
fi
echo "  Output:     $OUTPUT_DIR/videogamebench_dos/"
echo "================================================================"
