#!/usr/bin/env bash
#
# install_visualwebarena_sites.sh — bring up the VisualWebArena self-hosted
# sites on top of the existing WebArena stack.
#
# VWA shares 4 services with WebArena (Shopping, Reddit, Wikipedia,
# Homepage). This script:
#   1. Asserts those WA services are already up (run install_webarena_sites.sh
#      first), and
#   2. Stands up the VWA-only Classifieds stack (OSClass) via docker compose.
#
# Sites used by VWA:
#   shopping       (shared with WA, :7770)
#   reddit/forum   (shared with WA, :9999)
#   wikipedia      (shared with WA, :8888)
#   homepage       (VWA-specific homepage, :4400 to avoid clash with WA :4399)
#   classifieds    (OSClass + MySQL, :9980)  ← VWA only
#
# Usage:
#   bash install/install_webarena_sites.sh                      # WA first
#   bash install/install_visualwebarena_sites.sh                # then VWA
#
# Env vars / overrides:
#   VWA_DATA_DIR                where to download classifieds compose archive
#                               (default: /workspace/visualwebarena_data)
#   VWA_HOST                    hostname/IP to expose endpoints under
#                               (default: localhost)
#   VWA_CLASSIFIEDS_RESET_TOKEN reset token for the VWA classifieds reset API
#                               (default: 4b61655535e7ed388f0d40a93600254c
#                               — the canonical token used in the VWA paper)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VWA_DATA_DIR="${VWA_DATA_DIR:-/workspace/visualwebarena_data}"
VWA_HOST="${VWA_HOST:-localhost}"
VWA_CLASSIFIEDS_RESET_TOKEN="${VWA_CLASSIFIEDS_RESET_TOKEN:-4b61655535e7ed388f0d40a93600254c}"
ENV_FILE="${CODEBASE_ROOT}/cold_start/visualwebarena_env.sh"

mkdir -p "$VWA_DATA_DIR"

CLASSIFIEDS_URL="https://archive.org/download/classifieds_docker_compose/classifieds_docker_compose.zip"
CLASSIFIEDS_ZIP="${VWA_DATA_DIR}/classifieds_docker_compose.zip"
CLASSIFIEDS_DIR="${VWA_DATA_DIR}/classifieds_docker_compose"

# ── Step 1: assert shared WA services are up ──────────────────────────────
assert_wa_running() {
    echo "==> Checking that WebArena shared services are running"
    local missing=()
    for svc in shopping forum kiwix33; do
        if ! docker ps --format '{{.Names}}' | grep -qx "$svc"; then
            missing+=("$svc")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "[ERROR] These shared services are not running: ${missing[*]}"
        echo "        Bring up WebArena first:"
        echo "          bash ${SCRIPT_DIR}/install_webarena_sites.sh shopping reddit wiki"
        echo "        (Classifieds itself does not depend on these, but the"
        echo "        VWA tasks reference them, so without them most VWA tasks"
        echo "        will fail.)"
        return 1
    fi
    echo "    [ok] shopping, forum, kiwix33 all running"
}

# ── Step 2: download + unzip classifieds compose archive ──────────────────
download_classifieds() {
    if [ -d "$CLASSIFIEDS_DIR" ] && [ -f "${CLASSIFIEDS_DIR}/docker-compose.yml" ]; then
        echo "==> Classifieds compose dir already present (${CLASSIFIEDS_DIR})"
        return 0
    fi
    echo "==> Downloading classifieds docker-compose archive"
    [ -f "$CLASSIFIEDS_ZIP" ] || \
        wget --tries=10 --timeout=120 -c -O "$CLASSIFIEDS_ZIP" "$CLASSIFIEDS_URL"
    if ! command -v unzip >/dev/null 2>&1; then
        echo "[ERROR] unzip is required: sudo apt-get install -y unzip"
        return 1
    fi
    unzip -o "$CLASSIFIEDS_ZIP" -d "$VWA_DATA_DIR" >/dev/null
    echo "    [unzipped] -> ${CLASSIFIEDS_DIR}"
}

# ── Step 3: bring up classifieds via docker compose ───────────────────────
up_classifieds() {
    echo "==> Starting Classifieds stack (OSClass + MySQL on :9980)"
    local compose="${CLASSIFIEDS_DIR}/docker-compose.yml"
    [ -f "$compose" ] || { echo "[ERROR] $compose missing"; return 1; }

    # Edit the docker-compose.yml so the CLASSIFIEDS env var (consumed by the
    # OSClass image) points at the right host:port and uses our reset token.
    # Idempotent — re-running the sed is safe.
    sed -i.bak \
        -e "s|CLASSIFIEDS=http://[^[:space:]]*|CLASSIFIEDS=http://${VWA_HOST}:9980|g" \
        -e "s|CLASSIFIEDS_RESET_TOKEN=[A-Za-z0-9]*|CLASSIFIEDS_RESET_TOKEN=${VWA_CLASSIFIEDS_RESET_TOKEN}|g" \
        "$compose"

    (cd "$CLASSIFIEDS_DIR" && docker compose up --build -d) 2>&1 | tail -20 | sed 's/^/    /'

    echo "    Waiting 30s for MySQL + OSClass to start..."
    sleep 30

    # Populate the canonical Craigslist-style fixture.
    if docker ps --format '{{.Names}}' | grep -qx classifieds_db; then
        echo "    [seed] populating osclass DB with osclass_craigslist.sql"
        docker exec classifieds_db mysql -u root -ppassword osclass \
            -e 'source docker-entrypoint-initdb.d/osclass_craigslist.sql' \
            2>&1 | sed 's/^/      /' || true
    fi
}

# ── Step 4: write env file ────────────────────────────────────────────────
write_env_file() {
    echo "==> Writing $ENV_FILE"
    cat > "$ENV_FILE" <<EOF
# Auto-generated by install/install_visualwebarena_sites.sh on $(date)
# Source this file to activate the VisualWebArena URL endpoints.
# Most VWA URLs are aliased onto the shared WebArena services.
export VWA_HOMEPAGE="http://${VWA_HOST}:4399"
export VWA_SHOPPING="http://${VWA_HOST}:7770"
export VWA_REDDIT="http://${VWA_HOST}:9999"
export VWA_WIKIPEDIA="http://${VWA_HOST}:8888/viewer#wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export VWA_CLASSIFIEDS="http://${VWA_HOST}:9980"
export VWA_CLASSIFIEDS_RESET_TOKEN="${VWA_CLASSIFIEDS_RESET_TOKEN}"
EOF
    echo "    Done. ${ENV_FILE}"
}

# ── Step 5: smoke test ────────────────────────────────────────────────────
smoke_test() {
    echo "==> Smoke testing VWA endpoints"
    for url_var_port in \
        "VWA_SHOPPING=7770" "VWA_REDDIT=9999" "VWA_WIKIPEDIA=8888" \
        "VWA_HOMEPAGE=4399" "VWA_CLASSIFIEDS=9980"; do
        var="${url_var_port%%=*}"
        port="${url_var_port##*=}"
        code=$(curl --max-time 15 -s -o /dev/null -w "%{http_code}" \
            "http://${VWA_HOST}:${port}" 2>&1 || echo "000")
        printf "    %-22s :%-5s -> HTTP %s\n" "$var" "$port" "$code"
    done
}

# ── Main ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  VisualWebArena Classifieds installer"
echo "================================================================"
echo "  Data dir:           $VWA_DATA_DIR"
echo "  Host:               $VWA_HOST"
echo "  Classifieds reset:  $VWA_CLASSIFIEDS_RESET_TOKEN"
echo "================================================================"

assert_wa_running       || exit 1
download_classifieds    || exit 1
up_classifieds          || exit 1
write_env_file
smoke_test

echo ""
echo "================================================================"
echo "  VisualWebArena install complete"
echo "================================================================"
echo "  Activate both stacks with:"
echo "    source ${CODEBASE_ROOT}/cold_start/webarena_env.sh"
echo "    source ${ENV_FILE}"
echo ""
echo "  Then run VWA tasks:"
echo "    bash cold_start/run_coldstart_actor_browsergym.sh \\"
echo "        --tasks browsergym/visualwebarena.0 \\"
echo "        --episodes 1 --max_steps 12 --save_frames -v"
echo "================================================================"
