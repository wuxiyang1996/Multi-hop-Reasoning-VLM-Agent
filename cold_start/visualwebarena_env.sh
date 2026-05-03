# cold_start/visualwebarena_env.sh — VisualWebArena URL endpoints.
#
# This file is the same shape as the one written by
# ``install/install_visualwebarena_sites.sh``, but written by hand on
# 2026-05-03 so the cross-model VWA baseline can launch without first
# blocking on the (~10 GB / 30 min one-shot) Classifieds + VWA-homepage
# install. As a result there are TWO known limitations versus the
# fully-installed setup:
#
#   1. ``VWA_CLASSIFIEDS`` points at :9980 but no container is bound
#      there yet, so the 232 ``classifieds``-only tasks (and 2
#      ``classifieds`` + ``shopping`` mixes) will fail at reset.
#      In our 200-task pinned subset that's 52 tasks. Filter them out
#      with ``grep -v classifieds`` upstream, or accept the 52 errors.
#
#   2. ``VWA_HOMEPAGE`` is *aliased* onto the WebArena homepage at
#      :4399. The shared shopping/reddit/wikipedia URLs (:7770/:9999/
#      :8888) work fine — they're literally the same containers — but
#      :4399 does NOT serve VWA's task-specific input images at
#      ``/static/input_images/<site>/task_<id>/input_<i>.png``. So any
#      task whose ``image`` config is non-empty (72/200 in the pinned
#      subset) will 404 inside ``_build_goal``. Filter to image-free
#      tasks for the smoke / first baseline.
#
#   To LIFT both limitations, run:
#       bash install/install_visualwebarena_sites.sh
#       # plus a separate VWA-homepage container that mounts
#       # ``static/input_images/`` (currently DIY, ~20 min).
#
# Source this from your shell or rely on
# ``cold_start/run_coldstart_actor_browsergym.sh`` to source it
# automatically when ``--tasks browsergym/visualwebarena.*`` is passed.

# ── Required by ``browsergym.visualwebarena`` (VWA_*-prefixed names) ─────
export VWA_HOMEPAGE="http://localhost:4399"
export VWA_SHOPPING="http://localhost:7770"
export VWA_REDDIT="http://localhost:9999"
export VWA_WIKIPEDIA="http://localhost:8888/viewer#wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export VWA_CLASSIFIEDS="http://localhost:9980"
export VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"

# Optional full-stack reset endpoint (used by VisualWebArenaInstance
# .full_reset() — ours is unset so the upstream code skips the call).
# export VWA_FULL_RESET="http://localhost:7565"

# ── Required by upstream ``visualwebarena`` Python pkg (un-prefixed) ─────
# The upstream module reads these eagerly at import time. Keeping them
# in sync with the VWA_* vars above is what makes ``import
# visualwebarena.browser_env.env_config`` succeed without a KeyError.
export DATASET="visualwebarena"
export HOMEPAGE="${VWA_HOMEPAGE}"
export SHOPPING="${VWA_SHOPPING}"
export REDDIT="${VWA_REDDIT}"
export WIKIPEDIA="${VWA_WIKIPEDIA}"
export CLASSIFIEDS="${VWA_CLASSIFIEDS}"
export CLASSIFIEDS_RESET_TOKEN="${VWA_CLASSIFIEDS_RESET_TOKEN}"
