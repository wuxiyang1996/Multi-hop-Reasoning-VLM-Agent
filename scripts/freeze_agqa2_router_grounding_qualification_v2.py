#!/usr/bin/env python3
"""Freeze a fresh V68 grounding qualification after V67 development."""

from pathlib import Path

import scripts.freeze_agqa2_router_grounding_development_v1 as freezer


freezer.OUTPUT = freezer.REPO_ROOT / "configs/agqa2_router_grounding_qualification_v2_selection.json"
freezer.NONCE = "agqa2-router-v2-grounding-qualification-v2-after-some-choice-fix"
freezer.STATUS = "FROZEN_V68_QUALIFICATION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_OUTCOME_ACCESS"
freezer.SPLIT_NAME = "official_train_router_validation_fresh_grounding_qualification"


if __name__ == "__main__":
    freezer.main()
