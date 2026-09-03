#!/usr/bin/env python3
"""Freeze the final 120-video qualification of the core source executor."""

import scripts.freeze_agqa2_router_grounding_development_v1 as freezer


freezer.OUTPUT = freezer.REPO_ROOT / "configs/agqa2_source_executor_qualification_v13_selection.json"
freezer.COUNT = 120
freezer.NONCE = "agqa2-source-executor-final-qualification-v13-before-formal"
freezer.STATUS = "FROZEN_V72_QUALIFICATION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_OUTCOME_ACCESS"
freezer.SPLIT_NAME = "official_train_router_validation_source_executor_final_qualification_v13"


if __name__ == "__main__":
    freezer.main()
